#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MN40_DIR = os.path.join(REPO_ROOT, "PointKAN-elite", "classification_ModelNet40")
for path in (MN40_DIR, os.path.dirname(MN40_DIR), os.path.dirname(__file__)):
    if path not in sys.path:
        sys.path.insert(0, path)

from scannet_ply_specfps_scale_sweep import pc_normalize_np, read_scannet_xyz_ply  # type: ignore
from scanobjnn_specfps_scale_sweep import (  # type: ignore
    exact_fps_radii_sq,
    segfilter_stats_remaining_denominator,
    set_iou,
)
from spec_kan_fps_opt_kan import (  # type: ignore
    kan_prefix_len_from_K,
    load_kan_radii_model,
    predict_full_radii_squared_from_exact,
)


def setup_matplotlib(fontsize: int = 36):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    candidate_fonts = [
        "/root/.fonts/Arial.ttf",
        "/root/.fonts/arial.ttf",
        "/root/.fonts/ARIALN.TTF",
        "/root/.fonts/arialbd.ttf",
    ]
    font_path = next((fp for fp in candidate_fonts if os.path.exists(fp)), None)
    if font_path:
        font_manager.fontManager.addfont(font_path)
        arial_name = font_manager.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = arial_name
    plt.rcParams.update({
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": fontsize,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })
    return plt


def choose_source_path(ply_root: str, preferred: str = "") -> str:
    if preferred and os.path.exists(preferred):
        return preferred
    paths = sorted(glob.glob(os.path.join(ply_root, "*/scene*_vh_clean_2.ply")))
    if not paths:
        raise FileNotFoundError(f"No ScanNet PLY files found under {ply_root}")
    best_path = ""
    best_count = -1
    for path in paths:
        try:
            count = read_scannet_xyz_ply(path).shape[0]
        except Exception:
            continue
        if count > best_count:
            best_path = path
            best_count = count
    if not best_path:
        raise FileNotFoundError(f"No readable ScanNet PLY files found under {ply_root}")
    return best_path


def fixed_nested_points(source_path: str, max_points: int, seed: int) -> Dict[int, np.ndarray]:
    xyz = pc_normalize_np(read_scannet_xyz_ply(source_path))
    if xyz.shape[0] < max_points:
        raise ValueError(f"{source_path} has only {xyz.shape[0]} vertices, less than max_points={max_points}")
    rng = np.random.default_rng(seed)
    choice = rng.choice(xyz.shape[0], size=max_points, replace=False)
    sampled = xyz[choice].astype(np.float32, copy=True)
    return {n: sampled[:n].copy() for n in [1024, 2048, 4096, 8192, 16384, 32768] if n <= max_points}


@torch.no_grad()
def run_sweep(args) -> pd.DataFrame:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = load_kan_radii_model(args.kan_save_dir, device)
    cfg.prefix_ratio = float(args.prefix_ratio)

    source_path = choose_source_path(args.ply_root, args.source_path)
    points_by_n = fixed_nested_points(source_path, max(args.n_values), args.sample_seed)
    source_vertices = int(read_scannet_xyz_ply(source_path).shape[0])
    s0 = kan_prefix_len_from_K(args.K, args.prefix_ratio)

    rows: List[Dict[str, float]] = []
    for n in args.n_values:
        pts_np = points_by_n[n]
        pts = torch.from_numpy(pts_np).unsqueeze(0).to(device=device, dtype=torch.float32)
        idx_exact, radii_exact = exact_fps_radii_sq(pts, args.K, seed=args.seed)
        radii_kan = predict_full_radii_squared_from_exact(radii_exact, model, cfg, device)

        segment_lens = args.segment_lens if getattr(args, "segment_lens", None) else [int(np.ceil(float(args.K) / float(segcount))) for segcount in args.segcounts]
        for segment_len in segment_lens:
            segcount = int(np.ceil(float(args.K) / float(segment_len)))
            idx_spec, stats = segfilter_stats_remaining_denominator(
                pts,
                args.K,
                radii_kan,
                segment_len,
                s0,
                seed=args.seed,
            )
            rows.append({
                "dataset": "scannet_ply_fixed_sample",
                "source_mode": "scannet_ply",
                "source_path": source_path,
                "source_vertices": source_vertices,
                "num_points": int(n),
                "K": int(args.K),
                "prefix_ratio": float(args.prefix_ratio),
                "seg_filter_start_t": int(s0),
                "segcount": int(segcount),
                "segment_len": int(segment_len),
                "fallback_rate": float(stats["fallback_rate"]),
                "fallback_steps": int(stats["fallback_steps"]),
                "active_spec_steps": int(stats["active_spec_steps"]),
                "raw_candidates_mean": float(stats["raw_candidates_mean"]),
                "actual_candidates_mean": float(stats["actual_candidates_mean"]),
                "raw_candidate_rate_remaining": float(stats["raw_candidate_rate_remaining"]),
                "actual_candidate_rate_remaining": float(stats["actual_candidate_rate_remaining"]),
                "fps_iou": float(set_iou(idx_exact, idx_spec)[0]),
            })
    return pd.DataFrame(rows)


def plot_results(df: pd.DataFrame, out_stem: str, fontsize: int = 36) -> None:
    plt = setup_matplotlib(fontsize)
    import matplotlib.ticker as mticker

    fig, axes = plt.subplots(1, 3, figsize=(28, 8.6))
    colors = ["#d9f2d0", "#ffc000", "#00b0f0", "#7030a0", "#c00000", "#7f7f7f"]
    metrics = [
        ("fallback_rate", "Fallback rate (%)", 100.0),
        ("raw_candidate_rate_remaining", "Candidate rate (%)", 100.0),
        ("fps_iou", "IoU", 1.0),
    ]

    for ax, (metric, ylabel, scale) in zip(axes, metrics):
        for color, segment_len in zip(colors, sorted(df["segment_len"].unique())):
            sub = df[df["segment_len"] == segment_len].sort_values("num_points")
            y = sub[metric].to_numpy(dtype=float) * scale
            ax.plot(
                sub["num_points"],
                y,
                marker="o",
                markersize=12,
                linewidth=4.0,
                color=color,
                label=f"{int(segment_len)}",
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(df["num_points"].unique()))
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.set_xlabel("#Number of Point Clouds")
        ax.set_ylabel(ylabel, labelpad=2 if ax is axes[1] else 8)
        ax.grid(which="major", axis="both", linestyle="--", alpha=0.45, linewidth=1.2)
        ax.grid(which="minor", axis="both", linestyle=":", alpha=0.28, linewidth=0.9)
        ax.minorticks_on()
        for spine in ax.spines.values():
            spine.set_linewidth(3.0)
        ax.tick_params(width=3.0, length=8)

    axes[2].set_ylim(0.0, 1.05)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        [f"L={label}" for label in labels],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(labels),
        frameon=False,
        columnspacing=0.8,
        handlelength=1.6,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    fig.subplots_adjust(wspace=0.28)
    fig.savefig(out_stem + ".pdf", bbox_inches="tight", pad_inches=0.03, transparent=True)
    fig.savefig(out_stem + ".png", bbox_inches="tight", pad_inches=0.03, dpi=600)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Fixed real ScanNet PLY sample speculative FPS segment length sweep.")
    p.add_argument("--ply_root", default="/workspace/VoxelNet/scannet/scans_test")
    p.add_argument("--source_path", default="/workspace/VoxelNet/scannet/scans_test/scene0763_00/scene0763_00_vh_clean_2.ply")
    p.add_argument("--kan_save_dir", default="/workspace/PointKAN-pytorch/PointKAN-elite/classification_ModelNet40/runs/fps_radii_oneclick_all/fps_radii_kan_K512_scanobjectnn_pr0.10")
    p.add_argument("--n_list", default="1024,2048,4096,8192,16384,32768")
    p.add_argument("--segcounts", default="2,4,8,16,32")
    p.add_argument("--segment_lens", default="", help="Comma-separated segment_len values. If set, overrides segcounts.")
    p.add_argument("--K", type=int, default=512)
    p.add_argument("--prefix_ratio", type=float, default=0.10)
    p.add_argument("--seed", choices=("first", "max_norm"), default="first")
    p.add_argument("--sample_seed", type=int, default=123)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fontsize", type=int, default=36)
    p.add_argument("--out_dir", default="/workspace/PointKAN-pytorch/my_progressive_out/scannet_fixed_sample_segcount_sweep")
    args = p.parse_args()
    args.n_values = [int(x) for x in args.n_list.split(",") if x.strip()]
    args.segcounts = [int(x) for x in args.segcounts.split(",") if x.strip()]
    args.segment_lens = [int(x) for x in args.segment_lens.split(",") if x.strip()] if args.segment_lens else []
    return args


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    df = run_sweep(args)
    stem = f"fixed_sample_K{args.K}_segment_len_sweep" if getattr(args, "segment_lens", None) else f"fixed_sample_K{args.K}_segcount_sweep"
    csv_path = os.path.join(args.out_dir, stem + ".csv")
    df.to_csv(csv_path, index=False)
    plot_results(df, os.path.join(args.out_dir, stem), fontsize=args.fontsize)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("WROTE", csv_path)
    print("WROTE", os.path.join(args.out_dir, stem + ".pdf"))
    print("WROTE", os.path.join(args.out_dir, stem + ".png"))


if __name__ == "__main__":
    main()
