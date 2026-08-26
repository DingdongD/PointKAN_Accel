#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import font_manager


REPO_ROOT = Path(__file__).resolve().parents[1]
MN40_DIR = REPO_ROOT / "PointKAN-elite" / "classification_ModelNet40"
for path in (str(MN40_DIR), str(MN40_DIR.parent), str(Path(__file__).resolve().parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

from evaluate_specfps_corrected_multidataset import _auto_kan_save_dir  # type: ignore
from make_specfps_corrected_summary_and_plots import segfilter_trace  # type: ignore
from scanobjnn_specfps_scale_sweep import exact_fps_radii_sq  # type: ignore
from spec_kan_fps import Config  # type: ignore
from spec_kan_fps_multidataset import build_dataset_pair, finalize_preset_paths  # type: ignore
from spec_kan_fps_opt_kan import (  # type: ignore
    kan_prefix_len_from_K,
    load_kan_radii_model,
    predict_full_radii_squared_from_exact,
)


class _PrintLogger:
    def info(self, msg: str, *args) -> None:
        if args:
            msg = msg % args
        print(msg)


def setup_matplotlib(fontsize: int = 24) -> None:
    candidates = [
        "/root/.fonts/Arial.ttf",
        "/root/.fonts/arial.ttf",
        "/root/.fonts/ARIALN.TTF",
        "/root/.fonts/arialbd.ttf",
    ]
    font_path = next((fp for fp in candidates if os.path.exists(fp)), None)
    if font_path:
        font_manager.fontManager.addfont(font_path)
        family = font_manager.FontProperties(fname=font_path).get_name()
    else:
        family = "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize - 2,
            "ytick.labelsize": fontsize - 2,
            "legend.fontsize": fontsize - 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def make_dataset_args(dataset: str, num_points: int) -> argparse.Namespace:
    args = argparse.Namespace(
        dataset=dataset,
        partition="test",
        preset="voxelnet",
        data_root="",
        shapenet_root="",
        shapenet_normalize=False,
        scanobjectnn_train_h5="",
        scanobjectnn_test_h5="",
        num_points=num_points,
        K=512,
        kitti_use_roi=False,
        roi_x0=0.0,
        roi_x1=70.4,
        roi_y0=-40.0,
        roi_y1=40.0,
        roi_z0=-3.0,
        roi_z1=1.0,
        outdoor_train_ratio=0.8,
        outdoor_eval_seed=12345,
        max_list_files=0,
        max_train_files=0,
        max_test_files=0,
    )
    finalize_preset_paths(args)
    return args


def modelnet_label_index(category_name: str) -> int:
    shape_names_path = Path("/workspace/VoxelNet/pc_data/modelnet40_ply_hdf5_2048/shape_names.txt")
    names = [line.strip() for line in shape_names_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    normalized = category_name.strip().lower()
    if normalized not in names:
        raise ValueError(f"ModelNet40 category {category_name!r} not found")
    return names.index(normalized)


def modelnet_label_name(label: int) -> str:
    shape_names_path = Path("/workspace/VoxelNet/pc_data/modelnet40_ply_hdf5_2048/shape_names.txt")
    names = [line.strip() for line in shape_names_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return names[label] if 0 <= label < len(names) else f"label_{label}"


def choose_sample(dataset: str, test_ds, sample_index: int, modelnet_category: str) -> int:
    if dataset != "modelnet40" or not modelnet_category:
        return int(sample_index)
    target = modelnet_label_index(modelnet_category)
    labels = None
    if hasattr(test_ds, "ds") and hasattr(test_ds.ds, "label"):
        labels = np.asarray(test_ds.ds.label).reshape(-1)
    elif hasattr(test_ds, "label"):
        labels = np.asarray(test_ds.label).reshape(-1)
    if labels is None:
        raise RuntimeError("Cannot inspect ModelNet40 labels")
    hits = np.where(labels == target)[0]
    if hits.size == 0:
        raise RuntimeError(f"No ModelNet40 test sample found for {modelnet_category!r}")
    return int(hits[0])


def load_sample(dataset: str, num_points: int, sample_index: int, modelnet_category: str) -> Tuple[torch.Tensor, int, str, str, int]:
    args = make_dataset_args(dataset, num_points)
    cfg = Config()
    cfg.num_points = num_points
    _, test_ds, tag = build_dataset_pair(args, cfg, _PrintLogger())
    sample_index = choose_sample(tag, test_ds, sample_index, modelnet_category)
    pts, label = test_ds[sample_index]
    label = int(np.asarray(label).reshape(-1)[0])
    if tag == "modelnet40":
        category = modelnet_label_name(label)
    elif tag == "shapenet_part" and hasattr(test_ds, "datapath"):
        category, _path = test_ds.datapath[sample_index]
    else:
        category = f"label_{label}"
    if not torch.is_tensor(pts):
        pts = torch.as_tensor(pts)
    return pts[:, :3].float().contiguous(), label, tag, str(category), sample_index


@torch.no_grad()
def evaluate_trace(
    dataset: str,
    num_points: int,
    sample_index: int,
    modelnet_category: str,
    K: int,
    segment_len: int,
    prefix_ratio: float,
    seed: str,
    device: torch.device,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    pts_cpu, label, tag, category, resolved_sample_index = load_sample(dataset, num_points, sample_index, modelnet_category)
    pts = pts_cpu.unsqueeze(0).to(device=device, dtype=torch.float32)
    idx_exact, radii_exact = exact_fps_radii_sq(pts, K, seed=seed)
    kan_dir = _auto_kan_save_dir(tag, K, prefix_ratio)
    model, cfg = load_kan_radii_model(kan_dir, device)
    cfg.prefix_ratio = float(prefix_ratio)
    s0 = kan_prefix_len_from_K(K, prefix_ratio)
    radii_kan = predict_full_radii_squared_from_exact(radii_exact, model, cfg, device)
    idx_spec, trace = segfilter_trace(pts, K, radii_kan, segment_len, s0, seed=seed)

    exact_r = torch.sqrt(radii_exact[0].clamp_min(0)).detach().cpu().numpy()
    pred_r = torch.sqrt(radii_kan[0].clamp_min(0)).detach().cpu().numpy()
    trace["dataset"] = tag
    trace["sample_index"] = resolved_sample_index
    trace["label"] = label
    trace["category"] = category
    trace["num_points"] = int(pts.size(1))
    trace["K"] = K
    trace["segment_len"] = segment_len
    trace["seg_filter_start_t"] = s0
    trace["exact_radius"] = exact_r[: len(trace)]
    trace["kan_pred_radius"] = pred_r[: len(trace)]
    trace["raw_candidates_plot"] = trace["raw_candidates"].where((trace["iter"] >= s0) & (trace["raw_candidates"] > 0), np.nan)
    trace["actual_candidates_plot"] = trace["actual_candidates"].where(trace["iter"] >= s0, trace["remaining_points"])
    active = trace[trace["iter"] >= s0]
    meta = {
        "dataset": tag,
        "sample_index": resolved_sample_index,
        "label": label,
        "category": category,
        "num_points": int(pts.size(1)),
        "K": K,
        "segment_len": segment_len,
        "seg_filter_start_t": s0,
        "fallback_rate": float(active["fallback"].mean()),
        "fallback_steps": int(active["fallback"].sum()),
        "active_spec_steps": int(len(active)),
        "raw_candidates_mean": float(active["raw_candidates"].mean()),
        "actual_candidates_mean": float(active["actual_candidates"].mean()),
        "fps_iou": float(len(set(idx_exact[0].detach().cpu().tolist()) & set(idx_spec[0].detach().cpu().tolist())) / len(set(idx_exact[0].detach().cpu().tolist()) | set(idx_spec[0].detach().cpu().tolist()))),
    }
    return trace, meta


def plot_trace(trace: pd.DataFrame, meta: Dict[str, float], out_stem: Path) -> None:
    setup_matplotlib(24)
    x = trace["iter"].to_numpy()
    s0 = int(meta["seg_filter_start_t"])
    raw_mean = float(meta["raw_candidates_mean"])
    actual_mean = float(meta["actual_candidates_mean"])
    fallback_pct = 100.0 * float(meta["fallback_rate"])

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.2), sharex=True)
    axes[0].plot(x, trace["exact_radius"], color="#222222", linewidth=3.0, label="Exact radius")
    axes[0].plot(x, trace["kan_pred_radius"], color="#00b0f0", linewidth=2.8, linestyle="--", label="KAN pred")
    axes[0].axvspan(0, s0, color="#d9d9d9", alpha=0.35, label="Warmup")
    axes[0].axvline(s0, color="#444444", linewidth=2.0, linestyle=":", label="KAN start")
    axes[0].text(
        0.985,
        0.08,
        f"{meta['category']}, N={int(meta['num_points'])}",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=20,
    )
    axes[0].set_ylabel("Radius")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)
    axes[0].legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.04), fontsize=14)

    axes[1].plot(trace["iter"], trace["actual_candidates_plot"], color="#ffc000", linewidth=2.8, label="Eff. Cand.")
    axes[1].plot(trace["iter"], trace["raw_candidates_plot"], color="#00b0f0", linewidth=2.8, linestyle="--", label="Raw Cand.")
    axes[1].axhline(actual_mean, color="#d62728", linewidth=2.4, linestyle="-.", label=f"Eff. mean={actual_mean:.1f}")
    axes[1].axhline(raw_mean, color="#555555", linewidth=2.0, linestyle=":", label=f"Raw mean={raw_mean:.1f}")
    fb = trace[(trace["iter"] >= s0) & (trace["fallback"] > 0)]
    if not fb.empty:
        axes[1].scatter(
            fb["iter"],
            fb["actual_candidates_plot"],
            marker="x",
            s=70,
            linewidths=2.2,
            color="#d62728",
            label=f"FB={fallback_pct:.1f}%",
            zorder=5,
        )
    else:
        axes[1].plot([], [], marker="x", linestyle="None", color="#d62728", label="FB=0.0%")
    axes[1].axvspan(0, s0, color="#d9d9d9", alpha=0.35)
    axes[1].axvline(s0, color="#444444", linewidth=2.0, linestyle=":")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Cand. count")
    axes[1].set_xlabel("FPS iteration")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    axes[1].legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.04), fontsize=13)
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04, transparent=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot ModelNet40 and ShapeNet speculative FPS threshold gap traces.")
    p.add_argument("--num_points", type=int, default=2048)
    p.add_argument("--K", type=int, default=512)
    p.add_argument("--segment_len", type=int, default=32)
    p.add_argument("--prefix_ratio", type=float, default=0.10)
    p.add_argument("--sample_index", type=int, default=0)
    p.add_argument("--modelnet_category", default="chair")
    p.add_argument("--seed", choices=("first", "max_norm"), default="first")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", default=str(REPO_ROOT / "my_progressive_out" / "speculative_fps_process"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    summaries = []
    jobs = [
        ("modelnet40", f"speculative_fps_threshold_gap_modelnet40_{args.modelnet_category}_N{args.num_points}"),
        ("shapenet_part", f"speculative_fps_threshold_gap_shapenet_sample{args.sample_index}_N{args.num_points}"),
    ]
    for dataset, stem in jobs:
        trace, meta = evaluate_trace(
            dataset=dataset,
            num_points=args.num_points,
            sample_index=args.sample_index,
            modelnet_category=args.modelnet_category,
            K=args.K,
            segment_len=args.segment_len,
            prefix_ratio=args.prefix_ratio,
            seed=args.seed,
            device=device,
        )
        out_stem = Path(args.out_dir) / stem
        trace.to_csv(out_stem.with_suffix(".csv"), index=False)
        plot_trace(trace, meta, out_stem)
        summaries.append(meta)
        print(
            f"{meta['dataset']} sample={meta['sample_index']} category={meta['category']} "
            f"N={meta['num_points']} L={meta['segment_len']} fallback={100.0 * meta['fallback_rate']:.2f}% "
            f"IoU={meta['fps_iou']:.6f}"
        )
        print("WROTE", out_stem.with_suffix(".pdf"))
        print("WROTE", out_stem.with_suffix(".png"))
        print("WROTE", out_stem.with_suffix(".csv"))
    pd.DataFrame(summaries).to_csv(Path(args.out_dir) / "speculative_fps_threshold_gap_modelnet_shapenet_summary.csv", index=False)


if __name__ == "__main__":
    main()
