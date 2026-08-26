#!/usr/bin/env python3
from __future__ import annotations

import csv
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import font_manager
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[1]
MN40_DIR = REPO_ROOT / "PointKAN-elite" / "classification_ModelNet40"
for path in (str(MN40_DIR), str(MN40_DIR.parent), str(Path(__file__).resolve().parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

from scanobjnn_specfps_scale_sweep import exact_fps_radii_sq, set_iou  # type: ignore
from scannet_ply_specfps_scale_sweep import pc_normalize_np, read_scannet_xyz_ply  # type: ignore
from spec_kan_fps_opt_kan import (  # type: ignore
    kan_prefix_len_from_K,
    load_kan_radii_model,
    predict_full_radii_squared_from_exact,
)


OUT_MULTI = REPO_ROOT / "my_progressive_out" / "specfps_corrected_multidataset"
OUT_SCALE = REPO_ROOT / "my_progressive_out" / "scannet_ply_specfps_scale_sweep_64files"
OUT_PROCESS = REPO_ROOT / "my_progressive_out" / "speculative_fps_process"
SCANOBJ_H5 = Path("/workspace/VoxelNet/scannet_kan/main_split/test_objectdataset_augmentedrot_scale75.h5")
KAN_SAVE_DIR = REPO_ROOT / "PointKAN-elite" / "classification_ModelNet40" / "runs" / "fps_radii_oneclick_all" / "fps_radii_kan_K512_scanobjectnn_pr0.10"
PLY_ROOT = Path("/workspace/VoxelNet/scannet/scans_test")


def setup_matplotlib(fontsize: int = 28) -> None:
    candidates = [
        "/root/.fonts/Arial.ttf",
        "/root/.fonts/arial.ttf",
        "/root/.fonts/ARIALN.TTF",
        "/root/.fonts/arialbd.ttf",
    ]
    font_path = next((fp for fp in candidates if os.path.exists(fp)), None)
    if font_path is not None:
        font_manager.fontManager.addfont(font_path)
        family = font_manager.FontProperties(fname=font_path).get_name()
    else:
        family = "DejaVu Sans"
    plt.rcParams.update({
        "font.family": family,
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize - 2,
        "ytick.labelsize": fontsize - 2,
        "legend.fontsize": fontsize - 4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })


def save_fig(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04, transparent=True)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def make_summary_csv() -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for path in sorted(OUT_MULTI.glob("corrected_*_test_N*_K512_seg32.csv")):
        rows.append(pd.read_csv(path))
    if not rows:
        raise FileNotFoundError(f"No corrected CSV files found in {OUT_MULTI}")
    df = pd.concat(rows, ignore_index=True)
    dataset_map = {
        "modelnet40": "ModelNet40",
        "shapenet_part": "ShapeNetPart",
        "scanobjectnn": "ScanObjectNN",
    }
    df["dataset_name"] = df["dataset"].map(dataset_map).fillna(df["dataset"])
    df["fallback_rate_pct"] = df["fallback_rate"] * 100.0
    df["raw_candidate_rate_remaining_pct"] = df["raw_candidate_rate_remaining"] * 100.0
    df["actual_candidate_rate_remaining_pct"] = df["actual_candidate_rate_remaining"] * 100.0
    cols = [
        "dataset_name",
        "dataset",
        "partition",
        "num_points",
        "branch",
        "num_samples",
        "K",
        "segment_len",
        "seg_filter_start_t",
        "active_spec_steps",
        "fallback_steps",
        "fallback_rate",
        "fallback_rate_pct",
        "raw_candidates_mean",
        "actual_candidates_mean",
        "raw_candidate_rate_remaining",
        "raw_candidate_rate_remaining_pct",
        "actual_candidate_rate_remaining",
        "actual_candidate_rate_remaining_pct",
        "fps_iou_mean",
        "kan_save_dir",
    ]
    df = df[[c for c in cols if c in df.columns]]
    out = OUT_MULTI / "corrected_fallback_summary_modelnet_shapenet_scanobjectnn.csv"
    df.to_csv(out, index=False)
    print(f"WROTE {out}")
    return df


def plot_scanobjectnn_scale(summary_df: pd.DataFrame) -> None:
    setup_matplotlib(30)
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    h5 = summary_df[(summary_df["dataset"] == "scanobjectnn") & (summary_df["branch"] == "kan")].sort_values("num_points")
    ax.plot(
        h5["num_points"],
        h5["fallback_rate_pct"],
        marker="o",
        markersize=12,
        linewidth=3.2,
        label="ScanObjectNN h5",
        color="#00b0f0",
    )
    scale_csv = OUT_SCALE / "scannet_ply_specfps_scale_K512_seg32_files64.csv"
    if scale_csv.exists():
        ply = pd.read_csv(scale_csv)
        ply = ply[ply["branch"] == "kan"].sort_values("num_points")
        ax.plot(
            ply["num_points"],
            ply["fallback_rate"] * 100.0,
            marker="s",
            markersize=11,
            linewidth=3.2,
            label="Real ScanNet PLY",
            color="#ffc000",
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks([1024, 2048, 4096, 8192, 16384, 32768])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Input points")
    ax.set_ylabel("Fallback rate (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    for spine in ax.spines.values():
        spine.set_linewidth(2.2)
    ax.legend(frameon=False, loc="upper right")
    save_fig(fig, OUT_PROCESS / "scanobjectnn_corrected_fallback_scale_trend")


@torch.no_grad()
def segfilter_trace(
    X: torch.Tensor,
    K: int,
    radii_ref_sq: torch.Tensor,
    segment_len: int,
    seg_filter_start_t: int,
    seed: str = "first",
    upper_mode: str = "none",
) -> Tuple[torch.Tensor, pd.DataFrame]:
    B, N, _ = X.shape
    if B != 1:
        raise ValueError("segfilter_trace expects one sample")
    K = min(int(K), int(N))
    device, dtype = X.device, X.dtype
    x2 = (X * X).sum(dim=-1)
    idx = torch.zeros((B, K), dtype=torch.long, device=device)
    min_dist = torch.full((B, N), float("inf"), dtype=dtype, device=device)
    selected = torch.zeros((B, N), dtype=torch.bool, device=device)
    batch = torch.arange(B, device=device)
    L = max(1, int(segment_len))
    s0 = max(1, min(int(seg_filter_start_t), K))
    if seed == "first":
        farthest = torch.zeros(B, dtype=torch.long, device=device)
    elif seed == "max_norm":
        farthest = torch.argmax(x2, dim=1)
    else:
        raise ValueError(f"Unsupported seed: {seed}")
    rows: List[Dict[str, float]] = []

    for t in range(K):
        raw_count = float("nan")
        actual_count = float("nan")
        remaining = float((~selected).sum().item())
        fallback = False
        tau = float("nan")
        tau_high = float("nan")
        if t == 0:
            pass
        elif t < s0:
            cand = min_dist.masked_fill(selected, -float("inf"))
            farthest = torch.argmax(cand, dim=1)
        else:
            j = t // L
            seg_end = min((j + 1) * L - 1, K - 1)
            tau_tensor = radii_ref_sq[:, seg_end].to(device=device, dtype=dtype)
            tau = float(tau_tensor[0].item())
            valid = ~selected
            mask = (min_dist >= tau_tensor.unsqueeze(1)) & valid
            if upper_mode == "prev_segment":
                prev_end = max(j * L - 1, 0)
                tau_high_tensor = radii_ref_sq[:, prev_end].to(device=device, dtype=dtype)
                tau_high_tensor = torch.maximum(tau_high_tensor, tau_tensor)
                tau_high = float(tau_high_tensor[0].item())
                mask = mask & (min_dist <= tau_high_tensor.unsqueeze(1))
            elif upper_mode != "none":
                raise ValueError(f"Unsupported upper_mode: {upper_mode}")
            raw_tensor = mask.sum(dim=1).to(dtype)
            raw_count = float(raw_tensor[0].item())
            fallback = raw_count <= 0
            actual_count = raw_count if not fallback else remaining
            if fallback:
                mask = valid
            cand = min_dist.masked_fill(~mask, -float("inf"))
            farthest = torch.argmax(cand, dim=1)

        idx[:, t] = farthest
        selected[batch, farthest] = True
        C = X[batch, farthest]
        c2 = (C * C).sum(dim=-1, keepdim=True)
        d2 = (x2 + c2 - 2.0 * (X @ C.unsqueeze(-1)).squeeze(-1)).clamp_min_(0.0)
        min_dist = torch.minimum(min_dist, d2)
        min_dist[selected] = 0.0
        rows.append({
            "iter": t,
            "raw_candidates": raw_count,
            "actual_candidates": actual_count,
            "remaining_points": remaining,
            "fallback": int(fallback),
            "tau_sq": tau,
            "tau_high_sq": tau_high,
            "tau_radius": float(np.sqrt(max(tau, 0.0))) if np.isfinite(tau) else np.nan,
            "tau_high_radius": float(np.sqrt(max(tau_high, 0.0))) if np.isfinite(tau_high) else np.nan,
        })
    return idx, pd.DataFrame(rows)


def load_scanobjectnn_sample(num_points: int = 2048, index: int = 0) -> np.ndarray:
    with h5py.File(SCANOBJ_H5, "r") as f:
        pts = f["data"][index, :num_points, :3].astype(np.float32)
    return pts


def sample_ply_points(num_points: int, seed: int = 123) -> Tuple[np.ndarray, str]:
    paths = sorted(glob.glob(str(PLY_ROOT / "*" / "scene*_vh_clean_2.ply")))
    if not paths:
        raise FileNotFoundError(f"No real ScanNet PLY files found under {PLY_ROOT}")
    xyz = pc_normalize_np(read_scannet_xyz_ply(paths[0]))
    rng = np.random.default_rng(seed + num_points)
    choice = rng.choice(xyz.shape[0], size=num_points, replace=xyz.shape[0] < num_points)
    return xyz[choice].astype(np.float32), paths[0]


def ply_vertex_count(path: str) -> int:
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Malformed PLY header: {path}")
            s = line.decode("latin1").strip()
            if s.startswith("element vertex"):
                return int(s.split()[-1])
            if s == "end_header":
                break
    raise ValueError(f"No vertex count in PLY: {path}")


def choose_distinct_ply_by_scale(n_values: List[int]) -> Dict[int, Tuple[str, int]]:
    paths = sorted(glob.glob(str(PLY_ROOT / "*" / "scene*_vh_clean_2.ply")))
    if not paths:
        raise FileNotFoundError(f"No real ScanNet PLY files found under {PLY_ROOT}")
    meta = [(p, ply_vertex_count(p)) for p in paths]
    chosen: Dict[int, Tuple[str, int]] = {}
    used = set()
    for n in sorted(n_values):
        candidates = [m for m in meta if m[0] not in used]
        if not candidates:
            candidates = meta
            used.clear()
        path, count = min(candidates, key=lambda item: (abs(item[1] - n), item[1]))
        chosen[n] = (path, count)
        used.add(path)
    return chosen


def sample_ply_points_from_path(path: str, num_points: int, seed: int = 123) -> np.ndarray:
    xyz = pc_normalize_np(read_scannet_xyz_ply(path))
    rng = np.random.default_rng(seed + num_points)
    choice = rng.choice(xyz.shape[0], size=num_points, replace=xyz.shape[0] < num_points)
    return xyz[choice].astype(np.float32)


def plot_sample_vanilla_vs_spec(device: torch.device, model, cfg) -> None:
    setup_matplotlib(28)
    K = 512
    segment_len = 32
    s0 = kan_prefix_len_from_K(K, 0.10)
    pts_np = load_scanobjectnn_sample(2048, 0)
    pts = torch.from_numpy(pts_np).unsqueeze(0).to(device=device, dtype=torch.float32)
    idx_exact, radii_exact = exact_fps_radii_sq(pts, K, seed="first")
    radii_kan = predict_full_radii_squared_from_exact(radii_exact, model, cfg, device)
    idx_spec, trace = segfilter_trace(pts, K, radii_kan, segment_len, s0, seed="first")
    iou = set_iou(idx_exact, idx_spec)[0]
    exact_sel = idx_exact[0].detach().cpu().numpy()
    spec_sel = idx_spec[0].detach().cpu().numpy()
    same = np.intersect1d(exact_sel, spec_sel)

    fig = plt.figure(figsize=(16, 5.4))
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
    panels = [
        ("Original", None, "#8d99ae"),
        ("Vanilla FPS", exact_sel, "#ffc000"),
        (f"Speculative FPS\nIoU={iou:.3f}", spec_sel, "#00b0f0"),
    ]
    for ax, (title, sel, color) in zip(axes, panels):
        ax.scatter(pts_np[:, 0], pts_np[:, 1], pts_np[:, 2], s=2, c="#cfd3d7", alpha=0.22, linewidths=0)
        if sel is not None:
            ax.scatter(pts_np[sel, 0], pts_np[sel, 1], pts_np[sel, 2], s=9, c=color, alpha=0.92, linewidths=0)
        ax.set_title(title, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_box_aspect((1, 1, 1))
    fig.text(
        0.5,
        0.02,
        f"ScanObjectNN sample0, N=2048, K=512, matched sampled points={len(same)}/512",
        ha="center",
        fontsize=24,
    )
    save_fig(fig, OUT_PROCESS / "scanobjectnn_sample0_vanilla_vs_speculative_fps")
    trace.to_csv(OUT_PROCESS / "scanobjectnn_sample0_speculative_trace.csv", index=False)
    print(f"WROTE {OUT_PROCESS / 'scanobjectnn_sample0_vanilla_vs_speculative_fps.png'}")


def plot_threshold_gap_multiscale(device: torch.device, model, cfg) -> None:
    setup_matplotlib(24)
    K = 512
    segment_len = 32
    s0 = kan_prefix_len_from_K(K, 0.10)
    n_values = [1024, 2048, 4096, 8192, 16384, 32768]
    scale_sources = choose_distinct_ply_by_scale(n_values)
    all_rows: List[pd.DataFrame] = []
    for n in n_values:
        source_path, source_vertices = scale_sources[n]
        pts_np = sample_ply_points_from_path(source_path, n, seed=123)
        pts = torch.from_numpy(pts_np).unsqueeze(0).to(device=device, dtype=torch.float32)
        _idx_exact, radii_exact = exact_fps_radii_sq(pts, K, seed="first")
        radii_kan = predict_full_radii_squared_from_exact(radii_exact, model, cfg, device)
        _idx_spec, trace = segfilter_trace(pts, K, radii_kan, segment_len, s0, seed="first")
        exact_r = torch.sqrt(radii_exact[0].clamp_min(0)).detach().cpu().numpy()
        pred_r = torch.sqrt(radii_kan[0].clamp_min(0)).detach().cpu().numpy()
        x = np.arange(K)

        t = trace.copy()
        t["num_points"] = n
        t["source"] = "scannet_ply_real"
        t["source_path"] = source_path
        t["source_vertices"] = source_vertices
        t["exact_radius"] = exact_r[: len(t)]
        t["kan_pred_radius"] = pred_r[: len(t)]
        t["raw_candidates_plot"] = t["raw_candidates"].where((t["iter"] >= s0) & (t["raw_candidates"] > 0), np.nan)
        t["actual_candidates_plot"] = t["actual_candidates"].where(t["iter"] >= s0, t["remaining_points"])
        active = t[t["iter"] >= s0]
        raw_mean = float(active["raw_candidates"].mean())
        actual_mean = float(active["actual_candidates"].mean())
        fallback_rate = float(active["fallback"].mean() * 100.0)
        all_rows.append(t)

        fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.2), sharex=True)
        axes[0].plot(x, exact_r, color="#222222", linewidth=3.0, label="Exact radius")
        axes[0].plot(x, pred_r, color="#00b0f0", linewidth=2.8, linestyle="--", label="KAN pred")
        axes[0].axvspan(0, s0, color="#d9d9d9", alpha=0.35, label="Warmup")
        axes[0].axvline(s0, color="#444444", linewidth=2.0, linestyle=":", label="KAN start")
        axes[0].text(
            0.985,
            0.08,
            f"N={n}",
            transform=axes[0].transAxes,
            ha="right",
            va="bottom",
            fontsize=20,
        )
        axes[0].set_ylabel("Radius")
        axes[0].grid(axis="y", linestyle="--", alpha=0.35)
        axes[0].legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.04), fontsize=14)

        axes[1].plot(t["iter"], t["actual_candidates_plot"], color="#ffc000", linewidth=2.8, label="Eff. Cand.")
        axes[1].plot(t["iter"], t["raw_candidates_plot"], color="#00b0f0", linewidth=2.8, linestyle="--", label="Raw Cand.")
        axes[1].axhline(actual_mean, color="#d62728", linewidth=2.4, linestyle="-.", label=f"Eff. mean={actual_mean:.1f}")
        if raw_mean > 0:
            axes[1].axhline(raw_mean, color="#555555", linewidth=2.0, linestyle=":", label=f"Raw mean={raw_mean:.1f}")
        fb = t[(t["iter"] >= s0) & (t["fallback"] > 0)]
        if not fb.empty:
            axes[1].scatter(
                fb["iter"],
                fb["actual_candidates_plot"],
                marker="x",
                s=70,
                linewidths=2.2,
                color="#d62728",
                label=f"FB={fallback_rate:.1f}%",
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
        save_fig(fig, OUT_PROCESS / f"speculative_fps_threshold_gap_sample0_scannet_ply_N{n}")

    OUT_PROCESS.mkdir(parents=True, exist_ok=True)
    trace_all = pd.concat(all_rows, ignore_index=True)
    trace_all.to_csv(OUT_PROCESS / "speculative_fps_threshold_gap_sample0_scannet_ply_multiscale.csv", index=False)
    # Keep the historical filename as a compact default view. The full per-scale
    # figures live next to it with the N suffix.
    default_png = OUT_PROCESS / "speculative_fps_threshold_gap_sample0_scannet_ply_N2048.png"
    default_pdf = OUT_PROCESS / "speculative_fps_threshold_gap_sample0_scannet_ply_N2048.pdf"
    if default_png.exists():
        (OUT_PROCESS / "speculative_fps_threshold_gap_sample0.png").write_bytes(default_png.read_bytes())
    if default_pdf.exists():
        (OUT_PROCESS / "speculative_fps_threshold_gap_sample0.pdf").write_bytes(default_pdf.read_bytes())
    print(f"WROTE {OUT_PROCESS / 'speculative_fps_threshold_gap_sample0.png'}")


def evaluate_adjacent_window_feasibility(device: torch.device, model, cfg) -> None:
    setup_matplotlib(24)
    K = 512
    segment_len = 32
    s0 = kan_prefix_len_from_K(K, 0.10)
    n_values = [1024, 2048, 4096, 8192, 16384, 32768]
    scale_sources = choose_distinct_ply_by_scale(n_values)
    rows: List[Dict[str, float]] = []
    trace_rows: List[pd.DataFrame] = []
    for n in n_values:
        source_path, source_vertices = scale_sources[n]
        pts_np = sample_ply_points_from_path(source_path, n, seed=123)
        pts = torch.from_numpy(pts_np).unsqueeze(0).to(device=device, dtype=torch.float32)
        idx_exact, radii_exact = exact_fps_radii_sq(pts, K, seed="first")
        radii_kan = predict_full_radii_squared_from_exact(radii_exact, model, cfg, device)
        for mode_name, upper_mode in (("lower_only", "none"), ("adjacent_window", "prev_segment")):
            idx_spec, trace = segfilter_trace(pts, K, radii_kan, segment_len, s0, seed="first", upper_mode=upper_mode)
            active = trace[trace["iter"] >= s0]
            iou = set_iou(idx_exact, idx_spec)[0]
            rows.append({
                "num_points": n,
                "mode": mode_name,
                "source": "scannet_ply_real",
                "source_path": source_path,
                "source_vertices": source_vertices,
                "raw_candidates_mean": float(active["raw_candidates"].mean()),
                "actual_candidates_mean": float(active["actual_candidates"].mean()),
                "fallback_rate": float(active["fallback"].mean()),
                "fps_iou": float(iou),
                "exact": bool(iou == 1.0),
            })
            t = trace.copy()
            t["num_points"] = n
            t["mode"] = mode_name
            t["source_path"] = source_path
            t["source_vertices"] = source_vertices
            trace_rows.append(t)

    summary = pd.DataFrame(rows)
    trace_df = pd.concat(trace_rows, ignore_index=True)
    summary_path = OUT_PROCESS / "speculative_fps_adjacent_window_feasibility_scannet_ply.csv"
    trace_path = OUT_PROCESS / "speculative_fps_adjacent_window_trace_scannet_ply.csv"
    summary.to_csv(summary_path, index=False)
    trace_df.to_csv(trace_path, index=False)

    pivot_actual = summary.pivot(index="num_points", columns="mode", values="actual_candidates_mean")
    pivot_raw = summary.pivot(index="num_points", columns="mode", values="raw_candidates_mean")
    x = np.arange(len(n_values))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.bar(x - width / 2, pivot_actual.loc[n_values, "lower_only"], width, color="#00b0f0", label="Lower-only actual")
    ax.bar(x + width / 2, pivot_actual.loc[n_values, "adjacent_window"], width, color="#ffc000", label="Adjacent-window actual")
    ax.plot(x, pivot_raw.loc[n_values, "lower_only"], color="#005f7f", marker="o", linewidth=2.4, label="Lower-only raw")
    ax.plot(x, pivot_raw.loc[n_values, "adjacent_window"], color="#a66a00", marker="s", linewidth=2.4, label="Adjacent-window raw")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in n_values])
    ax.set_xlabel("Input points")
    ax.set_ylabel("Mean Cand.")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(ncol=2, frameon=False, loc="upper left", fontsize=15)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    save_fig(fig, OUT_PROCESS / "speculative_fps_adjacent_window_candidate_compare_scannet_ply")
    print(f"WROTE {summary_path}")


def main() -> None:
    OUT_PROCESS.mkdir(parents=True, exist_ok=True)
    df = make_summary_csv()
    plot_scanobjectnn_scale(df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_kan_radii_model(str(KAN_SAVE_DIR), device)
    cfg.prefix_ratio = 0.10
    plot_sample_vanilla_vs_spec(device, model, cfg)
    plot_threshold_gap_multiscale(device, model, cfg)
    evaluate_adjacent_window_feasibility(device, model, cfg)


if __name__ == "__main__":
    main()
