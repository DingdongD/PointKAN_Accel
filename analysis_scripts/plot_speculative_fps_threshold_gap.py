#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


def _setup_font(font_size: int) -> None:
    candidates = [
        "/root/.fonts/Arial.ttf",
        "/root/.fonts/arial.ttf",
        "/root/.fonts/ARIALN.TTF",
        "/root/.fonts/arialbd.ttf",
    ]
    font_path = next((p for p in candidates if os.path.exists(p)), None)
    family = "Arial"
    if font_path:
        font_manager.fontManager.addfont(font_path)
        family = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 2,
            "ytick.labelsize": font_size - 2,
            "legend.fontsize": font_size - 4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def _style(ax) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(2.4)
    ax.tick_params(width=2.2, length=7)
    ax.grid(True, linestyle="--", linewidth=1.0, alpha=0.35)
    ax.set_axisbelow(True)


def _read_rows(path: str) -> Dict[str, np.ndarray]:
    rows: List[Dict[str, str]] = list(csv.DictReader(open(path, "r", encoding="utf-8")))
    out: Dict[str, np.ndarray] = {}
    for key in rows[0].keys():
        if key == "phase":
            out[key] = np.array([r[key] for r in rows])
        else:
            out[key] = np.array([float(r[key]) for r in rows], dtype=np.float64)
    return out


def plot(args: argparse.Namespace) -> Dict[str, str]:
    _setup_font(args.fontsize)
    d = _read_rows(args.per_step_csv)
    t = d["t"]
    phase = d["phase"]
    fallback = d["kan_fallback"].astype(bool)
    warmup = phase == "warmup_exact"
    spec_start = int(t[~warmup][0]) if np.any(~warmup) else 0

    exact_th = d["exact_segment_threshold"]
    kan_th = d["kan_segment_threshold"]
    exact_r = d["exact_radius"]
    kan_r = d["kan_pred_radius"]
    gap = kan_th - exact_th
    cand_actual = d["kan_fps_candidates_actual"]
    cand_raw = d["kan_fps_candidates_raw_pass"]

    fig, axes = plt.subplots(3, 1, figsize=(18, 16), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.0, 1.0]})
    ax0, ax1, ax2 = axes

    for ax in axes:
        ax.axvspan(0, spec_start - 1, color="#eeeeee", alpha=0.78)
        ax.axvline(spec_start, color="#444444", lw=2.0, linestyle="--")

    ax0.plot(t, exact_r, color="#00b0f0", lw=3.0, label="Exact radius")
    ax0.plot(t, kan_r, color="#ffc000", lw=2.8, linestyle="--", label="KAN predicted radius")
    ax0.step(t, exact_th, where="post", color="#0070c0", lw=2.2, alpha=0.9, label="Exact segment threshold")
    ax0.step(t, kan_th, where="post", color="#c08000", lw=2.2, alpha=0.9, linestyle=":", label="KAN segment threshold")
    ax0.scatter(t[fallback], kan_th[fallback], marker="x", s=42, color="#d62728", label="Fallback step")
    ax0.set_ylabel("Radius")
    ax0.set_title("KAN Threshold Curve vs Exact Reference")
    ax0.legend(frameon=False, loc="upper right", ncol=2)
    _style(ax0)

    ax1.axhline(0.0, color="#333333", lw=1.8)
    ax1.fill_between(t, 0.0, gap, where=gap >= 0, color="#ffb3b3", alpha=0.55, step="post", label="KAN higher than exact")
    ax1.fill_between(t, 0.0, gap, where=gap < 0, color="#b7e4c7", alpha=0.55, step="post", label="KAN lower than exact")
    ax1.plot(t, gap, color="#7f3b08", lw=2.5)
    ax1.scatter(t[fallback], gap[fallback], marker="x", s=46, color="#d62728", label="Fallback")
    ax1.set_ylabel("Threshold gap\nKAN - exact")
    ax1.legend(frameon=False, loc="best")
    _style(ax1)

    ax2.plot(t, cand_raw, color="#ffc000", lw=2.4, linestyle="--", label="Raw pass candidates")
    ax2.plot(t, cand_actual, color="#c08000", lw=2.7, label="Actual candidates after fallback")
    ax2.scatter(t[fallback], cand_actual[fallback], marker="x", s=42, color="#d62728", label="Fallback")
    ax2.set_xlabel("FPS iteration")
    ax2.set_ylabel("Candidate count")
    ax2.legend(frameon=False, loc="best")
    _style(ax2)

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, args.output_name)
    fig.tight_layout()
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": base + ".pdf", "png": base + ".png"}


def plot_squared_domain(args: argparse.Namespace) -> Dict[str, str]:
    _setup_font(args.fontsize)
    d = _read_rows(args.per_step_csv)
    t = d["t"]
    phase = d["phase"]
    fallback = d["kan_fallback"].astype(bool)
    warmup = phase == "warmup_exact"
    spec_start = int(t[~warmup][0]) if np.any(~warmup) else 0

    exact_th_sq = d["exact_segment_threshold"] ** 2
    kan_th_sq = d["kan_segment_threshold"] ** 2
    exact_r_sq = d["exact_radius"] ** 2
    kan_r_sq = d["kan_pred_radius"] ** 2
    gap_sq = kan_th_sq - exact_th_sq
    cand_actual = d["kan_fps_candidates_actual"]
    cand_raw = d["kan_fps_candidates_raw_pass"]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(18, 16),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0, 1.0]},
    )
    ax0, ax1, ax2 = axes

    for ax in axes:
        ax.axvspan(0, spec_start - 1, color="#eeeeee", alpha=0.78)
        ax.axvline(spec_start, color="#444444", lw=2.0, linestyle="--")

    ax0.plot(t, exact_r_sq, color="#00b0f0", lw=3.0, label="Exact radius squared")
    ax0.plot(t, kan_r_sq, color="#ffc000", lw=2.8, linestyle="--", label="KAN predicted radius squared")
    ax0.step(t, exact_th_sq, where="post", color="#0070c0", lw=2.2, alpha=0.9, label="Exact threshold squared")
    ax0.step(t, kan_th_sq, where="post", color="#c08000", lw=2.2, alpha=0.9, linestyle=":", label="KAN threshold squared")
    ax0.scatter(t[fallback], kan_th_sq[fallback], marker="x", s=42, color="#d62728", label="Fallback step")
    ax0.set_ylabel("Squared distance")
    ax0.set_title("KAN Threshold vs Exact Reference in Squared-Distance Domain")
    ax0.legend(frameon=False, loc="upper right", ncol=2)
    _style(ax0)

    ax1.axhline(0.0, color="#333333", lw=1.8)
    ax1.fill_between(t, 0.0, gap_sq, where=gap_sq >= 0, color="#ffb3b3", alpha=0.55, step="post", label="KAN higher than exact")
    ax1.fill_between(t, 0.0, gap_sq, where=gap_sq < 0, color="#b7e4c7", alpha=0.55, step="post", label="KAN lower than exact")
    ax1.plot(t, gap_sq, color="#7f3b08", lw=2.5)
    ax1.scatter(t[fallback], gap_sq[fallback], marker="x", s=46, color="#d62728", label="Fallback")
    ax1.set_ylabel("Squared gap\nKAN - exact")
    ax1.legend(frameon=False, loc="best")
    _style(ax1)

    ax2.plot(t, cand_raw, color="#ffc000", lw=2.4, linestyle="--", label="Raw pass candidates")
    ax2.plot(t, cand_actual, color="#c08000", lw=2.7, label="Actual candidates after fallback")
    ax2.scatter(t[fallback], cand_actual[fallback], marker="x", s=42, color="#d62728", label="Fallback")
    ax2.set_xlabel("FPS iteration")
    ax2.set_ylabel("Candidate count")
    ax2.legend(frameon=False, loc="best")
    _style(ax2)

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, args.output_name + "_squared")
    fig.tight_layout()
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": base + ".pdf", "png": base + ".png"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot KAN speculative-FPS threshold gap against exact reference.")
    p.add_argument(
        "--per_step_csv",
        default="/workspace/PointKAN-pytorch/my_progressive_out/speculative_fps_process/speculative_fps_process_sample0_per_step.csv",
    )
    p.add_argument("--out_dir", default="/workspace/PointKAN-pytorch/my_progressive_out/speculative_fps_process")
    p.add_argument("--output_name", default="speculative_fps_threshold_gap_sample0")
    p.add_argument("--fontsize", type=int, default=28)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = plot(args)
    sq_paths = plot_squared_domain(args)
    print(paths["pdf"])
    print(paths["png"])
    print(sq_paths["pdf"])
    print(sq_paths["png"])


if __name__ == "__main__":
    main()
