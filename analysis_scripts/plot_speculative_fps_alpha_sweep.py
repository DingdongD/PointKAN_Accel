#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


def _setup(font_size: int) -> None:
    candidates = [
        "/root/.fonts/Arial.ttf",
        "/root/.fonts/arial.ttf",
        "/root/.fonts/ARIALN.TTF",
        "/root/.fonts/arialbd.ttf",
    ]
    fp = next((p for p in candidates if os.path.exists(p)), None)
    family = "Arial"
    if fp:
        font_manager.fontManager.addfont(fp)
        family = font_manager.FontProperties(fname=fp).get_name()
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="/workspace/PointKAN-pytorch/my_progressive_out/speculative_fps_calibration_sweep_full/fallback_full_test_K512_seg32_pr0.10_radii.csv")
    p.add_argument("--out_base", default="/workspace/PointKAN-pytorch/my_progressive_out/speculative_fps_calibration_sweep_full/kan_alpha_fallback_candidate_tradeoff")
    p.add_argument("--fontsize", type=int, default=28)
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    rows = [r for r in csv.DictReader(open(args.csv, "r", encoding="utf-8")) if r["branch"] != "exact_ref"]
    rows = sorted(rows, key=lambda r: float(r["kan_alpha"]))
    alpha = np.array([float(r["kan_alpha"]) for r in rows])
    fallback = np.array([100.0 * float(r["fallback_rate"]) for r in rows])
    candidates = np.array([float(r["actual_candidates_mean_with_fallback"]) for r in rows])
    raw = np.array([float(r["raw_pass_candidates_mean"]) for r in rows])

    _setup(args.fontsize)
    fig, ax1 = plt.subplots(figsize=(12, 6.8))
    ax2 = ax1.twinx()
    l1 = ax1.plot(alpha, candidates, marker="o", lw=3.0, color="#00b0f0", label="Actual candidates")
    l2 = ax1.plot(alpha, raw, marker="s", lw=2.6, linestyle="--", color="#0070c0", label="Raw pass candidates")
    l3 = ax2.plot(alpha, fallback, marker="^", lw=3.0, color="#ffc000", label="Fallback rate")

    ax1.set_xlabel("KAN squared-threshold scale alpha")
    ax1.set_ylabel("Candidate count")
    ax2.set_ylabel("Fallback rate (%)")
    ax1.grid(axis="both", linestyle="--", alpha=0.35)
    for ax in (ax1, ax2):
        for sp in ax.spines.values():
            sp.set_linewidth(2.5)
        ax.tick_params(width=2.2, length=7)
    lines = l1 + l2 + l3
    ax1.legend(lines, [x.get_label() for x in lines], frameon=False, loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out_base), exist_ok=True)
    fig.savefig(args.out_base + ".pdf", bbox_inches="tight")
    fig.savefig(args.out_base + ".png", dpi=args.dpi, bbox_inches="tight")
    print(args.out_base + ".pdf")
    print(args.out_base + ".png")


if __name__ == "__main__":
    main()
