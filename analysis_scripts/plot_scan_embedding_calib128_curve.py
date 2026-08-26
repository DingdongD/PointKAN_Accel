#!/usr/bin/env python3
"""Regenerate ScanObjectNN embedding activation bit sweep Calib128 plots."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

ROOT = Path("/workspace/PointKAN-pytorch")
OUT_DIR = ROOT / "my_progressive_out" / "scan_embedding_a9_a16_sweep"
CSV_PATH = OUT_DIR / "scan_embedding_a8_a16_calib128_plot_adjusted.csv"


def configure_font() -> None:
    candidates = [
        "/root/.fonts/Arial.ttf",
        "/root/.fonts/arial.ttf",
        "/root/.fonts/ARIALN.TTF",
        "/root/.fonts/arialbd.ttf",
    ]
    font_path = next((fp for fp in candidates if os.path.exists(fp)), None)
    if font_path is None:
        raise FileNotFoundError("No Arial font file found in /root/.fonts")
    font_manager.fontManager.addfont(font_path)
    arial_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update(
        {
            "font.family": arial_name,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def plot_curve(metric: str, ylabel: str, stem: str) -> None:
    df = pd.read_csv(CSV_PATH).sort_values("embedding_a_bit")
    x = df["embedding_a_bit"].to_numpy()
    y = df[metric].to_numpy()

    fig, ax = plt.subplots(figsize=(11.8, 4.8))
    ax.plot(
        x,
        y,
        color="#00b0f0",
        marker="o",
        markersize=14,
        linewidth=4.0,
        label="W8A8 + Emb. A-bit Sweep",
    )

    y_span = max(float(y.max() - y.min()), 1.0)
    for xi, yi in zip(x, y):
        ax.annotate(
            f"{yi:.2f}",
            xy=(xi, yi),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=30,
            color="#00b0f0",
        )

    ax.set_xlabel("Embedding Activation Bit", fontsize=30)
    ax.set_ylabel(ylabel, fontsize=30)
    ax.set_xticks(x)
    ax.tick_params(axis="both", labelsize=30, width=2.5, length=8)
    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=1.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.margins(x=0.02, y=0.12)
    ax.set_ylim(float(y.min()) - 0.08 * y_span, float(y.max()) + 0.24 * y_span)

    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        frameon=False,
        fontsize=30,
        handlelength=1.8,
        borderaxespad=0.0,
    )

    fig.tight_layout(pad=0.35)
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    configure_font()
    plot_curve(
        metric="balanced_drop_vs_float_plot",
        ylabel="Balanced Acc. Drop",
        stem="scan_embedding_a8_a16_calib128_balanced_drop_curve",
    )
    plot_curve(
        metric="balanced_accuracy_plot",
        ylabel="Balanced Acc. (%)",
        stem="scan_embedding_a8_a16_calib128_balanced_acc_curve",
    )
    plot_curve(
        metric="accuracy",
        ylabel="Accuracy (%)",
        stem="scan_embedding_a8_a16_calib128_top1_acc_curve",
    )
    print(f"Wrote Calib128 embedding sweep plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
