#!/usr/bin/env python3
"""Regenerate pointkan_quant_compare with consistent labels and fonts."""

from __future__ import annotations

import os
import csv

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


def configure_font() -> None:
    candidate_fonts = [
        "/root/.fonts/Arial.ttf",
        "/root/.fonts/arial.ttf",
        "/root/.fonts/ARIALN.TTF",
        "/root/.fonts/arialbd.ttf",
    ]
    font_path = next((fp for fp in candidate_fonts if os.path.exists(fp)), None)
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
            "hatch.linewidth": 2.5,
        }
    )


def autolabel(ax, rects, fontsize: int, dy: float = 0.15) -> None:
    for rect in rects:
        h = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            h + dy,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def main() -> None:
    configure_font()

    colors = ["#d9f2d0", "#ffc000", "#00b0f0"]
    patterns = ["/", "\\", "x"]
    font_size = 20
    width = 0.22

    configs = ["Baseline", "+RTN@W8A8", "+OBAS-GRF", "+Spec FPS-KNN"]

    # Progressive accuracy data:
    # Baseline is float vanilla GRF-KAN; stage 2/3 are rerun with the same
    # W8A8 RTN setup and CUDA-aligned PWL/OBAS injection. Spec FPS-KNN is
    # exact, so it inherits the stage-3 model accuracy.
    modelnet_bal_acc = [88.72383720930233, 88.90, 88.57383720930233, 88.57383720930233]
    scanobjectnn_bal_acc = [76.65, 73.13, 73.3136326698253, 73.3136326698253]
    shapenet_acc = [94.07198699547669, 94.05, 94.07674153645833, 94.07674153645833]
    shapenet_miou = [84.47, 84.08, 84.25828481690777, 84.25828481690777]

    with open("pointkan_quant_compare_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "modelnet40_bal_acc", "shapenet_acc", "scanobjectnn_bal_acc", "shapenet_miou"])
        for row in zip(configs, modelnet_bal_acc, shapenet_acc, scanobjectnn_bal_acc, shapenet_miou):
            writer.writerow(row)

    x = np.arange(len(configs))
    fig, ax1 = plt.subplots(figsize=(12.5, 4.6))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(
        x - width,
        modelnet_bal_acc,
        width=width,
        label="ModelNet40",
        color=colors[0],
        edgecolor="white",
        hatch=patterns[0],
        linewidth=0.8,
    )
    bars2 = ax1.bar(
        x,
        shapenet_acc,
        width=width,
        label="ShapeNet",
        color=colors[1],
        edgecolor="white",
        hatch=patterns[1],
        linewidth=0.8,
    )
    bars3 = ax1.bar(
        x + width,
        scanobjectnn_bal_acc,
        width=width,
        label="ScanObjectNN",
        color=colors[2],
        edgecolor="white",
        hatch=patterns[2],
        linewidth=0.8,
    )

    line = ax2.plot(
        x,
        shapenet_miou,
        marker="*",
        markersize=16,
        linewidth=3.0,
        linestyle="--",
        color="red",
        label="ShapeNet mIoU",
    )[0]

    autolabel(ax1, bars1, fontsize=font_size)
    autolabel(ax1, bars2, fontsize=font_size)
    autolabel(ax1, bars3, fontsize=font_size)
    for xi, yi in zip(x, shapenet_miou):
        ax2.text(
            xi,
            yi + 0.15,
            f"{yi:.2f}",
            ha="center",
            va="bottom",
            fontsize=font_size,
            color="red",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=font_size)
    ax1.set_ylabel("Accuracy (%)", fontsize=font_size)
    ax2.set_ylabel("ShapeNet mIoU (%)", fontsize=font_size)
    ax1.set_ylim(72, 96)
    ax2.set_ylim(78, 86)
    ax1.tick_params(axis="both", labelsize=font_size, width=2.0, length=6)
    ax2.tick_params(axis="y", labelsize=font_size, width=2.0, length=6)
    ax1.grid(axis="y", alpha=0.45, linestyle="--", linewidth=1.2)
    ax1.set_axisbelow(True)
    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)

    handles1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(
        handles1 + [line],
        labels1 + ["ShapeNet mIoU"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
        fontsize=font_size,
        columnspacing=0.8,
        handlelength=1.6,
    )

    fig.tight_layout()
    fig.savefig("pointkan_quant_compare.pdf", transparent=True, bbox_inches="tight", format="pdf", pad_inches=0.02)
    fig.savefig("pointkan_quant_compare.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
