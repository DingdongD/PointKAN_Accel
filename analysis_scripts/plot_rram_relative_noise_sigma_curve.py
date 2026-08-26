#!/usr/bin/env python3
"""Plot relative RRAM weight-noise sensitivity from full-dataset summaries."""

from __future__ import annotations

import os
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


ROOT = Path("/workspace/PointKAN-pytorch")
OUT_DIR = ROOT / "my_progressive_out" / "rram_relative_noise_sigma_curve"

DATASETS = {
    "ModelNet40": ROOT
    / "analysis_rram_weight_noise_relative_percent_modelnet40"
    / "rram_weight_noise_summary.csv",
    "ShapeNet": ROOT
    / "analysis_rram_weight_noise_relative_percent_partseg"
    / "rram_weight_noise_summary.csv",
    "ScanObjectNN": ROOT
    / "analysis_rram_weight_noise_relative_percent_scanobj"
    / "rram_weight_noise_summary.csv",
}

COLORS = {
    "ModelNet40": "#2ca02c",
    "ShapeNet": "#ffbf00",
    "ScanObjectNN": "#00a6d6",
}

MARKERS = {
    "ModelNet40": "o",
    "ShapeNet": "s",
    "ScanObjectNN": "^",
}


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


def load_relative_rows() -> pd.DataFrame:
    rows = []
    for dataset, csv_path in DATASETS.items():
        df = pd.read_csv(csv_path)
        rel = df[df["kind"] == "relative"].copy()
        if rel.empty:
            raise RuntimeError(f"No relative rows found in {csv_path}")
        rel["dataset"] = dataset
        metadata_path = csv_path.with_name("rram_weight_noise_metadata.json")
        with metadata_path.open() as f:
            metadata = json.load(f)
        baseline = next(row for row in metadata["rows"] if row["kind"] == "baseline")
        base_acc = float(baseline["accuracy"])
        base_bal = float(baseline["balanced_accuracy"])
        rel["accuracy_drop_from_sigma0"] = base_acc - rel["accuracy_mean"]
        rel["balanced_drop_from_sigma0"] = base_bal - rel["balanced_accuracy_mean"]
        rel["device_variation_percent"] = rel["sigma"] * 100.0
        rows.append(rel)
    out = pd.concat(rows, ignore_index=True)
    return out[
        [
            "dataset",
            "sigma",
            "device_variation_percent",
            "accuracy_mean",
            "accuracy_std",
            "accuracy_drop_from_sigma0",
            "balanced_accuracy_mean",
            "balanced_accuracy_std",
            "balanced_drop_from_sigma0",
        ]
    ].sort_values(["dataset", "sigma"])


def plot_metric(df: pd.DataFrame, metric: str, drop_metric: str, stem: str, ylabel: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12.5, 5.2))
    x_ticks = sorted(df["device_variation_percent"].unique())

    for dataset in DATASETS:
        sub = df[df["dataset"] == dataset].sort_values("sigma")
        ax.plot(
            sub["device_variation_percent"],
            sub[metric],
            color=COLORS[dataset],
            marker=MARKERS[dataset],
            markersize=14,
            linewidth=4.0,
            label=dataset,
        )
        if dataset == "ScanObjectNN" and metric == "accuracy_mean":
            for x_val, y_val in zip(sub["device_variation_percent"], sub[metric]):
                ax.annotate(
                    f"{y_val:.2f}",
                    xy=(x_val, y_val),
                    xytext=(0, 12),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=20,
                    color=COLORS[dataset],
                )
    ax.set_ylabel(ylabel, fontsize=36)
    ax.set_xlabel("Device Variation (%)", fontsize=36)

    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.12),
        mode="expand",
        ncol=3,
        frameon=False,
        fontsize=25,
        handlelength=1.8,
        columnspacing=0.75,
        borderaxespad=0.0,
    )

    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{x:.0f}" for x in x_ticks])
    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=1.5, alpha=0.5)
    ax.tick_params(axis="both", labelsize=30, width=2.5, length=8)
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    configure_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_relative_rows()
    df.to_csv(OUT_DIR / "rram_relative_noise_sigma_summary.csv", index=False)
    plot_metric(
        df,
        metric="accuracy_mean",
        drop_metric="accuracy_drop_from_sigma0",
        stem="rram_relative_sigma_accuracy_curve",
        ylabel="Accuracy (%)",
    )
    plot_metric(
        df,
        metric="balanced_accuracy_mean",
        drop_metric="balanced_drop_from_sigma0",
        stem="rram_relative_sigma_balanced_curve",
        ylabel="Balanced Acc (%)",
    )
    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
