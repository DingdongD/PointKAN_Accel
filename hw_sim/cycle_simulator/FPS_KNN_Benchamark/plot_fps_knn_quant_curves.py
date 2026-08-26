#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取 run_fps_knn_quant_recall.py 输出的 JSON，绘制「每轴量化 bit → 相对 vanilla 的召回/IoU」
以及「量化坐标 MSE」曲线。可用 --layout rows 将 recall 与 MSE 上下对齐便于看拐点关系。
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))


def _setup_font():
    candidates = [
        "/root/.fonts/Arial.ttf",
        "/root/.fonts/arial.ttf",
        "/root/.fonts/ARIALN.TTF",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    font_path = next((fp for fp in candidates if os.path.exists(fp)), None)
    if font_path and "DejaVu" not in font_path:
        font_manager.fontManager.addfont(font_path)
        name = font_manager.FontProperties(fname=font_path).get_name()
    else:
        name = "DejaVu Sans"
    return name


def _metrics_block_per_dataset(block: dict) -> dict:
    """兼容主基准 ``metrics_vs_vanilla`` 与固定中心脚本 ``metrics_fixed_centers_vs_vanilla_knn``。"""
    return block.get("metrics_vs_vanilla") or block.get("metrics_fixed_centers_vs_vanilla_knn") or {}


def load_series(data: dict, metric: str) -> dict:
    """返回 {dataset_key: (bits_sorted, values)}"""
    out = {}
    for key, block in data.get("datasets", {}).items():
        mvs = _metrics_block_per_dataset(block)
        bits = sorted(int(b) for b in mvs.keys())
        vals = []
        for b in bits:
            row = mvs.get(str(b)) or {}
            vals.append(float(row.get(metric, float("nan"))))
        label = block.get("display_name") or key
        out[label] = (bits, vals)
    return out


def _metric_ylabel(metric: str) -> str:
    if "fps_idx_recall" in metric:
        return "FPS index recall (|Sv∩Sq|/K)"
    if metric == "knn_recall_mean" or ("knn" in metric and "recall" in metric):
        return "KNN neighbor recall (mean)"
    if "recall" in metric:
        return "Recall (mean)"
    if "fps_set_iou" in metric:
        return "FPS index set IoU"
    if "knn_iou" in metric:
        return "KNN neighbor set IoU"
    if metric == "coord_quant_mse_mean":
        return r"Coord. MSE "
    if metric == "coord_quant_rmse_mean":
        return r"Coord. RMSE "
    return metric.replace("_", " ")


def _metric_use_log_y(metric: str) -> bool:
    return "coord_quant_mse" in metric or "coord_quant_rmse" in metric


def _metric_clamp_01_y(metric: str) -> bool:
    return not _metric_use_log_y(metric) and ("recall" in metric or "iou" in metric)


def parse_args():
    p = argparse.ArgumentParser(description="绘制 FPS+KNN 量化召回曲线")
    p.add_argument(
        "--json",
        type=str,
        default=os.path.join(BENCH_DIR, "fps_knn_quant_recall_results.json"),
    )
    p.add_argument(
        "--metrics",
        type=str,
        default="knn_recall_mean,fps_idx_recall_mean,coord_quant_mse_mean",
        help="逗号分隔；键来自各数据集 metrics_vs_vanilla 或 metrics_fixed_centers_vs_vanilla_knn",
    )
    p.add_argument(
        "--layout",
        type=str,
        choices=("cols", "rows"),
        default="rows",
        help="cols=并排子图；rows=共用横轴上下排列（适合 recall+MSE 对照）",
    )
    p.add_argument("--fname", type=str, default="fps_knn_quant_recall", help="输出文件名前缀（无扩展名）")
    p.add_argument("--figsize", type=str, default="9.2,9.0", help="宽,高 英寸（rows 时高度含多子图）")
    return p.parse_args()


def main():
    args = parse_args()
    arial_name = _setup_font()
    FONT_SIZE = 16
    plt.rcParams.update(
        {
            "font.family": arial_name,
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE - 1,
            "ytick.labelsize": FONT_SIZE - 1,
            "legend.fontsize": FONT_SIZE - 2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    w, h = [float(x) for x in args.figsize.split(",")]
    n = len(metrics)
    if n == 0:
        raise SystemExit("未指定 --metrics")
    sharex = args.layout == "rows"
    if args.layout == "rows":
        fig, axes = plt.subplots(n, 1, figsize=(w, h), squeeze=False, sharex=sharex)
        axes = axes[:, 0] if axes.ndim == 2 else axes
    else:
        fig, axes = plt.subplots(1, n, figsize=(w, h), squeeze=False)
        axes = axes[0]

    colors = ["#00b0f0", "#ffc000", "#d9f2d0", "#c0504d", "#7030a0", "#4f81bd", "#9bbb59"]
    for ax, metric in zip(axes, metrics):
        series = load_series(data, metric)
        all_bits = set()
        for i, (label, (bits, vals)) in enumerate(series.items()):
            all_bits.update(bits)
            c = colors[i % len(colors)]
            ax.plot(bits, vals, marker="o", linewidth=2.0, markersize=7, label=label, color=c)
        if args.layout == "rows" and ax is not axes[-1]:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Bits per axis (xyz)")
        ax.set_ylabel(_metric_ylabel(metric))
        xt = sorted(all_bits)
        if xt:
            ax.set_xticks(xt)
        if len(xt) > 10:
            plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.45, linestyle="--")
        ax.set_axisbelow(True)
        ax.legend(loc="best", frameon=False)
        if _metric_use_log_y(metric):
            ax.set_yscale("log")
            ax.margins(y=0.12)
        elif _metric_clamp_01_y(metric):
            ax.set_ylim(0.0, 1.05)

    plt.tight_layout(pad=0.35)
    base = os.path.join(BENCH_DIR, args.fname)
    plt.savefig(f"{base}.pdf", transparent=True, bbox_inches="tight", format="pdf", pad_inches=0.02)
    plt.savefig(f"{base}.png", dpi=300, bbox_inches="tight", format="png", pad_inches=0.02)
    print(f"已保存: {base}.pdf , {base}.png")


if __name__ == "__main__":
    main()
