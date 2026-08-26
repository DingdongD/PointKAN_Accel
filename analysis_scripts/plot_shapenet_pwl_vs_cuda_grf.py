#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


def _setup_arial(font_size: int = 24) -> str:
    candidates = [
        os.path.expanduser("~/.fonts/Arial.ttf"),
        os.path.expanduser("~/.fonts/arial.ttf"),
        os.path.expanduser("~/.fonts/ARIALN.TTF"),
        os.path.expanduser("~/.fonts/arialbd.ttf"),
    ]
    font_path = next((p for p in candidates if os.path.exists(p)), None)
    if font_path:
        fm.fontManager.addfont(font_path)
        font_name = fm.FontProperties(fname=font_path).get_name()
    else:
        font_name = "Arial"
    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.size": font_size,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "hatch.linewidth": 2.5,
        }
    )
    return font_name


def _eval_cuda_grf(x: np.ndarray, weight_numerator: np.ndarray, weight_denominator: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    num = np.asarray(weight_numerator, dtype=np.float64)
    den = np.asarray(weight_denominator, dtype=np.float64)
    if num.ndim == 1:
        num = num[None, :]
    if den.ndim == 1:
        den = den[None, :]
    xp_num = np.stack([x**k for k in range(num.shape[1])], axis=0)
    out = []
    for g in range(den.shape[0]):
        p = num[min(g, num.shape[0] - 1)] @ xp_num
        q = np.ones_like(x)
        for k in range(den.shape[1]):
            q = q + np.abs(float(den[g, k]) * (x ** (k + 1)))
        out.append(p / np.maximum(q, 1e-8))
    return np.stack(out, axis=0)


def _piecewise_linear(x: np.ndarray, split_points: np.ndarray, slopes: np.ndarray, intercepts: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(split_points[1:-1], x, side="right")
    idx = np.clip(idx, 0, len(slopes) - 1)
    return slopes[idx] * x + intercepts[idx]


def _eval_group_pwl(x: np.ndarray, x_min: float, x_max: float, group: Dict) -> np.ndarray:
    mode = str(group.get("mode", "pwl"))
    x_eff = np.clip(x, x_min, x_max)
    if mode == "zero":
        return np.zeros_like(x_eff)
    if mode == "const":
        return np.full_like(x_eff, float(group["intercepts"][0]))
    if mode == "linear":
        return float(group["slopes"][0]) * x_eff + float(group["intercepts"][0])
    if not group.get("valid", True):
        raise ValueError(f"Invalid PWL group: {group.get('reason')}")
    inner = np.asarray(group.get("breakpoints", []), dtype=np.float64)
    inner = np.clip(inner, x_min, x_max)
    inner.sort()
    split_points = np.concatenate(([x_min], inner, [x_max])).astype(np.float64)
    split_points.sort()
    slopes = np.asarray(group["slopes"], dtype=np.float64).reshape(-1)
    intercepts = np.asarray(group["intercepts"], dtype=np.float64).reshape(-1)
    return _piecewise_linear(x_eff, split_points, slopes, intercepts)


def _group_breakpoint_markers(x_min: float, x_max: float, group: Dict) -> Tuple[np.ndarray, np.ndarray]:
    mode = str(group.get("mode", "pwl"))
    if mode != "pwl" or not group.get("valid", True):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    inner = np.asarray(group.get("breakpoints", []), dtype=np.float64)
    if inner.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    inner = np.unique(np.clip(inner, x_min, x_max))
    inner.sort()
    y = _eval_group_pwl(inner, x_min, x_max, group)
    return inner, y


def _eval_pwl_layer(x: np.ndarray, layer: Dict, bit_key: str) -> np.ndarray:
    x_min = float(layer["x_range"]["min"])
    x_max = float(layer["x_range"]["max"])
    groups = layer["bits"][bit_key]["groups"]
    return np.stack([_eval_group_pwl(x, x_min, x_max, g) for g in groups], axis=0)


def _load_layers(grf_json: str, pwl_json: str) -> Tuple[List[Dict], Dict[str, Dict]]:
    grf = json.load(open(grf_json, "r", encoding="utf-8"))
    pwl = json.load(open(pwl_json, "r", encoding="utf-8"))
    pwl_by_name = {str(l["name"]): l for l in pwl["layers"]}
    return grf["layers"], pwl_by_name


def _write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_all_layers(args: argparse.Namespace) -> None:
    _setup_arial(args.fontsize)
    grf_layers, pwl_by_name = _load_layers(args.grf_json, args.pwl_json)
    bit_key = str(args.bit)
    layer_names = [l["name"] for l in grf_layers if l["name"] in pwl_by_name]
    ncols = 4
    nrows = int(np.ceil(len(layer_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(40, 30), squeeze=False)

    rows: List[Dict] = []
    for li, grf_layer in enumerate([l for l in grf_layers if l["name"] in pwl_by_name]):
        name = grf_layer["name"]
        pwl_layer = pwl_by_name[name]
        ax = axes[li // ncols][li % ncols]
        x_min = float(pwl_layer["x_range"]["min"])
        x_max = float(pwl_layer["x_range"]["max"])
        x = np.linspace(x_min, x_max, args.n_points, dtype=np.float64)

        y_grf = _eval_cuda_grf(
            x,
            np.asarray(grf_layer["weight_numerator"], dtype=np.float64),
            np.asarray(grf_layer["weight_denominator"], dtype=np.float64),
        )
        y_pwl = _eval_pwl_layer(x, pwl_layer, bit_key)
        diff = y_pwl - y_grf
        marker_group = int(np.argmax(np.mean(diff**2, axis=1)))
        mse = float(np.mean(diff**2))
        mae = float(np.mean(np.abs(diff)))
        max_abs = float(np.max(np.abs(diff)))
        rel_rmse = float(np.sqrt(mse) / (np.sqrt(np.mean(y_grf**2)) + 1e-12))
        rows.append(
            {
                "layer_idx": li,
                "layer": name,
                "bit": bit_key,
                "mse": mse,
                "mae": mae,
                "max_abs": max_abs,
                "rel_rmse": rel_rmse,
                "x_min": x_min,
                "x_max": x_max,
                "breakpoint_marker_group": marker_group,
            }
        )

        ax.set_facecolor("#f4f4f4")
        groups = pwl_layer["bits"][bit_key]["groups"]
        for g in range(y_grf.shape[0]):
            if g == marker_group:
                continue
            ax.plot(x, y_grf[g], color="#00b0f0", alpha=0.20, lw=1.4)
            ax.plot(x, y_pwl[g], color="#ffc000", alpha=0.24, lw=1.4, linestyle="--")
        ax.plot(x, y_grf[marker_group], color="#0070c0", alpha=0.95, lw=3.0, zorder=8)
        ax.plot(x, y_pwl[marker_group], color="#c08000", alpha=0.98, lw=3.0, linestyle="--", zorder=9)
        bx, by = _group_breakpoint_markers(x_min, x_max, groups[marker_group])
        if bx.size:
            ax.scatter(
                bx,
                by,
                s=72,
                marker="o",
                facecolor="#c08000",
                edgecolor="white",
                linewidth=1.2,
                alpha=0.95,
                zorder=12,
            )
        ax.plot(x, y_grf.mean(axis=0), color="#0070c0", lw=2.4, alpha=0.60, zorder=10)
        ax.plot(x, y_pwl.mean(axis=0), color="#c08000", lw=2.4, alpha=0.65, linestyle="--", zorder=11)
        ax.set_title(f"L{li}: {name}\nMSE={mse:.2e}", fontsize=args.fontsize)
        ax.tick_params(axis="both", labelsize=args.fontsize, width=2.8, length=8)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        for spine in ax.spines.values():
            spine.set_linewidth(2.8)
        ax.grid(axis="both", alpha=0.45, linestyle="--", linewidth=1.2)

    for li in range(len(layer_names), nrows * ncols):
        axes[li // ncols][li % ncols].axis("off")

    handles = [
        Line2D([0], [0], color="#0070c0", lw=4.0, label="Selected GRF group"),
        Line2D([0], [0], color="#c08000", lw=4.0, linestyle="--", label="Selected PWL group"),
        Line2D(
            [0],
            [0],
            color="none",
            marker="o",
            markersize=12,
            markerfacecolor="#c08000",
            markeredgecolor="white",
            label="Breakpoints",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        fontsize=args.fontsize,
        frameon=False,
        columnspacing=1.4,
        handlelength=2.2,
    )
    fig.tight_layout(rect=[0.015, 0.02, 1.0, 0.89])
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, f"shapenet_pwl_v2_vs_cuda_grf_bit{bit_key}")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    _write_csv(base + "_layer_error.csv", rows)
    print(base + ".pdf")
    print(base + ".png")
    print(base + "_layer_error.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize ShapeNetPart PWL V2 against CUDA-aligned GRF.")
    p.add_argument(
        "--grf_json",
        default="/workspace/PointKAN-pytorch/PointKAN-elite/part_segmentation/grf_export_out/grf_coefficients.json",
    )
    p.add_argument(
        "--pwl_json",
        default="/workspace/PointKAN-pytorch/PointKAN-elite/part_segmentation/grf_export_out/pwl_v2.json",
    )
    p.add_argument("--bit", default="8")
    p.add_argument("--n_points", type=int, default=2048)
    p.add_argument("--fontsize", type=int, default=30)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--out_dir", default="/workspace/PointKAN-pytorch/my_progressive_out/shapenet_pwl_v2_visualization")
    return p.parse_args()


if __name__ == "__main__":
    plot_all_layers(parse_args())
