#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.ticker import FormatStrFormatter


@dataclass
class TaskSpec:
    name: str
    coeff_json: str


def _setup_font(font_size: int = 24) -> str:
    candidates = [
        os.path.expanduser("~/.fonts/Arial.ttf"),
        os.path.expanduser("~/.fonts/arial.ttf"),
        os.path.expanduser("~/.fonts/arialbd.ttf"),
        os.path.expanduser("~/.fonts/ARIALN.TTF"),
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
            "axes.linewidth": 2.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    return font_name


def _eval_formula(
    x: np.ndarray,
    weight_numerator: np.ndarray,
    weight_denominator: np.ndarray,
    formula: str,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    num = np.asarray(weight_numerator, dtype=np.float64)
    den = np.asarray(weight_denominator, dtype=np.float64)
    if num.ndim == 1:
        num = num[None, :]
    if den.ndim == 1:
        den = den[None, :]

    xp_num = np.stack([x**k for k in range(num.shape[1])], axis=0)
    out = []
    for g in range(den.shape[0]):
        num_row = num[min(g, num.shape[0] - 1)]
        p = num_row @ xp_num
        if formula == "legacy_abs_coeff":
            q = np.ones_like(x, dtype=np.float64)
            for k in range(den.shape[1]):
                q = q + abs(float(den[g, k])) * (x ** (k + 1))
        elif formula == "cuda_abs_term":
            q = np.ones_like(x, dtype=np.float64)
            for k in range(den.shape[1]):
                q = q + np.abs(float(den[g, k]) * (x ** (k + 1)))
        else:
            raise ValueError(f"Unknown formula: {formula}")
        out.append(p / np.maximum(q, 1e-8))
    return np.stack(out, axis=0)


def _layer_range(layer: Dict, default_range: Tuple[float, float], min_width: float = 1e-3) -> Tuple[float, float]:
    r = layer.get("input_range") or {}
    lo = float(r.get("min", default_range[0]))
    hi = float(r.get("max", default_range[1]))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < min_width:
        return default_range
    return lo, hi


def _metrics(y_legacy: np.ndarray, y_cuda: np.ndarray) -> Dict[str, float]:
    diff = y_cuda - y_legacy
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    denom = float(np.sqrt(np.mean(y_cuda**2))) + 1e-12
    rel_rmse = float(np.sqrt(mse) / denom)
    return {"mse": mse, "mae": mae, "max_abs": max_abs, "rel_rmse": rel_rmse}


def _write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _style_ax(ax) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(2.8)
    ax.tick_params(axis="both", labelsize=18, width=2.4, length=7)
    ax.grid(True, linestyle="--", linewidth=1.0, alpha=0.35)


def _plot_heatmap(task: str, layers: List[str], heat: np.ndarray, out_path: str, title: str) -> None:
    fig_w = max(12, 0.62 * len(layers))
    fig, ax = plt.subplots(figsize=(fig_w, 6.5))
    im = ax.imshow(heat, aspect="auto", cmap="magma")
    ax.set_title(title, fontsize=24)
    ax.set_xlabel("Layer", fontsize=22)
    ax.set_ylabel("Group", fontsize=22)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(i) for i in range(len(layers))], fontsize=16)
    ax.set_yticks(range(heat.shape[0]))
    ax.set_yticklabels([str(i) for i in range(heat.shape[0])], fontsize=16)
    for spine in ax.spines.values():
        spine.set_linewidth(2.8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=16, width=2.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_task_worst_curves(
    task: str,
    coeff_data: Dict,
    rows: List[Dict],
    out_dir: str,
    top_k: int,
    default_range: Tuple[float, float],
    n_points: int,
) -> None:
    layers = coeff_data["layers"]
    top_rows = sorted(rows, key=lambda r: r["input_rel_rmse"], reverse=True)[:top_k]
    for rank, row in enumerate(top_rows, start=1):
        layer_idx = int(row["layer_idx"])
        group_idx = int(row["group_idx"])
        layer = layers[layer_idx]
        lo, hi = _layer_range(layer, default_range)
        x = np.linspace(lo, hi, n_points)
        wn = np.asarray(layer["weight_numerator"], dtype=np.float64)
        wd = np.asarray(layer["weight_denominator"], dtype=np.float64)
        y_legacy = _eval_formula(x, wn, wd, "legacy_abs_coeff")[group_idx]
        y_cuda = _eval_formula(x, wn, wd, "cuda_abs_term")[group_idx]
        diff = y_cuda - y_legacy

        fig, axes = plt.subplots(2, 1, figsize=(13.5, 10), sharex=True, height_ratios=[2.2, 1.0])
        axes[0].plot(x, y_legacy, linewidth=3.0, label="Infer legacy", color="#1f77b4")
        axes[0].plot(x, y_cuda, linewidth=3.0, label="CUDA formula", color="#d62728", linestyle="--")
        axes[0].set_ylabel("GRF output", fontsize=22)
        axes[0].set_title(
            f"{task} rank {rank}: L{layer_idx} G{group_idx} relRMSE={row['input_rel_rmse']:.3e}",
            fontsize=22,
        )
        axes[0].legend(fontsize=18, frameon=False)
        axes[1].plot(x, diff, linewidth=3.0, color="#2ca02c")
        axes[1].axhline(0.0, color="#444444", linewidth=2.0, linestyle="--")
        axes[1].set_xlabel("x", fontsize=22)
        axes[1].set_ylabel("CUDA - legacy", fontsize=22)
        for ax in axes:
            _style_ax(ax)
        fig.tight_layout()
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in layer["name"])
        base = os.path.join(out_dir, f"{task}_worst_rank{rank:02d}_L{layer_idx:02d}_G{group_idx}_{safe_name}")
        fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(base + ".pdf", bbox_inches="tight")
        plt.close(fig)


def _plot_all_layer_curves(
    task: str,
    coeff_data: Dict,
    out_dir: str,
    default_range: Tuple[float, float],
    n_points: int,
) -> None:
    layers = coeff_data["layers"]
    n_layers = len(layers)
    ncols = 4
    nrows = int(np.ceil(n_layers / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(40, 30), squeeze=False)
    legacy_label = "Inference GRF (legacy Python/PWL)"
    cuda_label = "Train GRF (Triton CUDA)"
    highlight_label = "Different region"
    highlight_label_added = False

    for li, layer in enumerate(layers):
        ax = axes[li // ncols][li % ncols]
        lo, hi = _layer_range(layer, default_range)
        x = np.linspace(lo, hi, n_points)
        wn = np.asarray(layer["weight_numerator"], dtype=np.float64)
        wd = np.asarray(layer["weight_denominator"], dtype=np.float64)
        y_legacy = _eval_formula(x, wn, wd, "legacy_abs_coeff")
        y_cuda = _eval_formula(x, wn, wd, "cuda_abs_term")
        diff_abs = np.max(np.abs(y_cuda - y_legacy), axis=0)
        diff_thr = max(float(np.max(diff_abs)) * 0.05, 1e-8)
        diff_mask = diff_abs > diff_thr
        ax.set_facecolor("#eeeeee")
        if np.any(diff_mask):
            start = None
            for idx, flag in enumerate(diff_mask):
                if flag and start is None:
                    start = idx
                is_last = idx == len(diff_mask) - 1
                if start is not None and ((not flag) or is_last):
                    end = idx if not flag else idx + 1
                    ax.axvspan(
                        x[start],
                        x[min(end, len(x) - 1)],
                        color="#ffe680",
                        alpha=0.42,
                        linewidth=0,
                        label=highlight_label if not highlight_label_added else None,
                    )
                    highlight_label_added = True
                    start = None
        for gi in range(y_legacy.shape[0]):
            ax.plot(
                x,
                y_legacy[gi],
                color="#1f77b4",
                linewidth=1.3,
                alpha=0.34,
                label=legacy_label if li == 0 and gi == 0 else None,
            )
            ax.plot(
                x,
                y_cuda[gi],
                color="#d62728",
                linewidth=1.3,
                alpha=0.34,
                linestyle="--",
                label=cuda_label if li == 0 and gi == 0 else None,
            )
        ax.axvline(0.0, color="#444444", linewidth=0.9, alpha=0.35)
        ax.set_title(f"L{li}: {layer['name']}", fontsize=36)
        ax.tick_params(axis="both", labelsize=36, width=2.8, length=8)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        for spine in ax.spines.values():
            spine.set_linewidth(2.8)
        ax.grid(True, linestyle="--", linewidth=1.2, alpha=0.25)

    for li in range(n_layers, nrows * ncols):
        axes[li // ncols][li % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=len(labels),
        fontsize=36,
        frameon=False,
    )
    fig.tight_layout(rect=[0.02, 0.025, 1.0, 0.89])
    base = os.path.join(out_dir, f"{task}_all_layers_inference_vs_train_grf")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def _analyze_task(
    spec: TaskSpec,
    out_dir: str,
    default_range: Tuple[float, float],
    fixed_range: Tuple[float, float],
    n_points: int,
    top_k: int,
) -> Tuple[List[Dict], Dict]:
    coeff_data = json.load(open(spec.coeff_json))
    layers = coeff_data["layers"]
    rows = []
    heat = None
    for li, layer in enumerate(layers):
        wn = np.asarray(layer["weight_numerator"], dtype=np.float64)
        wd = np.asarray(layer["weight_denominator"], dtype=np.float64)
        lo, hi = _layer_range(layer, default_range)
        x_input = np.linspace(lo, hi, n_points)
        x_fixed = np.linspace(fixed_range[0], fixed_range[1], n_points)
        y_legacy_input = _eval_formula(x_input, wn, wd, "legacy_abs_coeff")
        y_cuda_input = _eval_formula(x_input, wn, wd, "cuda_abs_term")
        y_legacy_fixed = _eval_formula(x_fixed, wn, wd, "legacy_abs_coeff")
        y_cuda_fixed = _eval_formula(x_fixed, wn, wd, "cuda_abs_term")
        if heat is None:
            heat = np.zeros((wd.shape[0], len(layers)), dtype=np.float64)
        for gi in range(wd.shape[0]):
            mi = _metrics(y_legacy_input[gi], y_cuda_input[gi])
            mf = _metrics(y_legacy_fixed[gi], y_cuda_fixed[gi])
            heat[gi, li] = mi["rel_rmse"]
            rows.append(
                {
                    "task": spec.name,
                    "layer_idx": li,
                    "layer": layer["name"],
                    "group_idx": gi,
                    "input_min": lo,
                    "input_max": hi,
                    "input_mse": mi["mse"],
                    "input_mae": mi["mae"],
                    "input_max_abs": mi["max_abs"],
                    "input_rel_rmse": mi["rel_rmse"],
                    "fixed_mse": mf["mse"],
                    "fixed_mae": mf["mae"],
                    "fixed_max_abs": mf["max_abs"],
                    "fixed_rel_rmse": mf["rel_rmse"],
                    "b1": float(wd[gi, 0]),
                    "b2": float(wd[gi, 1]),
                    "b3": float(wd[gi, 2]),
                    "b4": float(wd[gi, 3]),
                }
            )

    task_dir = os.path.join(out_dir, spec.name)
    os.makedirs(task_dir, exist_ok=True)
    _write_csv(os.path.join(task_dir, "formula_compare_layer_group.csv"), rows)
    _plot_heatmap(
        spec.name,
        [l["name"] for l in layers],
        np.log10(heat + 1e-12),
        os.path.join(task_dir, "formula_compare_input_range_relrmse_heatmap_log10.png"),
        f"{spec.name}: log10 relRMSE, input range",
    )
    _plot_all_layer_curves(spec.name, coeff_data, task_dir, default_range, n_points)
    _plot_task_worst_curves(spec.name, coeff_data, rows, task_dir, top_k, default_range, n_points)

    summary = {
        "task": spec.name,
        "num_layers": len(layers),
        "num_groups": int(np.asarray(layers[0]["weight_denominator"]).shape[0]),
        "mean_input_rel_rmse": float(np.mean([r["input_rel_rmse"] for r in rows])),
        "median_input_rel_rmse": float(np.median([r["input_rel_rmse"] for r in rows])),
        "max_input_rel_rmse": float(np.max([r["input_rel_rmse"] for r in rows])),
        "mean_fixed_rel_rmse": float(np.mean([r["fixed_rel_rmse"] for r in rows])),
        "max_fixed_rel_rmse": float(np.max([r["fixed_rel_rmse"] for r in rows])),
        "worst_input": sorted(rows, key=lambda r: r["input_rel_rmse"], reverse=True)[0],
    }
    return rows, summary


def _plot_overall_summary(summaries: List[Dict], out_dir: str) -> None:
    tasks = [s["task"] for s in summaries]
    mean_vals = [s["mean_input_rel_rmse"] for s in summaries]
    max_vals = [s["max_input_rel_rmse"] for s in summaries]
    x = np.arange(len(tasks))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.bar(x - width / 2, mean_vals, width=width, label="Mean", color="#1f77b4")
    ax.bar(x + width / 2, max_vals, width=width, label="Max", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=18)
    ax.set_ylabel("Relative RMSE", fontsize=22)
    ax.set_title("CUDA vs Legacy GRF Formula Difference", fontsize=24)
    ax.legend(fontsize=18, frameon=False)
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "formula_compare_task_summary.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "formula_compare_task_summary.pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare legacy Infer GRF formula against CUDA-aligned formula.")
    p.add_argument("--out_dir", default="/workspace/PointKAN-pytorch/my_progressive_out/grf_formula_compare")
    p.add_argument("--n_points", type=int, default=2048)
    p.add_argument("--top_k", type=int, default=6)
    p.add_argument("--default_range", type=float, nargs=2, default=[-3.0, 3.0])
    p.add_argument("--fixed_range", type=float, nargs=2, default=[-3.0, 3.0])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _setup_font(font_size=36)
    specs = [
        TaskSpec(
            "ModelNet40",
            "/workspace/PointKAN-pytorch/PointKAN-elite/classification_ModelNet40/grf_export_out/grf_coefficients.json",
        ),
        TaskSpec(
            "ShapeNetPart",
            "/workspace/PointKAN-pytorch/PointKAN-elite/part_segmentation/grf_export_out/grf_coefficients.json",
        ),
        TaskSpec(
            "ScanObjectNN",
            "/workspace/PointKAN-pytorch/PointKAN-elite/classification_ScanObjectNN/grf_export_out/grf_coefficients.json",
        ),
    ]
    os.makedirs(args.out_dir, exist_ok=True)
    all_rows: List[Dict] = []
    summaries: List[Dict] = []
    for spec in specs:
        rows, summary = _analyze_task(
            spec,
            args.out_dir,
            tuple(args.default_range),
            tuple(args.fixed_range),
            args.n_points,
            args.top_k,
        )
        all_rows.extend(rows)
        summaries.append(summary)
        print(json.dumps(summary, indent=2))
    _write_csv(os.path.join(args.out_dir, "formula_compare_all_layer_group.csv"), all_rows)
    _write_csv(os.path.join(args.out_dir, "formula_compare_task_summary.csv"), summaries)
    with open(os.path.join(args.out_dir, "formula_compare_summary.json"), "w") as f:
        json.dump({"summaries": summaries}, f, indent=2)
    _plot_overall_summary(summaries, args.out_dir)
    print("WROTE", args.out_dir)


if __name__ == "__main__":
    main()
