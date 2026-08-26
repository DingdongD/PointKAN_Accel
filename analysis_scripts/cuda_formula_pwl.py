#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CUDA-formula GRF -> PWL utilities.

This script is intentionally outside the original PointKAN model folders. It can
generate a deployable PWL JSON and run MSE sweeps over segment count, breakpoint
bits, and slope/intercept coefficient bits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def eval_cuda_style_grf(x: np.ndarray, numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    num = np.asarray(numerator, dtype=np.float64).reshape(-1)
    den = np.asarray(denominator, dtype=np.float64).reshape(-1)
    p = np.zeros_like(x, dtype=np.float64)
    for k, a in enumerate(num):
        p += float(a) * (x ** k)
    q = np.ones_like(x, dtype=np.float64)
    for k, b in enumerate(den, start=1):
        q += np.abs(float(b) * (x ** k))
    return p / np.maximum(q, 1e-8)


def quantize(values: np.ndarray, bits: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if bits <= 0 or values.size == 0:
        return values
    max_abs = float(np.max(np.abs(values)))
    if max_abs == 0.0:
        return values
    qmax = (1 << (int(bits) - 1)) - 1
    scale = qmax / max_abs
    return np.round(values * scale) / scale


def layer_x_range(layer: dict, fallback: tuple[float, float]) -> tuple[float, float]:
    xr = layer.get("input_range") or layer.get("x_range")
    if xr is None:
        return fallback
    return float(xr["min"]), float(xr["max"])


def numerator_row(weight_numerator: np.ndarray, group_idx: int) -> np.ndarray:
    arr = np.asarray(weight_numerator, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    return arr[min(group_idx, arr.shape[0] - 1)]


def fit_group(
    *,
    x_min: float,
    x_max: float,
    numerator: np.ndarray,
    denominator: np.ndarray,
    num_splits: int,
    breakpoint_bits: int,
    coef_bits: int,
    x_step: float,
) -> dict:
    n_intervals = int(num_splits) + 1
    edges = np.linspace(x_min, x_max, n_intervals + 1, dtype=np.float64)
    if breakpoint_bits > 0:
        levels = (1 << int(breakpoint_bits)) - 1
        q = np.round((edges - x_min) / (x_max - x_min) * levels)
        edges = x_min + q / levels * (x_max - x_min)
        edges[0] = x_min
        edges[-1] = x_max
        edges = np.maximum.accumulate(edges)

    slopes = []
    intercepts = []
    for i in range(n_intervals):
        lo, hi = float(edges[i]), float(edges[i + 1])
        xs = np.arange(lo, hi + (x_step * 0.5 if i == n_intervals - 1 else 0.0), x_step)
        if xs.size < 2:
            xs = np.asarray([lo, hi], dtype=np.float64)
        ys = eval_cuda_style_grf(xs, numerator, denominator)
        a = np.stack([xs, np.ones_like(xs)], axis=1)
        m, b = np.linalg.lstsq(a, ys, rcond=None)[0]
        slopes.append(float(m))
        intercepts.append(float(b))

    slopes_q = quantize(np.asarray(slopes, dtype=np.float64), int(coef_bits))
    intercepts_q = quantize(np.asarray(intercepts, dtype=np.float64), int(coef_bits))

    sqerr = []
    for i in range(n_intervals):
        lo, hi = float(edges[i]), float(edges[i + 1])
        xs = np.arange(lo, hi + (x_step * 0.5 if i == n_intervals - 1 else 0.0), x_step)
        if xs.size < 2:
            xs = np.asarray([lo, hi], dtype=np.float64)
        ys = eval_cuda_style_grf(xs, numerator, denominator)
        pred = slopes_q[i] * xs + intercepts_q[i]
        sqerr.extend(((pred - ys) ** 2).tolist())

    return {
        "group_id": None,
        "mode": "pwl",
        "valid": True,
        "reason": "",
        "breakpoints": [float(v) for v in edges[1:-1]],
        "slopes": [float(v) for v in slopes_q],
        "intercepts": [float(v) for v in intercepts_q],
        "mse": float(np.mean(sqerr)) if sqerr else math.nan,
    }


def build_pwl_payload(
    grf_payload: dict,
    *,
    num_splits: int,
    breakpoint_bits: int,
    coef_bits: int,
    x_step: float,
    fallback_x_range: tuple[float, float],
) -> dict:
    layers_out = []
    for layer in grf_payload.get("layers", []):
        name = str(layer["name"])
        x_min, x_max = layer_x_range(layer, fallback_x_range)
        wn = np.asarray(layer["weight_numerator"], dtype=np.float64)
        wd = np.asarray(layer["weight_denominator"], dtype=np.float64)
        if wd.ndim == 1:
            wd = wd[np.newaxis, :]
        groups = []
        for g in range(wd.shape[0]):
            spec = fit_group(
                x_min=x_min,
                x_max=x_max,
                numerator=numerator_row(wn, g),
                denominator=wd[g],
                num_splits=num_splits,
                breakpoint_bits=breakpoint_bits,
                coef_bits=coef_bits,
                x_step=x_step,
            )
            spec["group_id"] = g
            groups.append(spec)
        layers_out.append(
            {
                "name": name,
                "x_range": {"min": float(x_min), "max": float(x_max)},
                "num_splits": int(num_splits),
                "num_intervals": int(num_splits) + 1,
                "num_groups": int(wd.shape[0]),
                "group_specific_breakpoints": True,
                "fit_method": "uniform_least_squares_per_segment_cuda_formula",
                "quantization_aware_fitness": False,
                "pruning_enabled": False,
                "groups": [],
                "bits": {
                    str(int(breakpoint_bits)): {
                        "breakpoint_bits": int(breakpoint_bits),
                        "groups": groups,
                        "summary": {
                            "mean_mse": float(np.mean([g["mse"] for g in groups])),
                            "max_mse": float(np.max([g["mse"] for g in groups])),
                            "coef_bits": int(coef_bits),
                            "formula": "cuda_abs_term",
                        },
                    }
                },
                "search_summary": {
                    "formula": "P(x)/(1+sum(abs(b_i*x**i)))",
                    "x_step": float(x_step),
                },
            }
        )

    return {
        "meta": {
            "source": os.path.abspath(str(grf_payload.get("_source", ""))),
            "num_splits": int(num_splits),
            "num_intervals": int(num_splits) + 1,
            "decimal_bit_range": [0, int(breakpoint_bits)],
            "deploy_breakpoint_bits": int(breakpoint_bits),
            "w_b_bit": int(coef_bits),
            "x_step": float(x_step),
            "formula": "P(x)/(1+sum(abs(b_i*x**i)))",
            "fit_method": "uniform_least_squares_per_segment_cuda_formula",
        },
        "layers": layers_out,
    }


def load_grf(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["_source"] = os.path.abspath(path)
    return payload


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def run_generate(args: argparse.Namespace) -> None:
    payload = build_pwl_payload(
        load_grf(args.grf_json),
        num_splits=args.num_splits,
        breakpoint_bits=args.breakpoint_bits,
        coef_bits=args.coef_bits,
        x_step=args.x_step,
        fallback_x_range=(args.x_range[0], args.x_range[1]),
    )
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"wrote: {args.output_json}")


def plot_heatmap(matrix: np.ndarray, row_labels: list[int], col_labels: list[int], title: str, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(col_labels)), [str(x) for x in col_labels])
    ax.set_yticks(range(len(row_labels)), [str(x) for x in row_labels])
    ax.set_xlabel("coefficient bits")
    ax.set_ylabel("breakpoint bits")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.1e}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=ax, label="mean MSE")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run_sweep(args: argparse.Namespace) -> None:
    grf = load_grf(args.grf_json)
    splits_values = parse_int_list(args.num_splits_values)
    bp_values = parse_int_list(args.breakpoint_bits_values)
    coef_values = parse_int_list(args.coef_bits_values)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    per_layer_rows = []
    for ns in splits_values:
        matrix = np.zeros((len(bp_values), len(coef_values)), dtype=np.float64)
        for bi, bp in enumerate(bp_values):
            for ci, cb in enumerate(coef_values):
                payload = build_pwl_payload(
                    grf,
                    num_splits=ns,
                    breakpoint_bits=bp,
                    coef_bits=cb,
                    x_step=args.x_step,
                    fallback_x_range=(args.x_range[0], args.x_range[1]),
                )
                layer_mses = []
                for layer in payload["layers"]:
                    summary = layer["bits"][str(bp)]["summary"]
                    layer_mses.append(float(summary["mean_mse"]))
                    per_layer_rows.append(
                        {
                            "num_splits": ns,
                            "breakpoint_bits": bp,
                            "coef_bits": cb,
                            "layer": layer["name"],
                            "mean_mse": summary["mean_mse"],
                            "max_mse": summary["max_mse"],
                        }
                    )
                overall = float(np.mean(layer_mses))
                matrix[bi, ci] = overall
                rows.append(
                    {
                        "num_splits": ns,
                        "breakpoint_bits": bp,
                        "coef_bits": cb,
                        "overall_mean_mse": overall,
                        "overall_max_layer_mean_mse": float(np.max(layer_mses)),
                    }
                )
        plot_heatmap(matrix, bp_values, coef_values, f"CUDA-formula PWL MSE, num_splits={ns}", out_dir / f"overall_mse_heatmap_s{ns}.png")

    with open(out_dir / "overall_mse_long.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "per_layer_mse_long.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_layer_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_layer_rows)
    with open(out_dir / "sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "grf_json": os.path.abspath(args.grf_json),
                "num_splits_values": splits_values,
                "breakpoint_bits_values": bp_values,
                "coef_bits_values": coef_values,
                "x_step": args.x_step,
                "formula": "P(x)/(1+sum(abs(b_i*x**i)))",
                "overall_csv": str(out_dir / "overall_mse_long.csv"),
                "per_layer_csv": str(out_dir / "per_layer_mse_long.csv"),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"wrote sweep: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CUDA-formula GRF -> PWL generator/sweeper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate")
    gen.add_argument("--grf_json", required=True)
    gen.add_argument("--output_json", required=True)
    gen.add_argument("--num_splits", type=int, default=8)
    gen.add_argument("--breakpoint_bits", type=int, default=8)
    gen.add_argument("--coef_bits", type=int, default=8)
    gen.add_argument("--x_step", type=float, default=0.005)
    gen.add_argument("--x_range", type=float, nargs=2, default=(-4.0, 4.0))
    gen.set_defaults(func=run_generate)

    sweep = sub.add_parser("sweep")
    sweep.add_argument("--grf_json", required=True)
    sweep.add_argument("--out_dir", required=True)
    sweep.add_argument("--num_splits_values", default="4,8,16,32")
    sweep.add_argument("--breakpoint_bits_values", default="6,8,10,12")
    sweep.add_argument("--coef_bits_values", default="6,8,10,12")
    sweep.add_argument("--x_step", type=float, default=0.01)
    sweep.add_argument("--x_range", type=float, nargs=2, default=(-4.0, 4.0))
    sweep.set_defaults(func=run_sweep)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
