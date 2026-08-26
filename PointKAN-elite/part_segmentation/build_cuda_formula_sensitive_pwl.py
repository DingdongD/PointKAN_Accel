#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build PWL layers against the CUDA KAT denominator convention.

The existing GQA-LUT trainer fits Q(x)=sum(abs(b_i) * x^i), while the CUDA KAT
kernel behaves much closer to Q(x)=1+sum(abs(b_i * x^i)). This utility creates
least-squares PWL entries for selected sensitive layers using that CUDA-style
formula, then emits a mixed JSON that can be consumed by the existing injector.
"""

import argparse
import copy
import json
import math
import os
from pathlib import Path

import numpy as np


def _eval_cuda_style_grf(x: np.ndarray, numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    num = np.asarray(numerator, dtype=np.float64).reshape(-1)
    den = np.asarray(denominator, dtype=np.float64).reshape(-1)
    p = sum(float(num[k]) * (x ** k) for k in range(num.size))
    q = np.ones_like(x, dtype=np.float64)
    for k, b in enumerate(den, start=1):
        q += np.abs(float(b) * (x ** k))
    q = np.maximum(q, 1e-8)
    return p / q


def _quantize(values: np.ndarray, bits: int | None) -> np.ndarray:
    if bits is None or bits <= 0:
        return values.astype(np.float64)
    max_abs = float(np.max(np.abs(values))) if values.size else 0.0
    if max_abs == 0.0:
        return values.astype(np.float64)
    qmax = (1 << (bits - 1)) - 1
    scale = qmax / max_abs
    return np.round(values * scale) / scale


def _fit_group(
    *,
    x_min: float,
    x_max: float,
    numerator: np.ndarray,
    denominator: np.ndarray,
    num_splits: int,
    breakpoint_bits: int,
    coef_bits: int | None,
    x_step: float,
) -> dict:
    n_intervals = int(num_splits) + 1
    edges = np.linspace(x_min, x_max, n_intervals + 1, dtype=np.float64)
    if breakpoint_bits > 0:
        levels = (1 << breakpoint_bits) - 1
        q = np.round((edges - x_min) / (x_max - x_min) * levels)
        edges = x_min + q / levels * (x_max - x_min)
        edges[0] = x_min
        edges[-1] = x_max
        edges = np.maximum.accumulate(edges)

    slopes = []
    intercepts = []
    sqerr = []
    npts = []
    for i in range(n_intervals):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == n_intervals - 1:
            xs = np.arange(lo, hi + x_step * 0.5, x_step, dtype=np.float64)
        else:
            xs = np.arange(lo, hi, x_step, dtype=np.float64)
        if xs.size < 2:
            xs = np.asarray([lo, hi], dtype=np.float64)
        ys = _eval_cuda_style_grf(xs, numerator, denominator)
        a = np.stack([xs, np.ones_like(xs)], axis=1)
        m, b = np.linalg.lstsq(a, ys, rcond=None)[0]
        slopes.append(float(m))
        intercepts.append(float(b))
        pred = m * xs + b
        sqerr.extend(((pred - ys) ** 2).tolist())
        npts.append(int(xs.size))

    slopes_arr = _quantize(np.asarray(slopes, dtype=np.float64), coef_bits)
    intercepts_arr = _quantize(np.asarray(intercepts, dtype=np.float64), coef_bits)

    # Recompute MSE after coefficient quantization.
    sqerr_q = []
    for i in range(n_intervals):
        lo, hi = float(edges[i]), float(edges[i + 1])
        xs = np.arange(lo, hi + (x_step * 0.5 if i == n_intervals - 1 else 0.0), x_step, dtype=np.float64)
        if xs.size < 2:
            xs = np.asarray([lo, hi], dtype=np.float64)
        ys = _eval_cuda_style_grf(xs, numerator, denominator)
        pred = slopes_arr[i] * xs + intercepts_arr[i]
        sqerr_q.extend(((pred - ys) ** 2).tolist())

    return {
        "group_id": None,
        "mode": "pwl",
        "valid": True,
        "reason": "",
        "breakpoints": [float(v) for v in edges[1:-1]],
        "slopes": [float(v) for v in slopes_arr],
        "intercepts": [float(v) for v in intercepts_arr],
        "mse": float(np.mean(sqerr_q)) if sqerr_q else math.nan,
    }


def _num_row(weight_numerator: np.ndarray, group_idx: int) -> np.ndarray:
    arr = np.asarray(weight_numerator, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    return arr[min(group_idx, arr.shape[0] - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mixed PWL JSON using CUDA-style GRF formula")
    parser.add_argument("--grf_json", required=True)
    parser.add_argument("--base_json", required=True)
    parser.add_argument("--sensitive_layers", nargs="+", required=True)
    parser.add_argument("--base_bit", default="8")
    parser.add_argument("--deploy_bit", default="10")
    parser.add_argument("--num_splits", type=int, default=32)
    parser.add_argument("--coef_bits", type=int, default=12)
    parser.add_argument("--x_step", type=float, default=0.0025)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    with open(args.grf_json, "r", encoding="utf-8") as f:
        grf = json.load(f)
    with open(args.base_json, "r", encoding="utf-8") as f:
        out = json.load(f)

    grf_by_name = {str(l["name"]): l for l in grf.get("layers", [])}
    sensitive_set = set(args.sensitive_layers)
    deploy_bit = str(args.deploy_bit)

    out = copy.deepcopy(out)
    meta = copy.deepcopy(out.get("meta", {}))
    meta["cuda_formula_sensitive_pwl"] = {
        "grf_json": os.path.abspath(args.grf_json),
        "base_json": os.path.abspath(args.base_json),
        "sensitive_layers": args.sensitive_layers,
        "base_bit_source": str(args.base_bit),
        "deploy_bit_key": deploy_bit,
        "num_splits_sensitive": int(args.num_splits),
        "coef_bits_sensitive": int(args.coef_bits),
        "formula": "P(x) / (1 + sum(abs(b_i * x**i)))",
    }
    meta["deploy_breakpoint_bits"] = int(deploy_bit)
    out["meta"] = meta

    for layer in out.get("layers", []):
        name = str(layer["name"])
        if name not in sensitive_set:
            bits = layer.get("bits", {})
            bits[deploy_bit] = copy.deepcopy(bits[str(args.base_bit)])
            bits[deploy_bit]["breakpoint_bits"] = int(deploy_bit)
            continue

        if name not in grf_by_name:
            raise KeyError(f"{name!r} missing in {args.grf_json}")
        coeff = grf_by_name[name]
        x_min = float(layer["x_range"]["min"])
        x_max = float(layer["x_range"]["max"])
        den = np.asarray(coeff["weight_denominator"], dtype=np.float64)
        if den.ndim == 1:
            den = den[np.newaxis, :]
        groups = []
        for g in range(den.shape[0]):
            spec = _fit_group(
                x_min=x_min,
                x_max=x_max,
                numerator=_num_row(np.asarray(coeff["weight_numerator"], dtype=np.float64), g),
                denominator=den[g],
                num_splits=args.num_splits,
                breakpoint_bits=int(args.deploy_bit),
                coef_bits=int(args.coef_bits),
                x_step=float(args.x_step),
            )
            spec["group_id"] = g
            groups.append(spec)
        layer["num_splits"] = int(args.num_splits)
        layer["num_intervals"] = int(args.num_splits) + 1
        layer["bits"] = {
            deploy_bit: {
                "breakpoint_bits": int(args.deploy_bit),
                "groups": groups,
                "summary": {
                    "mean_mse": float(np.mean([g["mse"] for g in groups])),
                    "max_mse": float(np.max([g["mse"] for g in groups])),
                    "coef_bits": int(args.coef_bits),
                    "formula": "cuda_abs_term",
                },
            }
        }

    missing = sorted(sensitive_set - {str(l["name"]) for l in out.get("layers", [])})
    if missing:
        raise KeyError(f"sensitive layers missing in base json: {missing}")

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"wrote: {args.output_json}")


if __name__ == "__main__":
    main()
