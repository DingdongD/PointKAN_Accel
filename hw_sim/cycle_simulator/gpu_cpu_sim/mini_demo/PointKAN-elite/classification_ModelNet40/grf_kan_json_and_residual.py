#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRF-KAN 系数 JSON 导出、value 空间 base+残差分解、f(x) 曲线量化分析

- 将每层每组的 numerator/denominator 导出为 JSON，可从 JSON 直接得到 GRF-KAN 的 f(x)。
- 残差分解框架（与 inference_and_kan_analysis 中一致）：以组间 GRF 的 mean / min / max curve 为 base，
  残差 curve = f_g(x) - base(x)；重建 f_g(x) = base(x) + residual_g(x)，用于验证与存储分析。
- 量化针对 f(x) curve：对每个激活 x 对应的 f(x) 值做量化/反量化，分析并可视化原始 f(x) 与量化后 f(x) 的差异。
"""

import json
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
for d in (SCRIPT_DIR, ROOT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)


# ---------- 1) 从系数计算 f(x) = P(x)/Q(x) ----------

def eval_grf_vanilla(x, weight_numerator, weight_denominator):
    """
    Vanilla GRF-KAN：用原始分子/分母系数计算 f(x) = P(x)/Q(x)。
    返回: (n_groups, n_points)
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        x = x.reshape(1)
    num_w = np.asarray(weight_numerator, dtype=np.float64)
    den_w = np.asarray(weight_denominator, dtype=np.float64)
    if num_w.ndim == 1:
        num_w = num_w[np.newaxis, :]
    n_groups = den_w.shape[0]
    num_rows = num_w.shape[0]
    xp_num = np.stack([x ** k for k in range(6)], axis=0)
    xp_den = np.stack([x ** k for k in range(5)], axis=0)
    out = []
    for g in range(n_groups):
        num_row = num_w[min(g, num_rows - 1)]
        P = num_row @ xp_num
        q_coeff = np.concatenate([[1.0], np.abs(den_w[g])])
        Q = q_coeff @ xp_den
        Q = np.maximum(Q, 1e-8)
        out.append(P / Q)
    return np.stack(out, axis=0)


# ---------- 2) Value 空间 base + 残差曲线（与 inference_and_kan_analysis 组间 GRF 框架一致）----------

def get_base_and_residual_curves(f_curves, base_kind="mean"):
    """
    给定各组 f(x) 曲线 f_curves shape (G, N)，按 base_kind 取 base 曲线，得到残差曲线。
    base_kind: "mean" | "min" | "max"
    - mean: base(x) = mean over groups of f_g(x)
    - min:  base(x) = min over groups of f_g(x)
    - max:  base(x) = max over groups of f_g(x)
    返回: base (N,), residuals (G, N)。满足 f_curves = base + residuals（广播）。
    """
    f_curves = np.asarray(f_curves, dtype=np.float64)
    if base_kind == "mean":
        base = f_curves.mean(axis=0) 
    elif base_kind == "min":
        base = f_curves.min(axis=0)
    elif base_kind == "max":
        base = f_curves.max(axis=0)
    else:
        raise ValueError("base_kind must be mean | min | max")
    residuals = f_curves - base
    return base, residuals


def verify_base_plus_residual_equals_vanilla(f_curves, base_kind="mean", atol=1e-10):
    """
    验证 base(x) + residual_g(x) == f_g(x)。由定义恒成立，仅做数值检查。
    f_curves: (G, N)
    返回 (passed, max_abs_diff)
    """
    base, residuals = get_base_and_residual_curves(f_curves, base_kind=base_kind)
    reconstructed = base + residuals
    diff = np.abs(reconstructed - f_curves)
    max_diff = float(np.max(diff))
    passed = bool(np.allclose(reconstructed, f_curves, atol=atol))
    return passed, max_diff


def residual_curve_dynamic_range(f_curves, base_kind="mean"):
    """
    返回 base 与 residual 的动态范围（min, max, span），用于存储分析。
    """
    base, residuals = get_base_and_residual_curves(f_curves, base_kind=base_kind)
    def _range(a):
        a = np.asarray(a)
        lo, hi = float(np.min(a)), float(np.max(a))
        return {"min": round(lo, 6), "max": round(hi, 6), "span": round(hi - lo, 6)}
    return {"base": _range(base), "residual": _range(residuals)}


# ---------- 3) f(x) 曲线量化（对每个 x 处的 f(x) 值做量化）----------

def quantize_dequantize_values(arr, bits=8, symmetric=True):
    """
    对数组中的每个值做线性量化再反量化（用于 f(x) curve 的量化）。
    arr: 任意 shape，一般为 (G, N) 的 f(x) 曲线值
    """
    arr = np.asarray(arr, dtype=np.float64)
    n = 2 ** bits
    if symmetric:
        q_min, q_max = -(n // 2), (n // 2) - 1
        amax = np.abs(arr).max()
        scale = amax / max(q_max, 1) if amax > 1e-12 else 1.0
        zp = 0
    else:
        q_min, q_max = 0, n - 1
        lo, hi = arr.min(), arr.max()
        scale = (hi - lo) / max(q_max - q_min, 1) if (hi - lo) > 1e-12 else 1.0
        zp = q_min - lo / scale
    q = np.round(arr / scale + zp).astype(np.int32)
    q = np.clip(q, q_min, q_max)
    return (q.astype(np.float64) - zp) * scale


def quantize_fx_curve(f_curves, bits=8, symmetric=True):
    """
    对 f(x) 曲线数组 (G, N) 做量化再反量化，得到量化后的曲线。
    """
    return quantize_dequantize_values(f_curves, bits=bits, symmetric=symmetric)


def quantization_analysis_fx_curve(f_curves, bits_list=(8, 6, 4)):
    """
    对单层 f(x) 曲线做多比特量化分析。
    f_curves: (G, N)
    返回 list of dict: {bits, mse, max_abs_err, mean_abs_err}
    """
    results = []
    for b in bits_list:
        f_q = quantize_fx_curve(f_curves, bits=b, symmetric=True)
        err = f_q - f_curves
        results.append({
            "bits": b,
            "mse": round(float(np.mean(err ** 2)), 8),
            "max_abs_err": round(float(np.max(np.abs(err))), 8),
            "mean_abs_err": round(float(np.mean(np.abs(err))), 8),
        })
    return results


# ---------- 3.1) base + offset 分量分别量化（探索 offset 低比特）----------

def quantize_base_and_offset(base, offset, base_bits=8, offset_bits=8, symmetric=True):
    """
    分别对 base 曲线和 offset（残差）曲线做量化再反量化。
    base: (N,), offset: (G, N)
    返回 base_q (N,), offset_q (G, N)。重建 f_recon = base_q + offset_q。
    """
    base = np.asarray(base, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    base_q = quantize_dequantize_values(base, bits=base_bits, symmetric=symmetric)
    offset_q = quantize_dequantize_values(offset, bits=offset_bits, symmetric=symmetric)
    return base_q, offset_q


def analysis_base_offset_quantization(f_curves, base_kind="mean", base_bits=8, offset_bits_list=(8, 6, 4), symmetric=True):
    """
    对 base 与 offset 分别量化：base 用 base_bits，offset 用多种低比特，比较 base_q+offset_q 与原始 f 的误差。
    f_curves: (G, N)
    返回: list of dict {offset_bits, mse, max_abs_err, mean_abs_err}，以及 (base, offset) 用于可视化。
    """
    base, offset = get_base_and_residual_curves(f_curves, base_kind=base_kind)
    results = []
    for ob in offset_bits_list:
        base_q, offset_q = quantize_base_and_offset(base, offset, base_bits=base_bits, offset_bits=ob, symmetric=symmetric)
        f_recon = base_q + offset_q
        err = f_recon - f_curves
        results.append({
            "base_bits": base_bits,
            "offset_bits": ob,
            "mse": round(float(np.mean(err ** 2)), 8),
            "max_abs_err": round(float(np.max(np.abs(err))), 8),
            "mean_abs_err": round(float(np.mean(np.abs(err))), 8),
        })
    return results, base, offset


# ---------- 4) JSON 导出与加载 ----------

def export_grf_coefficients_to_json(kan_weights_list, activation_ranges=None, path="grf_coefficients.json"):
    """将每层每组的 numerator/denominator 导出为 JSON。"""
    layers = []
    for item in kan_weights_list:
        name = item["name"]
        num_w = np.asarray(item["weight_numerator"])
        den_w = np.asarray(item["weight_denominator"])
        if num_w.ndim == 1:
            num_w = num_w[np.newaxis, :]
        layer = {
            "name": name,
            "weight_numerator": num_w.tolist(),
            "weight_denominator": den_w.tolist(),
        }
        if activation_ranges and name in activation_ranges:
            r = activation_ranges[name]
            layer["input_range"] = {"min": float(r["min"]), "max": float(r["max"])}
        layers.append(layer)
    den_w = np.asarray(kan_weights_list[-1]["weight_denominator"])
    payload = {"layers": layers, "num_groups": int(den_w.shape[0]), "num_numerator": 6, "num_denominator": 4}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def load_grf_from_json(path):
    """从 JSON 加载 GRF 系数。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for layer in data["layers"]:
        out.append({
            "name": layer["name"],
            "weight_numerator": np.array(layer["weight_numerator"], dtype=np.float64),
            "weight_denominator": np.array(layer["weight_denominator"], dtype=np.float64),
        })
    return out
