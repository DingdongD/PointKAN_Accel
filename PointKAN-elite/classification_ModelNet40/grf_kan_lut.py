#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRF-KAN 均匀分割 LUT 查表实现

- 对每层 GRF 在 [x_min, x_max] 上均匀分割 n_bins，在格点上预计算 f(x) 得到 LUT，查表时线性插值。
- 支持两种形式：(1) 原始 f(x) 整表 LUT；(2) base 表 + offset 表，查表后合并 base + offset 恢复 f(x)。
- 复用 grf_kan_json_and_residual 的 eval_grf_vanilla、get_base_and_residual_curves。
"""

import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
for d in (SCRIPT_DIR, ROOT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

from grf_kan_json_and_residual import eval_grf_vanilla, get_base_and_residual_curves


def build_lut_vanilla(weight_numerator, weight_denominator, x_min, x_max, n_bins):
    """
    构建原始 f(x) 的均匀分割 LUT。
    x_edges = linspace(x_min, x_max, n_bins+1)，在 x_edges 上计算 f(x)，得到 table (G, n_bins+1)。
    返回 dict: x_min, x_max, n_bins, table (list/array), num_groups。
    """
    x_edges = np.linspace(float(x_min), float(x_max), int(n_bins) + 1)
    f_table = eval_grf_vanilla(x_edges, weight_numerator, weight_denominator)
    G = f_table.shape[0]
    return {
        "x_min": float(x_min),
        "x_max": float(x_max),
        "n_bins": int(n_bins),
        "table": np.asarray(f_table, dtype=np.float64),
        "num_groups": G,
    }


def build_lut_base_offset(f_curves, x_orig, x_min, x_max, n_bins, base_kind="mean"):
    """
    在已有 f_curves (G, N) 与 x_orig (N,) 上得到 base 与 offset，再在均匀格点 x_edges 上重采样得到 base_lut、offset_lut。
    f_curves: (G, N), x_orig: (N,)
    返回: base_lut (n_bins+1,), offset_lut (G, n_bins+1), x_edges, 以及 dict 形式供存储。
    """
    base, offset = get_base_and_residual_curves(f_curves, base_kind=base_kind)
    x_edges = np.linspace(float(x_min), float(x_max), int(n_bins) + 1)
    base_lut = np.interp(x_edges, x_orig, base)
    G = offset.shape[0]
    offset_lut = np.zeros((G, len(x_edges)), dtype=np.float64)
    for g in range(G):
        offset_lut[g] = np.interp(x_edges, x_orig, offset[g])
    lut_dict = {
        "x_min": float(x_min),
        "x_max": float(x_max),
        "n_bins": int(n_bins),
        "base_kind": base_kind,
        "base_table": np.asarray(base_lut, dtype=np.float64),
        "offset_table": np.asarray(offset_lut, dtype=np.float64),
        "num_groups": G,
    }
    return lut_dict, x_edges


def lookup_lut_numpy(x, lut_dict, kind="vanilla"):
    """
    用 LUT 线性插值得到 f(x)。x 为标量或一维数组。
    kind: "vanilla" 使用 lut_dict["table"]；(G, n_bins+1) -> 输出 (G, len(x))
          "base_offset" 使用 base_table + offset_table，输出 (G, len(x))。
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        x = x.reshape(1)
    x_flat = x.ravel()
    x_min = lut_dict["x_min"]
    x_max = lut_dict["x_max"]
    n_bins = lut_dict["n_bins"]
    G = lut_dict["num_groups"]
    span = x_max - x_min
    if span < 1e-12:
        span = 1e-12
    idx_float = (x_flat - x_min) / span * n_bins
    idx_float = np.clip(idx_float, 0, n_bins)
    i0 = np.floor(idx_float).astype(np.int32)
    i0 = np.clip(i0, 0, n_bins - 1)
    i1 = np.minimum(i0 + 1, n_bins)
    t = idx_float - i0.astype(np.float64)
    out_list = []
    if kind == "vanilla":
        table = lut_dict["table"]
        for g in range(G):
            v0 = table[g, i0]
            v1 = table[g, i1]
            out_g = (1 - t) * v0 + t * v1
            out_list.append(out_g)
    else:
        base_table = lut_dict["base_table"]
        offset_table = lut_dict["offset_table"]
        base_v = (1 - t) * base_table[i0] + t * base_table[i1]
        for g in range(G):
            off_v = (1 - t) * offset_table[g, i0] + t * offset_table[g, i1]
            out_list.append(base_v + off_v)
    out = np.stack(out_list, axis=0)
    out = out.reshape(G, *x.shape)
    return out


def verify_lut_vs_float(lut_dict, weight_numerator, weight_denominator, x_test=None, kind="vanilla", n_test=500):
    """
    在随机或给定 x 上比较 LUT 查表结果与浮点 f(x) 的误差。
    返回 mse, max_abs_err, mean_abs_err。
    """
    if x_test is None:
        x_test = np.linspace(lut_dict["x_min"], lut_dict["x_max"], n_test)
    x_test = np.asarray(x_test)
    if kind == "vanilla":
        f_lut = lookup_lut_numpy(x_test, lut_dict, kind="vanilla")
    else:
        f_lut = lookup_lut_numpy(x_test, lut_dict, kind="base_offset")
    f_float = eval_grf_vanilla(x_test, weight_numerator, weight_denominator)
    err = f_lut - f_float
    mse = float(np.mean(err ** 2))
    max_err = float(np.max(np.abs(err)))
    mean_err = float(np.mean(np.abs(err)))
    return {"mse": mse, "max_abs_err": max_err, "mean_abs_err": mean_err}


# --------------- 量化 LUT ---------------

def quantize_table_symmetric(table, bits=8):
    """
    对称量化：scale = max(abs(table)) / (2**(bits-1) - 1)，q = round(table/scale)，范围 [-2**(bits-1), 2**(bits-1)-1]。
    返回 table_q (int8/int16), scale (float), zero_point=0。
    """
    table = np.asarray(table, dtype=np.float64)
    max_val = np.max(np.abs(table))
    if max_val < 1e-12:
        return np.zeros(table.shape, dtype=np.int8 if bits <= 8 else np.int16), 1.0, 0
    high = (1 << (bits - 1)) - 1
    low = -(1 << (bits - 1))
    scale = max_val / high
    table_q = np.round(table / scale).astype(np.float64).clip(low, high).astype(np.int8 if bits <= 8 else np.int16)
    return table_q, float(scale), 0


def dequantize_table(table_q, scale, zero_point=0):
    """反量化: value = (table_q - zero_point) * scale"""
    return (np.asarray(table_q, dtype=np.float64) - zero_point) * scale


def lookup_lut_numpy_quantized(x, lut_dict, kind="vanilla"):
    """
    使用量化表查表（先整表反量化再线性插值，等价于先插值再反量化）。
    lut_dict 需包含 kind 对应表的 _q 与 _scale、_zp，如 vanilla: table_q, table_scale, table_zp。
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        x = x.reshape(1)
    x_flat = x.ravel()
    x_min = lut_dict["x_min"]
    x_max = lut_dict["x_max"]
    n_bins = lut_dict["n_bins"]
    G = lut_dict["num_groups"]
    span = x_max - x_min
    if span < 1e-12:
        span = 1e-12
    idx_float = (x_flat - x_min) / span * n_bins
    idx_float = np.clip(idx_float, 0, n_bins)
    i0 = np.floor(idx_float).astype(np.int32)
    i0 = np.clip(i0, 0, n_bins - 1)
    i1 = np.minimum(i0 + 1, n_bins)
    t = idx_float - i0.astype(np.float64)
    out_list = []
    if kind == "vanilla":
        table_q = lut_dict["table_q"]
        scale = lut_dict["table_scale"]
        zp = lut_dict.get("table_zp", 0)
        table = dequantize_table(table_q, scale, zp)
        for g in range(G):
            v0 = table[g, i0]
            v1 = table[g, i1]
            out_g = (1 - t) * v0 + t * v1
            out_list.append(out_g)
    else:
        base_q = lut_dict["base_table_q"]
        base_scale = lut_dict["base_table_scale"]
        base_zp = lut_dict.get("base_table_zp", 0)
        base_table = dequantize_table(base_q, base_scale, base_zp)
        offset_q = lut_dict["offset_table_q"]
        offset_scale = lut_dict["offset_table_scale"]
        offset_zp = lut_dict.get("offset_table_zp", 0)
        offset_table = dequantize_table(offset_q, offset_scale, offset_zp)
        base_v = (1 - t) * base_table[i0] + t * base_table[i1]
        for g in range(G):
            off_v = (1 - t) * offset_table[g, i0] + t * offset_table[g, i1]
            out_list.append(base_v + off_v)
    out = np.stack(out_list, axis=0)
    out = out.reshape(G, *x.shape)
    return out


def verify_lut_quantized_vs_float(lut_dict_quantized, weight_numerator, weight_denominator, x_test=None, kind="vanilla", n_test=500):
    """
    量化 LUT 查表（反量化后）与浮点 f(x) 的误差。
    返回 mse, max_abs_err, mean_abs_err。
    """
    if x_test is None:
        x_test = np.linspace(lut_dict_quantized["x_min"], lut_dict_quantized["x_max"], n_test)
    x_test = np.asarray(x_test)
    f_lut = lookup_lut_numpy_quantized(x_test, lut_dict_quantized, kind=kind)
    f_float = eval_grf_vanilla(x_test, weight_numerator, weight_denominator)
    err = f_lut - f_float
    mse = float(np.mean(err ** 2))
    max_err = float(np.max(np.abs(err)))
    mean_err = float(np.mean(np.abs(err)))
    return {"mse": mse, "max_abs_err": max_err, "mean_abs_err": mean_err}
