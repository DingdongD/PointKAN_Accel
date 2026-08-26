"""离散熵、浮点指数熵、联合坐标熵等工具。"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np


def shannon_entropy_counts(counts: np.ndarray) -> float:
    c = counts.astype(np.float64)
    c = c[c > 0]
    if c.size == 0:
        return 0.0
    p = c / c.sum()
    return float(-np.sum(p * np.log2(p)))


def discretize_axis(values: np.ndarray, n_bits: int, vmin: float, vmax: float) -> np.ndarray:
    """将 float 轴量化到 [0, 2^n_bits - 1] 的整数 bin 索引。"""
    n = int(2**n_bits)
    t = (values.astype(np.float64) - vmin) / (vmax - vmin + 1e-12)
    t = np.clip(t, 0.0, 1.0)
    idx = np.floor(t * (n - 1)).astype(np.int64)
    return idx


def marginal_entropies_xyz(
    xyz: np.ndarray, n_bits: int, bounds: Tuple[float, float] = (-1.0, 1.0)
) -> Tuple[float, float, float]:
    vmin, vmax = bounds
    hx = []
    for d in range(3):
        idx = discretize_axis(xyz[:, d], n_bits, vmin, vmax)
        c = np.bincount(idx, minlength=2**n_bits)
        hx.append(shannon_entropy_counts(c))
    return float(hx[0]), float(hx[1]), float(hx[2])


def joint_entropy_xyz(
    xyz: np.ndarray, n_bits: int, bounds: Tuple[float, float] = (-1.0, 1.0)
) -> float:
    vmin, vmax = bounds
    ix = discretize_axis(xyz[:, 0], n_bits, vmin, vmax)
    iy = discretize_axis(xyz[:, 1], n_bits, vmin, vmax)
    iz = discretize_axis(xyz[:, 2], n_bits, vmin, vmax)
    keys = ix.astype(np.uint64) | (iy.astype(np.uint64) << np.uint64(n_bits))
    keys |= iz.astype(np.uint64) << np.uint64(2 * n_bits)
    uniq, cnt = np.unique(keys, return_counts=True)
    return shannon_entropy_counts(cnt)


def float32_exponents(arr: np.ndarray) -> np.ndarray:
    u = arr.astype(np.float32).view(np.uint32)
    exp = ((u >> 23) & 0xFF).astype(np.int32)
    return exp


def exponent_entropy_and_topk(arr: np.ndarray, k: int = 3) -> Tuple[float, Dict[str, float]]:
    exp = float32_exponents(arr.ravel())
    mx = int(exp.max()) + 1
    c = np.bincount(exp, minlength=mx)
    h = shannon_entropy_counts(c)
    total = float(c.sum())
    top_idx = np.argsort(-c)[:k]
    cov = {f"top{j+1}_exp_cov": float(c[top_idx[j]] / total) for j in range(k) if j < len(top_idx)}
    return h, cov


def delta_along_sequence(xyz: np.ndarray) -> np.ndarray:
    d = np.diff(xyz.astype(np.float64), axis=0)
    return d.astype(np.float32)


def entropy_of_deltas(
    deltas: np.ndarray,
    n_bits: int,
    delta_range: float,
) -> Tuple[float, float, float]:
    """将差分 clip 到 [-R,R] 再均匀量化到 n_bits，估计 H(dx),H(dy),H(dz)。"""
    n = int(2**n_bits)
    out = []
    for d in range(3):
        t = (deltas[:, d].astype(np.float64) + delta_range) / (2.0 * delta_range + 1e-12)
        t = np.clip(t, 0.0, 1.0)
        idx = np.floor(t * (n - 1)).astype(np.int64)
        c = np.bincount(idx, minlength=n)
        out.append(shannon_entropy_counts(c))
    return float(out[0]), float(out[1]), float(out[2])


def small_delta_ratios(
    deltas: np.ndarray, tau_axis: Iterable[float], tau_norm: Iterable[float]
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for j, t in enumerate(tau_axis):
        out[f"p_abs_dx_lt_tau_{j}"] = float(np.mean(np.abs(deltas[:, 0]) < t))
        out[f"p_abs_dy_lt_tau_{j}"] = float(np.mean(np.abs(deltas[:, 1]) < t))
        out[f"p_abs_dz_lt_tau_{j}"] = float(np.mean(np.abs(deltas[:, 2]) < t))
    nrm = np.linalg.norm(deltas.astype(np.float64), axis=1)
    for j, tr in enumerate(tau_norm):
        out[f"p_norm_lt_tau_r_{j}"] = float(np.mean(nrm < tr))
    return out


def signed_bits_required(max_abs: int) -> int:
    if max_abs <= 0:
        return 1
    return int(np.ceil(np.log2(max_abs + 1.0))) + 1
