"""离散 Shannon 熵与直方图工具。"""

from __future__ import annotations

from typing import Dict

import numpy as np


def shannon_entropy_counts(counts: np.ndarray) -> float:
    c = counts.astype(np.float64)
    c = c[c > 0]
    if c.size == 0:
        return 0.0
    p = c / c.sum()
    return float(-np.sum(p * np.log2(p)))


def entropy_from_labels(labels: np.ndarray, max_label: int) -> float:
    c = np.bincount(labels.astype(np.int64).ravel(), minlength=max_label + 1)
    return shannon_entropy_counts(c)


def joint_entropy_two(a: np.ndarray, b: np.ndarray, na: int, nb: int) -> float:
    """H(A,B)，a in [0,na), b in [0,nb)。"""
    a = a.astype(np.int64).ravel()
    b = b.astype(np.int64).ravel()
    key = a * np.int64(nb) + b
    uniq, cnt = np.unique(key, return_counts=True)
    return shannon_entropy_counts(cnt)


def topk_coverage_from_counts(counts: np.ndarray, k: int = 3) -> Dict[str, float]:
    total = float(counts.sum())
    if total <= 0:
        return {f"top{i+1}_exp_cov": 0.0 for i in range(k)}
    idx = np.argsort(-counts)[:k]
    out = {}
    for i in range(k):
        key = f"top{i+1}_exp_cov"
        if i < len(idx):
            out[key] = float(counts[idx[i]] / total)
        else:
            out[key] = 0.0
    return out


def marginal_coord_entropy_12bit(vals: np.ndarray, vmin: float = -1.0, vmax: float = 1.0) -> float:
    n_bins = 4096
    t = (vals.astype(np.float64) - vmin) / (vmax - vmin + 1e-12)
    t = np.clip(t, 0.0, 1.0)
    idx = np.floor(t * (n_bins - 1)).astype(np.int64)
    c = np.bincount(idx, minlength=n_bins)
    return shannon_entropy_counts(c)


def marginal_delta_entropy_12bit(vals: np.ndarray, half_range: float) -> float:
    """差分标量在 [-half_range, half_range] 内做 12-bit 均匀分桶的边际熵。"""
    n_bins = 4096
    hr = max(float(half_range), 1e-12)
    t = (vals.astype(np.float64) + hr) / (2.0 * hr + 1e-12)
    t = np.clip(t, 0.0, 1.0)
    idx = np.floor(t * (n_bins - 1)).astype(np.int64)
    c = np.bincount(idx, minlength=n_bins)
    return shannon_entropy_counts(c)
