"""Morton / Hilbert 空间填充曲线排序（3D）。"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def morton_code_3d(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, bits: int = 10) -> np.ndarray:
    """3D Morton / Z-order：按位交织 x,y,z（低位优先）。"""
    ix = ix.astype(np.uint64) & np.uint64((1 << bits) - 1)
    iy = iy.astype(np.uint64) & np.uint64((1 << bits) - 1)
    iz = iz.astype(np.uint64) & np.uint64((1 << bits) - 1)
    code = np.zeros(ix.shape[0], dtype=np.uint64)
    one = np.uint64(1)
    for b in range(bits):
        sh = np.uint64(3 * b)
        code |= ((ix >> np.uint64(b)) & one) << sh
        code |= ((iy >> np.uint64(b)) & one) << (sh + one)
        code |= ((iz >> np.uint64(b)) & one) << (sh + np.uint64(2))
    return code


def lattice_coords_unit(xyz: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """每样本独立：将 xyz 线性映射到 [0, 2^p-1]^3 整数格点。"""
    lo = xyz.min(axis=0, keepdims=True)
    hi = xyz.max(axis=0, keepdims=True)
    span = np.maximum(hi - lo, 1e-8)
    g = (2**p) - 1
    q = ((xyz - lo) / span * g).astype(np.float64)
    ix = np.clip(np.round(q[:, 0]), 0, g).astype(np.uint64)
    iy = np.clip(np.round(q[:, 1]), 0, g).astype(np.uint64)
    iz = np.clip(np.round(q[:, 2]), 0, g).astype(np.uint64)
    return ix, iy, iz


def sort_indices_morton(xyz: np.ndarray, p: int = 10) -> np.ndarray:
    ix, iy, iz = lattice_coords_unit(xyz, p)
    code = morton_code_3d(ix, iy, iz)
    return np.argsort(code, kind="mergesort")


def sort_indices_hilbert(xyz: np.ndarray, p: int = 10):
    try:
        from hilbertcurve.hilbertcurve import HilbertCurve
    except ImportError:
        return None
    ix, iy, iz = lattice_coords_unit(xyz, p)
    hc = HilbertCurve(p, 3)
    dists = np.empty(xyz.shape[0], dtype=np.uint64)
    for i in range(xyz.shape[0]):
        dists[i] = np.uint64(hc.distance_from_point([int(ix[i]), int(iy[i]), int(iz[i])]))
    return np.argsort(dists, kind="mergesort")


def apply_order(xyz: np.ndarray, order: np.ndarray) -> np.ndarray:
    return xyz[order].copy()
