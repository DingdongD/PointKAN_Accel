"""Morton 序列上的固定长度块与多种残差表示。"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .entropy import (
    discretize_axis,
    entropy_of_deltas,
    exponent_entropy_and_topk,
    shannon_entropy_counts,
    signed_bits_required,
)


def _quantize_xyz_int16(xyz: np.ndarray, scale: float = 1024.0) -> np.ndarray:
    return np.clip(np.round(xyz.astype(np.float64) * scale), -32768, 32767).astype(np.int32)


def collect_residuals(
    block_xyz: np.ndarray, repr_type: str
) -> Tuple[np.ndarray, str]:
    if repr_type == "absolute":
        return block_xyz.astype(np.float32), "xyz"
    if repr_type == "morton_delta":
        if block_xyz.shape[0] < 2:
            return np.zeros((0, 3), np.float32), "delta"
        return np.diff(block_xyz.astype(np.float64), axis=0).astype(np.float32), "delta"
    if repr_type == "block_centroid_delta":
        c = block_xyz.mean(axis=0, keepdims=True)
        return (block_xyz - c).astype(np.float32), "delta"
    if repr_type == "block_firstpoint_delta":
        return (block_xyz - block_xyz[0:1]).astype(np.float32), "delta"
    raise ValueError(repr_type)


def stats_on_values(
    vals: np.ndarray,
    kind: str,
    n_bits_entropy: int,
    delta_range: float,
    quant_scale: float,
) -> Dict[str, float]:
    if vals.size == 0:
        z = 0.0
        return {
            "Hdx": z,
            "Hdy": z,
            "Hdz": z,
            "Hexp_dx": z,
            "Hexp_dy": z,
            "Hexp_dz": z,
            "avg_required_bits_x": z,
            "avg_required_bits_y": z,
            "avg_required_bits_z": z,
            "p99_required_bits_x": z,
            "p99_required_bits_y": z,
            "p99_required_bits_z": z,
            "zero_ratio": z,
            "small_ratio": z,
        }
    if kind == "delta":
        hdx, hdy, hdz = entropy_of_deltas(vals, n_bits_entropy, delta_range)
    else:
        vmin, vmax = -1.0, 1.0
        hx = []
        for d in range(3):
            idx = discretize_axis(vals[:, d], n_bits_entropy, vmin, vmax)
            c = np.bincount(idx, minlength=2**n_bits_entropy)
            hx.append(shannon_entropy_counts(c))
        hdx, hdy, hdz = float(hx[0]), float(hx[1]), float(hx[2])

    hex_dx, _ = exponent_entropy_and_topk(vals[:, 0:1], k=3)
    hex_dy, _ = exponent_entropy_and_topk(vals[:, 1:2], k=3)
    hex_dz, _ = exponent_entropy_and_topk(vals[:, 2:3], k=3)

    qi = _quantize_xyz_int16(vals, scale=quant_scale)
    bits_x = [signed_bits_required(int(v)) for v in np.abs(qi[:, 0])]
    bits_y = [signed_bits_required(int(v)) for v in np.abs(qi[:, 1])]
    bits_z = [signed_bits_required(int(v)) for v in np.abs(qi[:, 2])]

    return {
        "Hdx": hdx,
        "Hdy": hdy,
        "Hdz": hdz,
        "Hexp_dx": float(hex_dx),
        "Hexp_dy": float(hex_dy),
        "Hexp_dz": float(hex_dz),
        "avg_required_bits_x": float(np.mean(bits_x)),
        "avg_required_bits_y": float(np.mean(bits_y)),
        "avg_required_bits_z": float(np.mean(bits_z)),
        "p99_required_bits_x": float(np.percentile(bits_x, 99)),
        "p99_required_bits_y": float(np.percentile(bits_y, 99)),
        "p99_required_bits_z": float(np.percentile(bits_z, 99)),
        "zero_ratio": float(np.mean(np.all(qi == 0, axis=1))),
        "small_ratio": float(np.mean(np.max(np.abs(qi), axis=1) <= 2)),
    }


def sweep_blocks(
    xyz_morton: np.ndarray,
    block_sizes: List[int],
    repr_types: List[str],
    n_bits_entropy: int = 12,
    delta_range: float = 0.5,
    quant_scale: float = 2048.0,
    block_type: str = "morton_fixed_length",
) -> List[Dict[str, float]]:
    out_rows: List[Dict[str, float]] = []
    n = xyz_morton.shape[0]
    for bsz in block_sizes:
        n_bl = n // bsz
        if n_bl == 0:
            continue
        for rt in repr_types:
            hs: List[List[float]] = []
            exp_sums = [0.0, 0.0, 0.0]
            bits_x_all: List[int] = []
            bits_y_all: List[int] = []
            bits_z_all: List[int] = []
            zr: List[float] = []
            sr: List[float] = []
            for bi in range(n_bl):
                blk = xyz_morton[bi * bsz : (bi + 1) * bsz]
                vals, kind = collect_residuals(blk, rt)
                st = stats_on_values(vals, kind, n_bits_entropy, delta_range, quant_scale)
                hs.append([st["Hdx"], st["Hdy"], st["Hdz"]])
                exp_sums[0] += st["Hexp_dx"]
                exp_sums[1] += st["Hexp_dy"]
                exp_sums[2] += st["Hexp_dz"]
                if vals.size:
                    qi = _quantize_xyz_int16(vals, scale=quant_scale)
                    bits_x_all.extend(int(v) for v in np.abs(qi[:, 0]))
                    bits_y_all.extend(int(v) for v in np.abs(qi[:, 1]))
                    bits_z_all.extend(int(v) for v in np.abs(qi[:, 2]))
                zr.append(st["zero_ratio"])
                sr.append(st["small_ratio"])
            hs_arr = np.array(hs, dtype=np.float64)
            mean_h = hs_arr.mean(axis=0)
            bx = [signed_bits_required(v) for v in bits_x_all]
            by = [signed_bits_required(v) for v in bits_y_all]
            bz = [signed_bits_required(v) for v in bits_z_all]
            out_rows.append(
                {
                    "block_type": block_type,
                    "block_size": float(bsz),
                    "repr_type": rt,
                    "Hdx": float(mean_h[0]),
                    "Hdy": float(mean_h[1]),
                    "Hdz": float(mean_h[2]),
                    "Hexp_dx": float(exp_sums[0] / n_bl),
                    "Hexp_dy": float(exp_sums[1] / n_bl),
                    "Hexp_dz": float(exp_sums[2] / n_bl),
                    "avg_required_bits_x": float(np.mean(bx)) if bx else 0.0,
                    "avg_required_bits_y": float(np.mean(by)) if by else 0.0,
                    "avg_required_bits_z": float(np.mean(bz)) if bz else 0.0,
                    "p99_required_bits_x": float(np.percentile(bx, 99)) if bx else 0.0,
                    "p99_required_bits_y": float(np.percentile(by, 99)) if by else 0.0,
                    "p99_required_bits_z": float(np.percentile(bz, 99)) if bz else 0.0,
                    "zero_ratio": float(np.mean(zr)) if zr else 0.0,
                    "small_ratio": float(np.mean(sr)) if sr else 0.0,
                }
            )
    return out_rows
