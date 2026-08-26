"""Morton 固定块长下，将各块表示拼接后做池化统计（阶段 3）。"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .entropy_stats import marginal_coord_entropy_12bit, marginal_delta_entropy_12bit
from .float_bits import quantize_xyz_int16, signed_bits_required
from .ieee32_stats import ieee32_stream_field_stats


def _collect_residuals(block_xyz: np.ndarray, repr_type: str) -> Tuple[np.ndarray, str]:
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


def pooled_block_metrics(
    xyz_morton: np.ndarray,
    block_size: int,
    repr_type: str,
    quant_scale: float = 2048.0,
) -> Dict[str, float]:
    n = xyz_morton.shape[0]
    n_bl = n // block_size
    if n_bl == 0:
        return _empty_row()

    parts: List[np.ndarray] = []
    for bi in range(n_bl):
        blk = xyz_morton[bi * block_size : (bi + 1) * block_size]
        v, kind = _collect_residuals(blk, repr_type)
        if v.size:
            parts.append(v)
    if not parts:
        return _empty_row()
    v = np.vstack(parts).astype(np.float32)

    axes = ["dx", "dy", "dz"]
    out: Dict[str, float] = {}
    for d, an in enumerate(axes):
        col = v[:, d]
        if repr_type == "absolute":
            h_marg = float(marginal_coord_entropy_12bit(col))
        else:
            dr = float(np.percentile(np.abs(col), 99) + 1e-6)
            dr = max(dr, 1e-9)
            h_marg = float(marginal_delta_entropy_12bit(col, dr))
        out[f"H{an}"] = h_marg
        st = ieee32_stream_field_stats(col)
        out[f"Hexp_{an}"] = float(st["Hexp"])
        out[f"Hsign_{an}"] = float(st["Hsign"])
        out[f"Hmant8_{an}"] = float(st["Hmant8"])
        out[f"avg_effective_bits_{an}"] = float(st["avg_effective_bits"])

    qi = quantize_xyz_int16(v, scale=quant_scale)
    bx = [signed_bits_required(int(abs(int(t)))) for t in qi[:, 0]]
    by = [signed_bits_required(int(abs(int(t)))) for t in qi[:, 1]]
    bz = [signed_bits_required(int(abs(int(t)))) for t in qi[:, 2]]
    out["avg_required_bits_x"] = float(np.mean(bx)) if bx else 0.0
    out["avg_required_bits_y"] = float(np.mean(by)) if by else 0.0
    out["avg_required_bits_z"] = float(np.mean(bz)) if bz else 0.0
    out["p99_required_bits_x"] = float(np.percentile(bx, 99)) if bx else 0.0
    out["p99_required_bits_y"] = float(np.percentile(by, 99)) if by else 0.0
    out["p99_required_bits_z"] = float(np.percentile(bz, 99)) if bz else 0.0

    out["zero_ratio"] = float(np.mean(np.all(qi == 0, axis=1))) if qi.size else 0.0
    out["small_ratio"] = float(np.mean(np.max(np.abs(qi), axis=1) <= 2)) if qi.size else 0.0
    return out


def _empty_row() -> Dict[str, float]:
    z = 0.0
    keys = []
    for an in ["dx", "dy", "dz"]:
        keys.extend(
            [
                f"H{an}",
                f"Hexp_{an}",
                f"Hsign_{an}",
                f"Hmant8_{an}",
                f"avg_effective_bits_{an}",
            ]
        )
    keys.extend(
        [
            "avg_required_bits_x",
            "avg_required_bits_y",
            "avg_required_bits_z",
            "p99_required_bits_x",
            "p99_required_bits_y",
            "p99_required_bits_z",
            "zero_ratio",
            "small_ratio",
        ]
    )
    return {k: z for k in keys}
