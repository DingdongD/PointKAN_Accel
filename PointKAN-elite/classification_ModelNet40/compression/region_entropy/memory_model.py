"""基于统计量粗估 DRAM bytes 与 burst transaction（Phase E）。"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def ceil_div(a: float, b: float) -> int:
    return int(np.ceil(a / b))


def build_memory_model_rows(
    n_points: int = 1024,
    n_centers: int = 256,
    k_neighbors: int = 32,
    bytes_coord_raw_per_point: float = 12.0,
    bytes_index_raw: float = 4.0,
    avg_bits_per_delta_axis: float = 10.0,
    burst_sizes: List[int] = None,
) -> List[Dict[str, float]]:
    """
    使用「块基 + 整点 delta」与「原始邻域表 vs pool+bitmask」两类方案做归一化对比。
    avg_bits_per_delta_axis: 来自 Phase C 某配置下三轴平均位宽之和的 1/3 近似单轴。
    """
    burst_sizes = burst_sizes or [32, 64, 128]

    def txns(bytes_total: float, burst: int) -> int:
        return ceil_div(bytes_total, burst)

    rows: List[Dict[str, float]] = []

    bytes_raw_coords = n_points * bytes_coord_raw_per_point
    bits_total_per_point = avg_bits_per_delta_axis * 3.0
    bytes_per_point_delta = bits_total_per_point / 8.0
    bytes_base = 6.0
    n_blocks = 32
    bytes_block_scheme = n_blocks * bytes_base + n_points * bytes_per_point_delta

    bytes_raw_knn = n_centers * k_neighbors * bytes_index_raw
    pool_sz = float(np.mean([k_neighbors * gs * 0.55 for gs in [2, 4, 8]]))
    pool_sz = max(pool_sz, float(k_neighbors))
    bytes_pool = pool_sz * bytes_index_raw + n_centers * np.ceil(pool_sz / 8.0)

    schemes = [
        ("coords", "float32_xyz", bytes_raw_coords, bytes_raw_coords),
        ("coords", "block_base_int16_delta", bytes_raw_coords, bytes_block_scheme),
        ("knn_index", "int32_full_table", bytes_raw_knn, bytes_raw_knn),
        ("knn_index", "shared_pool_bitmask_est", bytes_raw_knn, float(bytes_pool)),
    ]

    for obj, scheme, raw_b, cmp_b in schemes:
        ratio = raw_b / max(cmp_b, 1e-6)
        row: Dict[str, float] = {
            "object_type": obj,
            "scheme": scheme,
            "bytes_raw": float(raw_b),
            "bytes_compressed": float(cmp_b),
            "compression_ratio": float(ratio),
        }
        for b in burst_sizes:
            row[f"txn_{b}B_raw"] = float(txns(raw_b, b))
            row[f"txn_{b}B_cmp"] = float(txns(cmp_b, b))
        rows.append(row)
    return rows


