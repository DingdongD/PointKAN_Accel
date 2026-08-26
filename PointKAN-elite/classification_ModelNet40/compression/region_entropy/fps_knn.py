"""FPS 采样中心与 KNN 邻域重叠统计。"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .spatial_order import lattice_coords_unit, morton_code_3d


def farthest_point_sample_indices(xyz: np.ndarray, npoint: int, seed: int = 0) -> np.ndarray:
    """返回长度为 npoint 的原始点索引（FPS）。"""
    rng = np.random.default_rng(seed)
    n = xyz.shape[0]
    if npoint > n:
        raise ValueError("npoint > N")
    centroids = np.zeros(npoint, dtype=np.int64)
    distance = np.ones(n, dtype=np.float64) * 1e10
    farthest = int(rng.integers(0, n))
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest]
        dist = np.sum((xyz - centroid) ** 2, axis=1)
        distance = np.minimum(distance, dist)
        farthest = int(np.argmax(distance))
    return centroids


def knn_indices(
    xyz: np.ndarray, query: np.ndarray, k: int
) -> np.ndarray:
    """query: (M,3) 在 xyz (N,3) 中找 K 近邻，返回 (M,K) 索引。"""
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=-1)
    nn.fit(xyz)
    _, idx = nn.kneighbors(query, return_distance=True)
    return idx.astype(np.int64)


def order_centers_morton(centers: np.ndarray, p: int = 10) -> np.ndarray:
    ix, iy, iz = lattice_coords_unit(centers, p)
    code = morton_code_3d(ix, iy, iz)
    return np.argsort(code, kind="mergesort")


def apply_center_order(xyz: np.ndarray, center_idx: np.ndarray, order: np.ndarray) -> np.ndarray:
    return center_idx[order]


def neighbor_overlap_stats(
    neigh_idx: np.ndarray,
    n_random_pairs: int = 32,
    rng: Optional[np.random.default_rng] = None,
) -> Dict[str, float]:
    """
    neigh_idx: (M, K)
    相邻行：中心按当前顺序已排列。
    """
    rng = rng or np.random.default_rng(0)
    m, k = neigh_idx.shape
    if m < 2:
        return {
            "avg_overlap_jaccard": 0.0,
            "avg_shared_at_k": 0.0,
            "avg_added": 0.0,
            "avg_removed": 0.0,
            "avg_edit_distance": 0.0,
            "avg_overlap_jaccard_random": 0.0,
            "avg_shared_at_k_random": 0.0,
        }
    jac = []
    sh = []
    add_l = []
    rem_l = []
    ed = []
    for i in range(m - 1):
        a = set(neigh_idx[i].tolist())
        b = set(neigh_idx[i + 1].tolist())
        inter = len(a & b)
        union = len(a | b)
        jac.append(inter / union if union else 0.0)
        sh.append(inter / k)
        add_l.append(len(b - a))
        rem_l.append(len(a - b))
        ed.append(len(b - a) + len(a - b))

    r_jac = []
    r_sh = []
    for _ in range(n_random_pairs):
        i = int(rng.integers(0, m - 1))
        j = int(rng.integers(0, m))
        if j == i or j == i + 1:
            continue
        a = set(neigh_idx[i].tolist())
        b = set(neigh_idx[j].tolist())
        inter = len(a & b)
        union = len(a | b)
        r_jac.append(inter / union if union else 0.0)
        r_sh.append(inter / k)

    return {
        "avg_overlap_jaccard": float(np.mean(jac)),
        "avg_shared_at_k": float(np.mean(sh)),
        "avg_added": float(np.mean(add_l)),
        "avg_removed": float(np.mean(rem_l)),
        "avg_edit_distance": float(np.mean(ed)),
        "avg_overlap_jaccard_random": float(np.mean(r_jac)) if r_jac else 0.0,
        "avg_shared_at_k_random": float(np.mean(r_sh)) if r_sh else 0.0,
    }


def pool_group_stats(neigh_idx: np.ndarray, group_sizes: List[int]) -> List[Dict[str, float]]:
    """连续 G 个中心的邻居并集大小与 bitmask 粗估压缩比。"""
    m, k = neigh_idx.shape
    rows = []
    for g in group_sizes:
        pool_sizes = []
        ratios = []
        for s in range(0, m - g + 1, g):
            pool: set = set()
            for j in range(g):
                pool.update(neigh_idx[s + j].tolist())
            ps = len(pool)
            pool_sizes.append(ps)
            raw_bytes = g * k * 4
            bitmask_bytes = g * int(np.ceil(ps / 8.0))
            pool_bytes = ps * 4
            cmp_bytes = pool_bytes + bitmask_bytes
            ratios.append(raw_bytes / max(cmp_bytes, 1.0))
        rows.append(
            {
                "pool_group_size": float(g),
                "avg_pool_size": float(np.mean(pool_sizes)) if pool_sizes else 0.0,
                "pool_compression_ratio": float(np.mean(ratios)) if ratios else 0.0,
            }
        )
    return rows
