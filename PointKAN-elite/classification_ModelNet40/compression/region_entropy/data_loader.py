"""从 ModelNet40 HDF5（与 classification_ModelNet40/data.py 一致）加载测试点云。"""

from __future__ import annotations

import glob
import os
from typing import Iterator, List, Optional, Tuple

import h5py
import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def default_data_dir() -> str:
    return "/workspace/VoxelNet/pc_data/modelnet40_ply_hdf5_2048"


def load_modelnet40_test(
    data_dir: Optional[str] = None,
    num_points: int = 1024,
    max_samples: Optional[int] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    返回:
        points: (S, N, 3) float32 — 与训练时测试集一致，仅截断前 N 点（无增强）
        labels: (S,) int64
        sample_ids: 每个样本的稳定字符串 id（h5 文件名 + 索引）
    """
    data_dir = data_dir or default_data_dir()
    paths = sorted(glob.glob(os.path.join(data_dir, "ply_data_test*.h5")))
    if not paths:
        raise FileNotFoundError(f"未找到测试 HDF5: {data_dir}/ply_data_test*.h5")

    rng = np.random.default_rng(seed)
    all_data: List[np.ndarray] = []
    all_label: List[np.ndarray] = []
    id_parts: List[Tuple[str, int]] = []

    for h5_path in paths:
        with h5py.File(h5_path, "r") as f:
            data = f["data"][:].astype(np.float32)
            label = f["label"][:].astype(np.int64)
        base = os.path.basename(h5_path)
        for i in range(data.shape[0]):
            id_parts.append((base, i))
        all_data.append(data[:, :num_points, :])
        all_label.append(label.reshape(-1))

    pts = np.concatenate(all_data, axis=0)
    labs = np.concatenate(all_label, axis=0)
    n = pts.shape[0]
    order = np.arange(n)
    if max_samples is not None and max_samples < n:
        order = rng.choice(n, size=max_samples, replace=False)
        order.sort()
        pts = pts[order]
        labs = labs[order]
        id_parts = [id_parts[i] for i in order.tolist()]

    sample_ids = [f"{b}#{idx}" for b, idx in id_parts]
    return pts, labs, sample_ids


def iter_samples(
    data_dir: Optional[str] = None,
    num_points: int = 1024,
    max_samples: Optional[int] = None,
    seed: int = 0,
) -> Iterator[Tuple[str, np.ndarray, int]]:
    pts, labs, ids = load_modelnet40_test(data_dir, num_points, max_samples, seed)
    for sid, p, y in zip(ids, pts, labs):
        yield sid, p, int(y)


def normalize_unit_sphere(xyz: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """每样本中心化并缩放到最大半径为 1 的单位球内。"""
    c = xyz.mean(axis=0, keepdims=True)
    x = xyz - c
    r = np.linalg.norm(x, axis=1).max()
    return x / (r + eps)
