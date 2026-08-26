#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S3DIS — PointNet++ 语义分割用的 Indoor3D HDF5 管线（与 ISEDA26/data_config.load_data_semseg 一致）。

- ``pc_root`` 下应有 ``indoor3d_sem_seg_hdf5_data`` 与 ``indoor3d_sem_seg_hdf5_data_test``。
- 每块 ``4096`` 点、HDF5 内 ``data`` 为 **9 通道**（与官方 ``room2blocks_plus_normalized`` 一致）：
    - **0:3**：块内对 **x、y** 做平移，使水平坐标相对 **块中心**（每块用该块 ``minx/miny`` 与 ``block_size`` 定中心）；**z 不在这段里再除尺度**，仍为房间/场景尺度下的高度。因此全局统计常见 **x、y 落在约 ±block_size/2 附近**，**z 跨度可较大（米级）**。
    - **3:6**：RGB（已 ``/255``）等（与 PointNet 输入一致）。
    - **6:9**：相对 **整间房** ``max_room_{x,y,z}`` 归一化的坐标（``x/max_room_x`` 等），供网络作附加特征。
- **本基准**的 FPS/KNN 与 ``stats_dataset_xyz_bounds`` 仅读取 **``data[..., :3]``**，即上表 **0:3**，**不使用 6:9**；若需与「整房归一化坐标」对齐的量化实验，须改读 ``[..., 6:9]`` 并另跑统计。

- ``Dataset`` 按块 **惰性读 HDF5**（顺序迭代时复用打开的文件句柄），避免 ``partition=all`` 时整集载入内存。

partition:
  - ``all``: 训练+测试目录中全部块，不做 Area 划分（适合全局 min/max 统计）。
  - ``train`` / ``test``: 按 ``room_filelist.txt`` + ``test_area`` 与 PointNet++ 一致。
"""
from __future__ import annotations

import os
from typing import List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _resolve_h5_path(pc_root: str, data_dir: str, rel: str) -> str:
    rel = rel.rstrip()
    if os.path.isabs(rel) and os.path.isfile(rel):
        return rel
    p = os.path.join(data_dir, rel)
    if os.path.isfile(p):
        return p
    p2 = os.path.join(pc_root, rel)
    if os.path.isfile(p2):
        return p2
    base = os.path.join(data_dir, os.path.basename(rel))
    if os.path.isfile(base):
        return base
    raise FileNotFoundError(f"找不到 HDF5: rel={rel!r} data_dir={data_dir}")


def _iter_block_specs(pc_root: str, data_dir: str) -> List[Tuple[str, int]]:
    """按 all_files.txt 顺序枚举每一块 (h5_path, row_in_that_file)。"""
    list_path = os.path.join(data_dir, "all_files.txt")
    if not os.path.isfile(list_path):
        raise FileNotFoundError(f"缺少 all_files.txt: {list_path}")
    with open(list_path, "r", encoding="utf-8") as f:
        all_files = [line.rstrip() for line in f if line.strip()]
    specs: List[Tuple[str, int]] = []
    for fn in all_files:
        path = _resolve_h5_path(pc_root, data_dir, fn)
        with h5py.File(path, "r") as hf:
            n = int(hf["data"].shape[0])
        for i in range(n):
            specs.append((path, i))
    return specs


def _read_room_lines(data_dir: str) -> List[str]:
    rp = os.path.join(data_dir, "room_filelist.txt")
    if not os.path.isfile(rp):
        raise FileNotFoundError(f"缺少 room_filelist.txt: {rp}")
    with open(rp, "r", encoding="utf-8") as f:
        return [line.rstrip() for line in f if line.strip()]


def build_s3dis_indoor3d_block_specs(
    pc_root: str,
    partition: str,
    test_area: str = "5",
) -> List[Tuple[str, int]]:
    """
    partition: ``all`` | ``train`` | ``test``
    返回按基准顺序的 (h5_path, row) 列表。
    """
    pc_root = os.path.abspath(pc_root)
    train_dir = os.path.join(pc_root, "indoor3d_sem_seg_hdf5_data")
    test_dir = os.path.join(pc_root, "indoor3d_sem_seg_hdf5_data_test")

    if partition == "all":
        out: List[Tuple[str, int]] = []
        for d in (train_dir, test_dir):
            if os.path.isdir(d):
                out.extend(_iter_block_specs(pc_root, d))
        if not out:
            raise FileNotFoundError(
                f"未找到 indoor3d HDF5 目录（至少一个）: {train_dir} 或 {test_dir}"
            )
        return out

    if partition == "train":
        data_dir = train_dir
    elif partition == "test":
        data_dir = test_dir
    else:
        raise ValueError(f"partition 须为 all|train|test，当前: {partition!r}")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"缺少目录: {data_dir}")

    specs = _iter_block_specs(pc_root, data_dir)
    room_lines = _read_room_lines(data_dir)
    if len(room_lines) != len(specs):
        raise ValueError(
            f"room_filelist 行数 {len(room_lines)} 与块数 {len(specs)} 不一致 ({data_dir})"
        )

    test_area_name = "Area_" + str(test_area)
    out = []
    for i, spec in enumerate(specs):
        in_test_area = test_area_name in room_lines[i]
        if partition == "train" and not in_test_area:
            out.append(spec)
        if partition == "test" and in_test_area:
            out.append(spec)
    return out


class S3DISIndoor3DH5Dataset(Dataset):
    """
    每个样本为一块 (num_points, 3)；默认 num_points=4096。
    不做训练时的逐样本点洗牌（FPS/KNN 可复现）。
    """

    def __init__(
        self,
        pc_root: str,
        partition: str = "all",
        test_area: str = "5",
        num_points: int = 4096,
        max_blocks: int = -1,
    ):
        self.pc_root = os.path.abspath(pc_root)
        self.partition = partition
        self.test_area = str(test_area)
        self.num_points = int(num_points)

        self._specs = build_s3dis_indoor3d_block_specs(
            self.pc_root, partition, test_area=self.test_area
        )
        if not self._specs:
            raise RuntimeError(f"S3DIS Indoor3D: 0 块 (partition={partition})")
        if max_blocks > 0:
            self._specs = self._specs[: int(max_blocks)]

        self._hf = None  # type: ignore
        self._hf_path: str | None = None

    def __len__(self) -> int:
        return len(self._specs)

    def _ensure_file(self, path: str) -> h5py.File:
        if self._hf_path != path:
            if self._hf is not None:
                self._hf.close()
            self._hf = h5py.File(path, "r")
            self._hf_path = path
        return self._hf

    def __getitem__(self, idx: int):
        path, row = self._specs[idx]
        hf = self._ensure_file(path)
        # 0:3 = 块中心化 xy + 房间尺度 z；非 6:9 整房归一化坐标（见模块 docstring）
        raw = hf["data"][row, : self.num_points, :3]
        x = np.asarray(raw, dtype=np.float32, order="C")
        return torch.from_numpy(x.copy()), 0

    def __del__(self) -> None:
        if self._hf is not None:
            self._hf.close()
