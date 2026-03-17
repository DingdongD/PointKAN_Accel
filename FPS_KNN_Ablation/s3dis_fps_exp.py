#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S3DIS 点云 FPS 实验脚本。

与 waymo_fps_exp / kitti_fps_exp 相同流程：加载、FPS、指标、BEV、半径曲线、
点云数直方图、每场景 xyz 区间统计与直方图。数据目录默认 pc_data/s3dis_trainval，
文件为 .npy（[N, 7] 取前 3 列为 xyz）。室内场景不做 ROI 过滤。
"""

import os
import glob
import argparse
import numpy as np

from kitti_fps_exp import get_parser, run_main


def find_npy_files(data_dir: str):
    """返回 data_dir 下所有 .npy 文件路径（已排序）。"""
    return sorted(glob.glob(os.path.join(data_dir, "*.npy")))


def load_s3dis_npy(path: str) -> np.ndarray:
    """S3DIS .npy：格式 [N, 7]，取前 3 列为 xyz，返回 [N, 3] 供 FPS/ROI 逻辑使用。"""
    data = np.load(path)
    if data.ndim == 1:
        data = data.reshape(-1, 7)
    xyz = data[:, :3].astype(np.float32)
    return xyz


def main():
    parser = get_parser(description="S3DIS 点云 FPS 实验：统计、可视化与批处理")
    parser.set_defaults(
        data_dir="/workspace/VoxelNet/pc_data/s3dis_trainval",
        out_dir="./s3dis_fps_outputs",
        no_roi=True,  # 室内场景不做 ROI
    )
    args = parser.parse_args()
    args._load_fn = load_s3dis_npy
    args._find_files = find_npy_files
    run_main(args)


if __name__ == "__main__":
    main()
