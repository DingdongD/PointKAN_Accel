#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NuScenes 点云 FPS 实验脚本。

与 kitti_fps_exp 相同流程：加载、ROI（可选）、FPS、指标、BEV、半径曲线、
点云数直方图、每帧 xyz 区间统计与直方图。数据目录默认 NuScenes LIDAR_TOP，
文件匹配 *.pcd.bin；NuScenes 与 KITTI 相同为每点 4×float32（x,y,z,intensity），
保留原始点云（不做预下采样）。
"""

import os
import glob
import argparse

# 复用 KITTI 实验的全部逻辑，仅替换加载与查找
from kitti_fps_exp import (
    get_parser,
    run_main,
    load_kitti_bin,
    find_bin_files,
)


def load_nuscenes_pcd_bin(bin_path: str):
    """NuScenes .pcd.bin 与 KITTI 相同：每点 4×float32（x,y,z,intensity）。"""
    return load_kitti_bin(bin_path)


def find_nuscenes_files(data_dir: str):
    """返回 data_dir 下所有 .pcd.bin 文件路径（已排序）。"""
    return sorted(glob.glob(os.path.join(data_dir, "*.pcd.bin")))


def main():
    parser = get_parser(description="NuScenes 点云 FPS 实验：统计、可视化与批处理")
    parser.set_defaults(
        data_dir="/workspace/VoxelNet/Nuscenes_Mini/sweeps/LIDAR_TOP",
        out_dir="./nuscenes_fps_outputs",
    )
    args = parser.parse_args()
    args._load_fn = load_nuscenes_pcd_bin
    args._find_files = find_nuscenes_files
    run_main(args)


if __name__ == "__main__":
    main()
