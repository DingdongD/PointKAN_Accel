#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Waymo（KITTI 格式）点云 FPS 实验脚本。

与 kitti_fps_exp 相同流程：加载、ROI（可选）、FPS、指标、BEV、半径曲线、
点云数直方图、每帧 xyz 区间统计与直方图。数据目录默认 Waymo kitti_format velodyne，
文件为 .bin；Waymo 每点 6×float32（与 KITTI/NuScenes 的 4 维不同），保留原始点云。
"""

import argparse
import numpy as np

from kitti_fps_exp import get_parser, run_main, find_bin_files


def load_waymo_bin(bin_path: str):
    """Waymo .bin：每点 6×float32，roi 用前 3 列。"""
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 6)


def main():
    parser = get_parser(description="Waymo 点云 FPS 实验：统计、可视化与批处理")
    parser.set_defaults(
        data_dir="/workspace/VoxelNet/Waymo_Mini/Waymo_mini/kitti_format/training/velodyne",
        out_dir="./waymo_fps_outputs",
    )
    args = parser.parse_args()
    args._load_fn = load_waymo_bin
    args._find_files = find_bin_files
    run_main(args)


if __name__ == "__main__":
    main()
