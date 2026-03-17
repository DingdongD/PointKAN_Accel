#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计 S3DIS 点云数据集（s3dis_trainval）的点云数目。
数据路径：/workspace/VoxelNet/pc_data/s3dis_trainval/*.npy
"""
import os
import glob
import argparse
import numpy as np
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser(description="统计 S3DIS 点云数目")
    p.add_argument("--data_dir", type=str, default="/workspace/VoxelNet/pc_data/s3dis_trainval",
                   help="S3DIS .npy 目录")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.npy")))
    if not files:
        print(f"未找到 .npy 文件: {args.data_dir}")
        return

    total_pts = 0
    pts_per_file = []
    for path in tqdm(files, desc="统计点数"):
        arr = np.load(path)
        n = int(arr.shape[0]) if arr.ndim >= 1 else 0
        total_pts += n
        pts_per_file.append(n)

    pts_per_file = np.array(pts_per_file)
    n_files = len(files)

    print("=" * 60)
    print("S3DIS 点云数目统计")
    print("=" * 60)
    print(f"  数据目录: {args.data_dir}")
    print(f"  场景/文件数: {n_files}")
    print(f"  总点数: {total_pts:,}")
    print(f"  每文件点数: min={pts_per_file.min():,}  max={pts_per_file.max():,}  mean={pts_per_file.mean():,.1f}  std={pts_per_file.std():,.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
