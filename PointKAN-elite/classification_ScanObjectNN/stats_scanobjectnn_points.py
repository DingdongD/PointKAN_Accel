#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计 ScanObjectNN 数据集中每个样本的点云数目（每样本点数）。

支持与 ScanObjectNN.py 相同的数据路径，也可通过参数指定。输出各分区的
样本数、点数的 min/max/mean/std/median 及可选 CSV。
"""

import argparse
import os
import sys

import numpy as np

# 与 ScanObjectNN 同目录，便于复用路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# 默认数据目录（与 ScanObjectNN.load_scanobjectnn_data 中使用的根目录一致）
DEFAULT_DATA_ROOT = "/workspace/VoxelNet/scannet_kan/main_split"
DEFAULT_H5_PATTERN = "{partition}_objectdataset_augmentedrot_scale75.h5"


def get_h5_path(data_root, partition):
    return os.path.join(data_root, DEFAULT_H5_PATTERN.format(partition=partition))


def load_point_counts(h5_path):
    """从 h5 中读取每个样本的点数，返回 (num_samples,) 的数组。"""
    try:
        import h5py
    except ImportError:
        raise RuntimeError("需要 h5py，请安装: pip install h5py")
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    with h5py.File(h5_path, "r") as f:
        data = f["data"]
        # data shape: (N, P, 3) 或 类似，每样本点数为 data[i].shape[0]
        n = data.shape[0]
        counts = np.zeros(n, dtype=np.int64)
        for i in range(n):
            counts[i] = data[i].shape[0]
            print(counts[i])
    return counts


def main():
    parser = argparse.ArgumentParser(description="统计 ScanObjectNN 每样本点云数目")
    parser.add_argument(
        "--data_root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="h5 所在目录，默认与 ScanObjectNN.py 一致",
    )
    parser.add_argument(
        "--partitions",
        type=str,
        nargs="+",
        default=["training", "test"],
        help="要统计的分区，如 training test",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="若指定，将汇总统计写入该 CSV 文件",
    )
    parser.add_argument(
        "--csv_per_sample",
        type=str,
        default=None,
        help="若指定，将每个样本的点数写入该 CSV（列: partition, sample_idx, num_points）",
    )
    args = parser.parse_args()

    all_rows = []
    per_sample_rows = []
    for partition in args.partitions:
        h5_path = get_h5_path(args.data_root, partition)
        if not os.path.isfile(h5_path):
            print("跳过（文件不存在）: {}".format(h5_path))
            continue
        counts = load_point_counts(h5_path)
        n_samples = len(counts)
        min_p, max_p = int(counts.min()), int(counts.max())
        mean_p, std_p = float(counts.mean()), float(counts.std())
        median_p = int(np.median(counts))

        line = (
            "分区 {}: 样本数={}, 点数 min={}, max={}, mean={:.2f}, std={:.2f}, median={}".format(
                partition, n_samples, min_p, max_p, mean_p, std_p, median_p
            )
        )
        print(line)
        all_rows.append(
            {
                "partition": partition,
                "num_samples": n_samples,
                "min_points": min_p,
                "max_points": max_p,
                "mean_points": round(mean_p, 4),
                "std_points": round(std_p, 4),
                "median_points": median_p,
            }
        )
        for idx in range(len(counts)):
            per_sample_rows.append((partition, idx, int(counts[idx])))

    if args.csv_per_sample and per_sample_rows:
        with open(args.csv_per_sample, "w", encoding="utf-8") as f:
            f.write("partition,sample_idx,num_points\n")
            for partition, idx, num_p in per_sample_rows:
                f.write("{},{},{}\n".format(partition, idx, num_p))
        print("每样本点数 CSV 已保存: {}".format(os.path.abspath(args.csv_per_sample)))

    if args.csv and all_rows:
        import csv
        with open(args.csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "partition",
                    "num_samples",
                    "min_points",
                    "max_points",
                    "mean_points",
                    "std_points",
                    "median_points",
                ],
            )
            w.writeheader()
            w.writerows(all_rows)
        print("汇总 CSV 已保存: {}".format(os.path.abspath(args.csv)))


if __name__ == "__main__":
    main()
