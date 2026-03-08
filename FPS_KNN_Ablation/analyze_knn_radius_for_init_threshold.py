#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通过分析每个数据集上 Vanilla KNN 的「第 k 近邻距离」分布，用分位数（99%、75% 等）
推荐 init_threshold 的合理范围，供 adaptive/coarse_fine KNN 使用。

思路：对每帧做 FPS + Vanilla KNN，得到每个 FPS 采样点的第 k 近邻距离（knn_radii）。
收集所有帧、所有 FPS 点的该距离，做直方图并计算分位数；init_threshold 选在
该分布的低分位（如 50%）到高分位（如 99%）之间，可覆盖多数点的邻居范围。

用法：
  python analyze_knn_radius_for_init_threshold.py --dataset waymo --max_samples 50
  python analyze_knn_radius_for_init_threshold.py --dataset kitti --out_dir ./radius_analysis
  python analyze_knn_radius_for_init_threshold.py --dataset nuscenes --percentiles "50,75,90,95,99"

输出中的 suggested_init_threshold_range_* 可直接用作 sweep_knn_params.py 的 --init_threshold
扫描范围（在该范围内取若干点如 0.01,0.05,0.1 等，需按实际分位数尺度调整）。
"""

import os
import sys
import json
import argparse
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
MODELNET40_DIR = os.path.join(REPO_ROOT, "PointKAN-elite", "classification_ModelNet40")
for d in (SCRIPT_DIR, REPO_ROOT, MODELNET40_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kitti_fps_exp import load_kitti_bin, roi_filter
from spec_kan_fps import fps_with_knn_vanilla, collate_points

# 复用 sweep 脚本里的数据集与 get_loader
from sweep_knn_params import get_loader


def parse_args():
    p = argparse.ArgumentParser(description="分析 KNN 第 k 近邻距离分布，推荐 init_threshold 范围")
    p.add_argument("--dataset", type=str, required=True, choices=["kitti", "nuscenes", "waymo"])
    p.add_argument("--data_dir", type=str, default="")
    p.add_argument("--num_points", type=int, default=50000)
    p.add_argument("--fps_steps", type=int, default=2048)
    p.add_argument("--k_neighbors", type=int, default=32)
    p.add_argument("--max_samples", type=int, default=50, help="参与统计的帧数")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--no_roi", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--percentiles", type=str, default="25,50,75,90,95,99",
                   help="逗号分隔的分位数，用于统计与输出")
    p.add_argument("--out_dir", type=str, default="", help="输出目录，默认当前目录")
    p.add_argument("--out_json", type=str, default="", help="统计结果 JSON 路径")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def collect_knn_radii(loader, device, K, k_neighbors, max_samples):
    """收集所有 batch 的 vanilla KNN 第 k 近邻距离，返回一维数组。"""
    radii_list = []
    n_processed = 0
    for pts, _ in tqdm(loader, desc="Collecting KNN radii", ncols=100):
        if max_samples > 0 and n_processed >= max_samples:
            break
        pts = pts.to(device, dtype=torch.float32)
        B = pts.size(0)
        if max_samples > 0 and n_processed + B > max_samples:
            pts = pts[: max_samples - n_processed]
            B = pts.size(0)
        n_processed += B
        _, _, _, _, knn_radii = fps_with_knn_vanilla(
            pts, K, k_neighbors=k_neighbors, seed="first"
        )
        # knn_radii: (B, K)
        r = knn_radii.cpu().numpy().ravel()
        r = r[np.isfinite(r)]
        radii_list.append(r)
    if not radii_list:
        return np.array([], dtype=np.float64)
    return np.concatenate(radii_list, axis=0)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if not args.data_dir.strip():
        if args.dataset == "kitti":
            args.data_dir = "/workspace/VoxelNet/KITTI_Mini/training/velodyne"
        elif args.dataset == "nuscenes":
            args.data_dir = "/workspace/VoxelNet/Nuscenes_Mini/sweeps/LIDAR_TOP"
        else:
            args.data_dir = "/workspace/VoxelNet/Waymo_Mini/Waymo_mini/kitti_format/training/velodyne"

    out_dir = args.out_dir.strip() or "."
    os.makedirs(out_dir, exist_ok=True)

    percentiles = [int(x.strip()) for x in args.percentiles.split(",") if x.strip()]
    if not percentiles:
        percentiles = [25, 50, 75, 90, 95, 99]

    device = torch.device(args.device)
    loader, n_total = get_loader(
        args.dataset,
        args.data_dir,
        args.num_points,
        do_roi=not args.no_roi,
        max_frames=-1,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    K = min(args.fps_steps, args.num_points)
    k = args.k_neighbors

    radii = collect_knn_radii(loader, device, K, k, args.max_samples)
    if radii.size == 0:
        print("No valid radii collected.")
        return

    # 分位数
    perc_vals = {p: float(np.percentile(radii, p)) for p in percentiles}
    stats = {
        "min": float(np.min(radii)),
        "max": float(np.max(radii)),
        "mean": float(np.mean(radii)),
        "std": float(np.std(radii)),
        "median": float(np.median(radii)),
        "count": int(radii.size),
    }
    stats["percentiles"] = perc_vals

    # 建议 init_threshold 范围：常用 50%–99% 或 75%–95%
    p50 = perc_vals.get(50, stats["median"])
    p99 = perc_vals.get(99, stats["max"])
    p75 = perc_vals.get(75)
    p95 = perc_vals.get(95)
    p90 = perc_vals.get(90)
    stats["suggested_init_threshold_range_p50_p99"] = [p50, p99]
    stats["suggested_init_threshold_range_p75_p95"] = [p75, p95] if (p75 is not None and p95 is not None) else [p50, p99]
    stats["suggested_init_threshold_range_p50_p95"] = [p50, p95] if p95 is not None else [p50, p99]
    stats["suggested_init_threshold_median"] = stats["median"]
    stats["suggested_init_threshold_p75"] = p75
    stats["suggested_init_threshold_p90"] = p90

    out = {
        "dataset": args.dataset,
        "data_dir": args.data_dir,
        "n_frames_used": min(args.max_samples, n_total) if args.max_samples > 0 else n_total,
        "n_total_frames": n_total,
        "num_points": args.num_points,
        "fps_steps": K,
        "k_neighbors": k,
        "do_roi": not args.no_roi,
        "n_radius_samples": int(radii.size),
        "stats": stats,
    }

    # 打印表
    print("\n" + "=" * 60)
    print(f"KNN 第 k 近邻距离分布（dataset={args.dataset}, k={k}, 样本数={radii.size}）")
    print("=" * 60)
    print(f"  min   = {stats['min']:.6f}")
    print(f"  max   = {stats['max']:.6f}")
    print(f"  mean  = {stats['mean']:.6f}")
    print(f"  std   = {stats['std']:.6f}")
    print(f"  median= {stats['median']:.6f}")
    print("  分位数:")
    for p in sorted(perc_vals.keys()):
        print(f"    {p}%   = {perc_vals[p]:.6f}")
    print("  建议 init_threshold 范围（基于分位数）:")
    print(f"    [p50, p99] = [{stats['suggested_init_threshold_range_p50_p99'][0]:.6f}, {stats['suggested_init_threshold_range_p50_p99'][1]:.6f}]")
    print(f"    [p75, p95] = [{stats['suggested_init_threshold_range_p75_p95'][0]:.6f}, {stats['suggested_init_threshold_range_p75_p95'][1]:.6f}]")
    p75s = stats['suggested_init_threshold_p75']
    p90s = stats['suggested_init_threshold_p90']
    p75_str = f"{p75s:.6f}" if p75s is not None else "N/A"
    p90_str = f"{p90s:.6f}" if p90s is not None else "N/A"
    print(f"    单点建议: median={stats['suggested_init_threshold_median']:.6f}, p75={p75_str}, p90={p90_str}")
    print("=" * 60)

    # 直方图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.hist(radii, bins=min(80, max(20, radii.size // 200)), density=True, color="steelblue", alpha=0.8, edgecolor="black", linewidth=0.3)
    ax.axvline(stats["median"], color="green", linestyle="--", label=f"median={stats['median']:.4f}")
    if 75 in perc_vals:
        ax.axvline(perc_vals[75], color="orange", linestyle="--", label=f"p75={perc_vals[75]:.4f}")
    if 95 in perc_vals:
        ax.axvline(perc_vals[95], color="red", linestyle=":", label=f"p95={perc_vals[95]:.4f}")
    ax.set_xlabel("k-th neighbor distance (radius)")
    ax.set_ylabel("density")
    ax.set_title(f"KNN radius distribution ({args.dataset}, k={k})")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    sorted_r = np.sort(radii)
    cdf = np.arange(1, len(sorted_r) + 1, dtype=np.float64) / len(sorted_r)
    ax.plot(sorted_r, cdf, color="steelblue", linewidth=0.8)
    for p in sorted(perc_vals.keys()):
        v = perc_vals[p]
        ax.axvline(v, color="gray", linestyle=":", alpha=0.5)
        ax.axhline(p / 100.0, color="gray", linestyle=":", alpha=0.5)
        ax.scatter([v], [p / 100.0], s=15, zorder=5)
        ax.annotate(f"p{p}={v:.3f}", (v, p / 100.0), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("k-th neighbor distance (radius)")
    ax.set_ylabel("CDF")
    ax.set_title(f"CDF with percentiles ({args.dataset})")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    hist_path = os.path.join(out_dir, f"knn_radius_distribution_{args.dataset}.png")
    plt.savefig(hist_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n直方图与 CDF 已保存: {hist_path}")

    json_path = args.out_json.strip() or os.path.join(out_dir, f"knn_radius_stats_{args.dataset}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"统计结果已保存: {json_path}")


if __name__ == "__main__":
    main()
