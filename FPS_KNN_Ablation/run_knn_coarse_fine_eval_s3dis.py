#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KNN 粗筛+细筛 IoU 分析 — S3DIS 点云版

在 S3DIS .npy 点云上做与 KITTI 版相同的分析：
  - FPS + vanilla / adaptive / coarse_fine 三种 KNN 对比
  - coarse_fine、adaptive 相对 vanilla 的邻居集合 IoU
  - radii MSE

每场景：加载 .npy（取前 3 列为 xyz）→ 按 num_points 裁剪/填充 → 送入 FPS+KNN。
S3DIS 为室内场景，不做 ROI 过滤。默认 num_points=50000、batch_size=1。

数据路径：默认使用 pc_data 下的 s3dis_trainval/*.npy（原始点云）；
与 ISEDA26 中 indoor3d 处理后的 HDF5（indoor3d_sem_seg_hdf5_data*）为不同格式，本脚本仅支持 .npy。

用法：
  python run_knn_coarse_fine_eval_s3dis.py --data_dir /workspace/VoxelNet/pc_data/s3dis_trainval
  python run_knn_coarse_fine_eval_s3dis.py --max_samples 50 --num_points 40960 --batch_size 1
  或使用配套脚本：./run_knn_coarse_fine_eval_s3dis.sh
"""

import os
import sys
import json
import argparse
import time
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
MODELNET40_DIR = os.path.join(REPO_ROOT, "PointKAN-elite", "classification_ModelNet40")
for d in (SCRIPT_DIR, REPO_ROOT, MODELNET40_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from spec_kan_fps import (
    fps_with_knn_adaptive,
    fps_with_knn_coarse_fine,
    fps_with_knn_vanilla,
    knn_neighbor_set_iou,
    collate_points,
)


def find_npy_files(data_dir: str):
    return sorted(glob.glob(os.path.join(data_dir, "*.npy")))


def load_s3dis_npy(path: str) -> np.ndarray:
    """
    加载 S3DIS .npy 点云。格式为 [N, 7]（xyz + 其他特征），取前 3 列为 xyz。
    Returns:
        xyz: [N, 3] float32
    """
    data = np.load(path)
    if data.ndim == 1:
        data = data.reshape(-1, 7)  # 常见为 7 维
    xyz = data[:, :3].astype(np.float32)
    return xyz


class S3DISPointCloudWrapper(Dataset):
    """
    S3DIS .npy 点云数据集：每场景裁剪/填充到 num_points，输出 [num_points, 3] + 伪标签 0。
    室内场景不做 ROI 过滤。
    """
    def __init__(
        self,
        data_dir: str,
        num_points: int = 50000,
        max_scenes: int = -1,
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.num_points = num_points
        self.npy_paths = find_npy_files(data_dir)
        if max_scenes > 0:
            self.npy_paths = self.npy_paths[:max_scenes]
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, idx):
        path = self.npy_paths[idx]
        xyz = load_s3dis_npy(path)
        n = xyz.shape[0]
        if n == 0:
            pts = np.zeros((self.num_points, 3), dtype=np.float32)
            return torch.from_numpy(pts), 0
        if n >= self.num_points:
            choice = self.rng.choice(n, size=self.num_points, replace=False)
            pts = xyz[choice]
        else:
            repeat = self.num_points - n
            pad_idx = self.rng.integers(0, n, size=repeat)
            pts = np.concatenate([xyz, xyz[pad_idx]], axis=0)
        return torch.from_numpy(pts), 0


def parse_args():
    p = argparse.ArgumentParser(description="KNN 粗筛+细筛 IoU 分析（S3DIS 点云）")
    p.add_argument("--data_dir", type=str, default="/workspace/VoxelNet/pc_data/s3dis_trainval",
                   help="S3DIS .npy 目录")
    p.add_argument("--num_points", type=int, default=50000,
                   help="每场景点数 N；超出则随机裁到 N，不足则重复填充")
    p.add_argument("--fps_steps", type=int, default=512, help="FPS 采样数 K")
    p.add_argument("--k_neighbors", type=int, default=32, help="KNN 邻居数 k")
    p.add_argument("--batch_size", type=int, default=1, help="批大小")
    p.add_argument("--max_samples", type=int, default=0, help="最多评估场景数，0 表示全部")
    p.add_argument("--min_candidates_ratio", type=float, default=1.1,
                   help="粗筛+细筛：候选数 >= min_candidates_ratio * k_neighbors")
    p.add_argument("--tol", type=float, default=0.05, help="adaptive 的 tol")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_json", type=str, default="", help="结果 JSON 路径")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def run_iou_analysis(args):
    device = torch.device(args.device)
    ds = S3DISPointCloudWrapper(
        args.data_dir,
        num_points=args.num_points,
        max_scenes=args.max_samples if args.max_samples > 0 else -1,
        seed=args.seed,
    )
    n_total = len(ds)
    n_samples = min(args.max_samples, n_total) if args.max_samples > 0 else n_total
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_points,
    )

    k = args.k_neighbors
    K = min(args.fps_steps, args.num_points)
    iou_cf_v_list = []
    iou_ad_v_list = []
    radii_mse_cf_v_list = []
    radii_mse_ad_v_list = []
    time_vanilla_list = []
    time_adaptive_list = []
    time_coarse_fine_list = []
    n_batches = 0
    n_pts_processed = 0
    is_cuda = device.type == "cuda"

    def _sync():
        if is_cuda:
            torch.cuda.synchronize()

    pbar = tqdm(loader, desc="KNN IoU & radii (S3DIS)", ncols=100)
    for pts, _ in pbar:
        if args.max_samples > 0 and n_pts_processed >= args.max_samples:
            break
        pts = pts.to(device, dtype=torch.float32)
        B = pts.size(0)
        if args.max_samples > 0 and n_pts_processed + B > args.max_samples:
            pts = pts[: args.max_samples - n_pts_processed]
            B = pts.size(0)
        n_pts_processed += B

        _sync()
        t0 = time.perf_counter()
        _, _, knn_vanilla, dist_vanilla, radii_vanilla = fps_with_knn_vanilla(
            pts, K, k_neighbors=k, seed="first"
        )
        _sync()
        time_vanilla_list.append(time.perf_counter() - t0)

        _sync()
        t0 = time.perf_counter()
        _, _, knn_adaptive, dist_adaptive, radii_adaptive, _, _ = fps_with_knn_adaptive(
            pts, K, k_neighbors=k, init_threshold=2.4, max_iter=10, tol=args.tol, seed="first"
        )
        _sync()
        time_adaptive_list.append(time.perf_counter() - t0)

        _sync()
        t0 = time.perf_counter()
        _, _, knn_cf, dist_cf, radii_cf, _ = fps_with_knn_coarse_fine(
            pts, K, k_neighbors=k, init_threshold=0.1, max_iter=10,
            min_candidates_ratio=args.min_candidates_ratio, seed="first"
        )
        _sync()
        time_coarse_fine_list.append(time.perf_counter() - t0)

        iou_cf_v = knn_neighbor_set_iou(knn_cf, knn_vanilla)
        iou_ad_v = knn_neighbor_set_iou(knn_adaptive, knn_vanilla)
        iou_cf_v_list.append(iou_cf_v.cpu().numpy())
        iou_ad_v_list.append(iou_ad_v.cpu().numpy())

        ok = torch.isfinite(radii_vanilla) & torch.isfinite(radii_cf) & torch.isfinite(radii_adaptive)
        if ok.any():
            mse_cf_v = ((radii_cf - radii_vanilla).float() ** 2).masked_select(ok).mean().item()
            mse_ad_v = ((radii_adaptive - radii_vanilla).float() ** 2).masked_select(ok).mean().item()
        else:
            mse_cf_v = float("nan")
            mse_ad_v = float("nan")
        radii_mse_cf_v_list.append(mse_cf_v)
        radii_mse_ad_v_list.append(mse_ad_v)
        n_batches += 1

    iou_cf_v_all = np.concatenate(iou_cf_v_list, axis=0)
    iou_ad_v_all = np.concatenate(iou_ad_v_list, axis=0)
    iou_cf_v_per_sample = iou_cf_v_all.mean(axis=1)
    iou_ad_v_per_sample = iou_ad_v_all.mean(axis=1)

    results = {
        "dataset": "S3DIS",
        "device": args.device,
        "data_dir": args.data_dir,
        "n_samples": int(iou_cf_v_per_sample.size),
        "n_total_scenes": n_total,
        "num_points": args.num_points,
        "fps_steps": K,
        "k_neighbors": k,
        "min_candidates_ratio": args.min_candidates_ratio,
        "tol_adaptive": args.tol,
        "iou_coarse_fine_vs_vanilla": {
            "mean": float(np.mean(iou_cf_v_per_sample)),
            "std": float(np.std(iou_cf_v_per_sample)),
            "min": float(np.min(iou_cf_v_per_sample)),
            "max": float(np.max(iou_cf_v_per_sample)),
        },
        "iou_adaptive_vs_vanilla": {
            "mean": float(np.mean(iou_ad_v_per_sample)),
            "std": float(np.std(iou_ad_v_per_sample)),
            "min": float(np.min(iou_ad_v_per_sample)),
            "max": float(np.max(iou_ad_v_per_sample)),
        },
        "radii_mse_coarse_fine_vs_vanilla": float(np.nanmean(radii_mse_cf_v_list)) if np.any(np.isfinite(radii_mse_cf_v_list)) else float("nan"),
        "radii_mse_adaptive_vs_vanilla": float(np.nanmean(radii_mse_ad_v_list)) if np.any(np.isfinite(radii_mse_ad_v_list)) else float("nan"),
        "inference_time_vanilla_sec": {
            "total": float(np.sum(time_vanilla_list)),
            "mean_per_batch": float(np.mean(time_vanilla_list)),
            "std_per_batch": float(np.std(time_vanilla_list)),
        },
        "inference_time_adaptive_sec": {
            "total": float(np.sum(time_adaptive_list)),
            "mean_per_batch": float(np.mean(time_adaptive_list)),
            "std_per_batch": float(np.std(time_adaptive_list)),
        },
        "inference_time_coarse_fine_sec": {
            "total": float(np.sum(time_coarse_fine_list)),
            "mean_per_batch": float(np.mean(time_coarse_fine_list)),
            "std_per_batch": float(np.std(time_coarse_fine_list)),
        },
        "speedup_adaptive_vs_vanilla": float(np.sum(time_vanilla_list) / np.sum(time_adaptive_list)) if np.sum(time_adaptive_list) > 0 else float("nan"),
        "speedup_coarse_fine_vs_vanilla": float(np.sum(time_vanilla_list) / np.sum(time_coarse_fine_list)) if np.sum(time_coarse_fine_list) > 0 else float("nan"),
    }
    return results


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    npy_files = find_npy_files(args.data_dir)
    if len(npy_files) == 0:
        raise FileNotFoundError(f"未找到 .npy 文件: {args.data_dir}")

    # 统计数据集点数（mmap 只读头信息，不载入全部数据）
    pts_per_file = []
    for path in npy_files:
        arr = np.load(path, mmap_mode="r")
        n = int(arr.shape[0]) if arr.ndim >= 1 else 0
        pts_per_file.append(n)
    pts_per_file = np.array(pts_per_file)
    total_pts = int(pts_per_file.sum())
    dataset_stats = {
        "n_scenes": len(npy_files),
        "total_points": total_pts,
        "points_per_scene_min": int(pts_per_file.min()),
        "points_per_scene_max": int(pts_per_file.max()),
        "points_per_scene_mean": float(pts_per_file.mean()),
    }
    pts_min, pts_max = dataset_stats["points_per_scene_min"], dataset_stats["points_per_scene_max"]
    num_points_requested = args.num_points
    num_points_eff = min(args.num_points, pts_max)
    args.num_points = num_points_eff

    print("=" * 60)
    print("S3DIS 数据集点数统计")
    print("=" * 60)
    print(f"  场景/文件数: {dataset_stats['n_scenes']}")
    print(f"  总点数: {dataset_stats['total_points']:,}")
    print(f"  每场景点数: 最少={pts_min:,}  最多={pts_max:,}  平均={dataset_stats['points_per_scene_mean']:,.1f}")
    print("=" * 60)
    print("KNN 粗筛+细筛 与 Vanilla KNN IoU / radii 分析（S3DIS）")
    print("=" * 60)
    t0 = time.perf_counter()
    iou_results = run_iou_analysis(args)
    iou_results["dataset_points_stats"] = dataset_stats
    iou_results["num_points_requested"] = num_points_requested
    iou_results["num_points_effective"] = num_points_eff
    t1 = time.perf_counter()

    print(f"  数据目录: {iou_results['data_dir']}")
    print(f"  分析场景数: {iou_results['n_samples']} (目录总场景数: {iou_results['n_total_scenes']})")
    ds_stats = iou_results.get("dataset_points_stats", {})
    pts_min = ds_stats.get("points_per_scene_min", 0)
    pts_max = ds_stats.get("points_per_scene_max", 0)
    n_req = iou_results.get("num_points_requested", iou_results["num_points"])
    n_eff = iou_results.get("num_points_effective", iou_results["num_points"])
    if n_req > pts_max:
        print(f"  每场景点数 N: {n_eff:,} (设定 {n_req:,}，已按数据集最大点数截断)；数据集每场景点数范围: [{pts_min:,}, {pts_max:,}]")
    else:
        print(f"  每场景点数 N: {n_eff:,}；数据集每场景点数范围: [{pts_min:,}, {pts_max:,}]")
    print(f"  FPS 步数 K: {iou_results['fps_steps']}, 邻居数 k: {iou_results['k_neighbors']}")
    print(f"  粗筛+细筛 min_candidates_ratio: {iou_results['min_candidates_ratio']}")
    print("  --- IoU (相对 Vanilla) ---")
    print(f"  Coarse-Fine vs Vanilla: mean={iou_results['iou_coarse_fine_vs_vanilla']['mean']:.4f} "
          f"std={iou_results['iou_coarse_fine_vs_vanilla']['std']:.4f} "
          f"[{iou_results['iou_coarse_fine_vs_vanilla']['min']:.4f}, {iou_results['iou_coarse_fine_vs_vanilla']['max']:.4f}]")
    print(f"  Adaptive vs Vanilla:    mean={iou_results['iou_adaptive_vs_vanilla']['mean']:.4f} "
          f"std={iou_results['iou_adaptive_vs_vanilla']['std']:.4f} "
          f"[{iou_results['iou_adaptive_vs_vanilla']['min']:.4f}, {iou_results['iou_adaptive_vs_vanilla']['max']:.4f}]")
    print("  --- Radii MSE (相对 Vanilla) ---")
    print(f"  Coarse-Fine vs Vanilla: {iou_results['radii_mse_coarse_fine_vs_vanilla']:.6e}")
    print(f"  Adaptive vs Vanilla:    {iou_results['radii_mse_adaptive_vs_vanilla']:.6e}")
    print("  --- 推理时间对比 (FPS+KNN) ---")
    tv = iou_results["inference_time_vanilla_sec"]
    ta = iou_results["inference_time_adaptive_sec"]
    tc = iou_results["inference_time_coarse_fine_sec"]
    print(f"  Vanilla:      total={tv['total']:.4f}s  mean_per_batch={tv['mean_per_batch']:.4f}s (±{tv['std_per_batch']:.4f})")
    print(f"  Adaptive:     total={ta['total']:.4f}s  mean_per_batch={ta['mean_per_batch']:.4f}s (±{ta['std_per_batch']:.4f})")
    print(f"  Coarse-Fine:  total={tc['total']:.4f}s  mean_per_batch={tc['mean_per_batch']:.4f}s (±{tc['std_per_batch']:.4f})")
    print(f"  加速比 Adaptive vs Vanilla:     {iou_results['speedup_adaptive_vs_vanilla']:.4f}x")
    print(f"  加速比 Coarse-Fine vs Vanilla:  {iou_results['speedup_coarse_fine_vs_vanilla']:.4f}x")
    print(f"  总耗时: {t1 - t0:.2f}s")
    print("=" * 60)

    out_path = args.out_json.strip()
    if not out_path:
        out_path = os.path.join(SCRIPT_DIR, "knn_coarse_fine_iou_results_s3dis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(iou_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已写入: {out_path}")


if __name__ == "__main__":
    main()
