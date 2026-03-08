#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KNN 粗筛+细筛 IoU 分析 — NuScenes 点云版

在 NuScenes LIDAR_TOP .pcd.bin 点云上做与 KITTI 版相同的分析：
  - FPS + vanilla / adaptive / coarse_fine 三种 KNN 对比
  - coarse_fine、adaptive 相对 vanilla 的邻居集合 IoU
  - radii MSE

每帧：加载 .pcd.bin → ROI 过滤 → 按 num_points 裁剪/填充（保留原始点云尺度）
→ 送入 FPS+KNN。默认 num_points=50000、batch_size=1。
不跑分类推理。

用法：
  python run_knn_coarse_fine_eval_nuscenes.py --data_dir /path/to/LIDAR_TOP
  python run_knn_coarse_fine_eval_nuscenes.py --max_samples 50 --num_points 40000 --batch_size 1
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

from kitti_fps_exp import load_kitti_bin, roi_filter
from spec_kan_fps import (
    fps_with_knn_adaptive,
    fps_with_knn_coarse_fine,
    fps_with_knn_vanilla,
    knn_neighbor_set_iou,
    collate_points,
)


def find_nuscenes_files(data_dir: str):
    return sorted(glob.glob(os.path.join(data_dir, "*.pcd.bin")))


def load_nuscenes_pcd_bin(path: str):
    """NuScenes .pcd.bin 与 KITTI 相同：每点 4×float32（x,y,z,intensity）。"""
    return load_kitti_bin(path)


class NuScenesPointCloudWrapper(Dataset):
    """
    NuScenes LIDAR_TOP .pcd.bin：每帧 ROI 后裁剪/填充到 num_points，保留原始尺度，
    输出 [num_points, 3] + 伪标签 0。
    """
    def __init__(
        self,
        data_dir: str,
        num_points: int = 50000,
        do_roi: bool = True,
        max_frames: int = -1,
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.num_points = num_points
        self.do_roi = do_roi
        self.bin_paths = find_nuscenes_files(data_dir)
        if max_frames > 0:
            self.bin_paths = self.bin_paths[:max_frames]
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.bin_paths)

    def __getitem__(self, idx):
        path = self.bin_paths[idx]
        points = load_nuscenes_pcd_bin(path)
        if self.do_roi:
            points = roi_filter(points)
        xyz = points[:, :3].astype(np.float32)
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
    p = argparse.ArgumentParser(description="KNN 粗筛+细筛 IoU 分析（NuScenes 点云）")
    p.add_argument("--data_dir", type=str, default="/workspace/VoxelNet/Nuscenes_Mini/sweeps/LIDAR_TOP",
                   help="NuScenes LIDAR_TOP 目录")
    p.add_argument("--num_points", type=int, default=50000,
                   help="每帧点数 N，保留原始尺度；超出则随机裁到 N，不足则重复填充")
    p.add_argument("--fps_steps", type=int, default=512, help="FPS 采样数 K")
    p.add_argument("--k_neighbors", type=int, default=24, help="KNN 邻居数 k")
    p.add_argument("--batch_size", type=int, default=1, help="批大小")
    p.add_argument("--max_samples", type=int, default=0, help="最多评估帧数，0 表示全部")
    p.add_argument("--min_candidates_ratio", type=float, default=1.2,
                   help="粗筛+细筛：候选数 >= min_candidates_ratio * k_neighbors")
    p.add_argument("--tol", type=float, default=0.05, help="adaptive 的 tol")
    p.add_argument("--no_roi", action="store_true", help="不做 ROI 过滤")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_json", type=str, default="", help="结果 JSON 路径")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def run_iou_analysis(args):
    device = torch.device(args.device)
    ds = NuScenesPointCloudWrapper(
        args.data_dir,
        num_points=args.num_points,
        do_roi=not args.no_roi,
        max_frames=args.max_samples if args.max_samples > 0 else -1,
        seed=args.seed,
    )
    n_total = len(ds)
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
    n_pts_processed = 0
    is_cuda = device.type == "cuda"

    def _sync():
        if is_cuda:
            torch.cuda.synchronize()

    pbar = tqdm(loader, desc="KNN IoU & radii (NuScenes)", ncols=100)
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
            pts, K, k_neighbors=k, init_threshold=0.1, max_iter=10, tol=args.tol, seed="first"
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

    iou_cf_v_all = np.concatenate(iou_cf_v_list, axis=0)
    iou_ad_v_all = np.concatenate(iou_ad_v_list, axis=0)
    iou_cf_v_per_sample = iou_cf_v_all.mean(axis=1)
    iou_ad_v_per_sample = iou_ad_v_all.mean(axis=1)
    results = {
        "dataset": "NuScenes",
        "data_dir": args.data_dir,
        "n_samples": int(iou_cf_v_per_sample.size),
        "n_total_frames": n_total,
        "num_points": args.num_points,
        "fps_steps": K,
        "k_neighbors": k,
        "do_roi": not args.no_roi,
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
        "radii_mse_coarse_fine_vs_vanilla": float(np.nanmean(radii_mse_cf_v_list)),
        "radii_mse_adaptive_vs_vanilla": float(np.nanmean(radii_mse_ad_v_list)),
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
    files = find_nuscenes_files(args.data_dir)
    if len(files) == 0:
        raise FileNotFoundError(f"未找到 .pcd.bin 文件: {args.data_dir}")
    print("=" * 60)
    print("KNN 粗筛+细筛 与 Vanilla KNN IoU / radii 分析（NuScenes）")
    print("=" * 60)
    t0 = time.perf_counter()
    iou_results = run_iou_analysis(args)
    t1 = time.perf_counter()
    print(f"  数据目录: {iou_results['data_dir']}")
    print(f"  分析帧数: {iou_results['n_samples']} (目录总帧数: {iou_results['n_total_frames']})")
    print(f"  每帧点数 N: {iou_results['num_points']}, FPS 步数 K: {iou_results['fps_steps']}, 邻居数 k: {iou_results['k_neighbors']}")
    print(f"  ROI 过滤: {iou_results['do_roi']}")
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
        out_path = os.path.join(SCRIPT_DIR, "knn_coarse_fine_iou_results_nuscenes.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(iou_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已写入: {out_path}")


if __name__ == "__main__":
    main()
