#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KNN 超参数扫描：为 KITTI / NuScenes / Waymo 寻找最佳 init_threshold（及可选 tol、max_iter）。

以 init_threshold 为主扫描维度，在少量帧上评估每种参数下的
Coarse-Fine vs Vanilla 与 Adaptive vs Vanilla 的 IoU，选出 IoU 最高的参数并写入 JSON。

用法：
  # 仅扫描 init_threshold（默认网格）
  python sweep_knn_params.py --dataset waymo --max_samples 30 --out_json ./sweep_waymo.json
  python sweep_knn_params.py --dataset kitti --data_dir /path/to/velodyne --max_samples 20
  python sweep_knn_params.py --dataset nuscenes --max_samples 25

  # 自定义 init_threshold 网格
  python sweep_knn_params.py --dataset waymo --init_threshold "0.02,0.05,0.1,0.2,0.3" --max_samples 30

  # 同时扫描 tol（adaptive）或 max_iter
  python sweep_knn_params.py --dataset waymo --sweep_tol --tol_list "0.02,0.05,0.1" --max_samples 20
  python sweep_knn_params.py --dataset waymo --sweep_max_iter --max_iter_list "5,10,20" --max_samples 20
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

# --------------- 数据集：KITTI ---------------
def find_bin_files(data_dir: str):
    return sorted(glob.glob(os.path.join(data_dir, "*.bin")))


class KITTIPointCloudWrapper(Dataset):
    def __init__(self, data_dir: str, num_points: int, do_roi: bool, max_frames: int, seed: int):
        self.num_points = num_points
        self.do_roi = do_roi
        self.bin_paths = find_bin_files(data_dir)[:max_frames] if max_frames > 0 else find_bin_files(data_dir)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.bin_paths)

    def __getitem__(self, idx):
        path = self.bin_paths[idx]
        points = load_kitti_bin(path)
        if self.do_roi:
            points = roi_filter(points)
        xyz = points[:, :3].astype(np.float32)
        n = xyz.shape[0]
        if n == 0:
            return torch.from_numpy(np.zeros((self.num_points, 3), dtype=np.float32)), 0
        if n >= self.num_points:
            choice = self.rng.choice(n, size=self.num_points, replace=False)
            pts = xyz[choice]
        else:
            pts = np.concatenate([xyz, xyz[self.rng.integers(0, n, size=self.num_points - n)]], axis=0)
        return torch.from_numpy(pts), 0


# --------------- 数据集：NuScenes ---------------
def find_nuscenes_files(data_dir: str):
    return sorted(glob.glob(os.path.join(data_dir, "*.pcd.bin")))


def load_nuscenes_pcd_bin(path: str):
    return load_kitti_bin(path)


class NuScenesPointCloudWrapper(Dataset):
    def __init__(self, data_dir: str, num_points: int, do_roi: bool, max_frames: int, seed: int):
        self.num_points = num_points
        self.do_roi = do_roi
        paths = find_nuscenes_files(data_dir)
        self.bin_paths = paths[:max_frames] if max_frames > 0 else paths
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
            return torch.from_numpy(np.zeros((self.num_points, 3), dtype=np.float32)), 0
        if n >= self.num_points:
            choice = self.rng.choice(n, size=self.num_points, replace=False)
            pts = xyz[choice]
        else:
            pts = np.concatenate([xyz, xyz[self.rng.integers(0, n, size=self.num_points - n)]], axis=0)
        return torch.from_numpy(pts), 0


# --------------- 数据集：Waymo ---------------
def load_waymo_bin(path: str):
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 6)
    return points


class WaymoPointCloudWrapper(Dataset):
    def __init__(self, data_dir: str, num_points: int, do_roi: bool, max_frames: int, seed: int):
        self.num_points = num_points
        self.do_roi = do_roi
        paths = find_bin_files(data_dir)
        self.bin_paths = paths[:max_frames] if max_frames > 0 else paths
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.bin_paths)

    def __getitem__(self, idx):
        path = self.bin_paths[idx]
        points = load_waymo_bin(path)
        if self.do_roi:
            points = roi_filter(points)
        xyz = points[:, :3].astype(np.float32)
        n = xyz.shape[0]
        if n == 0:
            return torch.from_numpy(np.zeros((self.num_points, 3), dtype=np.float32)), 0
        if n >= self.num_points:
            choice = self.rng.choice(n, size=self.num_points, replace=False)
            pts = xyz[choice]
        else:
            pts = np.concatenate([xyz, xyz[self.rng.integers(0, n, size=self.num_points - n)]], axis=0)
        return torch.from_numpy(pts), 0


# --------------- 构建 DataLoader ---------------
def get_loader(dataset_name: str, data_dir: str, num_points: int, do_roi: bool,
               max_frames: int, batch_size: int, seed: int):
    max_f = max_frames if max_frames > 0 else 1000000
    if dataset_name == "kitti":
        ds = KITTIPointCloudWrapper(data_dir, num_points, do_roi, max_f, seed)
    elif dataset_name == "nuscenes":
        ds = NuScenesPointCloudWrapper(data_dir, num_points, do_roi, max_f, seed)
    elif dataset_name == "waymo":
        ds = WaymoPointCloudWrapper(data_dir, num_points, do_roi, max_f, seed)
    else:
        raise ValueError("dataset must be kitti | nuscenes | waymo")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_points,
    ), len(ds)


# --------------- 单次评估：给定 init_threshold, tol, max_iter 的 IoU ---------------
@torch.no_grad()
def run_one_config(
    loader,
    device,
    K,
    k_neighbors,
    init_threshold,
    tol,
    max_iter,
    min_candidates_ratio,
    max_samples,
):
    iou_cf_list = []
    iou_ad_list = []
    n_processed = 0
    for pts, _ in loader:
        if max_samples > 0 and n_processed >= max_samples:
            break
        pts = pts.to(device, dtype=torch.float32)
        B = pts.size(0)
        if max_samples > 0 and n_processed + B > max_samples:
            pts = pts[: max_samples - n_processed]
            B = pts.size(0)
        n_processed += B

        _, _, knn_vanilla, _, _ = fps_with_knn_vanilla(pts, K, k_neighbors=k_neighbors, seed="first")
        _, _, knn_adaptive, _, _, _, _ = fps_with_knn_adaptive(
            pts, K, k_neighbors=k_neighbors,
            init_threshold=init_threshold, max_iter=max_iter, tol=tol, seed="first"
        )
        _, _, knn_cf, _, _, _ = fps_with_knn_coarse_fine(
            pts, K, k_neighbors=k_neighbors,
            init_threshold=init_threshold, max_iter=max_iter,
            min_candidates_ratio=min_candidates_ratio, seed="first"
        )
        iou_cf_list.append(knn_neighbor_set_iou(knn_cf, knn_vanilla).cpu().numpy())
        iou_ad_list.append(knn_neighbor_set_iou(knn_adaptive, knn_vanilla).cpu().numpy())

    if not iou_cf_list:
        return float("nan"), float("nan")
    iou_cf_all = np.concatenate(iou_cf_list, axis=0)
    iou_ad_all = np.concatenate(iou_ad_list, axis=0)
    mean_cf = float(iou_cf_all.mean())
    mean_ad = float(iou_ad_all.mean())
    return mean_cf, mean_ad


def parse_args():
    p = argparse.ArgumentParser(description="KNN 超参数扫描（init_threshold 等）选最佳 IoU")
    p.add_argument("--dataset", type=str, required=True, choices=["kitti", "nuscenes", "waymo"],
                   help="数据集名称")
    p.add_argument("--data_dir", type=str, default="",
                   help="点云目录，不填则用该数据集默认路径")
    p.add_argument("--num_points", type=int, default=50000, help="每帧点数 N")
    p.add_argument("--fps_steps", type=int, default=2048, help="FPS 步数 K")
    p.add_argument("--k_neighbors", type=int, default=32, help="KNN 邻居数 k")
    p.add_argument("--max_samples", type=int, default=30,
                   help="扫描时每组参数评估的帧数，不宜过大以保证扫描速度")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--no_roi", action="store_true", help="不做 ROI 过滤")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # init_threshold 扫描（主）
    p.add_argument("--init_threshold", type=str, default="0.01,0.02,0.05,0.08,0.1,0.15,0.2,0.25,0.3,0.4,0.5",
                   help="逗号分隔的 init_threshold 列表，用于扫描")
    # 固定参数
    p.add_argument("--tol", type=float, default=0.05, help="adaptive 的 tol（不扫描时使用）")
    p.add_argument("--tol_list", type=str, default="",
                   help="若与 --sweep_tol 同用，则为逗号分隔的 tol 列表")
    p.add_argument("--sweep_tol", action="store_true", help="是否同时扫描 tol（与 init_threshold 做二维网格）")
    p.add_argument("--max_iter", type=int, default=10, help="adaptive/coarse_fine 的 max_iter（不扫描时）")
    p.add_argument("--max_iter_list", type=str, default="",
                   help="若与 --sweep_max_iter 同用，则为逗号分隔的 max_iter 列表")
    p.add_argument("--sweep_max_iter", action="store_true", help="是否同时扫描 max_iter")
    p.add_argument("--min_candidates_ratio", type=float, default=1.1, help="coarse_fine 候选比例")

    p.add_argument("--out_json", type=str, default="", help="结果 JSON 路径")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 默认数据目录
    if not args.data_dir.strip():
        if args.dataset == "kitti":
            args.data_dir = "/workspace/VoxelNet/KITTI_Mini/training/velodyne"
        elif args.dataset == "nuscenes":
            args.data_dir = "/workspace/VoxelNet/Nuscenes_Mini/sweeps/LIDAR_TOP"
        else:
            args.data_dir = "/workspace/VoxelNet/Waymo_Mini/Waymo_mini/kitti_format/training/velodyne"

    init_list = [float(x.strip()) for x in args.init_threshold.split(",") if x.strip()]
    tol_list = [float(x.strip()) for x in args.tol_list.split(",") if x.strip()] if args.tol_list else [args.tol]
    if args.sweep_tol and not args.tol_list:
        tol_list = [0.02, 0.05, 0.1]
    max_iter_list = [int(x.strip()) for x in args.max_iter_list.split(",") if x.strip()] if args.max_iter_list else [args.max_iter]
    if args.sweep_max_iter and not args.max_iter_list:
        max_iter_list = [5, 10, 20]

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

    # 网格：先以 init_threshold 为主；若开启 sweep_tol / sweep_max_iter 则做多维网格
    results_grid = []
    best_iou_cf = -1.0
    best_cfg = None
    total_runs = len(init_list) * len(tol_list) * len(max_iter_list)
    run_idx = 0

    for init_th in init_list:
        for tol in tol_list:
            for max_iter in max_iter_list:
                run_idx += 1
                th = time.perf_counter()
                mean_cf, mean_ad = run_one_config(
                    loader,
                    device,
                    K,
                    k,
                    init_threshold=init_th,
                    tol=tol,
                    max_iter=max_iter,
                    min_candidates_ratio=args.min_candidates_ratio,
                    max_samples=args.max_samples,
                )
                elapsed = time.perf_counter() - th
                results_grid.append({
                    "init_threshold": init_th,
                    "tol": tol,
                    "max_iter": max_iter,
                    "iou_coarse_fine_vs_vanilla": mean_cf,
                    "iou_adaptive_vs_vanilla": mean_ad,
                    "time_sec": round(elapsed, 4),
                })
                if not np.isnan(mean_cf) and mean_cf > best_iou_cf:
                    best_iou_cf = mean_cf
                    best_cfg = {
                        "init_threshold": init_th,
                        "tol": tol,
                        "max_iter": max_iter,
                        "iou_coarse_fine_vs_vanilla": mean_cf,
                        "iou_adaptive_vs_vanilla": mean_ad,
                    }
                pbar_str = f"[{run_idx}/{total_runs}] init_th={init_th} tol={tol} max_iter={max_iter} -> IoU_cf={mean_cf:.4f} IoU_ad={mean_ad:.4f} ({elapsed:.1f}s)"
                print(pbar_str)

    # 按 IoU (coarse_fine) 排序，便于查看
    results_grid.sort(key=lambda x: (-(x["iou_coarse_fine_vs_vanilla"] if not np.isnan(x["iou_coarse_fine_vs_vanilla"]) else -1)))

    out = {
        "dataset": args.dataset,
        "data_dir": args.data_dir,
        "n_total_frames": n_total,
        "max_samples_used": args.max_samples,
        "num_points": args.num_points,
        "fps_steps": K,
        "k_neighbors": k,
        "min_candidates_ratio": args.min_candidates_ratio,
        "best_by_iou_coarse_fine": best_cfg,
        "grid": results_grid,
    }

    print("\n" + "=" * 60)
    print("最佳参数（按 Coarse-Fine vs Vanilla IoU）")
    print("=" * 60)
    if best_cfg:
        print(f"  init_threshold = {best_cfg['init_threshold']}")
        print(f"  tol            = {best_cfg['tol']}")
        print(f"  max_iter       = {best_cfg['max_iter']}")
        print(f"  IoU (Coarse-Fine vs Vanilla) = {best_cfg['iou_coarse_fine_vs_vanilla']:.4f}")
        print(f"  IoU (Adaptive vs Vanilla)   = {best_cfg['iou_adaptive_vs_vanilla']:.4f}")
    print("=" * 60)

    out_path = args.out_json.strip()
    if not out_path:
        out_path = os.path.join(SCRIPT_DIR, f"sweep_knn_{args.dataset}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n结果已写入: {out_path}")


if __name__ == "__main__":
    main()
