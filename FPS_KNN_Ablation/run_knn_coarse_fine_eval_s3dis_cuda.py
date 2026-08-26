#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import time
import glob

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
MODELNET40_DIR = os.path.join(REPO_ROOT, "PointKAN-elite", "classification_ModelNet40")
ELITE_DIR = os.path.join(REPO_ROOT, "PointKAN-elite")
for d in (SCRIPT_DIR, REPO_ROOT, MODELNET40_DIR, ELITE_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

from spec_kan_fps import fps_with_knn_vanilla, fps_with_knn_coarse_fine, knn_neighbor_set_iou  # noqa: E402
from knn_fps_cuda import fps_knn_coarse_fine_cuda  # noqa: E402


def find_npy_files(data_dir: str):
    return sorted(glob.glob(os.path.join(data_dir, "*.npy")))


def load_s3dis_npy(path: str) -> np.ndarray:
    data = np.load(path)
    if data.ndim == 1:
        data = data.reshape(-1, 7)
    return data[:, :3].astype(np.float32)


class S3DISPointCloudWrapper(Dataset):
    def __init__(self, data_dir: str, num_points: int = 50000, max_scenes: int = -1, seed: int = 42):
        self.num_points = int(num_points)
        self.npy_paths = find_npy_files(data_dir)
        if max_scenes > 0:
            self.npy_paths = self.npy_paths[:max_scenes]
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, idx):
        xyz = load_s3dis_npy(self.npy_paths[idx])
        n = xyz.shape[0]
        if n >= self.num_points:
            c = self.rng.choice(n, size=self.num_points, replace=False)
            pts = xyz[c]
        else:
            pad = self.rng.integers(0, n, size=self.num_points - n)
            pts = np.concatenate([xyz, xyz[pad]], axis=0)
        return torch.from_numpy(pts), 0


def collate_points(batch):
    pts = torch.stack([x[0] for x in batch], dim=0)
    labels = torch.tensor([x[1] for x in batch], dtype=torch.long)
    return pts, labels


def parse_args():
    ap = argparse.ArgumentParser("S3DIS: vanilla/python_cf/cuda_cf compare")
    ap.add_argument("--data_dir", type=str, default="/workspace/VoxelNet/pc_data/s3dis_trainval")
    ap.add_argument("--num_points", type=int, default=50000)
    ap.add_argument("--fps_steps", type=int, default=512)
    ap.add_argument("--k_neighbors", type=int, default=24)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_samples", type=int, default=10)
    ap.add_argument("--min_candidates_ratio", type=float, default=1.2)
    ap.add_argument("--init_threshold", type=float, default=0.1)
    ap.add_argument("--max_iter", type=int, default=10)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--mmax_factor", type=int, default=4)
    ap.add_argument("--strict_semantic", action="store_true", help="CUDA路径使用严格语义（对齐python blockwise）")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", type=str, default="")
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ds = S3DISPointCloudWrapper(args.data_dir, num_points=args.num_points, max_scenes=args.max_samples, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_points)

    K = min(args.fps_steps, args.num_points)
    k = args.k_neighbors
    is_cuda = device.type == "cuda"

    t_v, t_py, t_cu = [], [], []
    iou_py_v_list, iou_cu_v_list, iou_cu_py_list = [], [], []
    mse_py_v, mse_cu_v = [], []

    def sync():
        if is_cuda:
            torch.cuda.synchronize()

    for pts, _ in tqdm(loader, desc="S3DIS cuda compare", ncols=100):
        pts = pts.to(device=device, dtype=torch.float32)

        sync()
        t0 = time.perf_counter()
        _, _, knn_v, _, radii_v = fps_with_knn_vanilla(pts, K, k_neighbors=k, seed="first")
        sync()
        t_v.append(time.perf_counter() - t0)

        sync()
        t0 = time.perf_counter()
        _, _, knn_py, _, radii_py, _ = fps_with_knn_coarse_fine(
            pts, K, k_neighbors=k, init_threshold=args.init_threshold, max_iter=args.max_iter,
            min_candidates_ratio=args.min_candidates_ratio, seed="first"
        )
        sync()
        t_py.append(time.perf_counter() - t0)

        sync()
        t0 = time.perf_counter()
        idx_cu, knn_cu = fps_knn_coarse_fine_cuda(
            pts.contiguous().float(), K,
            k_neighbors=k, init_threshold=args.init_threshold, max_iter=args.max_iter,
            min_candidates_ratio=args.min_candidates_ratio, seed="first", exclude_self=False,
            block_size=args.block_size, mmax_factor=args.mmax_factor,
            strict_semantic=bool(args.strict_semantic),
        )
        _ = idx_cu
        sync()
        t_cu.append(time.perf_counter() - t0)

        iou_py_v = knn_neighbor_set_iou(knn_py, knn_v)
        iou_cu_v = knn_neighbor_set_iou(knn_cu, knn_v)
        iou_cu_py = knn_neighbor_set_iou(knn_cu, knn_py)
        iou_py_v_list.append(iou_py_v.cpu().numpy())
        iou_cu_v_list.append(iou_cu_v.cpu().numpy())
        iou_cu_py_list.append(iou_cu_py.cpu().numpy())

        ok_py = torch.isfinite(radii_v) & torch.isfinite(radii_py)
        if ok_py.any():
            mse_py_v.append(((radii_py - radii_v).float() ** 2).masked_select(ok_py).mean().item())

        # cuda path目前未返回radii，用python cf近似对比即可；这里保持空
        mse_cu_v.append(float("nan"))

    iou_py_v_all = np.concatenate(iou_py_v_list, axis=0).mean(axis=1)
    iou_cu_v_all = np.concatenate(iou_cu_v_list, axis=0).mean(axis=1)
    iou_cu_py_all = np.concatenate(iou_cu_py_list, axis=0).mean(axis=1)

    res = {
        "dataset": "S3DIS",
        "num_points": args.num_points,
        "fps_steps": K,
        "k_neighbors": k,
        "min_candidates_ratio": args.min_candidates_ratio,
        "init_threshold": args.init_threshold,
        "max_iter": args.max_iter,
        "block_size": args.block_size,
        "mmax_factor": args.mmax_factor,
        "strict_semantic": bool(args.strict_semantic),
        "time_vanilla_sec": {"total": float(np.sum(t_v)), "mean": float(np.mean(t_v)), "std": float(np.std(t_v))},
        "time_python_cf_sec": {"total": float(np.sum(t_py)), "mean": float(np.mean(t_py)), "std": float(np.std(t_py))},
        "time_cuda_cf_sec": {"total": float(np.sum(t_cu)), "mean": float(np.mean(t_cu)), "std": float(np.std(t_cu))},
        "speedup_python_cf_vs_vanilla": float(np.sum(t_v) / np.sum(t_py)),
        "speedup_cuda_cf_vs_vanilla": float(np.sum(t_v) / np.sum(t_cu)),
        "speedup_cuda_cf_vs_python_cf": float(np.sum(t_py) / np.sum(t_cu)),
        "iou_python_cf_vs_vanilla": {
            "mean": float(np.mean(iou_py_v_all)), "std": float(np.std(iou_py_v_all)),
            "min": float(np.min(iou_py_v_all)), "max": float(np.max(iou_py_v_all)),
        },
        "iou_cuda_cf_vs_vanilla": {
            "mean": float(np.mean(iou_cu_v_all)), "std": float(np.std(iou_cu_v_all)),
            "min": float(np.min(iou_cu_v_all)), "max": float(np.max(iou_cu_v_all)),
        },
        "iou_cuda_cf_vs_python_cf": {
            "mean": float(np.mean(iou_cu_py_all)), "std": float(np.std(iou_cu_py_all)),
            "min": float(np.min(iou_cu_py_all)), "max": float(np.max(iou_cu_py_all)),
        },
        "radii_mse_python_cf_vs_vanilla": float(np.nanmean(mse_py_v)) if len(mse_py_v) else float("nan"),
    }

    print("=== S3DIS CUDA compare ===")
    print(f"Vanilla mean:    {res['time_vanilla_sec']['mean']:.4f}s")
    print(f"Python CF mean:  {res['time_python_cf_sec']['mean']:.4f}s")
    print(f"CUDA CF mean:    {res['time_cuda_cf_sec']['mean']:.4f}s")
    print(f"speedup(py/v):   {res['speedup_python_cf_vs_vanilla']:.4f}x")
    print(f"speedup(cuda/v): {res['speedup_cuda_cf_vs_vanilla']:.4f}x")
    print(f"speedup(cuda/py):{res['speedup_cuda_cf_vs_python_cf']:.4f}x")
    print(f"IoU py-v mean:   {res['iou_python_cf_vs_vanilla']['mean']:.4f}")
    print(f"IoU cuda-v mean: {res['iou_cuda_cf_vs_vanilla']['mean']:.4f}")
    print(f"IoU cuda-py mean:{res['iou_cuda_cf_vs_python_cf']['mean']:.4f}")

    out = args.out_json.strip() or os.path.join(SCRIPT_DIR, "knn_coarse_fine_iou_results_s3dis_cuda.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"结果已写入: {out}")


if __name__ == "__main__":
    main()

