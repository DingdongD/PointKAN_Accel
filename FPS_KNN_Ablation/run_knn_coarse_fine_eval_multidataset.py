#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多数据集 KNN 粗筛+细筛评测（IoU / radii MSE / 时间）

支持:
  - modelnet40
  - s3dis
  - partseg (ShapeNetPart)
  - scannet (当前按 ScanObjectNN H5 路径兼容)
  - scanobjectnn
"""

import os
import sys
import json
import glob
import time
import argparse
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
MODELNET40_DIR = os.path.join(REPO_ROOT, "PointKAN-elite", "classification_ModelNet40")
for d in (SCRIPT_DIR, REPO_ROOT, MODELNET40_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

from spec_kan_fps import (  # noqa: E402
    ModelNet40Wrapper,
    collate_points,
    fps_with_knn_adaptive,
    fps_with_knn_coarse_fine,
    fps_with_knn_vanilla,
    knn_neighbor_set_iou,
)


def load_s3dis_npy(path: str) -> np.ndarray:
    data = np.load(path)
    if data.ndim == 1:
        data = data.reshape(-1, 7)
    return data[:, :3].astype(np.float32)


def _sample_or_pad_xyz(xyz: np.ndarray, num_points: int, rng: np.random.Generator) -> np.ndarray:
    n = xyz.shape[0]
    if n == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    if n >= num_points:
        idx = rng.choice(n, size=num_points, replace=False)
        return xyz[idx]
    pad_idx = rng.integers(0, n, size=(num_points - n))
    return np.concatenate([xyz, xyz[pad_idx]], axis=0)


class S3DISPointCloudWrapper(Dataset):
    def __init__(self, data_dir: str, num_points: int = 50000, max_samples: int = -1, seed: int = 42):
        self.paths = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        if max_samples > 0:
            self.paths = self.paths[:max_samples]
        self.num_points = num_points
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        xyz = load_s3dis_npy(self.paths[idx])
        pts = _sample_or_pad_xyz(xyz, self.num_points, self.rng)
        return torch.from_numpy(pts), 0


class ScanObjectNNWrapper(Dataset):
    def __init__(self, h5_path: str, num_points: int = 1024):
        import h5py

        self.num_points = num_points
        with h5py.File(h5_path, "r") as f:
            self.data = f["data"][:].astype(np.float32)
            self.label = f["label"][:].astype(np.int64)

    def __len__(self):
        return int(self.data.shape[0])

    def __getitem__(self, idx):
        pts = self.data[idx][: self.num_points, :3]
        return torch.from_numpy(pts.copy()), int(self.label[idx])


class ShapeNetPartXYZDataset(Dataset):
    def __init__(self, root: str, npoints: int = 2048, split: str = "test", normalize: bool = False):
        self.npoints = npoints
        self.root = root
        self.normalize = normalize
        self.split = split

        catfile = os.path.join(self.root, "synsetoffset2category.txt")
        self.cat = {}
        with open(catfile, "r", encoding="utf-8") as f:
            for line in f:
                ls = line.strip().split()
                if len(ls) >= 2:
                    self.cat[ls[0]] = ls[1]

        tdir = os.path.join(self.root, "train_test_split")
        with open(os.path.join(tdir, "shuffled_test_file_list.json"), "r", encoding="utf-8") as f:
            test_ids = {str(d.split("/")[2]) for d in json.load(f)}

        self.meta = {item: [] for item in self.cat}
        for item in self.cat:
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            fns = [fn for fn in fns if fn[:-4] in test_ids]
            for fn in fns:
                token = os.path.splitext(os.path.basename(fn))[0]
                self.meta[item].append(os.path.join(dir_point, token + ".txt"))

        self.datapath: List[Tuple[str, str]] = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, idx):
        _, path = self.datapath[idx]
        data = np.loadtxt(path).astype(np.float32)
        pts = data[:, :3]
        if self.normalize:
            centroid = np.mean(pts, axis=0)
            pts = pts - centroid
            m = np.max(np.sqrt(np.sum(pts ** 2, axis=1)))
            pts = pts / (m + 1e-8)
        rng = np.random.default_rng(42 + idx)
        sel = rng.choice(pts.shape[0], self.npoints, replace=True)
        return torch.from_numpy(pts[sel].astype(np.float32)), 0


def parse_args():
    p = argparse.ArgumentParser(description="多数据集 KNN 粗筛+细筛 IoU 分析")
    p.add_argument("--dataset", type=str, required=True,
                   choices=["modelnet40", "s3dis", "partseg", "shapenet_part", "scannet", "scanobjectnn"])
    p.add_argument("--data_root", type=str, default="", help="数据根目录（按 dataset 解释）")
    p.add_argument("--num_points", type=int, default=2048)
    p.add_argument("--fps_steps", type=int, default=512)
    p.add_argument("--k_neighbors", type=int, default=24)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--min_candidates_ratio", type=float, default=1.2)
    p.add_argument("--tol", type=float, default=0.05)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_json", type=str, default="")
    p.add_argument("--partseg_normalize", action="store_true")
    return p.parse_args()


def resolve_defaults(args):
    if not args.data_root:
        if args.dataset == "modelnet40":
            args.data_root = "/workspace/VoxelNet/pc_data"
        elif args.dataset in ("s3dis",):
            args.data_root = "/workspace/VoxelNet/pc_data/s3dis_trainval"
        elif args.dataset in ("partseg", "shapenet_part"):
            args.data_root = "/workspace/VoxelNet/pc_data/shapenetcore_partanno_segmentation_benchmark_v0_normal"
        elif args.dataset in ("scannet", "scanobjectnn"):
            args.data_root = "/workspace/VoxelNet/scannet_kan/main_split"
    return args


def build_dataset(args):
    ds = args.dataset
    if ds == "modelnet40":
        return ModelNet40Wrapper(partition="test", num_points=args.num_points, root=args.data_root), "ModelNet40"
    if ds == "s3dis":
        return S3DISPointCloudWrapper(args.data_root, num_points=args.num_points, max_samples=-1, seed=args.seed), "S3DIS"
    if ds in ("partseg", "shapenet_part"):
        return ShapeNetPartXYZDataset(args.data_root, npoints=args.num_points, split="test",
                                      normalize=args.partseg_normalize), "ShapeNetPart"
    if ds in ("scannet", "scanobjectnn"):
        h5_path = os.path.join(args.data_root, "test_objectdataset_augmentedrot_scale75.h5")
        if not os.path.isfile(h5_path):
            raise FileNotFoundError(f"未找到 ScanObjectNN 测试集 h5: {h5_path}")
        return ScanObjectNNWrapper(h5_path, num_points=args.num_points), "ScanObjectNN"
    raise ValueError(f"未知 dataset: {ds}")


@torch.no_grad()
def run_iou_analysis(args):
    device = torch.device(args.device)
    ds, dataset_name = build_dataset(args)
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

    iou_cf_v_list, iou_ad_v_list = [], []
    radii_mse_cf_v_list, radii_mse_ad_v_list = [], []
    time_vanilla_list, time_adaptive_list, time_coarse_fine_list = [], [], []
    n_pts_processed = 0
    is_cuda = device.type == "cuda"

    def _sync():
        if is_cuda:
            torch.cuda.synchronize()

    pbar = tqdm(loader, desc=f"KNN IoU ({dataset_name})", ncols=100)
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
        _, _, knn_vanilla, _, radii_vanilla = fps_with_knn_vanilla(pts, K, k_neighbors=k, seed="first")
        _sync()
        time_vanilla_list.append(time.perf_counter() - t0)

        _sync()
        t0 = time.perf_counter()
        _, _, knn_adaptive, _, radii_adaptive, _, _ = fps_with_knn_adaptive(
            pts, K, k_neighbors=k, init_threshold=2.4, max_iter=10, tol=args.tol, seed="first"
        )
        _sync()
        time_adaptive_list.append(time.perf_counter() - t0)

        _sync()
        t0 = time.perf_counter()
        _, _, knn_cf, _, radii_cf, _ = fps_with_knn_coarse_fine(
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
            mse_cf_v, mse_ad_v = float("nan"), float("nan")
        radii_mse_cf_v_list.append(mse_cf_v)
        radii_mse_ad_v_list.append(mse_ad_v)

    iou_cf_v_all = np.concatenate(iou_cf_v_list, axis=0)
    iou_ad_v_all = np.concatenate(iou_ad_v_list, axis=0)
    iou_cf_v_per_sample = iou_cf_v_all.mean(axis=1)
    iou_ad_v_per_sample = iou_ad_v_all.mean(axis=1)

    return {
        "dataset": dataset_name,
        "dataset_arg": args.dataset,
        "device": args.device,
        "data_root": args.data_root,
        "n_samples": int(iou_cf_v_per_sample.size),
        "n_total_samples": n_total,
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
        "radii_mse_coarse_fine_vs_vanilla": float(np.nanmean(radii_mse_cf_v_list)),
        "radii_mse_adaptive_vs_vanilla": float(np.nanmean(radii_mse_ad_v_list)),
        "inference_time_vanilla_sec": float(np.sum(time_vanilla_list)),
        "inference_time_adaptive_sec": float(np.sum(time_adaptive_list)),
        "inference_time_coarse_fine_sec": float(np.sum(time_coarse_fine_list)),
        "speedup_adaptive_vs_vanilla": float(np.sum(time_vanilla_list) / max(np.sum(time_adaptive_list), 1e-12)),
        "speedup_coarse_fine_vs_vanilla": float(np.sum(time_vanilla_list) / max(np.sum(time_coarse_fine_list), 1e-12)),
    }


def main():
    args = resolve_defaults(parse_args())
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    t0 = time.perf_counter()
    res = run_iou_analysis(args)
    t1 = time.perf_counter()
    res["elapsed_sec"] = float(t1 - t0)

    print("=" * 60)
    print(f"dataset={res['dataset']} ({res['dataset_arg']})  n={res['n_samples']}/{res['n_total_samples']}")
    print(f"K={res['fps_steps']} k={res['k_neighbors']} device={res['device']}")
    print(f"IoU(cf vs v): mean={res['iou_coarse_fine_vs_vanilla']['mean']:.4f} "
          f"std={res['iou_coarse_fine_vs_vanilla']['std']:.4f}")
    print(f"IoU(ad vs v): mean={res['iou_adaptive_vs_vanilla']['mean']:.4f} "
          f"std={res['iou_adaptive_vs_vanilla']['std']:.4f}")
    print(f"Radii MSE cf/ad vs v: {res['radii_mse_coarse_fine_vs_vanilla']:.3e} / "
          f"{res['radii_mse_adaptive_vs_vanilla']:.3e}")
    print(f"Timing (sec): vanilla={res['inference_time_vanilla_sec']:.2f} "
          f"adaptive={res['inference_time_adaptive_sec']:.2f} coarse_fine={res['inference_time_coarse_fine_sec']:.2f}")
    print(f"Speedup vs vanilla: adaptive={res['speedup_adaptive_vs_vanilla']:.2f}x "
          f"coarse_fine={res['speedup_coarse_fine_vs_vanilla']:.2f}x")
    print("=" * 60)

    out_path = args.out_json.strip() or os.path.join(SCRIPT_DIR, f"knn_coarse_fine_iou_results_{args.dataset}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"结果已写入: {out_path}")


if __name__ == "__main__":
    main()
