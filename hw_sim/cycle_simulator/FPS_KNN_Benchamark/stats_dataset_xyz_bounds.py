#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扫描 datasets.yaml 中的数据集，统计送入 FPS/KNN 基准前的点云 xyz **全局** min/max
（与 run_fps_knn_quant_recall 使用同一套 Dataset / collate，含 ROI、num_points 等）。

输出 JSON，便于为 global_per_axis 等方案填写 x_range/y_range/z_range。

用法（在本目录）:
  python stats_dataset_xyz_bounds.py
  python stats_dataset_xyz_bounds.py --datasets s3dis,scanobjectnn --max_batches 200
  python stats_dataset_xyz_bounds.py --margin 0.02 --out_json dataset_xyz_bounds.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)

import run_fps_knn_quant_recall as bench  # noqa: E402
from spec_kan_fps import collate_points  # noqa: E402


@torch.no_grad()
def accumulate_xyz_bounds(
    loader: DataLoader,
    max_batches: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, int, int, np.ndarray, np.ndarray]:
    """
    返回:
      xyz_min (3,), xyz_max (3,), n_batches_seen, n_points_total,
      mean_xyz (3,) 全体点平均, std_xyz (3,) 全体点总体 std（加权）
    """
    inf = float("inf")
    gmin = torch.tensor([inf, inf, inf], device=device, dtype=torch.float64)
    gmax = torch.tensor([-inf, -inf, -inf], device=device, dtype=torch.float64)
    sum_xyz = torch.zeros(3, device=device, dtype=torch.float64)
    sum_sq = torch.zeros(3, device=device, dtype=torch.float64)
    n_points = 0
    n_batches = 0

    for pts, _ in tqdm(loader, desc="xyz bounds", leave=False):
        if max_batches > 0 and n_batches >= max_batches:
            break
        pts = pts.to(device, dtype=torch.float64)
        flat = pts.reshape(-1, 3)
        pmin = flat.min(dim=0).values
        pmax = flat.max(dim=0).values
        gmin = torch.minimum(gmin, pmin)
        gmax = torch.maximum(gmax, pmax)
        sum_xyz += flat.sum(dim=0)
        sum_sq += (flat ** 2).sum(dim=0)
        n_points += flat.shape[0]
        n_batches += 1

    if n_points == 0:
        z = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        return z, z, 0, 0, z, z

    mean = (sum_xyz / float(n_points)).cpu().numpy()
    var = sum_sq / float(n_points) - (sum_xyz / float(n_points)) ** 2
    std = torch.sqrt(torch.clamp(var, min=0.0)).cpu().numpy()
    return (
        gmin.cpu().numpy(),
        gmax.cpu().numpy(),
        n_batches,
        n_points,
        mean,
        std,
    )


def suggest_ranges(
    xyz_min: np.ndarray,
    xyz_max: np.ndarray,
    margin: float,
) -> Dict[str, List[float]]:
    """按各轴 span * margin 外扩，供 global_per_axis 参考。"""
    span = np.maximum(xyz_max - xyz_min, 1e-8)
    pad = span * float(margin)
    lo = (xyz_min - pad).tolist()
    hi = (xyz_max + pad).tolist()
    return {
        "x_range": [float(lo[0]), float(hi[0])],
        "y_range": [float(lo[1]), float(hi[1])],
        "z_range": [float(lo[2]), float(hi[2])],
    }


def parse_args():
    p = argparse.ArgumentParser(description="统计各数据集点云 xyz 全局上下界")
    p.add_argument("--config", type=str, default=os.path.join(BENCH_DIR, "datasets.yaml"))
    p.add_argument("--datasets", type=str, default="", help="逗号分隔；空=yaml 中全部")
    p.add_argument(
        "--max_batches",
        type=int,
        default=0,
        help="每数据集最多 batch 数，0 表示扫完整 DataLoader",
    )
    p.add_argument(
        "--margin",
        type=float,
        default=0.01,
        help="建议 global_per_axis 时各轴按 (max-min)*margin 外扩比例",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--out_json",
        type=str,
        default="",
        help="默认写入本目录 dataset_xyz_bounds.json",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    g = cfg.get("defaults") or {}
    ds_all: Dict[str, Any] = cfg.get("datasets") or {}
    keys = [k.strip() for k in args.datasets.split(",") if k.strip()] if args.datasets.strip() else list(ds_all.keys())

    device = torch.device(args.device)
    out: Dict[str, Any] = {
        "config_path": os.path.abspath(args.config),
        "max_batches_per_dataset": args.max_batches,
        "margin_for_suggestion": args.margin,
        "datasets": {},
    }

    for key in keys:
        if key not in ds_all:
            print(f"[skip] 未知数据集键: {key}")
            continue
        entry = dict(ds_all[key])
        bs = int(entry.get("batch_size", g.get("batch_size", 4)))
        mb = args.max_batches

        try:
            ds, dname = bench.build_torch_dataset(key, entry, g, args.seed, mb if mb > 0 else -1)
        except FileNotFoundError as e:
            print(f"[skip] {key}: {e}")
            continue
        except Exception as e:
            print(f"[skip] {key}: {e}")
            continue

        loader = DataLoader(
            ds,
            batch_size=bs,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_points,
        )

        xyz_min, xyz_max, nb, npt, mean_xyz, std_xyz = accumulate_xyz_bounds(loader, mb, device)
        if npt == 0:
            print(f"[skip] {key}: 无点（空集）")
            continue

        sug = suggest_ranges(xyz_min, xyz_max, args.margin)
        out["datasets"][key] = {
            "display_name": dname,
            "type": entry.get("type"),
            "data_root": entry.get("data_root"),
            "num_points_config": entry.get("num_points", g.get("num_points")),
            "use_roi": entry.get("use_roi"),
            "xyz_min": xyz_min.tolist(),
            "xyz_max": xyz_max.tolist(),
            "xyz_mean_all_points": mean_xyz.tolist(),
            "xyz_std_all_points": std_xyz.tolist(),
            "n_batches_seen": nb,
            "n_points_total": npt,
            "suggested_global_per_axis_yaml": sug,
        }
        print(
            f"[ok] {key}  min={np.array2string(xyz_min, precision=3)}  "
            f"max={np.array2string(xyz_max, precision=3)}  points={npt} batches={nb}"
        )

    out_path = args.out_json.strip() or os.path.join(BENCH_DIR, "dataset_xyz_bounds.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"已写入: {out_path}")


if __name__ == "__main__":
    main()
