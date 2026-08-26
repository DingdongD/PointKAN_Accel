#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
固定 FPS 中心下的 KNN 量化一致性（与 FPS 分叉解耦）

在全精度点云 ``pts`` 上运行 ``fps_with_knn_vanilla(..., seed='first')``，得到每步 FPS 中心下标 ``idx_v``，
以及该路径下的参考邻居 ``knn_v``（与主基准一致）。

对同一批 ``idx_v``，在**反量化点云** ``pts_q`` 上用**相同中心下标**、相同欧氏距离做 top-k，
得到 ``knn_q``，再算 ``knn_neighbor_set_iou`` / ``knn_recall_vs_a``。

这样 KNN recall/IoU 的差异**不再包含**「量化导致 FPS 选错中心」的分支，只反映**几何量化对距离排序**
（及与 vanilla 相同的「中心点含于 kNN」等行为）。

用法（本目录）:
  python run_knn_fixed_fps_centers_quant_recall.py --device cuda --fast
  python run_knn_fixed_fps_centers_quant_recall.py --datasets modelnet40,kitti --bits 8,12,16
  python plot_fps_knn_quant_curves.py --json knn_fixed_centers_quant_recall_results.json \\
      --metrics knn_recall_mean,knn_iou_mean,coord_quant_mse_mean --fname knn_fixed_centers_quant
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)

import run_fps_knn_quant_recall as bench  # noqa: E402
from spec_kan_fps import (  # noqa: E402
    collate_points,
    fps_with_knn_vanilla,
    knn_neighbor_set_iou,
)


@torch.no_grad()
def knn_indices_at_centers(X: torch.Tensor, center_idx: torch.Tensor, k_neighbors: int) -> torch.Tensor:
    """
    X: (B, N, 3), center_idx: (B, K) long，每步一个中心点下标。
    返回 (B, K, k) 的 k 近邻下标（在最后一维 N 上 topk 最小距离；含自身，与 fps_with_knn_vanilla 一致）。
    """
    B, N, _ = X.shape
    device, dtype = X.device, X.dtype
    K = center_idx.shape[1]
    k = min(int(k_neighbors), N)
    b_idx = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1).expand(B, K)
    centers = X[b_idx, center_idx]  # (B, K, 3)
    diff = X.unsqueeze(1) - centers.unsqueeze(2)  # (B, K, N, 3)
    dist = diff.norm(dim=-1)
    _, knn_idx = torch.topk(dist, k=k, dim=-1, largest=False)
    return knn_idx


@torch.no_grad()
def eval_fixed_centers_on_loader(
    loader: DataLoader,
    device: torch.device,
    K: int,
    k_neighbors: int,
    bits: int,
    max_batches: int,
    quant_cfg: Dict[str, Any],
    verify_knn_v: bool = False,
) -> Dict[str, float]:
    """
    参考邻居 ``knn_v`` 直接取自 ``fps_with_knn_vanilla(pts)``（与主基准一致）。
    ``knn_q`` 在 ``pts_q`` 上对同一 ``idx_v`` 用 ``knn_indices_at_centers`` 计算。
    """
    knn_iou_list: List[np.ndarray] = []
    knn_recall_list: List[np.ndarray] = []
    coord_mse_batch: List[float] = []
    n_batches = 0
    max_mismatch = 0

    qscheme = str(quant_cfg.get("scheme", "per_cloud_minmax"))
    for pts, _ in tqdm(loader, desc=f"fixed-centers {qscheme} b={bits}", leave=False):
        if max_batches > 0 and n_batches >= max_batches:
            break
        pts = pts.to(device, dtype=torch.float32)
        n_batches += 1

        pts_q = bench.apply_quant_dequant(pts, bits, quant_cfg)
        err2 = (pts - pts_q).pow(2)
        mse_batch = err2.mean(dim=(1, 2))
        coord_mse_batch.extend(mse_batch.detach().cpu().tolist())

        idx_v, _, knn_v, _, _ = fps_with_knn_vanilla(pts, K, k_neighbors=k_neighbors, seed="first")
        knn_q = knn_indices_at_centers(pts_q, idx_v, k_neighbors)

        if verify_knn_v:
            knn_v_re = knn_indices_at_centers(pts, idx_v, k_neighbors)
            max_mismatch = max(max_mismatch, int((knn_v != knn_v_re).sum().item()))

        kiou = knn_neighbor_set_iou(knn_v, knn_q)
        krec = bench.knn_recall_vs_a(knn_v, knn_q)
        knn_iou_list.append(kiou.cpu().numpy())
        knn_recall_list.append(krec.mean(dim=-1).cpu().numpy())

    if not knn_iou_list:
        return {
            "knn_iou_mean": float("nan"),
            "knn_iou_std": float("nan"),
            "knn_recall_mean": float("nan"),
            "knn_recall_std": float("nan"),
            "coord_quant_mse_mean": float("nan"),
            "coord_quant_mse_std": float("nan"),
            "coord_quant_rmse_mean": float("nan"),
            "knn_v_vs_recompute_max_mismatch": 0,
            "n_batches": 0,
        }

    knn_iou_all = np.concatenate([x.reshape(-1) for x in knn_iou_list])
    knn_rec_all = np.concatenate([x.reshape(-1) for x in knn_recall_list])
    mse_arr = np.array(coord_mse_batch, dtype=np.float64)
    mse_mean = float(np.mean(mse_arr))
    return {
        "knn_iou_mean": float(np.mean(knn_iou_all)),
        "knn_iou_std": float(np.std(knn_iou_all)),
        "knn_recall_mean": float(np.mean(knn_rec_all)),
        "knn_recall_std": float(np.std(knn_rec_all)),
        "coord_quant_mse_mean": mse_mean,
        "coord_quant_mse_std": float(np.std(mse_arr)),
        "coord_quant_rmse_mean": float(np.sqrt(max(mse_mean, 0.0))),
        "knn_v_vs_recompute_max_mismatch": int(max_mismatch),
        "n_batches": int(n_batches),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="固定 vanilla FPS 中心下标，仅在 pts_q 上重算 KNN，测量化对 KNN recall/IoU 的影响"
    )
    p.add_argument("--config", type=str, default=os.path.join(BENCH_DIR, "datasets.yaml"))
    p.add_argument("--datasets", type=str, default="", help="逗号分隔；空=yaml 全部")
    p.add_argument(
        "--bits",
        type=str,
        default="1-16",
        help="解析后与各集 quant.bits_list+effective 得到的档位**取交集**；默认 1-16 即不额外缩小 yaml。仅跑 8,16 可写 --bits 8,16",
    )
    p.add_argument("--fps_steps", type=int, default=0)
    p.add_argument("--k_neighbors", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--fast_max_batches", type=int, default=None, metavar="N")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument(
        "--verify_knn_v",
        action="store_true",
        help="调试用：核对 fps_with_knn_vanilla 的 knn_v 与在 pts 上对 idx_v 显式 topk 是否一致",
    )
    p.add_argument(
        "--out_json",
        type=str,
        default="",
        help="默认 knn_fixed_centers_quant_recall_results.json",
    )
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    g = cfg.get("defaults") or {}
    bits_list = bench.parse_bits_list(args.bits)
    ds_cfg_all: Dict[str, Any] = cfg.get("datasets") or {}
    keys = [k.strip() for k in args.datasets.split(",") if k.strip()] if args.datasets.strip() else list(ds_cfg_all.keys())

    bits_allow = set(bits_list)
    print(f"[info] 固定 FPS 中心重算 KNN；命令行 --bits 允许集合: {sorted(bits_allow)}（将与 yaml 各集可用档位取交集）")
    print(f"[info] 数据集: {', '.join(keys)}")

    seed = int(args.seed) if args.seed >= 0 else int(g.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    max_batches_default = int(g.get("max_samples", 0))
    fast_cap_cfg = int(g.get("fast_max_batches", 20))
    if args.fast:
        if args.max_samples >= 0:
            max_batches = args.max_samples
            sampling_mode = "fast_explicit_max_samples"
        else:
            max_batches = int(args.fast_max_batches) if args.fast_max_batches is not None else fast_cap_cfg
            sampling_mode = "fast_capped"
        print(f"[info] 快速模式: 每数据集最多 {max_batches} batch")
    elif args.max_samples >= 0:
        max_batches = args.max_samples
        sampling_mode = "explicit_max_samples"
    else:
        max_batches = max_batches_default
        sampling_mode = "yaml_or_full"

    device = torch.device(args.device)
    K_def = int(g.get("fps_steps", 512))
    k_def = int(g.get("k_neighbors", 24))

    out: Dict[str, Any] = {
        "config_path": os.path.abspath(args.config),
        "bits_per_axis_cli_default": bits_list,
        "datasets": {},
        "experiment": {
            "name": "knn_fixed_vanilla_fps_centers",
            "note_zh": (
                "在全精度 pts 上 fps_with_knn_vanilla(seed=first) 得到每步中心下标 idx_v 与参考 knn_v；"
                "对同一 idx_v 在 pts_q 上用欧氏 topk 得到 knn_q。指标不含「量化导致 FPS 换中心」效应，"
                "仅反映量化几何下 k 近邻集合相对全精度的 recall/IoU。"
            ),
            "fps_seed": "first",
            "reference_knn": "fps_with_knn_vanilla(pts) 返回的 knn_idx",
            "quantized_knn": "knn_indices_at_centers(pts_q, idx_v)",
            "aggregation_note_zh": "knn_iou 对 (B,K) 展平后 mean/std；knn_recall 先对 K 步 mean 再对 batch 样本聚合，与 run_fps_knn_quant_recall 一致。",
            "bits_selection_zh": "各集先按 yaml quant.bits_list（若无则用 1–16）与 scheme 得到候选档，再与命令行 --bits 解析集合取交集；默认 --bits 1-16 等价于只受 yaml 约束。",
            "fast_mode": bool(args.fast),
            "sampling_mode": sampling_mode,
            "max_batches_cap_per_dataset": max_batches,
        },
        "dataset_keys_requested": keys,
    }

    for key in keys:
        if key not in ds_cfg_all:
            print(f"[skip] 未知数据集键: {key}")
            continue
        entry = dict(ds_cfg_all[key])
        quant_cfg = bench.resolve_quant_cfg(g, entry)
        try:
            quant_cfg = bench.finalize_quant_cfg(quant_cfg, g, key, BENCH_DIR)
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            print(f"[skip] {key}: 量化边界失败: {e}")
            continue

        K = int(args.fps_steps) if args.fps_steps > 0 else int(entry.get("fps_steps", K_def))
        k_neighbors = int(args.k_neighbors) if args.k_neighbors > 0 else int(entry.get("k_neighbors", k_def))
        bs = int(args.batch_size) if args.batch_size > 0 else int(entry.get("batch_size", g.get("batch_size", 4)))

        try:
            ds, dname = bench.build_torch_dataset(key, entry, g, seed, max_batches if max_batches > 0 else -1)
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

        ds_bits = bench.dataset_bits_list(list(range(1, 17)), quant_cfg)
        eff_bits = bench.effective_bits_list(ds_bits, quant_cfg)
        eff_bits = sorted(b for b in eff_bits if b in bits_allow)
        if not eff_bits:
            print(f"[skip] {key}: --bits 与 quant 组合后无可用档位（检查 yaml quant.bits_list 与 --bits 交集）")
            continue
        print(f"[info] {key} 本集实际评测 bits: {eff_bits}")

        per_bits: Dict[str, Any] = {}
        for b in eff_bits:
            per_bits[str(b)] = eval_fixed_centers_on_loader(
                loader,
                device,
                K,
                k_neighbors,
                b,
                max_batches,
                quant_cfg,
                verify_knn_v=bool(args.verify_knn_v),
            )

        out["datasets"][key] = {
            "display_name": dname,
            "type": entry.get("type"),
            "data_root": entry.get("data_root"),
            "num_points": entry.get("num_points", g.get("num_points")),
            "fps_steps": K,
            "k_neighbors": k_neighbors,
            "batch_size": bs,
            "max_batches": max_batches,
            "quant": bench.quant_cfg_for_json(quant_cfg),
            "bits_evaluated": eff_bits,
            "metrics_fixed_centers_vs_vanilla_knn": per_bits,
        }
        print(f"[ok] {key} ({dname})  K={K} k={k_neighbors} cap={max_batches}")

    out_path = args.out_json.strip() or os.path.join(BENCH_DIR, "knn_fixed_centers_quant_recall_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"已写入: {out_path}")


if __name__ == "__main__":
    main()
