#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
随机（或指定下标）取数据集中 **1 个样本**，对比全精度与量化后
`fps_with_knn_vanilla(..., seed='first')` 的 FPS 下标序列，便于核查低 IoU 是否合理。

用法（本目录）:
  python debug_fps_idx_single_sample.py --dataset modelnet40 --bits 8
  python debug_fps_idx_single_sample.py --dataset kitti --index 0 --bits 10
  python debug_fps_idx_single_sample.py --dataset s3dis --seed 123 --bits 12
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch
import yaml

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)

import run_fps_knn_quant_recall as bench  # noqa: E402
from spec_kan_fps import fps_with_knn_vanilla  # noqa: E402


def fps_set_metrics(idx_v: np.ndarray, idx_q: np.ndarray) -> Dict[str, float]:
    """idx_* 长度 K 的一维下标数组（每步 FPS 选中的点下标，理论上常互异；若输入退化可能大量重复）。"""
    sv: Set[int] = set(idx_v.tolist())
    sq: Set[int] = set(idx_q.tolist())
    inter = len(sv & sq)
    union = len(sv | sq)
    k = len(idx_v)
    len_sv = max(len(sv), 1)
    return {
        "K_fps_steps": float(k),
        "unique_in_idx_v": float(len(sv)),
        "unique_in_idx_q": float(len(sq)),
        "|intersection|": float(inter),
        "|union|": float(union),
        "fps_set_iou": float(inter) / float(max(union, 1)),
        # 与基准 JSON 一致：分母为 K（步数），若 |Sv|<<K 会显小
        "fps_idx_recall_json_style_inter_div_K": float(inter) / float(max(k, 1)),
        # 集合语义更直观：相对 vanilla 被选过的不同下标
        "set_recall_inter_div_|Sv|": float(inter) / float(len_sv),
        "ordered_match_rate": float(np.mean(idx_v == idx_q)),
    }


def print_step_table(
    idx_v: np.ndarray,
    idx_q: np.ndarray,
    max_rows: int,
) -> None:
    K = idx_v.shape[0]
    print(f"\n--- 逐步 FPS 下标（前 {min(max_rows, K)} / {K} 步）---")
    print(f"{'t':>5} {'idx_v':>8} {'idx_q':>8} {'同号?':>6}")
    for t in range(min(max_rows, K)):
        same = "Y" if idx_v[t] == idx_q[t] else ""
        print(f"{t:5d} {idx_v[t]:8d} {idx_q[t]:8d} {same:>6}")


def parse_args():
    p = argparse.ArgumentParser(description="单样本 FPS 下标 vanilla vs 量化 对比")
    p.add_argument("--config", type=str, default=os.path.join(BENCH_DIR, "datasets.yaml"))
    p.add_argument("--dataset", type=str, required=True, help="yaml 中 datasets 的 key，如 modelnet40")
    p.add_argument("--index", type=int, default=-1, help="样本下标，-1 表示随机")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bits", type=int, default=8)
    p.add_argument("--fps_steps", type=int, default=0, help="0=从 yaml 读")
    p.add_argument("--k_neighbors", type=int, default=0, help="0=从 yaml 读")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_step_rows", type=int, default=40, help="逐步打印最多行数")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    g = cfg.get("defaults") or {}
    ds_all: Dict[str, Any] = cfg.get("datasets") or {}
    if args.dataset not in ds_all:
        raise SystemExit(f"未知数据集 key: {args.dataset}")

    entry = dict(ds_all[args.dataset])
    quant_cfg = bench.resolve_quant_cfg(g, entry)
    quant_cfg = bench.finalize_quant_cfg(quant_cfg, g, args.dataset, BENCH_DIR)

    K = int(args.fps_steps) if args.fps_steps > 0 else int(entry.get("fps_steps", g.get("fps_steps", 512)))
    k_neighbors = int(args.k_neighbors) if args.k_neighbors > 0 else int(entry.get("k_neighbors", g.get("k_neighbors", 24)))

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    ds, dname = bench.build_torch_dataset(args.dataset, entry, g, args.seed, max_samples=-1)
    n = len(ds)
    if n == 0:
        raise SystemExit("数据集长度为 0")
    idx = int(rng.integers(0, n)) if args.index < 0 else int(args.index)
    if idx < 0 or idx >= n:
        raise SystemExit(f"index 越界: {idx} (len={n})")

    pts, _label = ds[idx]
    pts = pts.unsqueeze(0).to(args.device, dtype=torch.float32)

    pts_q = bench.apply_quant_dequant(pts, args.bits, quant_cfg)

    with torch.no_grad():
        idx_v, _, _, _, _ = fps_with_knn_vanilla(pts, K, k_neighbors=k_neighbors, seed="first")
        idx_q, _, _, _, _ = fps_with_knn_vanilla(pts_q, K, k_neighbors=k_neighbors, seed="first")

    iv = idx_v[0].detach().cpu().numpy().astype(np.int64)
    iq = idx_q[0].detach().cpu().numpy().astype(np.int64)
    m = fps_set_metrics(iv, iq)

    err = (pts - pts_q).abs()
    print("=" * 60)
    print(f"dataset={args.dataset} ({dname})  sample_index={idx}/{n}")
    print(f"N={pts.shape[1]}  K_fps={K}  k_neighbors={k_neighbors}  bits={args.bits}")
    print(f"quant.scheme={quant_cfg.get('scheme')}")
    if quant_cfg.get("stats_bounds_json"):
        print(f"stats_bounds_json={quant_cfg.get('stats_bounds_json')}")
    print(f"coord |p-pq| max={err.max().item():.6f}  mean={err.mean().item():.6f}")
    print("-" * 60)
    for k, v in m.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    print_step_table(iv, iq, args.max_step_rows)

    # 仅在集合中出现于一边的下标（便于扫一眼）
    sv, sq = set(iv.tolist()), set(iq.tolist())
    only_v = sorted(sv - sq)
    only_q = sorted(sq - sv)
    lim = 30
    print(f"\n仅在 vanilla FPS 集合中（共 {len(only_v)}）前 {min(lim, len(only_v))} 个: {only_v[:lim]}")
    print(f"仅在 quant FPS 集合中（共 {len(only_q)}）前 {min(lim, len(only_q))} 个: {only_q[:lim]}")


if __name__ == "__main__":
    main()
