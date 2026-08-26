#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单样本：对比全精度与量化后 `fps_with_knn_vanilla` 的 **KNN 邻居下标**（与 run_fps_knn_quant_recall 一致）。

说明（与 JSON 指标对齐）：
  - 第 t 步：vanilla 以 `idx_v[t]` 为中心在 **pts** 上做 kNN；quant 以 `idx_q[t]` 为中心在 **pts_q** 上做 kNN。
  - 邻居下标均为点张量列下标 0..N-1；若 FPS 在第 t 步已分叉，则两路 **物理中心不同**，KNN 集合差异会同时来自「中心偏移」与「距离形变」。

用法:
  python debug_knn_single_sample.py --dataset modelnet40 --bits 8
  python debug_knn_single_sample.py --dataset kitti --index 0 --bits 16
  python debug_knn_single_sample.py --dataset s3dis --seed 123 --bits 12 --max_step_rows 20
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
from spec_kan_fps import fps_with_knn_vanilla, knn_neighbor_set_iou  # noqa: E402


def knn_step_metrics_row(
    nv: np.ndarray, nq: np.ndarray
) -> Tuple[int, float, float]:
    """
    nv, nq: 长度 k 的邻居下标（允许重复时与 torch 实现略异，一般 topk 互异）。
    返回 (|交集|, recall_json_style, iou_json_style)。
    """
    sv: Set[int] = set(nv.tolist())
    sq: Set[int] = set(nq.tolist())
    inter = sv & sq
    k = int(nv.shape[0])
    recall = len(inter) / float(max(k, 1))
    union = len(sv | sq)
    iou = len(inter) / float(max(union, 1))
    return len(inter), recall, iou


def print_knn_step_table(
    idx_v: np.ndarray,
    idx_q: np.ndarray,
    knn_v: np.ndarray,
    knn_q: np.ndarray,
    max_rows: int,
    k_list: int,
) -> None:
    K = idx_v.shape[0]
    print(f"\n--- 逐步 KNN（前 {min(max_rows, K)} / {K} 步；每步 k={k_list}）---")
    print(
        f"{'t':>4} {'ctr_v':>7} {'ctr_q':>7} {'|∩|':>5} {'recall':>7} {'knn_IoU':>8} "
        f"{'ctr∈N_v':>7} {'ctr∈N_q':>7}"
    )
    for t in range(min(max_rows, K)):
        nv, nq = knn_v[t], knn_q[t]
        n_inter, rec, iou = knn_step_metrics_row(nv, nq)
        cv, cq = int(idx_v[t]), int(idx_q[t])
        in_v = cv in set(nv.tolist())
        in_q = cq in set(nq.tolist())
        print(
            f"{t:4d} {cv:7d} {cq:7d} {n_inter:5d} {rec:7.4f} {iou:8.4f} "
            f"{'Y' if in_v else '':>7} {'Y' if in_q else '':>7}"
        )


def parse_args():
    p = argparse.ArgumentParser(description="单样本 KNN 邻居 vanilla vs 量化（与 benchmark 定义一致）")
    p.add_argument("--config", type=str, default=os.path.join(BENCH_DIR, "datasets.yaml"))
    p.add_argument("--dataset", type=str, required=True, help="yaml 中 datasets 的 key")
    p.add_argument("--index", type=int, default=-1, help="样本下标，-1 表示随机")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bits", type=int, default=8)
    p.add_argument("--fps_steps", type=int, default=0, help="0=从 yaml 读")
    p.add_argument("--k_neighbors", type=int, default=0, help="0=从 yaml 读")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_step_rows", type=int, default=128, help="逐步打印最多行数")
    p.add_argument(
        "--dump_steps",
        type=str,
        default="",
        help="逗号分隔的步号，额外打印该步完整邻居列表，如 '0,1,127'",
    )
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
        idx_v, _, knn_v, _, _ = fps_with_knn_vanilla(pts, K, k_neighbors=k_neighbors, seed="first")
        idx_q, _, knn_q, _, _ = fps_with_knn_vanilla(pts_q, K, k_neighbors=k_neighbors, seed="first")

        kiou_t = knn_neighbor_set_iou(knn_v, knn_q)
        krec_t = bench.knn_recall_vs_a(knn_v, knn_q)

    iv = idx_v[0].detach().cpu().numpy().astype(np.int64)
    iq = idx_q[0].detach().cpu().numpy().astype(np.int64)
    kv = knn_v[0].detach().cpu().numpy().astype(np.int64)
    kq = knn_q[0].detach().cpu().numpy().astype(np.int64)
    kiou_1d = kiou_t[0].detach().cpu().numpy()
    krec_1d = krec_t[0].detach().cpu().numpy()
    k_list = kv.shape[1]

    err = (pts - pts_q).abs()
    print("=" * 72)
    print(f"dataset={args.dataset} ({dname})  sample_index={idx}/{n}")
    print(f"N={pts.shape[1]}  K_fps={K}  k_neighbors={k_list}  bits={args.bits}")
    print(f"quant.scheme={quant_cfg.get('scheme')}")
    if quant_cfg.get("stats_bounds_json"):
        print(f"stats_bounds_json={quant_cfg.get('stats_bounds_json')}")
    print(f"coord |p-pq| max={err.max().item():.6f}  mean={err.mean().item():.6f}")
    print("-" * 72)
    print("与 run_fps_knn_quant_recall 相同的张量级聚合（对 K 步展平后 mean/std）：")
    flat_iou = kiou_1d.reshape(-1)
    flat_rec = krec_1d.reshape(-1)
    print(f"  knn_iou_mean (per-step, mean over steps): {float(flat_iou.mean()):.6f}")
    print(f"  knn_iou_std:                               {float(flat_iou.std()):.6f}")
    print(f"  knn_recall_mean (per-step):                {float(flat_rec.mean()):.6f}")
    print(f"  knn_recall_std:                            {float(flat_rec.std()):.6f}")
    print("-" * 72)
    print("FPS 下标（便于对照）：")
    sv, sq = set(iv.tolist()), set(iq.tolist())
    print(f"  unique idx_v / idx_q: {len(sv)} / {len(sq)}  | 集合交: {len(sv & sq)}")
    print("=" * 72)

    print_knn_step_table(iv, iq, kv, kq, args.max_step_rows, k_list)

    if args.dump_steps.strip():
        steps = []
        for part in args.dump_steps.split(","):
            part = part.strip()
            if part:
                steps.append(int(part))
        for t in steps:
            if t < 0 or t >= K:
                print(f"\n[skip] dump step {t} 越界 [0,{K})")
                continue
            nv, nq = kv[t], kq[t]
            n_inter, rec, iou = knn_step_metrics_row(nv, nq)
            print(f"\n--- dump step t={t}  ctr_v={iv[t]} ctr_q={iq[t]}  |∩|={n_inter} recall={rec:.4f} knn_iou={iou:.4f} ---")
            print(f"  knn_v: {nv.tolist()}")
            print(f"  knn_q: {nq.tolist()}")
            only_v = sorted(set(nv.tolist()) - set(nq.tolist()))
            only_q = sorted(set(nq.tolist()) - set(nv.tolist()))
            print(f"  only_v ({len(only_v)}): {only_v[:40]}{'...' if len(only_v) > 40 else ''}")
            print(f"  only_q ({len(only_q)}): {only_q[:40]}{'...' if len(only_q) > 40 else ''}")


if __name__ == "__main__":
    main()
