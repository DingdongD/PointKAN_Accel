#!/usr/bin/env python3
"""
ModelNet40 上：先 FPS 最远点采样，再对采样点做 Vanilla KNN vs 预比较器 KNN，
评估 IoU、Recall 与过滤比例。

支持两种预比较器模式：
  1) safe   : 严格下界过滤（更接近无损）
  2) fastnn : 贴近 FastNN 论文的 MSS/MSB 预比较器

用法（在 classification_ModelNet40 目录下）：
  python run_knn_modelnet40_eval.py --partition test --num_points 1024 --n_ref 512 --k 16 --max_samples 100 --mode safe
  python run_knn_modelnet40_eval.py --partition test --num_points 1024 --n_ref 512 --k 16 --max_samples 100 --mode fastnn
"""
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

# 接入 inference 使用的 ModelNet40 数据
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# 接入 FastNN Bsearch（预比较器 KNN）
# 你自己的目录结构如果不同，改这里
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(SCRIPT_DIR)))
FASTNN_DIR = os.path.join(REPO_ROOT, "hw_sim", "SOTA-Accelerators", "FastNN")
if os.path.isdir(FASTNN_DIR) and FASTNN_DIR not in sys.path:
    sys.path.insert(0, FASTNN_DIR)

from data import ModelNet40

try:
    from Bsearch import (
        vanilla_knn_single_query,
        evaluate_knn_module,
        evaluate_knn_module_compare_all,
        knn_set_iou,
        knn_recall_at_k,
    )
except ImportError as e:
    raise ImportError(
        f"请确保 FastNN 路径可被导入。当前 FASTNN_DIR={FASTNN_DIR}\n{e}"
    ) from e


# ---------------------------------------------------------------------------
# 最远点采样 (FPS)，纯 NumPy，单点云
# ---------------------------------------------------------------------------
def fps_numpy(pc: np.ndarray, K: int, start_idx: int = 0) -> np.ndarray:
    """
    pc: [N, 3], float
    K: 采样点数
    start_idx: 第一个点的索引（默认 0）
    返回: 采样索引 idx [K], dtype int64
    """
    N = pc.shape[0]
    K = min(K, N)

    idx = np.zeros(K, dtype=np.int64)
    idx[0] = start_idx

    min_dist2 = np.sum((pc - pc[idx[0]: idx[0] + 1]) ** 2, axis=1)

    for i in range(1, K):
        min_dist2[idx[:i]] = -1.0
        farthest = np.argmax(min_dist2)
        idx[i] = farthest
        d2 = np.sum((pc - pc[farthest: farthest + 1]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, d2)

    return idx


def run_one_shape(
    pc: np.ndarray,
    n_ref: int,
    n_queries: int,
    k: int,
    coord_scale: float,
    rng: np.random.Generator,
    mode: str,
):
    """
    对单帧点云 pc [N, 3]：
      FPS 得到 ref，再取 query，跑 Vanilla / Safe / FastNN。
    mode: "safe" | "fastnn" | "all"
    """
    N = pc.shape[0]
    n_ref = min(n_ref, N)

    ref_idx = fps_numpy(pc, n_ref, start_idx=0)
    ref_xyz = pc[ref_idx].astype(np.float64)
    ref_xyz_int = (ref_xyz * coord_scale).astype(np.int64)

    if n_queries <= 0:
        query_idx = np.arange(N, dtype=np.int64)
    else:
        query_idx = rng.choice(N, size=min(n_queries, N), replace=False)

    queries_xyz = pc[query_idx].astype(np.float64)
    queries_int = (queries_xyz * coord_scale).astype(np.int64)
    scale_sq = coord_scale * coord_scale

    if mode == "all":
        return evaluate_knn_module_compare_all(
            queries_xyz, ref_xyz, queries_int, ref_xyz_int, k, scale_sq=scale_sq
        )
    res = evaluate_knn_module(
        queries_xyz, ref_xyz, queries_int, ref_xyz_int, k,
        mode=mode, scale_sq=scale_sq,
    )
    return res


def main():
    p = argparse.ArgumentParser(
        description="ModelNet40: FPS 后 Vanilla / Safe / FastNN KNN — IoU 与过滤对比"
    )
    p.add_argument("--partition", type=str, default="test", choices=("train", "test"))
    p.add_argument("--num_points", type=int, default=1024, help="每帧点云点数")
    p.add_argument("--n_ref", type=int, default=512, help="FPS 采样点数（参考集大小）")
    p.add_argument("--n_queries", type=int, default=0,
                   help="每帧 query 数，0 表示全部点")
    p.add_argument("--k", type=int, default=16, help="KNN 的 k")
    p.add_argument("--max_samples", type=int, default=0,
                   help="最多评估的样本数，0 表示全部")
    p.add_argument("--coord_scale", type=float, default=1000.0,
                   help="浮点坐标转整数的缩放，供预比较器使用")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", type=str, default="all", choices=("safe", "fastnn", "all"),
                   help="运行模式：safe / fastnn 单模式，all 同时跑三种并输出对比")
    args = p.parse_args()

    dataset = ModelNet40(partition=args.partition, num_points=args.num_points)
    total = len(dataset)
    n_samples = total if args.max_samples <= 0 else min(args.max_samples, total)
    rng = np.random.default_rng(args.seed)
    n_ref = args.n_ref
    n_q_per_shape = args.n_queries if args.n_queries > 0 else args.num_points

    if args.mode == "all":
        compare_list = []
        for i in tqdm(range(n_samples), desc="ModelNet40 KNN (Vanilla vs Safe vs FastNN)"):
            pc, _ = dataset[i]
            if pc.shape[0] < n_ref or pc.shape[0] < args.k:
                continue
            cmp = run_one_shape(
                pc, n_ref=n_ref, n_queries=args.n_queries, k=args.k,
                coord_scale=args.coord_scale, rng=rng, mode="all",
            )
            compare_list.append(cmp)
        if not compare_list:
            print("没有有效样本可评估。")
            return
        vanilla = compare_list[0]["vanilla"]
        safe = {
            "mean_iou": float(np.mean([d["safe"]["mean_iou"] for d in compare_list])),
            "std_iou": float(np.std([d["safe"]["mean_iou"] for d in compare_list])),
            "mean_recall": float(np.mean([d["safe"]["mean_recall"] for d in compare_list])),
            "std_recall": float(np.std([d["safe"]["mean_recall"] for d in compare_list])),
            "avg_full_compute": float(np.mean([d["safe"]["avg_full_compute"] for d in compare_list])),
            "avg_filtered": float(np.mean([d["safe"]["avg_filtered"] for d in compare_list])),
            "avg_filter_ratio": float(np.mean([d["safe"]["avg_filter_ratio"] for d in compare_list])),
        }
        fastnn = {
            "mean_iou": float(np.mean([d["fastnn"]["mean_iou"] for d in compare_list])),
            "std_iou": float(np.std([d["fastnn"]["mean_iou"] for d in compare_list])),
            "mean_recall": float(np.mean([d["fastnn"]["mean_recall"] for d in compare_list])),
            "std_recall": float(np.std([d["fastnn"]["mean_recall"] for d in compare_list])),
            "avg_full_compute": float(np.mean([d["fastnn"]["avg_full_compute"] for d in compare_list])),
            "avg_filtered": float(np.mean([d["fastnn"]["avg_filtered"] for d in compare_list])),
            "avg_filter_ratio": float(np.mean([d["fastnn"]["avg_filter_ratio"] for d in compare_list])),
        }
        print("\n" + "=" * 76)
        print("  ModelNet40 — Vanilla KNN vs Safe vs FastNN 三种模式对比")
        print("=" * 76)
        print(f"  数据: partition={args.partition}  num_points={args.num_points}  n_ref={n_ref}  k={args.k}")
        print(f"  有效样本数: {len(compare_list)}  每样本 query 数: ~{n_q_per_shape}")
        print("-" * 76)
        print(f"  {'模式':<10} | {'Mean IoU':>10} | {'Mean R@k':>10} | {'精确计算/ref':>14} | {'过滤比例':>10}")
        print("-" * 76)
        print(f"  {'Vanilla':<10} | {vanilla['mean_iou']:>10.6f} | {vanilla['mean_recall']:>10.6f} | {vanilla['avg_full_compute']:>14.1f} | {vanilla['avg_filter_ratio']*100:>9.2f}%")
        print(f"  {'Safe':<10} | {safe['mean_iou']:>10.6f} | {safe['mean_recall']:>10.6f} | {safe['avg_full_compute']:>14.1f} | {safe['avg_filter_ratio']*100:>9.2f}%")
        print(f"  {'FastNN':<10} | {fastnn['mean_iou']:>10.6f} | {fastnn['mean_recall']:>10.6f} | {fastnn['avg_full_compute']:>14.1f} | {fastnn['avg_filter_ratio']*100:>9.2f}%")
        print("=" * 76 + "\n")
        return
    # 单模式：safe 或 fastnn
    all_ious = []
    all_recalls = []
    all_full = []
    all_filtered = []
    valid_shapes = 0

    for i in tqdm(range(n_samples), desc=f"ModelNet40 KNN eval ({args.mode})"):
        pc, _ = dataset[i]
        if pc.shape[0] < n_ref or pc.shape[0] < args.k:
            continue
        res = run_one_shape(
            pc, n_ref=n_ref, n_queries=args.n_queries, k=args.k,
            coord_scale=args.coord_scale, rng=rng, mode=args.mode,
        )
        all_ious.extend(res["ious"])
        all_recalls.extend(res["recalls"])
        all_full.append(res["avg_full_compute"])
        all_filtered.append(res["avg_filtered"])
        valid_shapes += 1

    if valid_shapes == 0:
        print("没有有效样本可评估。")
        return

    mean_iou = float(np.mean(all_ious))
    std_iou = float(np.std(all_ious))
    mean_recall = float(np.mean(all_recalls))
    std_recall = float(np.std(all_recalls))
    avg_full = float(np.mean(all_full))
    avg_filtered = float(np.mean(all_filtered))
    avg_filter_ratio = avg_filtered / n_ref

    print("\n" + "=" * 72)
    print(f"  ModelNet40 — FPS 后 预比较器 KNN 评估 ({args.mode})")
    print("=" * 72)
    print(f"  数据: partition={args.partition}  num_points={args.num_points}  n_ref={n_ref}  k={args.k}")
    print(f"  有效评估样本数: {valid_shapes}  每样本 query 数: {args.n_queries or args.num_points}")
    print("-" * 72)
    print(f"    Mean IoU (vs Vanilla):   {mean_iou:.6f}  (std: {std_iou:.6f})")
    print(f"    Mean Recall@k:           {mean_recall:.6f}  (std: {std_recall:.6f})")
    print(f"    平均每 query 精确距离计算: {avg_full:.1f} / {n_ref}")
    print(f"    平均过滤比例:            {avg_filter_ratio * 100:.2f}%")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()