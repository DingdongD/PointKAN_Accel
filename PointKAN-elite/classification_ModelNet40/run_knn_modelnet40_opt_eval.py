#!/usr/bin/env python3
"""
ModelNet40 上评估四种模式:
  1) float_vanilla
  2) int_vanilla
  3) safe_strict
  4) fastnn

并输出三组关键对比:
  A. float_vanilla  vs int_vanilla   -> 量化损失
  B. int_vanilla    vs safe_strict   -> strict safe 是否严格无损
  C. float_vanilla  vs fastnn        -> 最终近似预比较器损失
"""
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(SCRIPT_DIR)))
FASTNN_DIR = os.path.join(REPO_ROOT, "hw_sim", "SOTA-Accelerators", "FastNN")
if os.path.isdir(FASTNN_DIR) and FASTNN_DIR not in sys.path:
    sys.path.insert(0, FASTNN_DIR)

from data import ModelNet40

try:
    from Bsearch_opt import (
        evaluate_knn_module,
        evaluate_knn_module_with_e2lsh,
        E2LSHTwoLevelIndex,
        quantize_points_fixed_scale,
        quantize_points_per_cloud_16bit,
        quantize_points_fixed_range,
        quantize_with_meta,
    )
except ImportError as e:
    raise ImportError(
        f"请确保 FastNN 路径可被导入。当前 FASTNN_DIR={FASTNN_DIR}\n{e}"
    ) from e


def fps_numpy(pc: np.ndarray, K: int, start_idx: int = 0) -> np.ndarray:
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


def quantize_ref_and_query(ref_xyz, queries_xyz, quant_mode, coord_scale,
                           coord_min, coord_max, quant_bits):
    if quant_mode == "fixed_scale":
        ref_int = quantize_points_fixed_scale(ref_xyz, coord_scale)
        query_int = quantize_points_fixed_scale(queries_xyz, coord_scale)
    elif quant_mode == "per_cloud_16bit":
        ref_int, meta = quantize_points_per_cloud_16bit(ref_xyz)
        query_int = quantize_with_meta(queries_xyz, meta)
    elif quant_mode == "dataset_range":
        # 按数据集空间 [coord_min, coord_max] 与量化比特数 bits 统一量化
        ref_int = quantize_points_fixed_range(
            ref_xyz, coord_min, coord_max, quant_bits, clip=True
        )
        query_int = quantize_points_fixed_range(
            queries_xyz, coord_min, coord_max, quant_bits, clip=True
        )
    else:
        raise ValueError(f"Unsupported quant_mode: {quant_mode}")
    return ref_int, query_int


def run_one_shape(
    pc: np.ndarray,
    n_ref: int,
    n_queries: int,
    k: int,
    rng: np.random.Generator,
    quant_mode: str,
    coord_scale: float,
    coord_min: float,
    coord_max: float,
    quant_bits: int,
):
    N = pc.shape[0]
    n_ref = min(n_ref, N)

    ref_idx = fps_numpy(pc, n_ref, start_idx=0)
    ref_xyz = pc[ref_idx].astype(np.float64)

    if n_queries <= 0:
        query_idx = np.arange(N, dtype=np.int64)
    else:
        query_idx = rng.choice(N, size=min(n_queries, N), replace=False)

    queries_xyz = pc[query_idx].astype(np.float64)

    ref_int, queries_int = quantize_ref_and_query(
        ref_xyz, queries_xyz, quant_mode=quant_mode, coord_scale=coord_scale,
        coord_min=coord_min, coord_max=coord_max, quant_bits=quant_bits,
    )

    # A. float vanilla vs int vanilla
    res_float_vs_int = evaluate_knn_module(
        queries_xyz, ref_xyz, queries_int, ref_int, k,
        mode="int_vanilla", gt_mode="float_vanilla"
    )

    # B. int vanilla vs strict safe
    res_int_vs_safe = evaluate_knn_module(
        queries_xyz, ref_xyz, queries_int, ref_int, k,
        mode="safe_strict", gt_mode="int_vanilla"
    )

    # C. float vanilla vs fastnn
    res_float_vs_fastnn = evaluate_knn_module(
        queries_xyz, ref_xyz, queries_int, ref_int, k,
        mode="fastnn", gt_mode="float_vanilla"
    )

    return {
        "float_vs_int": res_float_vs_int,
        "int_vs_safe": res_int_vs_safe,
        "float_vs_fastnn": res_float_vs_fastnn,
    }


def run_one_shape_e2lsh(
    pc: np.ndarray,
    n_ref: int,
    n_queries: int,
    k: int,
    rng: np.random.Generator,
    quant_mode: str,
    coord_scale: float,
    coord_min: float,
    coord_max: float,
    quant_bits: int,
    L: int,
    w: float,
    e2lsh_seed: int,
    e2lsh_K: int,
    e2lsh_table_size: int,
    e2lsh_C: int,
    candidate_mode: str,
    min_hits: int,
    search_mode: str,
    gt_mode: str,
):
    """FPS 得 ref -> 量化 -> 建 E2LSH (two-level) -> 在候选上做 KNN。"""
    N = pc.shape[0]
    n_ref = min(n_ref, N)

    ref_idx = fps_numpy(pc, n_ref, start_idx=0)
    ref_xyz = pc[ref_idx].astype(np.float64)

    if n_queries <= 0:
        query_idx = np.arange(N, dtype=np.int64)
    else:
        query_idx = rng.choice(N, size=min(n_queries, N), replace=False)
    queries_xyz = pc[query_idx].astype(np.float64)

    ref_int, queries_int = quantize_ref_and_query(
        ref_xyz, queries_xyz, quant_mode=quant_mode, coord_scale=coord_scale,
        coord_min=coord_min, coord_max=coord_max, quant_bits=quant_bits,
    )

    e2lsh = E2LSHTwoLevelIndex(
        dim=3,
        L=L,
        K=e2lsh_K,
        w=w,
        table_size=e2lsh_table_size,
        C=e2lsh_C,
        seed=e2lsh_seed,
    ).build(ref_xyz)

    res = evaluate_knn_module_with_e2lsh(
        queries_xyz=queries_xyz,
        refs_xyz=ref_xyz,
        queries_int=queries_int,
        refs_int=ref_int,
        k=k,
        e2lsh_index=e2lsh,
        candidate_mode=candidate_mode,
        min_hits=min_hits,
        search_mode=search_mode,
        gt_mode=gt_mode,
    )
    return res


def collect_metric(res_list, key):
    vals = []
    for r in res_list:
        vals.extend(r[key])
    return np.array(vals, dtype=np.float64)


def main():
    p = argparse.ArgumentParser(description="ModelNet40 KNN strict-safe / FastNN evaluation")
    p.add_argument("--partition", type=str, default="test", choices=("train", "test"))
    p.add_argument("--num_points", type=int, default=1024)
    p.add_argument("--n_ref", type=int, default=1024)
    p.add_argument("--n_queries", type=int, default=0, help="0 表示全部点")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--max_samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quant_mode", type=str, default="dataset_range",
                   choices=("fixed_scale", "per_cloud_16bit", "dataset_range"),
                   help="dataset_range: 按数据集空间 min/max 与 quant_bits 量化（ModelNet40 建议）")
    p.add_argument("--coord_scale", type=float, default=1000.0,
                   help="fixed_scale 模式下使用")
    p.add_argument("--coord_min", type=float, default=-1.0,
                   help="dataset_range 模式下数据空间下界（ModelNet40 一般为 -1）")
    p.add_argument("--coord_max", type=float, default=1.0,
                   help="dataset_range 模式下数据空间上界（ModelNet40 一般为 1）")
    p.add_argument("--quant_bits", type=int, default=16,
                   help="dataset_range 模式下每维量化比特数（如 8/12/16）")
    # E2LSH 模式：先 E2LSH 候选再 KNN
    p.add_argument("--mode", type=str, default="default",
                   choices=("default", "e2lsh"),
                   help="default: 三组 Float/Int/Safe/FastNN 对比; e2lsh: E2LSH 候选 + KNN 评测")
    p.add_argument("--e2lsh_L", type=int, default=8, help="E2LSH 表数量")
    p.add_argument("--e2lsh_K", type=int, default=4,
                   help="每张表的 primitive E2LSH 个数")
    p.add_argument("--e2lsh_w", type=float, default=0.2, help="E2LSH 桶宽")
    p.add_argument("--e2lsh_table_size", type=int, default=4096,
                   help="two-level 顶层表大小")
    p.add_argument("--e2lsh_C", type=int, default=(2**32 - 5),
                   help="two-level H1/H2 的模数，原文建议 2^32-5")
    p.add_argument("--e2lsh_seed", type=int, default=42, help="E2LSH 随机种子")
    p.add_argument("--e2lsh_candidate_mode", type=str, default="intersection",
                   choices=("union", "intersection", "min_hits"),
                   help="论文原文为 intersection")
    p.add_argument("--e2lsh_min_hits", type=int, default=2, help="candidate_mode=min_hits 时最少命中表数")
    p.add_argument("--e2lsh_search_mode", type=str, default="int_vanilla",
                   choices=("int_vanilla", "safe_strict", "fastnn"),
                   help="在候选集上的 KNN 方式")
    p.add_argument("--e2lsh_gt_mode", type=str, default="float_vanilla",
                   choices=("float_vanilla", "int_vanilla"))
    args = p.parse_args()

    dataset = ModelNet40(partition=args.partition, num_points=args.num_points)
    total = len(dataset)
    n_samples = total if args.max_samples <= 0 else min(args.max_samples, total)
    rng = np.random.default_rng(args.seed)

    if args.mode == "e2lsh":
        # E2LSH 模式：只跑 E2LSH 候选 + KNN，输出 IoU / Recall / 候选比例 / shortage 比例
        e2lsh_results = []
        for i in tqdm(range(n_samples), desc="ModelNet40 E2LSH+KNN eval"):
            pc, _ = dataset[i]
            if pc.shape[0] < args.n_ref or pc.shape[0] < args.k:
                continue
            res = run_one_shape_e2lsh(
                pc=pc,
                n_ref=args.n_ref,
                n_queries=args.n_queries,
                k=args.k,
                rng=rng,
                quant_mode=args.quant_mode,
                coord_scale=args.coord_scale,
                coord_min=args.coord_min,
                coord_max=args.coord_max,
                quant_bits=args.quant_bits,
                L=args.e2lsh_L,
                w=args.e2lsh_w,
                e2lsh_seed=args.e2lsh_seed,
                e2lsh_K=args.e2lsh_K,
                e2lsh_table_size=args.e2lsh_table_size,
                e2lsh_C=args.e2lsh_C,
                candidate_mode=args.e2lsh_candidate_mode,
                min_hits=args.e2lsh_min_hits,
                search_mode=args.e2lsh_search_mode,
                gt_mode=args.e2lsh_gt_mode,
            )
            e2lsh_results.append(res)
        if not e2lsh_results:
            print("没有有效样本可评估。")
            return
        mean_iou = float(np.mean(collect_metric(e2lsh_results, "ious")))
        mean_recall = float(np.mean(collect_metric(e2lsh_results, "recalls")))
        avg_cand = float(np.mean([r["avg_candidate_size"] for r in e2lsh_results]))
        avg_cand_ratio = float(np.mean([r["avg_candidate_ratio"] for r in e2lsh_results]))
        shortage_ratio = float(np.mean([r["shortage_query_ratio"] for r in e2lsh_results]))
        avg_full = float(np.mean([r["avg_full_compute"] for r in e2lsh_results]))
        print("\n" + "=" * 80)
        print("  ModelNet40 — E2LSH 候选 + KNN 评测")
        print("=" * 80)
        print(f"  数据: partition={args.partition}  n_ref={args.n_ref}  k={args.k}  有效样本数={len(e2lsh_results)}")
        print(f"  E2LSH: L={args.e2lsh_L}  w={args.e2lsh_w}  candidate_mode={args.e2lsh_candidate_mode}  search_mode={args.e2lsh_search_mode}  GT={args.e2lsh_gt_mode}")
        print("-" * 80)
        print(f"  Mean IoU (vs {args.e2lsh_gt_mode}):   {mean_iou:.6f}")
        print(f"  Mean Recall@k:                        {mean_recall:.6f}")
        print(f"  平均候选集大小:                        {avg_cand:.1f}  (ref={args.n_ref})")
        print(f"  平均候选比例 (candidate/ref):         {avg_cand_ratio*100:.2f}%")
        print(f"  候选不足 k 的 query 比例 (shortage):   {shortage_ratio*100:.2f}%")
        print(f"  平均每 query 精确计算量 (在候选内):     {avg_full:.1f}")
        print("=" * 80 + "\n")
        return

    all_float_vs_int = []
    all_int_vs_safe = []
    all_float_vs_fastnn = []
    valid_shapes = 0

    for i in tqdm(range(n_samples), desc="ModelNet40 strict-safe eval"):
        pc, _ = dataset[i]
        if pc.shape[0] < args.n_ref or pc.shape[0] < args.k:
            continue

        res = run_one_shape(
            pc=pc,
            n_ref=args.n_ref,
            n_queries=args.n_queries,
            k=args.k,
            rng=rng,
            quant_mode=args.quant_mode,
            coord_scale=args.coord_scale,
            coord_min=args.coord_min,
            coord_max=args.coord_max,
            quant_bits=args.quant_bits,
        )

        all_float_vs_int.append(res["float_vs_int"])
        all_int_vs_safe.append(res["int_vs_safe"])
        all_float_vs_fastnn.append(res["float_vs_fastnn"])
        valid_shapes += 1

    if valid_shapes == 0:
        print("没有有效样本可评估。")
        return

    def summarize(res_list):
        ious = collect_metric(res_list, "ious")
        recalls = collect_metric(res_list, "recalls")
        avg_full = float(np.mean([r["avg_full_compute"] for r in res_list]))
        avg_filter = float(np.mean([r["avg_filter_ratio"] for r in res_list]))
        return {
            "mean_iou": float(np.mean(ious)),
            "mean_recall": float(np.mean(recalls)),
            "avg_full": avg_full,
            "avg_filter": avg_filter,
        }

    s_float_vs_int = summarize(all_float_vs_int)
    s_int_vs_safe = summarize(all_int_vs_safe)
    s_float_vs_fastnn = summarize(all_float_vs_fastnn)

    print("\n" + "=" * 84)
    print("  ModelNet40 — Float / Int / Strict-Safe / FastNN 对比")
    print("=" * 84)
    print(f"  数据: partition={args.partition}  num_points={args.num_points}  n_ref={args.n_ref}  k={args.k}")
    quant_info = f"quant_mode={args.quant_mode}"
    if args.quant_mode == "dataset_range":
        quant_info += f"  range=[{args.coord_min},{args.coord_max}]  bits={args.quant_bits}"
    else:
        quant_info += f"  coord_scale={args.coord_scale}"
    print(f"  有效样本数: {valid_shapes}  {quant_info}")
    print("-" * 84)
    print("  对比项                          |   Mean IoU |   Mean R@k |    精确计算/ref |   过滤比例")
    print("-" * 84)
    print(f"  Float Vanilla vs Int Vanilla   | {s_float_vs_int['mean_iou']:10.6f} | {s_float_vs_int['mean_recall']:10.6f} |"
          f" {s_float_vs_int['avg_full']:14.1f} | {s_float_vs_int['avg_filter']*100:9.2f}%")
    print(f"  Int Vanilla   vs Safe Strict   | {s_int_vs_safe['mean_iou']:10.6f} | {s_int_vs_safe['mean_recall']:10.6f} |"
          f" {s_int_vs_safe['avg_full']:14.1f} | {s_int_vs_safe['avg_filter']*100:9.2f}%")
    print(f"  Float Vanilla vs FastNN        | {s_float_vs_fastnn['mean_iou']:10.6f} | {s_float_vs_fastnn['mean_recall']:10.6f} |"
          f" {s_float_vs_fastnn['avg_full']:14.1f} | {s_float_vs_fastnn['avg_filter']*100:9.2f}%")
    print("=" * 84 + "\n")


if __name__ == "__main__":
    main()