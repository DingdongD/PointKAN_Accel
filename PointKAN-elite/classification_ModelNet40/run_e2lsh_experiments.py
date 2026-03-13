#!/usr/bin/env python3
"""
E2LSH 实验脚本（审稿用）:
  1) E2LSH 参数扫: L, w, candidate_mode -> candidate_ratio, shortage_ratio, recall@k
  2) E2LSH + backend 对比: 固定较优 E2LSH，跑 int_vanilla / safe_strict / fastnn -> 拆解 E2LSH 损失 vs comparator 损失

日志与结果写入: runs/e2lsh_experiments/
"""
import os
import sys
import json
import argparse
from datetime import datetime
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
from run_knn_modelnet40_opt_eval import (
    run_one_shape_e2lsh,
    collect_metric,
)

OUT_DIR = os.path.join(SCRIPT_DIR, "runs", "e2lsh_experiments")


class ExperimentLogger:
    def __init__(self, log_path: str):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_path = log_path
        self._file = open(log_path, "w", encoding="utf-8")

    def log(self, msg: str):
        line = msg if msg.endswith("\n") else msg + "\n"
        self._file.write(line)
        self._file.flush()
        print(msg, end="" if msg.endswith("\n") else "\n")

    def close(self):
        self._file.close()


def run_e2lsh_single_config(
    dataset,
    n_samples,
    rng,
    n_ref,
    n_queries,
    k,
    quant_mode,
    coord_scale,
    coord_min,
    coord_max,
    quant_bits,
    L,
    w,
    e2lsh_seed,
    e2lsh_K,
    e2lsh_table_size,
    e2lsh_C,
    candidate_mode,
    min_hits,
    search_mode,
    gt_mode,
):
    results = []
    for i in range(n_samples):
        pc, _ = dataset[i]
        if pc.shape[0] < n_ref or pc.shape[0] < k:
            continue
        res = run_one_shape_e2lsh(
            pc=pc,
            n_ref=n_ref,
            n_queries=n_queries,
            k=k,
            rng=rng,
            quant_mode=quant_mode,
            coord_scale=coord_scale,
            coord_min=coord_min,
            coord_max=coord_max,
            quant_bits=quant_bits,
            L=L,
            w=w,
            e2lsh_seed=e2lsh_seed,
            e2lsh_K=e2lsh_K,
            e2lsh_table_size=e2lsh_table_size,
            e2lsh_C=e2lsh_C,
            candidate_mode=candidate_mode,
            min_hits=min_hits,
            search_mode=search_mode,
            gt_mode=gt_mode,
        )
        results.append(res)
    if not results:
        return None
    return {
        "mean_recall": float(np.mean(collect_metric(results, "recalls"))),
        "mean_iou": float(np.mean(collect_metric(results, "ious"))),
        "avg_candidate_ratio": float(np.mean([r["avg_candidate_ratio"] for r in results])),
        "shortage_query_ratio": float(np.mean([r["shortage_query_ratio"] for r in results])),
        "avg_candidate_size": float(np.mean([r["avg_candidate_size"] for r in results])),
        "n_shapes": len(results),
    }


def main():
    p = argparse.ArgumentParser(description="E2LSH 参数扫 + E2LSH+backend 对比实验")
    p.add_argument("--partition", type=str, default="test")
    p.add_argument("--num_points", type=int, default=1024)
    p.add_argument("--n_ref", type=int, default=1024)
    p.add_argument("--n_queries", type=int, default=0)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--max_samples", type=int, default=50, help="每配置评估的样本数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quant_mode", type=str, default="dataset_range")
    p.add_argument("--coord_min", type=float, default=-1.0)
    p.add_argument("--coord_max", type=float, default=1.0)
    p.add_argument("--quant_bits", type=int, default=16)
    p.add_argument("--e2lsh_seed", type=int, default=42)
    p.add_argument("--e2lsh_K", type=int, default=4)
    p.add_argument("--e2lsh_table_size", type=int, default=4096)
    p.add_argument("--e2lsh_C", type=int, default=(2**32 - 5))
    p.add_argument("--skip_sweep", action="store_true", help="只跑 backend 对比，不扫参数")
    p.add_argument("--pick_L", type=int, default=8, help="backend 对比时使用的 L")
    p.add_argument("--pick_w", type=float, default=0.5, help="backend 对比时使用的 w（建议选 Part1 中 recall 高、shortage 低的）")
    p.add_argument("--pick_candidate_mode", type=str, default="union",
                   help="two-level+intersection 极严易导致 shortage=100%%，实用建议 union")
    args = p.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(OUT_DIR, f"experiment_{ts}.log")
    logger = ExperimentLogger(log_path)

    logger.log("=" * 80)
    logger.log("  E2LSH 实验 (ModelNet40) — 参数扫 + E2LSH+backend 对比")
    logger.log("=" * 80)
    logger.log(f"  数据: partition={args.partition}  num_points={args.num_points}  n_ref={args.n_ref}  k={args.k}")
    logger.log(f"  max_samples={args.max_samples}  quant_mode={args.quant_mode}  range=[{args.coord_min},{args.coord_max}]  bits={args.quant_bits}")
    logger.log("")

    dataset = ModelNet40(partition=args.partition, num_points=args.num_points)
    total = len(dataset)
    n_samples = min(args.max_samples, total) if args.max_samples > 0 else total
    rng = np.random.default_rng(args.seed)

    coord_scale = 1000.0

    # ---------- Part 1: E2LSH 参数扫 ----------
    sweep_results = []
    if not args.skip_sweep:
        logger.log("--- Part 1: E2LSH 参数扫 (L, w, candidate_mode) ---")
        logger.log("  指标: candidate_ratio, shortage_ratio, recall@k")
        logger.log("")

        L_list = [4, 8, 12]
        w_list = [0.1, 0.2, 0.3, 0.5]
        candidate_modes = [
            ("union", 2),
            ("min_hits", 2),
        ]

        for L in L_list:
            for w in w_list:
                for cand_mode, min_hits in candidate_modes:
                    rec = run_e2lsh_single_config(
                        dataset=dataset,
                        n_samples=n_samples,
                        rng=rng,
                        n_ref=args.n_ref,
                        n_queries=args.n_queries,
                        k=args.k,
                        quant_mode=args.quant_mode,
                        coord_scale=coord_scale,
                        coord_min=args.coord_min,
                        coord_max=args.coord_max,
                        quant_bits=args.quant_bits,
                        L=L,
                        w=w,
                        e2lsh_seed=args.e2lsh_seed,
                        e2lsh_K=args.e2lsh_K,
                        e2lsh_table_size=args.e2lsh_table_size,
                        e2lsh_C=args.e2lsh_C,
                        candidate_mode=cand_mode,
                        min_hits=min_hits,
                        search_mode="int_vanilla",
                        gt_mode="float_vanilla",
                    )
                    if rec is None:
                        continue
                    row = {
                        "L": L,
                        "w": w,
                        "candidate_mode": cand_mode,
                        "min_hits": min_hits if cand_mode == "min_hits" else None,
                        "candidate_ratio": rec["avg_candidate_ratio"],
                        "shortage_ratio": rec["shortage_query_ratio"],
                        "recall_at_k": rec["mean_recall"],
                        "mean_iou": rec["mean_iou"],
                        "n_shapes": rec["n_shapes"],
                    }
                    sweep_results.append(row)
                    logger.log(
                        f"  L={L}  w={w}  mode={cand_mode}  ->  "
                        f"cand_ratio={rec['avg_candidate_ratio']*100:.2f}%  "
                        f"shortage={rec['shortage_query_ratio']*100:.2f}%  "
                        f"recall@k={rec['mean_recall']:.4f}"
                    )

        logger.log("")
        logger.log("  Part 1 汇总表 (candidate_ratio, shortage_ratio, recall@k):")
        logger.log("  " + "-" * 72)
        logger.log(f"  {'L':>3}  {'w':>5}  {'cand_mode':<10}  {'cand_ratio%':>12}  {'shortage%':>10}  {'recall@k':>10}")
        logger.log("  " + "-" * 72)
        for r in sweep_results:
            logger.log(
                f"  {r['L']:>3}  {r['w']:>5.2f}  {r['candidate_mode']:<10}  "
                f"{r['candidate_ratio']*100:>11.2f}%  {r['shortage_ratio']*100:>9.2f}%  {r['recall_at_k']:>10.4f}"
            )
        logger.log("")

        sweep_json = os.path.join(OUT_DIR, f"sweep_results_{ts}.json")
        with open(sweep_json, "w", encoding="utf-8") as f:
            json.dump({"sweep": sweep_results, "config": vars(args)}, f, indent=2)
        logger.log(f"  已保存: {sweep_json}")
        logger.log("")
    else:
        logger.log("  (跳过 Part 1 参数扫)")
        logger.log("")

    # ---------- Part 2: E2LSH + backend 对比 ----------
    logger.log("--- Part 2: E2LSH + backend 对比 (固定 E2LSH 参数) ---")
    L_fix = args.pick_L
    w_fix = args.pick_w
    cand_fix = args.pick_candidate_mode
    min_hits_fix = 2 if cand_fix == "min_hits" else 2
    logger.log(f"  固定 E2LSH: L={L_fix}  w={w_fix}  candidate_mode={cand_fix}  GT=float_vanilla")
    logger.log("")

    backend_results = {}
    for search_mode in ["int_vanilla", "safe_strict", "fastnn"]:
        rec = run_e2lsh_single_config(
            dataset=dataset,
            n_samples=n_samples,
            rng=rng,
            n_ref=args.n_ref,
            n_queries=args.n_queries,
            k=args.k,
            quant_mode=args.quant_mode,
            coord_scale=coord_scale,
            coord_min=args.coord_min,
            coord_max=args.coord_max,
            quant_bits=args.quant_bits,
            L=L_fix,
            w=w_fix,
            e2lsh_seed=args.e2lsh_seed,
            e2lsh_K=args.e2lsh_K,
            e2lsh_table_size=args.e2lsh_table_size,
            e2lsh_C=args.e2lsh_C,
            candidate_mode=cand_fix,
            min_hits=min_hits_fix,
            search_mode=search_mode,
            gt_mode="float_vanilla",
        )
        if rec is None:
            continue
        backend_results[search_mode] = rec
        logger.log(
            f"  E2LSH + {search_mode:<12}  ->  "
            f"IoU={rec['mean_iou']:.4f}  Recall@k={rec['mean_recall']:.4f}  "
            f"cand_ratio={rec['avg_candidate_ratio']*100:.2f}%  shortage={rec['shortage_query_ratio']*100:.2f}%"
        )

    logger.log("")
    logger.log("  Part 2 汇总:")
    logger.log("  " + "-" * 60)
    logger.log(f"  {'Backend':<16}  {'Mean IoU':>10}  {'Recall@k':>10}  {'cand_ratio%':>12}  {'shortage%':>10}")
    logger.log("  " + "-" * 60)
    for name, rec in backend_results.items():
        logger.log(
            f"  E2LSH+{name:<10}  {rec['mean_iou']:>10.4f}  {rec['mean_recall']:>10.4f}  "
            f"{rec['avg_candidate_ratio']*100:>11.2f}%  {rec['shortage_query_ratio']*100:>9.2f}%"
        )

    backend_json = os.path.join(OUT_DIR, f"backend_compare_{ts}.json")
    with open(backend_json, "w", encoding="utf-8") as f:
        json.dump({
            "backend_compare": backend_results,
            "e2lsh_config": {"L": L_fix, "w": w_fix, "candidate_mode": cand_fix},
            "config": vars(args),
        }, f, indent=2)
    logger.log(f"  已保存: {backend_json}")
    logger.log("")
    logger.log("=" * 80)
    logger.log(f"  日志已保存: {log_path}")
    logger.log("=" * 80)
    logger.close()


if __name__ == "__main__":
    main()
