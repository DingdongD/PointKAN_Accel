#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KNN 粗筛+细筛加强版：IoU 分析与 ModelNet40 推理评估

1. 在 FPS 基础上对比三种 KNN：
   - vanilla：精确 topk
   - adaptive：原 fps_with_knn_adaptive（阈值迭代，候选数在 [k*(1-tol), k*(1+tol)]）
   - coarse_fine：粗筛（候选数 >= min_candidates_ratio*k）+ 细筛 topk

2. 计算 coarse_fine / adaptive 相对 vanilla 的邻居集合 IoU。

3. 可选：调用与 inference_and_kan_analysis 一致的推理流程，汇报 ModelNet40 测试集准确率
   （当前分类模型内部仍为 vanilla KNN；本脚本主要验证粗筛+细筛与 vanilla 的 IoU 及 radii 差异）。

用法：
  python run_knn_coarse_fine_eval.py --k_neighbors 24 --fps_steps 512
  python run_knn_coarse_fine_eval.py --k_neighbors 24 --max_samples 500 --run_inference --checkpoint checkpoints/pointKANElite-pointkan_elite_exp-42
  python run_knn_coarse_fine_eval.py --no_inference --out_json ./knn_iou_results.json
"""

import os
import sys
import json
import argparse
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
for d in (SCRIPT_DIR, ROOT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from spec_kan_fps import (
    fps_with_knn_adaptive,
    fps_with_knn_coarse_fine,
    fps_with_knn_vanilla,
    knn_neighbor_set_iou,
    ModelNet40Wrapper,
    collate_points,
)


def parse_args():
    p = argparse.ArgumentParser(description="KNN 粗筛+细筛 IoU 分析与 ModelNet40 推理评估")
    p.add_argument("--num_points", type=int, default=1024, help="点云点数")
    p.add_argument("--fps_steps", type=int, default=512, help="FPS 采样数 K")
    p.add_argument("--k_neighbors", type=int, default=24, help="KNN 邻居数 k")
    p.add_argument("--batch_size", type=int, default=16, help="FPS/KNN 批大小")
    p.add_argument("--max_samples", type=int, default=0, help="最多评估样本数，0 表示全部测试集")
    p.add_argument("--min_candidates_ratio", type=float, default=1.2,
                   help="粗筛+细筛：候选数 >= min_candidates_ratio * k_neighbors")
    p.add_argument("--tol", type=float, default=0.05, help="adaptive 的 tol")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--root", type=str, default="./data", help="ModelNet40 数据根目录")
    p.add_argument("--out_json", type=str, default="", help="结果写入 JSON 路径")
    p.add_argument("--run_inference", action="store_true", help="是否运行分类推理（与 inference_and_kan_analysis 一致）")
    p.add_argument("--no_inference", action="store_true", help="仅做 IoU/radii 分析，不跑推理")
    p.add_argument("--checkpoint", type=str, default="checkpoints/pointKANElite-pointkan_elite_exp-42",
                   help="推理用检查点目录或 best_checkpoint.pth 路径")
    p.add_argument("--inference_batch_size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def run_iou_analysis(args):
    device = torch.device(args.device)
    ds = ModelNet40Wrapper(partition="test", num_points=args.num_points, root=args.root)
    n_total = len(ds)
    n_samples = min(args.max_samples, n_total) if args.max_samples > 0 else n_total
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
    iou_cf_v_list = []   # coarse_fine vs vanilla
    iou_ad_v_list = []   # adaptive vs vanilla
    radii_mse_cf_v_list = []
    radii_mse_ad_v_list = []
    n_batches = 0
    n_pts_processed = 0

    pbar = tqdm(loader, desc="KNN IoU & radii", ncols=100)
    for pts, _ in pbar:
        if args.max_samples > 0 and n_pts_processed >= args.max_samples:
            break
        pts = pts.to(device, dtype=torch.float32)
        B = pts.size(0)
        if args.max_samples > 0 and n_pts_processed + B > args.max_samples:
            pts = pts[: args.max_samples - n_pts_processed]
            B = pts.size(0)
        n_pts_processed += B

        # Vanilla
        _, _, knn_vanilla, dist_vanilla, radii_vanilla = fps_with_knn_vanilla(
            pts, K, k_neighbors=k, seed="first"
        )
        # Adaptive
        _, _, knn_adaptive, dist_adaptive, radii_adaptive, _, _ = fps_with_knn_adaptive(
            pts, K, k_neighbors=k, init_threshold=0.1, max_iter=10, tol=args.tol, seed="first"
        )
        # Coarse-fine
        _, _, knn_cf, dist_cf, radii_cf, _ = fps_with_knn_coarse_fine(
            pts, K, k_neighbors=k, init_threshold=0.1, max_iter=10,
            min_candidates_ratio=args.min_candidates_ratio, seed="first"
        )

        iou_cf_v = knn_neighbor_set_iou(knn_cf, knn_vanilla)
        iou_ad_v = knn_neighbor_set_iou(knn_adaptive, knn_vanilla)
        iou_cf_v_list.append(iou_cf_v.cpu().numpy())
        iou_ad_v_list.append(iou_ad_v.cpu().numpy())

        ok = torch.isfinite(radii_vanilla) & torch.isfinite(radii_cf) & torch.isfinite(radii_adaptive)
        if ok.any():
            mse_cf_v = ((radii_cf - radii_vanilla).float() ** 2).masked_select(ok).mean().item()
            mse_ad_v = ((radii_adaptive - radii_vanilla).float() ** 2).masked_select(ok).mean().item()
        else:
            mse_cf_v = float("nan")
            mse_ad_v = float("nan")
        radii_mse_cf_v_list.append(mse_cf_v)
        radii_mse_ad_v_list.append(mse_ad_v)
        n_batches += 1

    iou_cf_v_all = np.concatenate(iou_cf_v_list, axis=0)
    iou_ad_v_all = np.concatenate(iou_ad_v_list, axis=0)
    # 每个样本有 K 个 anchor，取平均
    iou_cf_v_per_sample = iou_cf_v_all.mean(axis=1)
    iou_ad_v_per_sample = iou_ad_v_all.mean(axis=1)

    results = {
        "n_samples": int(iou_cf_v_per_sample.size),
        "n_total_test": n_total,
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
    }
    return results


def run_inference_evaluation(args):
    """与 inference_and_kan_analysis 一致的推理评估，返回 eval_result dict。"""
    from inference_and_kan_analysis import (
        get_checkpoint_path,
        build_model_and_load_checkpoint,
        run_evaluation,
    )
    from data import ModelNet40
    from utils import cal_loss

    device = torch.device(args.device)
    checkpoint_path = get_checkpoint_path(args.checkpoint)
    net = build_model_and_load_checkpoint(checkpoint_path, device)
    test_dataset = ModelNet40(partition="test", num_points=args.num_points)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.inference_batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
    )
    eval_result = run_evaluation(net, test_loader, cal_loss, device)
    return eval_result


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    do_inference = args.run_inference and not args.no_inference

    # ---------- 1) IoU 与 radii 分析 ----------
    print("=" * 60)
    print("KNN 粗筛+细筛 与 Vanilla KNN IoU / radii 分析")
    print("=" * 60)
    t0 = time.perf_counter()
    iou_results = run_iou_analysis(args)
    t1 = time.perf_counter()
    print(f"  分析样本数: {iou_results['n_samples']} (测试集总数: {iou_results['n_total_test']})")
    print(f"  FPS 步数 K: {iou_results['fps_steps']}, 邻居数 k: {iou_results['k_neighbors']}")
    print(f"  粗筛+细筛 min_candidates_ratio: {iou_results['min_candidates_ratio']}")
    print("  --- IoU (相对 Vanilla) ---")
    print(f"  Coarse-Fine vs Vanilla: mean={iou_results['iou_coarse_fine_vs_vanilla']['mean']:.4f} "
          f"std={iou_results['iou_coarse_fine_vs_vanilla']['std']:.4f} "
          f"[{iou_results['iou_coarse_fine_vs_vanilla']['min']:.4f}, {iou_results['iou_coarse_fine_vs_vanilla']['max']:.4f}]")
    print(f"  Adaptive vs Vanilla:    mean={iou_results['iou_adaptive_vs_vanilla']['mean']:.4f} "
          f"std={iou_results['iou_adaptive_vs_vanilla']['std']:.4f} "
          f"[{iou_results['iou_adaptive_vs_vanilla']['min']:.4f}, {iou_results['iou_adaptive_vs_vanilla']['max']:.4f}]")
    print("  --- Radii MSE (相对 Vanilla) ---")
    print(f"  Coarse-Fine vs Vanilla: {iou_results['radii_mse_coarse_fine_vs_vanilla']:.6e}")
    print(f"  Adaptive vs Vanilla:    {iou_results['radii_mse_adaptive_vs_vanilla']:.6e}")
    print(f"  耗时: {t1 - t0:.2f}s")
    print("=" * 60)

    # ---------- 2) 可选：分类推理 ----------
    if do_inference:
        print("\n" + "=" * 60)
        print("ModelNet40 分类推理 (Vanilla 模型，与 inference_and_kan_analysis 一致)")
        print("=" * 60)
        try:
            eval_result = run_inference_evaluation(args)
            print(f"  测试样本数: {eval_result['num_samples']}")
            print(f"  Test Loss: {eval_result['loss']:.4f}")
            print(f"  Test Acc (accuracy): {eval_result['acc']:.2f}%")
            print(f"  Test Acc (balanced): {eval_result['acc_avg']:.2f}%")
            iou_results["inference"] = {
                "num_samples": eval_result["num_samples"],
                "loss": eval_result["loss"],
                "acc": eval_result["acc"],
                "acc_avg": eval_result["acc_avg"],
            }
        except FileNotFoundError as e:
            print(f"  推理跳过（未找到检查点或数据）: {e}")
            iou_results["inference"] = {"error": str(e)}
        except Exception as e:
            print(f"  推理异常: {e}")
            iou_results["inference"] = {"error": str(e)}
        print("=" * 60)

    # ---------- 3) 写出 JSON ----------
    out_path = args.out_json.strip()
    if not out_path:
        out_path = os.path.join(SCRIPT_DIR, "knn_coarse_fine_iou_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(iou_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已写入: {out_path}")


if __name__ == "__main__":
    main()
