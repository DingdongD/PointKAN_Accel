#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KITTI 点云 FPS+KNN GPU 显存与计算开销分析
===========================================
功能：
  - 固定使用 KITTI 原始大点云（ROI 后约 4 万–6.5 万点），用 --num_points 控制每帧截取数
  - 扫描 FPS 采样点数 K（--fps_steps_list 1024 2048 4096 8192），测试不同 K 下的开销
  - 使用 torch.cuda.Event 获取精确 GPU 计时（毫秒级，避免 perf_counter 的异步误差）
  - 使用 torch.cuda.reset_peak_memory_stats / max_memory_allocated 跟踪每算子峰值显存
  - 可选 NVTX 标注（--nvtx），开启后配合 Nsight Systems 查看 CUDA timeline；
    不开启时脚本正常运行，输出 console 表格 + JSON，无额外依赖
  - 输出：console 逐档表格 + 汇总 + JSON 文件

用法：
  python run_knn_gpu_profile_kitti.py
  python run_knn_gpu_profile_kitti.py --fps_steps_list 1024 2048 4096 8192
  python run_knn_gpu_profile_kitti.py --nvtx          # 开启 NVTX（需同步运行 nsys profile）
  python run_knn_gpu_profile_kitti.py --data_dir /path/to/velodyne --max_samples 30
"""

import os
import sys
import json
import argparse
import time
from contextlib import contextmanager

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
MODELNET40_DIR = os.path.join(REPO_ROOT, "PointKAN-elite", "classification_ModelNet40")
for d in (SCRIPT_DIR, REPO_ROOT, MODELNET40_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from kitti_fps_exp import load_kitti_bin, roi_filter, find_bin_files, collect_point_counts_and_xyz_ranges
from spec_kan_fps import (
    fps_with_knn_adaptive,
    fps_with_knn_coarse_fine,
    fps_with_knn_vanilla,
    collate_points,
)
from run_knn_coarse_fine_eval_kitti import KITTIPointCloudWrapper

# --------------------------------------------------------------------------- #
# NVTX 支持
# torch.cuda.nvtx 在 PyTorch 2.x + CUDA 中内置，提供 range_push / range_pop
# 以及 @contextmanager range()。无 CUDA 环境时用空 stub 代替。
# --------------------------------------------------------------------------- #
try:
    import torch.cuda.nvtx as _torch_nvtx
    _HAS_NVTX = True
except Exception:
    _HAS_NVTX = False


@contextmanager
def _nvtx_range(name: str, enabled: bool):
    """统一的 NVTX 范围上下文，enabled=False 时为零开销 stub"""
    if enabled and _HAS_NVTX:
        _torch_nvtx.range_push(name)
        try:
            yield
        finally:
            _torch_nvtx.range_pop()
    else:
        yield


# --------------------------------------------------------------------------- #
# CUDA 计时与显存工具
# --------------------------------------------------------------------------- #
def _timed_call(fn, *args, device, use_nvtx=False, nvtx_label="op", **kwargs):
    """
    用 CUDA Event 精确计时执行 fn(*args, **kwargs)，同时记录峰值显存。
    返回 (elapsed_ms, mem_peak_mb, mem_delta_mb)
    """
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    mem_before = torch.cuda.memory_allocated(device)

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)

    with _nvtx_range(nvtx_label, enabled=use_nvtx):
        s.record()
        fn(*args, **kwargs)
        e.record()

    torch.cuda.synchronize(device)
    elapsed_ms = s.elapsed_time(e)
    mem_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    mem_delta_mb = (torch.cuda.memory_allocated(device) - mem_before) / (1024 ** 2)

    return elapsed_ms, mem_peak_mb, mem_delta_mb


def bytes_to_mb(b: int) -> float:
    return b / (1024 ** 2)


# --------------------------------------------------------------------------- #
# 单 batch 三变体 profile
# --------------------------------------------------------------------------- #
@torch.no_grad()
def profile_one_batch(pts: torch.Tensor, K: int, k: int,
                      min_candidates_ratio: float,
                      use_nvtx: bool, device: torch.device) -> dict:
    """
    对一个 batch (B, N, 3) 执行 vanilla / adaptive / coarse_fine 三种 FPS+KNN，
    分别记录精确 GPU 时延和峰值显存。
    """
    N = pts.shape[1]
    K_eff = min(K, N)

    configs = [
        ("vanilla",     fps_with_knn_vanilla,
         dict(k_neighbors=k, seed="first")),
        ("adaptive",    fps_with_knn_adaptive,
         dict(k_neighbors=k, init_threshold=2.4, max_iter=10, tol=0.05, seed="first")),
        ("coarse_fine", fps_with_knn_coarse_fine,
         dict(k_neighbors=k, init_threshold=0.1, max_iter=10,
              min_candidates_ratio=min_candidates_ratio, seed="first")),
    ]

    results = {}
    for name, fn, kwargs in configs:
        elapsed, mem_peak, mem_delta = _timed_call(
            fn, pts, K_eff,
            device=device,
            use_nvtx=use_nvtx,
            nvtx_label=f"kitti_K{K_eff}_{name}",
            **kwargs,
        )
        results[name] = {
            "time_ms":      elapsed,
            "mem_peak_mb":  mem_peak,
            "mem_delta_mb": mem_delta,
        }
    return results


# --------------------------------------------------------------------------- #
# 扫描 fps_steps（K）列表
# --------------------------------------------------------------------------- #
def run_sweep(args) -> dict:
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    # 数据集统计
    bin_files = find_bin_files(args.data_dir)
    if not bin_files:
        raise FileNotFoundError(f"未找到 .bin 文件: {args.data_dir}")

    counts, _, _ = collect_point_counts_and_xyz_ranges(
        args.data_dir, do_roi=True, max_frames=-1,
        load_fn=load_kitti_bin, find_fn=find_bin_files,
    )
    counts = np.array(counts, dtype=np.int64)
    pts_real_min, pts_real_max = int(counts.min()), int(counts.max())
    pts_real_mean = float(counts.mean())
    print(f"数据集帧数: {len(counts)}, 每帧点数(ROI后): "
          f"min={pts_real_min:,}  max={pts_real_max:,}  mean={pts_real_mean:,.0f}")

    # 实际每帧点数（固定 num_points 模拟原始大点云，上限不超过数据集实际最大值）
    num_points_eff = min(args.num_points, pts_real_max)
    print(f"固定 num_points={num_points_eff:,}（原始大点云规模，超出则随机裁剪，不足则填充）\n")

    # 构建 Dataset / DataLoader（只建一次，各 K 共用同一批点云）
    ds = KITTIPointCloudWrapper(
        args.data_dir,
        num_points=num_points_eff,
        do_roi=True,
        max_frames=args.max_samples if args.max_samples > 0 else -1,
        seed=args.seed,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_points,
    )
    # 预先加载所有 batch 到 GPU，避免 IO 时间混入计时
    all_batches = []
    n_done = 0
    for pts, _ in loader:
        if args.max_samples > 0 and n_done >= args.max_samples:
            break
        pts = pts.to(device, dtype=torch.float32)
        B = pts.size(0)
        if args.max_samples > 0 and n_done + B > args.max_samples:
            pts = pts[: args.max_samples - n_done]
        all_batches.append(pts)
        n_done += pts.size(0)
    print(f"已加载 {n_done} 帧到 GPU（{device}），开始 K 扫描...\n")

    variants = ["vanilla", "adaptive", "coarse_fine"]
    k = args.k_neighbors

    # 表头
    col_w = 89
    header = (
        f"{'K(fps)':>8} | {'variant':>12} | "
        f"{'time_ms':>10} | {'±std':>8} | "
        f"{'peak_mem_MB':>12} | {'delta_mem_MB':>13} | "
        f"{'time_frac%':>11} | {'mem_frac%':>10}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    all_results = {}

    for K in args.fps_steps_list:
        accum = {v: {"times": [], "mem_peaks": [], "mem_deltas": []} for v in variants}

        for pts in tqdm(all_batches, desc=f"  K={K:>5}", ncols=80, leave=False):
            res = profile_one_batch(
                pts, K, k,
                min_candidates_ratio=args.min_candidates_ratio,
                use_nvtx=args.nvtx,
                device=device,
            )
            for v in variants:
                accum[v]["times"].append(res[v]["time_ms"])
                accum[v]["mem_peaks"].append(res[v]["mem_peak_mb"])
                accum[v]["mem_deltas"].append(res[v]["mem_delta_mb"])

        # 汇总
        summary = {}
        total_time = sum(np.mean(accum[v]["times"]) for v in variants)
        total_mem  = sum(np.mean(accum[v]["mem_peaks"]) for v in variants)

        for v in variants:
            t_arr = np.array(accum[v]["times"])
            m_arr = np.array(accum[v]["mem_peaks"])
            d_arr = np.array(accum[v]["mem_deltas"])
            t_mean = float(t_arr.mean())
            m_mean = float(m_arr.mean())
            summary[v] = {
                "time_ms_mean":      t_mean,
                "time_ms_std":       float(t_arr.std()),
                "mem_peak_mb_mean":  m_mean,
                "mem_peak_mb_std":   float(m_arr.std()),
                "mem_delta_mb_mean": float(d_arr.mean()),
                "time_frac":         t_mean / total_time if total_time > 0 else 0.0,
                "mem_peak_frac":     m_mean / total_mem  if total_mem  > 0 else 0.0,
                # 计算强度：peak_mem / time，越高说明内存带宽越是瓶颈
                "mem_intensity_mb_per_ms": m_mean / t_mean if t_mean > 0 else 0.0,
            }

        all_results[str(K)] = {
            "fps_steps_K":       K,
            "num_points":        num_points_eff,
            "k_neighbors":       k,
            "n_frames_evaluated": n_done,
            "variants":          summary,
        }

        # 打印表格行
        for v in variants:
            s = summary[v]
            print(
                f"{K:>8} | {v:>12} | "
                f"{s['time_ms_mean']:>9.2f}ms | "
                f"{s['time_ms_std']:>7.2f}ms | "
                f"{s['mem_peak_mb_mean']:>11.1f}MB | "
                f"{s['mem_delta_mb_mean']:>12.1f}MB | "
                f"{s['time_frac']:>10.1%} | "
                f"{s['mem_peak_frac']:>9.1%}"
            )
        print(sep)

    return all_results, {
        "num_points_effective": num_points_eff,
        "pts_real_min": pts_real_min,
        "pts_real_max": pts_real_max,
        "pts_real_mean": pts_real_mean,
    }


# --------------------------------------------------------------------------- #
# 汇总：随 K 增大的趋势
# --------------------------------------------------------------------------- #
def print_summary(all_results: dict):
    variants = ["vanilla", "adaptive", "coarse_fine"]
    print("\n" + "=" * 72)
    print("  汇总：各 K（FPS 采样数）下三种变体时延 & 显存开销")
    print("=" * 72)
    print(f"  {'K':>6}  {'variant':<14}  {'time_ms':>9}  {'peak_MB':>9}  "
          f"{'time%':>7}  {'mem%':>7}  {'MB/ms':>7}")
    print("  " + "-" * 68)
    for K_str, rec in all_results.items():
        for v in variants:
            s = rec["variants"][v]
            print(f"  {K_str:>6}  {v:<14}  "
                  f"{s['time_ms_mean']:>9.2f}  "
                  f"{s['mem_peak_mb_mean']:>9.1f}  "
                  f"{s['time_frac']:>7.1%}  "
                  f"{s['mem_peak_frac']:>7.1%}  "
                  f"{s['mem_intensity_mb_per_ms']:>7.2f}")
        print("  " + "-" * 68)

    print("\n  解读：")
    print("   · time% = 该变体时延 / 三种变体时延之和（衡量相对计算开销）")
    print("   · mem%  = 该变体峰值显存 / 三种变体峰值显存之和（衡量相对显存开销）")
    print("   · MB/ms = 峰值显存(MB) / 时延(ms)，越大说明内存带宽压力越重")
    print("   · K 增大 → FPS+KNN 计算量 & 显存均线性增长，可对比三种方法的缩放规律")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="KITTI FPS+KNN GPU 显存与计算开销分析（扫描 FPS 采样数 K）"
    )
    p.add_argument("--data_dir", type=str,
                   default="/workspace/VoxelNet/KITTI_Mini/training/velodyne",
                   help="KITTI velodyne 目录")
    p.add_argument("--num_points", type=int, default=50000,
                   help="固定每帧点数（模拟原始 KITTI 大点云，ROI 后约 4-6.5 万点）")
    p.add_argument("--fps_steps_list", type=int, nargs="+",
                   default=[1024, 2048, 4096, 8192],
                   help="扫描的 FPS 采样数 K 列表")
    p.add_argument("--k_neighbors", type=int, default=32, help="KNN 邻居数 k")
    p.add_argument("--min_candidates_ratio", type=float, default=1.2,
                   help="coarse_fine 粗筛候选比例")
    p.add_argument("--batch_size", type=int, default=1,
                   help="batch size，大点云建议 1")
    p.add_argument("--max_samples", type=int, default=20,
                   help="最多评估帧数（0=全部）；建议 10-30 快速测试")
    p.add_argument("--device", type=str,
                   default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--nvtx", action="store_true",
                   help="开启 NVTX 范围标注（配合 nsys profile 使用）")
    p.add_argument("--out_json", type=str, default="",
                   help="结果 JSON 输出路径（默认 ./knn_gpu_profile_kitti.json）")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("[错误] 未检测到 CUDA，本脚本需要 GPU 环境。")
        return

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    torch.cuda.manual_seed(args.seed)

    nvtx_status = "已开启（Nsight Systems 可见）" if (args.nvtx and _HAS_NVTX) else (
        "已请求但 torch.cuda.nvtx 不可用" if args.nvtx else "未开启（加 --nvtx 启用）"
    )

    print("=" * 72)
    print("  KITTI FPS+KNN GPU 显存 & 计算开销分析")
    print(f"  device:          {args.device}  ({torch.cuda.get_device_name(dev)})")
    print(f"  num_points:      {args.num_points:,}（固定，原始大点云规模）")
    print(f"  fps_steps_list:  {args.fps_steps_list}（扫描维度 = FPS 采样数 K）")
    print(f"  k_neighbors:     {args.k_neighbors}")
    print(f"  max_samples:     {args.max_samples if args.max_samples > 0 else '全部'} 帧/档")
    print(f"  NVTX:            {nvtx_status}")
    print("=" * 72)

    t0 = time.perf_counter()
    all_results, dataset_meta = run_sweep(args)
    t1 = time.perf_counter()

    print_summary(all_results)
    print(f"\n  总耗时: {t1 - t0:.1f}s")

    output = {
        "meta": {
            "gpu":              torch.cuda.get_device_name(dev),
            "total_gpu_mem_gb": round(
                torch.cuda.get_device_properties(dev).total_memory / 1024**3, 2),
            "device":           args.device,
            "nvtx_enabled":     args.nvtx and _HAS_NVTX,
            "num_points_fixed": dataset_meta["num_points_effective"],
            "fps_steps_list":   args.fps_steps_list,
            "k_neighbors":      args.k_neighbors,
            "batch_size":       args.batch_size,
            "max_samples":      args.max_samples,
            "data_dir":         args.data_dir,
            "dataset_pts_min":  dataset_meta["pts_real_min"],
            "dataset_pts_max":  dataset_meta["pts_real_max"],
            "dataset_pts_mean": round(dataset_meta["pts_real_mean"], 1),
        },
        "results": all_results,
    }

    out_path = args.out_json.strip() or os.path.join(SCRIPT_DIR, "knn_gpu_profile_kitti.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已写入: {out_path}")


if __name__ == "__main__":
    main()
