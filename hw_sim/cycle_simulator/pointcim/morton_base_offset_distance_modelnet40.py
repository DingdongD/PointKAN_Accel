#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
莫顿（Morton / Z-order）排序 + 分块 Base+Offset 量化重建后，对「点到查询点的平方距离」误差进行统计。

用途（与 PointCIM / 硬件仿真思路对齐）：
  1. 点云按莫顿码排序，使空间上相近的点在内存中更连续，便于分块与早期剪枝。
  2. 每块用「块内最小值作 base + 量化 offset」压缩坐标；重建坐标后，用 pairwise 平方距离
     对比原始点云，衡量该表示对距离计算的影响。

数据：默认加载与 inference.sh 相同的 ModelNet40 管线（classification_ModelNet40/data.py），
      即 HDF5 中的 modelnet40_ply_hdf5_2048；测试集 partition='test' 不做随机平移/打乱，
      与推理脚本中 test_dataset 一致。

运行示例（在仓库任意目录，或直接本文件所在目录）：
  python morton_base_offset_distance_modelnet40.py --num_samples 64 --num_points 1024
  python morton_base_offset_distance_modelnet40.py --no_morton   # 对照：不排序
  python morton_base_offset_distance_modelnet40.py --repeat_query_draws 5  # 查询随机性敏感性
  python morton_base_offset_distance_modelnet40.py --violation_taus 0.001,0.01,0.05  # 自定义违约阈值
  python morton_base_offset_distance_modelnet40.py --no_robust     # 关闭尾部分析（略快）
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 与 inference.sh / inference_and_kan_analysis.py 中 ModelNet40 默认一致的路径
# （data.py 内 load_data 使用固定 DATA_DIR，需保证该路径下已有 ply_data_*.h5）
_DEFAULT_MODELNET40_CODE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../PointKAN-elite/classification_ModelNet40")
)

# 鲁棒性分析默认：|Δd²| 超过该阈值视为「违约」（可改为与场景相关的半径平方等）
_DEFAULT_VIOLATION_TAUS = (1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1)
# 多档 |d_ref| 下限，用于掩码相对误差（避免极小真值主导）
_DEFAULT_REL_MASK_FLOORS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)


def quantize_uniform(
    x: np.ndarray, x_min: float, x_max: float, bits: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    均匀量化到整数格点再反量化回浮点。

    参数:
        x: 待量化数组（任意形状，按元素独立量化）。
        x_min, x_max: 量化动态范围上下界（超出部分会被 clip）。
        bits: 量化比特数；有效等级数为 2^bits - 1（与常见「含两端」均匀划分一致）。

    返回:
        q: int32 量化索引。
        x_hat: 反量化后的浮点重建。
    """
    if bits <= 0:
        raise ValueError("bits must be > 0")
    levels = 2**bits - 1
    x_clipped = np.clip(x, x_min, x_max)
    scale = levels / (x_max - x_min + 1e-12)
    q = np.round((x_clipped - x_min) * scale)
    x_hat = q / scale + x_min
    return q.astype(np.int32), x_hat


def interleave3(xi: int, yi: int, zi: int, bits: int) -> int:
    """
    三维 Morton 码（Z-order）：将三个坐标轴上的二进制位交错编码到一个整数。

    约定（与常见实现一致）：
      - 第 b 位：x 放在 3b+2，y 放在 3b+1，z 放在 3b+0。
      - 仅使用每个坐标的低 `bits` 位；调用方应保证 xi,yi,zi 已落在 [0, 2^bits-1]。

    注意:
        当 bits=10 时，码长最多 30 位，可放入 np.uint64；bits 更大时需确认不溢出。
    """
    code = 0
    for b in range(bits):
        code |= ((xi >> b) & 1) << (3 * b + 2)
        code |= ((yi >> b) & 1) << (3 * b + 1)
        code |= ((zi >> b) & 1) << (3 * b + 0)
    return code


def morton_sort_points(
    points: np.ndarray, bits: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将点云归一化到各维 [0, 2^bits-1] 的整数格，按 Morton 码排序。

    参数:
        points: [N, 3] 浮点坐标。
        bits: 用于 Morton 的格点分辨率位数；越大空间划分越细，排序更精细但更慢。

    返回:
        points_sorted: 按 Morton 码升序排列后的点云。
        order: argsort 索引，满足 points_sorted = points[order]。
        codes_sorted: 排序后每条点对应的 Morton 码（uint64）。
    """
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    norm = (points - pmin) / (pmax - pmin + 1e-12)
    max_int = 2**bits - 1
    ints = np.floor(norm * max_int).astype(np.int64)
    np.clip(ints, 0, max_int, out=ints)

    codes = np.empty(points.shape[0], dtype=np.uint64)
    for i in range(points.shape[0]):
        codes[i] = interleave3(int(ints[i, 0]), int(ints[i, 1]), int(ints[i, 2]), bits)

    order = np.argsort(codes)
    return points[order], order, codes[order]


def build_base_offset(
    points: np.ndarray,
    block_size: int = 64,
    base_bits: int = 16,
    offset_bits: int = 6,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    分块 Base + Offset 编码的简化仿真（与 LUT base_offset 思路类似）。

    每个块:
      - base = 块内各维最小值；
      - offset = 点坐标 - base（非负）；
      - base 在「全局 min/max」范围内用 base_bits 均匀量化；
      - 各维 offset 在 [0, off_max[d]] 内用 offset_bits 均匀量化。

    返回:
        recon: 与 points 同形状的重建坐标。
        meta: 每块的统计信息（原始范围、offset 上界等），用于分析压缩/剪枝潜力。
    """
    n = points.shape[0]
    recon = np.zeros_like(points)
    meta: List[Dict[str, Any]] = []
    global_min = points.min(axis=0)
    global_max = points.max(axis=0)

    for s in range(0, n, block_size):
        e = min(s + block_size, n)
        blk = points[s:e]

        base = blk.min(axis=0)
        offset = blk - base
        off_max = offset.max(axis=0)

        _, base_hat = quantize_uniform(base, global_min, global_max, base_bits)

        off_hat = np.zeros_like(offset)
        for d in range(3):
            if off_max[d] < 1e-15:
                off_hat[:, d] = 0.0
            else:
                _, hat = quantize_uniform(offset[:, d], 0.0, off_max[d], offset_bits)
                off_hat[:, d] = hat

        blk_hat = base_hat + off_hat
        recon[s:e] = blk_hat

        raw_range = blk.max(axis=0) - blk.min(axis=0)
        meta.append(
            {
                "start": s,
                "end": e,
                "base": base,
                "base_hat": base_hat,
                "offset_max": off_max,
                "raw_range": raw_range,
                "mean_offset_abs": np.mean(np.abs(offset), axis=0),
            }
        )

    return recon, meta


def pairwise_sqdist_to_query(points: np.ndarray, query: np.ndarray) -> np.ndarray:
    """点云中每个点到单个 query 的平方距离，形状 [N]。"""
    diff = points - query[None, :]
    return np.sum(diff * diff, axis=1)


def _violation_rates(abs_err_flat: np.ndarray, taus: Tuple[float, ...]) -> Dict[str, float]:
    """违约率：P(|Δd²| > τ)。对阈值列表返回可读键名。"""
    rates: Dict[str, float] = {}
    for tau in taus:
        key = f"tau_{tau:g}"
        rates[key] = float(np.mean(abs_err_flat > tau))
    return rates


def _average_robust_dicts(robust_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """对同一帧上多次随机 query 的 robust 字典做逐字段平均（违约率、分位数等标量）。"""
    if not robust_list:
        return {}
    scalar_keys = (
        "abs_d2_median",
        "abs_d2_mean",
        "abs_d2_std",
        "abs_d2_max",
        "abs_d2_p90",
        "abs_d2_p95",
        "abs_d2_p99",
        "abs_d2_p99_9",
        "per_query_abs_max_mean",
        "per_query_abs_max_p95",
        "coord_Linf_mean",
        "coord_Linf_p99",
        "coord_Linf_max",
    )
    out: Dict[str, Any] = {}
    for k in scalar_keys:
        vals = [float(r[k]) for r in robust_list if k in r]
        if vals:
            out[k] = float(np.mean(vals))
    # 违约率逐 τ 平均
    v0 = robust_list[0].get("violation_rates", {})
    out["violation_rates"] = {}
    for tau_key in v0:
        out["violation_rates"][tau_key] = float(
            np.mean([float(r["violation_rates"][tau_key]) for r in robust_list])
        )
    # 多档掩码相对误差：对子字典逐字段平均（跳过 None）
    rel0 = robust_list[0].get("rel_err_masked_multi", {})
    out["rel_err_masked_multi"] = {}
    for ek in rel0:
        subs = [r["rel_err_masked_multi"].get(ek) for r in robust_list]
        subs = [s for s in subs if s is not None]
        if not subs:
            out["rel_err_masked_multi"][ek] = None
            continue
        out["rel_err_masked_multi"][ek] = {
            "count": int(round(float(np.mean([s["count"] for s in subs])))),
            "mean": float(np.mean([s["mean"] for s in subs])),
            "median": float(np.mean([s["median"] for s in subs])),
            "p90": float(np.mean([s["p90"] for s in subs])),
            "p95": float(np.mean([s["p95"] for s in subs])),
            "p99": float(np.mean([s["p99"] for s in subs])),
        }
    return out


def evaluate_single_cloud(
    points: np.ndarray,
    block_size: int = 64,
    base_bits: int = 16,
    offset_bits: int = 6,
    use_morton: bool = True,
    morton_bits: int = 10,
    num_queries: int = 32,
    rng: Optional[np.random.Generator] = None,
    violation_taus: Tuple[float, ...] = _DEFAULT_VIOLATION_TAUS,
    rel_mask_floors: Tuple[float, ...] = _DEFAULT_REL_MASK_FLOORS,
    compute_robust: bool = True,
) -> Dict[str, Any]:
    """
    对单帧点云：可选莫顿排序 → Base+Offset 重建 → 相对随机查询点的距离误差统计。

    距离误差定义（与用户提供 simulate 一致）:
      - d_ref: 原始排序后点云到 query 的平方距离；
      - d_hat: 重建点云到同一 query 的平方距离；
      - 对每个 query 报告 mean(|d_ref - d_hat|) 与 mean(|d_ref-d_hat|/(|d_ref|+eps))。

    若 compute_robust=True，额外汇总所有 (query, point) 对的 |Δd²| 分布（分位数、最大、违约率），
    以及多档 |d_ref| 掩码下的相对误差分位数，用于尾部分析与硬件误差预算。
    """
    if rng is None:
        rng = np.random.default_rng(0)

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be [N, 3]")

    if use_morton:
        points_sorted, order, _ = morton_sort_points(pts, bits=morton_bits)
    else:
        points_sorted = pts.copy()
        order = np.arange(pts.shape[0])

    recon, meta = build_base_offset(
        points_sorted,
        block_size=block_size,
        base_bits=base_bits,
        offset_bits=offset_bits,
    )

    coord_mse = float(np.mean((points_sorted - recon) ** 2))
    coord_mae = float(np.mean(np.abs(points_sorted - recon)))

    n = points_sorted.shape[0]
    q_count = min(num_queries, n)
    query_idx = rng.choice(n, size=q_count, replace=False)
    queries = points_sorted[query_idx]

    dist_abs_err: List[float] = []
    dist_rel_err: List[float] = []
    dist_rel_err_masked: List[float] = []
    early_stop_proxy: List[float] = []
    abs_flat_chunks: List[np.ndarray] = []
    per_query_max_abs: List[float] = []
    rel_masked_chunks: Dict[float, List[np.ndarray]] = {eps: [] for eps in rel_mask_floors}

    global_range = points_sorted.max(axis=0) - points_sorted.min(axis=0)

    for q in queries:
        d_ref = pairwise_sqdist_to_query(points_sorted, q)
        d_hat = pairwise_sqdist_to_query(recon, q)
        abs_err = np.abs(d_ref - d_hat)
        # 与原始 simulate 一致：相对误差 = |Δd²| / (|d_ref|+eps)。当某点与 query 几乎重合时
        # d_ref→0，分母极小，相对误差会数值爆炸（不代表坐标重建一定很差）。
        rel_err = abs_err / (np.abs(d_ref) + 1e-12)
        dist_abs_err.append(float(np.mean(abs_err)))
        dist_rel_err.append(float(np.mean(rel_err)))
        mask = np.abs(d_ref) > 1e-4
        if np.any(mask):
            dist_rel_err_masked.append(float(np.mean(rel_err[mask])))
        else:
            dist_rel_err_masked.append(float("nan"))

        if compute_robust:
            abs_flat_chunks.append(abs_err.astype(np.float64))
            per_query_max_abs.append(float(np.max(abs_err)))
            for eps in rel_mask_floors:
                m = np.abs(d_ref) > eps
                if np.any(m):
                    rel_masked_chunks[eps].append(rel_err[m].astype(np.float64))

        per_blk_ratio = []
        for m in meta:
            ratio = float(np.max(m["offset_max"] / (global_range + 1e-12)))
            per_blk_ratio.append(ratio)
        early_stop_proxy.append(float(1.0 - np.mean(per_blk_ratio)))

    out: Dict[str, Any] = {
        "coord_mse": coord_mse,
        "coord_mae": coord_mae,
        "mean_dist_abs_err": float(np.mean(dist_abs_err)),
        "mean_dist_rel_err": float(np.mean(dist_rel_err)),
        "mean_dist_rel_err_masked": float(np.nanmean(dist_rel_err_masked)),
        "early_stop_proxy": float(np.mean(early_stop_proxy)),
        "num_points": n,
        "order": order,
    }

    if compute_robust and abs_flat_chunks:
        all_abs = np.concatenate(abs_flat_chunks, axis=0)
        out["robust"] = {
            "abs_d2_median": float(np.median(all_abs)),
            "abs_d2_mean": float(np.mean(all_abs)),
            "abs_d2_std": float(np.std(all_abs)),
            "abs_d2_max": float(np.max(all_abs)),
            "abs_d2_p90": float(np.percentile(all_abs, 90.0)),
            "abs_d2_p95": float(np.percentile(all_abs, 95.0)),
            "abs_d2_p99": float(np.percentile(all_abs, 99.0)),
            "abs_d2_p99_9": float(np.percentile(all_abs, 99.9)),
            "violation_rates": _violation_rates(all_abs, violation_taus),
            "per_query_abs_max_mean": float(np.mean(per_query_max_abs)),
            "per_query_abs_max_p95": float(np.percentile(per_query_max_abs, 95.0)),
        }
        # 坐标重建：逐点 L∞ 误差尾部分布（与距离误差互补）
        coord_abs = np.max(np.abs(points_sorted - recon), axis=1)
        out["robust"]["coord_Linf_mean"] = float(np.mean(coord_abs))
        out["robust"]["coord_Linf_p99"] = float(np.percentile(coord_abs, 99.0))
        out["robust"]["coord_Linf_max"] = float(np.max(coord_abs))

        rel_multi: Dict[str, Any] = {}
        for eps in rel_mask_floors:
            parts = rel_masked_chunks[eps]
            if not parts:
                rel_multi[f"eps_{eps:g}"] = None
                continue
            r = np.concatenate(parts, axis=0)
            rel_multi[f"eps_{eps:g}"] = {
                "count": int(r.size),
                "mean": float(np.mean(r)),
                "median": float(np.median(r)),
                "p90": float(np.percentile(r, 90.0)),
                "p95": float(np.percentile(r, 95.0)),
                "p99": float(np.percentile(r, 99.0)),
            }
        out["robust"]["rel_err_masked_multi"] = rel_multi

    return out


def import_modelnet40_dataset(modelnet40_code_dir: str):
    """
    动态将 classification_ModelNet40 加入 sys.path，并导入 data.ModelNet40。

    说明:
        data.py 中 load_data 的 HDF5 根目录写死为 /workspace/VoxelNet/pc_data/；
        与 inference 使用同一数据副本时，无需再传数据路径。
    """
    code_dir = os.path.abspath(modelnet40_code_dir)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    from data import ModelNet40  # noqa: E402 — 需在插入 sys.path 之后导入

    return ModelNet40


def evaluate_cloud_with_query_repeat(
    points: np.ndarray,
    block_size: int,
    base_bits: int,
    offset_bits: int,
    use_morton: bool,
    morton_bits: int,
    num_queries: int,
    cloud_index: int,
    base_seed: int,
    repeat_query_draws: int,
    violation_taus: Tuple[float, ...],
    rel_mask_floors: Tuple[float, ...],
    compute_robust: bool,
) -> Dict[str, Any]:
    """
    对单帧可重复随机抽取 query：repeat_query_draws>1 时衡量「查询点随机性」对标量指标的影响，
    并对 robust 子字典做逐字段平均，便于与单次抽样的尾部分解同时汇报。
    """
    outs: List[Dict[str, Any]] = []
    for r in range(repeat_query_draws):
        rng = np.random.default_rng(base_seed + cloud_index * 100_003 + r * 50_021)
        outs.append(
            evaluate_single_cloud(
                points,
                block_size=block_size,
                base_bits=base_bits,
                offset_bits=offset_bits,
                use_morton=use_morton,
                morton_bits=morton_bits,
                num_queries=num_queries,
                rng=rng,
                violation_taus=violation_taus,
                rel_mask_floors=rel_mask_floors,
                compute_robust=compute_robust,
            )
        )
    if repeat_query_draws == 1:
        return outs[0]

    merged = outs[0].copy()
    merged["mean_dist_abs_err"] = float(np.mean([o["mean_dist_abs_err"] for o in outs]))
    merged["mean_dist_rel_err"] = float(np.mean([o["mean_dist_rel_err"] for o in outs]))
    merged["mean_dist_rel_err_masked"] = float(np.nanmean([o["mean_dist_rel_err_masked"] for o in outs]))
    merged["early_stop_proxy"] = float(np.mean([o["early_stop_proxy"] for o in outs]))
    merged["query_repeat_std_mean_dist_abs"] = float(
        np.std([o["mean_dist_abs_err"] for o in outs])
    )
    if compute_robust and all("robust" in o for o in outs):
        merged["query_repeat_std_abs_d2_p99"] = float(
            np.std([o["robust"]["abs_d2_p99"] for o in outs])
        )
        merged["robust"] = _average_robust_dicts([o["robust"] for o in outs])
    elif compute_robust:
        merged["robust"] = outs[0]["robust"]
    return merged


def _print_robust_one_cloud(prefix: str, robust: Dict[str, Any]) -> None:
    """打印单帧鲁棒性摘要（ModelNet40 汇总前可先用于 synthetic）。"""
    print(f"{prefix}|Δd²| 分布: median={robust['abs_d2_median']:.6e}, "
          f"p90={robust['abs_d2_p90']:.6e}, p99={robust['abs_d2_p99']:.6e}, "
          f"p99.9={robust['abs_d2_p99_9']:.6e}, max={robust['abs_d2_max']:.6e}")
    print(
        f"{prefix}每 query 内 max|Δd²|: mean={robust['per_query_abs_max_mean']:.6e}, "
        f"p95={robust['per_query_abs_max_p95']:.6e}"
    )
    print(
        f"{prefix}坐标 L∞ 误差: mean={robust['coord_Linf_mean']:.6e}, "
        f"p99={robust['coord_Linf_p99']:.6e}, max={robust['coord_Linf_max']:.6e}"
    )
    vr = robust["violation_rates"]
    tau_line = ", ".join(f"{k}={v:.4f}" for k, v in sorted(vr.items()))
    print(f"{prefix}违约率 P(|Δd²|>τ)（本帧内）: {tau_line}")
    print(f"{prefix}掩码相对误差 |d_ref|>ε 时 r=|Δd²|/|d_ref| 的 p99（多档 ε）:")
    rel_keys = sorted(
        robust["rel_err_masked_multi"].keys(),
        key=lambda s: float(s.split("_", 1)[1]),
    )
    for ek in rel_keys:
        sub = robust["rel_err_masked_multi"][ek]
        if sub is None:
            print(f"    {ek}: (无样本)")
        else:
            print(
                f"    {ek}: n≈{sub['count']}, mean(r)={sub['mean']:.6e}, "
                f"p99(r)={sub['p99']:.6e}"
            )


def run_modelnet40_benchmark(
    partition: str = "test",
    num_points: int = 1024,
    num_samples: int = 256,
    seed: int = 42,
    block_size: int = 64,
    base_bits: int = 16,
    offset_bits: int = 6,
    use_morton: bool = True,
    morton_bits: int = 10,
    num_queries: int = 32,
    modelnet40_code_dir: str = _DEFAULT_MODELNET40_CODE_DIR,
    repeat_query_draws: int = 1,
    violation_taus: Tuple[float, ...] = _DEFAULT_VIOLATION_TAUS,
    rel_mask_floors: Tuple[float, ...] = _DEFAULT_REL_MASK_FLOORS,
    compute_robust: bool = True,
) -> None:
    """
    在 ModelNet40 上抽取若干样本，聚合「坐标重建误差」与「距离误差」指标。

    使用 test 分区可避免训练集上的 translate_pointcloud / shuffle，便于可重复对比。

    compute_robust: 为 True 时增加尾部分位数、违约率、多档掩码相对误差及 query 重复敏感性。
    """
    ModelNet40 = import_modelnet40_dataset(modelnet40_code_dir)
    ds = ModelNet40(partition=partition, num_points=num_points)
    rng = np.random.default_rng(seed)
    n_total = len(ds)
    indices = np.arange(n_total)
    if num_samples < n_total:
        indices = rng.choice(indices, size=num_samples, replace=False)

    agg_mse: List[float] = []
    agg_mae: List[float] = []
    agg_d_abs: List[float] = []
    agg_d_rel: List[float] = []
    agg_d_rel_m: List[float] = []
    agg_early: List[float] = []
    agg_q_std_mean: List[float] = []
    agg_q_std_p99: List[float] = []
    agg_r_p99: List[float] = []
    agg_r_p99_9: List[float] = []
    agg_r_max: List[float] = []
    agg_coord_linf_p99: List[float] = []
    violation_accum: Dict[str, List[float]] = {}

    print("=" * 80)
    print(
        f"ModelNet40 partition={partition!r}, num_points={num_points}, "
        f"samples={len(indices)}/{n_total}"
    )
    print(
        f"block_size={block_size}, base_bits={base_bits}, offset_bits={offset_bits}, "
        f"use_morton={use_morton}, morton_bits={morton_bits}, num_queries={num_queries}, "
        f"repeat_query_draws={repeat_query_draws}, compute_robust={compute_robust}"
    )
    print("=" * 80)

    for k, idx in enumerate(indices):
        pc, label = ds[int(idx)]
        points = np.asarray(pc, dtype=np.float64)
        out = evaluate_cloud_with_query_repeat(
            points,
            block_size=block_size,
            base_bits=base_bits,
            offset_bits=offset_bits,
            use_morton=use_morton,
            morton_bits=morton_bits,
            num_queries=num_queries,
            cloud_index=k,
            base_seed=seed,
            repeat_query_draws=repeat_query_draws,
            violation_taus=violation_taus,
            rel_mask_floors=rel_mask_floors,
            compute_robust=compute_robust,
        )
        agg_mse.append(out["coord_mse"])
        agg_mae.append(out["coord_mae"])
        agg_d_abs.append(out["mean_dist_abs_err"])
        agg_d_rel.append(out["mean_dist_rel_err"])
        agg_d_rel_m.append(out["mean_dist_rel_err_masked"])
        agg_early.append(out["early_stop_proxy"])
        if repeat_query_draws > 1:
            agg_q_std_mean.append(float(out["query_repeat_std_mean_dist_abs"]))
            if compute_robust and "query_repeat_std_abs_d2_p99" in out:
                agg_q_std_p99.append(float(out["query_repeat_std_abs_d2_p99"]))
        if compute_robust and "robust" in out:
            rb = out["robust"]
            agg_r_p99.append(rb["abs_d2_p99"])
            agg_r_p99_9.append(rb["abs_d2_p99_9"])
            agg_r_max.append(rb["abs_d2_max"])
            agg_coord_linf_p99.append(rb["coord_Linf_p99"])
            for tk, tv in rb["violation_rates"].items():
                violation_accum.setdefault(tk, []).append(tv)

    print(f"\n在 {len(indices)} 个样本上的平均值 ± 标准差:\n")
    print(f"  坐标 MSE : {np.mean(agg_mse):.6e} ± {np.std(agg_mse):.6e}")
    print(f"  坐标 MAE : {np.mean(agg_mae):.6e} ± {np.std(agg_mae):.6e}")
    print(f"  距离绝对误差(mean |Δd²|): {np.mean(agg_d_abs):.6e} ± {np.std(agg_d_abs):.6e}")
    print(f"  距离相对误差(mean, 原始): {np.mean(agg_d_rel):.6e} ± {np.std(agg_d_rel):.6e}")
    print(
        f"  距离相对误差(|d_ref|>1e-4): {np.nanmean(agg_d_rel_m):.6e} ± {np.nanstd(agg_d_rel_m):.6e}"
    )
    print(f"  Early-stop proxy:         {np.mean(agg_early):.4f} ± {np.std(agg_early):.4f}")
    print()

    # 存储比特粗算（与用户提供公式一致）
    n = num_points
    num_blocks = int(np.ceil(n / block_size))
    orig_bits = n * 3 * 16
    bo_bits = num_blocks * 3 * base_bits + n * 3 * offset_bits
    print("存储 proxy（单帧，与点数/块大小相关）:")
    print(f"  假定原始坐标 16b×3×N : {orig_bits}")
    print(f"  Base+Offset 比特数   : {bo_bits}")
    print(f"  比值 bo/orig         : {bo_bits / orig_bits:.4f}")
    print()

    if compute_robust and agg_r_p99:
        print("-" * 80)
        print("鲁棒性误差分析（每帧：所有 query × 点 的 |Δd²|；ModelNet40：跨帧 mean±std）")
        print("-" * 80)
        print(
            f"  |Δd²| 分位数（跨云聚合）: p99 mean={np.mean(agg_r_p99):.6e} ± {np.std(agg_r_p99):.6e}, "
            f"p99.9 mean={np.mean(agg_r_p99_9):.6e} ± {np.std(agg_r_p99_9):.6e}"
        )
        print(
            f"  |Δd²| 最坏尾: max 的跨云 mean={np.mean(agg_r_max):.6e} ± {np.std(agg_r_max):.6e}"
        )
        print(
            f"  坐标 L∞ 重建 p99（跨云）: mean={np.mean(agg_coord_linf_p99):.6e} ± "
            f"{np.std(agg_coord_linf_p99):.6e}"
        )
        print("  违约率 P(|Δd²|>τ)（每云算一次再对云平均；用于硬件阈值预算）:")
        print("    （τ 很小时比例常接近 1：多数点对的 |Δd²| 仍大于极小常数，属预期现象）")
        for tk in sorted(violation_accum.keys(), key=lambda s: float(s.replace("tau_", ""))):
            vals = violation_accum[tk]
            if vals:
                print(f"    τ={tk.replace('tau_', '')}: mean={np.mean(vals):.4f} ± {np.std(vals):.4f}")
        if repeat_query_draws > 1 and agg_q_std_mean:
            print(
                f"  查询随机性（每云 repeat={repeat_query_draws}）: "
                f"mean(|Δd²|) 的 std 对云平均={np.mean(agg_q_std_mean):.6e} ± {np.std(agg_q_std_mean):.6e}"
            )
            if agg_q_std_p99:
                print(
                    f"  同上，对 p99(|Δd²|) 的 std 对云平均="
                    f"{np.mean(agg_q_std_p99):.6e} ± {np.std(agg_q_std_p99):.6e}"
                )
        print(
            "  说明: 违约率为「单帧内比例」再对多样本平均；若要做全局比例可把多帧 |Δd²| 拼在一起重算。"
        )
        print()


def simulate_synthetic(
    n: int = 2048,
    block_size: int = 64,
    base_bits: int = 16,
    offset_bits: int = 6,
    use_morton: bool = True,
    seed: int = 42,
) -> None:
    """合成聚类点云（用户原始 simulate），用于快速自检。"""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-0.8, 0.8, size=(8, 3))
    cluster_ids = rng.integers(0, len(centers), size=n)
    points = centers[cluster_ids] + 0.05 * rng.normal(size=(n, 3))
    points = np.clip(points, -1.0, 1.0)

    out = evaluate_single_cloud(
        points,
        block_size=block_size,
        base_bits=base_bits,
        offset_bits=offset_bits,
        use_morton=use_morton,
        num_queries=32,
        rng=rng,
    )

    pts_sorted = morton_sort_points(points)[0] if use_morton else points.copy()
    _, meta = build_base_offset(
        pts_sorted,
        block_size=block_size,
        base_bits=base_bits,
        offset_bits=offset_bits,
    )
    raw_ranges = np.array([m["raw_range"] for m in meta])
    offset_maxs = np.array([m["offset_max"] for m in meta])
    global_range = pts_sorted.max(axis=0) - pts_sorted.min(axis=0)

    print("=" * 80)
    print(
        f"[Synthetic] N={n}, block_size={block_size}, base_bits={base_bits}, "
        f"offset_bits={offset_bits}, use_morton={use_morton}"
    )
    print("=" * 80)
    print(f"Coordinate reconstruction MSE : {out['coord_mse']:.6e}")
    print(f"Coordinate reconstruction MAE : {out['coord_mae']:.6e}")
    print(f"Mean distance absolute error    : {out['mean_dist_abs_err']:.6e}")
    print(f"Mean distance relative error    : {out['mean_dist_rel_err']:.6e}")
    print(
        f"Mean dist rel (|d_ref|>1e-4)     : {out['mean_dist_rel_err_masked']:.6e}"
    )
    print(f"Early-stop proxy                : {out['early_stop_proxy']:.4f}")
    if "robust" in out:
        print()
        _print_robust_one_cloud("[Synthetic] ", out["robust"])
    print("\nGlobal coordinate range per dim:", global_range)
    print("Average block raw range per dim:", np.mean(raw_ranges, axis=0))
    print("Average block offset_max per dim:", np.mean(offset_maxs, axis=0))
    comp = np.mean(offset_maxs / (global_range[None, :] + 1e-12), axis=0)
    print("Avg (offset_max / global_range) per dim:", comp)
    num_blocks = int(np.ceil(n / block_size))
    orig_bits = n * 3 * 16
    bo_bits = num_blocks * 3 * base_bits + n * 3 * offset_bits
    print("\nStorage proxy:")
    print(f"  Original bits (16b coords): {orig_bits}")
    print(f"  Base+Offset bits          : {bo_bits}")
    print(f"  Bit ratio                 : {bo_bits / orig_bits:.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="莫顿排序 + Base/Offset 量化下的距离误差验证（ModelNet40 或合成数据）"
    )
    p.add_argument(
        "--mode",
        choices=("modelnet40", "synthetic"),
        default="modelnet40",
        help="modelnet40: 读 HDF5；synthetic: 聚类随机点云自检",
    )
    p.add_argument(
        "--modelnet40_dir",
        type=str,
        default=_DEFAULT_MODELNET40_CODE_DIR,
        help="含 data.py 的 classification_ModelNet40 目录",
    )
    p.add_argument("--partition", type=str, default="test", choices=("train", "test"))
    p.add_argument("--num_points", type=int, default=1024, help="每样本点数（与 inference.sh NUM_POINTS 一致时常用 1024）")
    p.add_argument("--num_samples", type=int, default=128, help="ModelNet40 随机抽样的样本数")
    p.add_argument("--num_queries", type=int, default=32, help="每帧随机查询点数量")
    p.add_argument("--block_size", type=int, default=64)
    p.add_argument("--base_bits", type=int, default=16)
    p.add_argument("--offset_bits", type=int, default=6)
    p.add_argument("--morton_bits", type=int, default=10, help="Morton 格分辨率位数")
    p.add_argument("--no_morton", action="store_true", help="关闭莫顿排序（对照实验）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no_robust",
        action="store_true",
        help="关闭尾部分位数、违约率、多档掩码相对误差等鲁棒性统计（略快）",
    )
    p.add_argument(
        "--repeat_query_draws",
        type=int,
        default=1,
        help="每帧重复独立随机 query 的次数；>1 时汇报 query 随机性导致的指标波动（std）",
    )
    p.add_argument(
        "--violation_taus",
        type=str,
        default="",
        help="逗号分隔的 |Δd²| 违约阈值；默认内置多档。例: 1e-4,1e-3,0.01",
    )
    p.add_argument(
        "--rel_mask_floors",
        type=str,
        default="",
        help="逗号分隔的 |d_ref| 下限，用于掩码相对误差分位数；默认内置多档",
    )
    return p.parse_args()


def _parse_float_tuple(s: str, default: Tuple[float, ...]) -> Tuple[float, ...]:
    s = (s or "").strip()
    if not s:
        return default
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(float(p) for p in parts)


def main() -> None:
    args = parse_args()
    use_morton = not args.no_morton
    violation_taus = _parse_float_tuple(args.violation_taus, _DEFAULT_VIOLATION_TAUS)
    rel_mask_floors = _parse_float_tuple(args.rel_mask_floors, _DEFAULT_REL_MASK_FLOORS)
    compute_robust = not args.no_robust
    if args.mode == "synthetic":
        simulate_synthetic(
            n=2048,
            block_size=args.block_size,
            base_bits=args.base_bits,
            offset_bits=args.offset_bits,
            use_morton=use_morton,
            seed=args.seed,
        )
        return

    run_modelnet40_benchmark(
        partition=args.partition,
        num_points=args.num_points,
        num_samples=args.num_samples,
        seed=args.seed,
        block_size=args.block_size,
        base_bits=args.base_bits,
        offset_bits=args.offset_bits,
        use_morton=use_morton,
        morton_bits=args.morton_bits,
        num_queries=args.num_queries,
        modelnet40_code_dir=args.modelnet40_dir,
        repeat_query_draws=max(1, int(args.repeat_query_draws)),
        violation_taus=violation_taus,
        rel_mask_floors=rel_mask_floors,
        compute_robust=compute_robust,
    )


if __name__ == "__main__":
    main()
