#!/usr/bin/env python3
"""
ModelNet40 区域熵与压缩 — 算法级实验（Phase A–E）。
数据集默认与 classification_ModelNet40/data.py 一致。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from region_entropy.blocks import sweep_blocks
from region_entropy.data_loader import default_data_dir, load_modelnet40_test
from region_entropy.entropy import (
    delta_along_sequence,
    entropy_of_deltas,
    exponent_entropy_and_topk,
    float32_exponents,
    joint_entropy_xyz,
    marginal_entropies_xyz,
    shannon_entropy_counts,
    small_delta_ratios,
)
from region_entropy.fps_knn import (
    apply_center_order,
    farthest_point_sample_indices,
    knn_indices,
    neighbor_overlap_stats,
    order_centers_morton,
    pool_group_stats,
)
from region_entropy.memory_model import build_memory_model_rows
from region_entropy.spatial_order import apply_order, sort_indices_hilbert, sort_indices_morton


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def phase_a_raw(xyz: np.ndarray, sample_id: str, n_bits: int = 12) -> Dict[str, Any]:
    bounds = (-1.0, 1.0)
    hx, hy, hz = marginal_entropies_xyz(xyz, n_bits, bounds)
    hj = joint_entropy_xyz(xyz, n_bits, bounds)
    hex_x, cx = exponent_entropy_and_topk(xyz[:, 0], k=3)
    hex_y, cy = exponent_entropy_and_topk(xyz[:, 1], k=3)
    hex_z, cz = exponent_entropy_and_topk(xyz[:, 2], k=3)
    d = delta_along_sequence(xyz)
    if len(d) == 0:
        h_delta = (0.0, 0.0, 0.0)
        dr0 = 0.0
    else:
        dr0 = float(np.percentile(np.abs(d), 99) + 1e-6)
        dr0 = max(dr0, 1e-3)
        h_delta = entropy_of_deltas(d, n_bits, delta_range=dr0)
    row: Dict[str, Any] = {
        "sample_id": sample_id,
        "Hx_raw": hx,
        "Hy_raw": hy,
        "Hz_raw": hz,
        "Hxyz_joint_raw": hj,
        "Hexp_x_raw": float(hex_x),
        "Hexp_y_raw": float(hex_y),
        "Hexp_z_raw": float(hex_z),
        "delta_entropy_input_order_Hdx": h_delta[0],
        "delta_entropy_input_order_Hdy": h_delta[1],
        "delta_entropy_input_order_Hdz": h_delta[2],
        "delta_entropy_input_order_sum": sum(h_delta),
        "delta_bin_half_range_p99_input": dr0,
    }
    for k, v in cx.items():
        row[f"{k}_x"] = v
    for k, v in cy.items():
        row[f"{k}_y"] = v
    for k, v in cz.items():
        row[f"{k}_z"] = v
    return row


def phase_b_sfc(xyz: np.ndarray, sample_id: str, rng: np.random.Generator, n_bits: int = 12):
    rows = []
    # 注意：ModelNet40 HDF5 点序往往已接近/等于 Morton 序，直接对 xyz 做 morton 可能与 unordered 相同。
    # 下面用「先打乱再 SFC」验证曲线重排对差分熵的影响（与实验计划 Phase B 一致）。
    perm = rng.permutation(xyz.shape[0])
    xyz_shuf = xyz[perm]
    morton_on_shuf = sort_indices_morton(xyz_shuf, p=10)
    hil_on_shuf = sort_indices_hilbert(xyz_shuf, p=10)

    orders = {
        "unordered": np.arange(xyz.shape[0]),
        "random_shuffle": rng.permutation(xyz.shape[0]),
        "morton_sorted": sort_indices_morton(xyz, p=10),
        "shuffled_input": np.arange(xyz.shape[0]),
        "morton_after_shuffle": morton_on_shuf,
    }
    if hil_on_shuf is not None:
        orders["hilbert_after_shuffle"] = hil_on_shuf

    hil = sort_indices_hilbert(xyz, p=10)
    if hil is not None:
        orders["hilbert_sorted"] = hil

    tau_axis = (1e-3, 1e-2, 5e-2)
    tau_norm = (1e-2, 5e-2, 1e-1)

    for name, ord_idx in orders.items():
        base = xyz_shuf if name in ("shuffled_input", "morton_after_shuffle", "hilbert_after_shuffle") else xyz
        seq = apply_order(base, ord_idx)
        d = delta_along_sequence(seq)
        if len(d) == 0:
            continue
        # 固定 0.5 会在「差分尺度」随排序剧烈变化时扭曲熵；用分位数自适应范围保持桶宽与数据尺度可比。
        dr = float(np.percentile(np.abs(d), 99) + 1e-6)
        dr = max(dr, 1e-3)
        hdx, hdy, hdz = entropy_of_deltas(d, n_bits, delta_range=dr)
        edx, _ = exponent_entropy_and_topk(d[:, 0:1], k=3)
        edy, _ = exponent_entropy_and_topk(d[:, 1:2], k=3)
        edz, _ = exponent_entropy_and_topk(d[:, 2:3], k=3)
        dist = np.linalg.norm(d.astype(np.float64), axis=1)
        sm = small_delta_ratios(d, tau_axis, tau_norm)
        row = {
            "sample_id": sample_id,
            "order_type": name,
            "delta_bin_half_range_p99": dr,
            "Hdx": hdx,
            "Hdy": hdy,
            "Hdz": hdz,
            "Hexp_dx": float(edx),
            "Hexp_dy": float(edy),
            "Hexp_dz": float(edz),
            "mean_neighbor_dist_in_sequence": float(np.mean(dist)),
            "median_neighbor_dist_in_sequence": float(np.median(dist)),
        }
        row.update(sm)
        rows.append(row)
    return rows


def phase_c_blocks(xyz: np.ndarray, sample_id: str) -> List[Dict[str, Any]]:
    ord_m = sort_indices_morton(xyz, p=10)
    xm = apply_order(xyz, ord_m)
    block_sizes = [8, 16, 32, 64, 128]
    repr_types = ["absolute", "morton_delta", "block_centroid_delta", "block_firstpoint_delta"]
    out = sweep_blocks(
        xm,
        block_sizes,
        repr_types,
        n_bits_entropy=12,
        delta_range=0.5,
        quant_scale=2048.0,
    )
    for r in out:
        r["sample_id"] = sample_id
    return out


def phase_d_knn(xyz: np.ndarray, sample_id: str, rng: np.random.Generator):
    rows = []
    center_idx = farthest_point_sample_indices(xyz, 256, seed=int(rng.integers(0, 2**31)))
    centers = xyz[center_idx]
    for k in (16, 32):
        idx = knn_indices(xyz, centers, k)
        for order_name, c_ord in (
            ("fps_order", np.arange(256)),
            ("morton_sorted_centers", order_centers_morton(centers, p=10)),
        ):
            idx_o = idx[c_ord]
            st = neighbor_overlap_stats(idx_o, n_random_pairs=48, rng=rng)
            base = {
                "sample_id": sample_id,
                "K": k,
                "center_order_type": order_name,
                **st,
            }
            for pg in pool_group_stats(idx_o, [2, 4, 8]):
                rows.append({**base, **pg})
    return rows


def plot_figures(summary_dir: str, plot_dir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_dir(plot_dir)
    sfc = pd.read_csv(os.path.join(summary_dir, "sfc_entropy_mean.csv"))
    if not sfc.empty:
        plt.figure(figsize=(7, 4))
        sub = sfc.groupby("order_type")[["Hdx", "Hdy", "Hdz"]].mean()
        sub.plot(kind="bar", rot=20)
        plt.ylabel("entropy (bits)")
        plt.title("SFC / order: mean marginal delta entropy")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "fig1_sfc_delta_entropy.png"), dpi=150)
        plt.close()

    blk = pd.read_csv(os.path.join(summary_dir, "block_residual_entropy_mean.csv"))
    if not blk.empty:
        for rt, g in blk.groupby("repr_type"):
            plt.figure(figsize=(6, 4))
            gg = g.groupby("block_size")["sum_Hxyz"].mean()
            gg.plot(marker="o")
            plt.xlabel("block size")
            plt.ylabel("H(dx)+H(dy)+H(dz) mean")
            plt.title(f"Block residual entropy — {rt}")
            plt.tight_layout()
            safe = rt.replace("/", "_")
            plt.savefig(os.path.join(plot_dir, f"fig2_block_entropy_{safe}.png"), dpi=150)
            plt.close()

        cent = blk[blk["repr_type"] == "block_centroid_delta"]
        if not cent.empty:
            plt.figure(figsize=(6, 4))
            g = cent.groupby("block_size")[["avg_required_bits_x", "avg_required_bits_y", "avg_required_bits_z"]].mean()
            g.plot(marker="o")
            plt.xlabel("block size")
            plt.ylabel("mean required bits (quant int)")
            plt.title("Centroid delta: avg required bits per axis")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "fig3_block_bits_centroid.png"), dpi=150)
            plt.close()

    knn_path = os.path.join(summary_dir, "knn_overlap_mean.csv")
    if os.path.isfile(knn_path):
        knn = pd.read_csv(knn_path)
        if not knn.empty and "pool_group_size" in knn.columns:
            sub = knn[knn["pool_group_size"] == knn["pool_group_size"].min()].copy()
            plt.figure(figsize=(8, 4))
            labels = []
            vals = []
            for (kk, ot), g in sub.groupby(["K", "center_order_type"]):
                labels.append(f"K={int(kk)}\n{ot}")
                vals.append(g["avg_shared_at_k"].mean())
            plt.bar(range(len(vals)), vals)
            plt.xticks(range(len(labels)), labels, rotation=15, ha="right")
            plt.ylabel("Shared@K (adjacent centers)")
            plt.title("KNN overlap (adjacent vs random in table)")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "fig4_knn_shared_at_k.png"), dpi=150)
            plt.close()


def write_final_summary_md(
    path: str,
    raw_stats: pd.DataFrame,
    sfc_mean: pd.DataFrame,
    blk_mean: pd.DataFrame,
    knn_mean: pd.DataFrame,
    mem: pd.DataFrame,
) -> None:
    def _md(df: pd.DataFrame, max_rows: int = 30) -> str:
        if df is None or df.empty:
            return "_无数据_"
        try:
            return df.head(max_rows).to_markdown(index=False)
        except Exception:
            return df.head(max_rows).to_csv(index=False)

    lines = [
        "# ModelNet40 区域熵实验 — 自动摘要",
        "",
        "## Phase A：原始坐标（测试集顺序，数值列 mean/std）",
        _md(raw_stats.T.reset_index().rename(columns={"index": "metric"}), 40),
        "",
        "## Phase B：差分熵（按排序类型）",
        _md(sfc_mean),
        "",
        "## Phase C：块残差（按 block_size × repr_type 的样本均值）",
        _md(blk_mean, 50),
        "",
        "## Phase D：KNN 重叠（节选）",
        _md(knn_mean, 36),
        "",
        "## Phase E：访存粗模",
        _md(mem, 20),
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=None, help="ModelNet HDF5 目录")
    ap.add_argument("--num_points", type=int, default=1024)
    ap.add_argument("--max_samples", type=int, default=200, help="子采样数量；<=0 表示全测试集")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(_SCRIPT_DIR, "results"),
    )
    args = ap.parse_args()
    data_dir = args.data_dir or default_data_dir()
    max_s = args.max_samples if args.max_samples > 0 else None

    rng = np.random.default_rng(args.seed)
    pts, _, sample_ids = load_modelnet40_test(
        data_dir, num_points=args.num_points, max_samples=max_s, seed=args.seed
    )

    res_root = args.out_dir
    per_sample = os.path.join(res_root, "per_sample")
    summary = os.path.join(res_root, "summary")
    plots = os.path.join(res_root, "plots")
    _ensure_dir(per_sample)
    _ensure_dir(summary)
    _ensure_dir(plots)

    raw_rows, sfc_rows, blk_rows, knn_rows = [], [], [], []

    for i, sid in enumerate(sample_ids):
        xyz = pts[i]
        raw_rows.append(phase_a_raw(xyz, sid))
        sfc_rows.extend(phase_b_sfc(xyz, sid, rng))
        blk_rows.extend(phase_c_blocks(xyz, sid))
        knn_rows.extend(phase_d_knn(xyz, sid, rng))

    df_raw = pd.DataFrame(raw_rows)
    df_sfc = pd.DataFrame(sfc_rows)
    df_blk = pd.DataFrame(blk_rows)
    df_knn = pd.DataFrame(knn_rows)

    df_raw.to_csv(os.path.join(per_sample, "raw_entropy.csv"), index=False)
    df_sfc.to_csv(os.path.join(per_sample, "sfc_entropy.csv"), index=False)
    df_blk.to_csv(os.path.join(per_sample, "block_residual_entropy.csv"), index=False)
    df_knn.to_csv(os.path.join(per_sample, "knn_overlap.csv"), index=False)

    raw_mean = df_raw.drop(columns=["sample_id"]).agg(["mean", "std"])
    raw_mean.T.to_csv(os.path.join(summary, "raw_entropy_mean.csv"))

    sfc_mean = df_sfc.groupby("order_type").mean(numeric_only=True).reset_index()
    sfc_mean.to_csv(os.path.join(summary, "sfc_entropy_mean.csv"), index=False)

    blk_mean = (
        df_blk.groupby(["block_size", "repr_type", "block_type"]).mean(numeric_only=True).reset_index()
    )
    blk_mean["sum_Hxyz"] = blk_mean["Hdx"] + blk_mean["Hdy"] + blk_mean["Hdz"]
    blk_mean.to_csv(os.path.join(summary, "block_residual_entropy_mean.csv"), index=False)

    knn_mean = df_knn.groupby(["K", "center_order_type", "pool_group_size"]).mean(numeric_only=True).reset_index()
    knn_mean.to_csv(os.path.join(summary, "knn_overlap_mean.csv"), index=False)

    ref = blk_mean[
        (blk_mean["repr_type"] == "block_centroid_delta") & (blk_mean["block_size"] == 32.0)
    ]
    if not ref.empty:
        avg_axis_bits = float(
            (
                ref["avg_required_bits_x"].iloc[0]
                + ref["avg_required_bits_y"].iloc[0]
                + ref["avg_required_bits_z"].iloc[0]
            )
            / 3.0
        )
    else:
        avg_axis_bits = 10.0

    mem_rows = build_memory_model_rows(
        n_points=args.num_points,
        n_centers=256,
        k_neighbors=32,
        avg_bits_per_delta_axis=avg_axis_bits,
    )
    df_mem = pd.DataFrame(mem_rows)
    df_mem.to_csv(os.path.join(summary, "memory_model.csv"), index=False)

    with open(os.path.join(summary, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "data_dir": data_dir,
                "num_points": args.num_points,
                "max_samples": max_s,
                "seed": args.seed,
                "n_samples": len(sample_ids),
            },
            f,
            indent=2,
        )

    plot_figures(summary, plots)

    write_final_summary_md(
        os.path.join(res_root, "final_summary.md"),
        raw_mean,
        sfc_mean,
        blk_mean,
        knn_mean,
        df_mem,
    )

    print(f"完成：{len(sample_ids)} 个样本 -> {res_root}")


if __name__ == "__main__":
    main()
