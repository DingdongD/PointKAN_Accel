#!/usr/bin/env python3
"""阶段 3：Morton 固定块上的多种区域表示 + 池化字段熵/位宽；生成可视化。"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_COMPRESSION = os.path.normpath(os.path.join(_ROOT, ".."))
if _COMPRESSION not in sys.path:
    sys.path.insert(0, _COMPRESSION)

from region_entropy.spatial_order import apply_order, sort_indices_morton  # noqa: E402

from scripts.utils.block_pool_stats import pooled_block_metrics  # noqa: E402
from scripts.utils.io import load_test_npz  # noqa: E402

DEFAULT_BLOCK_SIZES = [8, 16, 32, 64, 128]
DEFAULT_REPR = ["absolute", "morton_delta", "block_centroid_delta", "block_firstpoint_delta"]


def plot_phase_c(per_sample_path: str, fig_dir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)
    df = pd.read_csv(per_sample_path)
    if df.empty:
        return

    g = df.groupby(["block_size", "repr_type"], as_index=False).mean(numeric_only=True)
    g["sum_Hxyz"] = g["Hdx"] + g["Hdy"] + g["Hdz"]
    g["sum_Hexp"] = g["Hexp_dx"] + g["Hexp_dy"] + g["Hexp_dz"]
    g["avg_req_bits_xyz"] = (g["avg_required_bits_x"] + g["avg_required_bits_y"] + g["avg_required_bits_z"]) / 3.0
    g["avg_eff_bits_xyz"] = (
        g["avg_effective_bits_dx"] + g["avg_effective_bits_dy"] + g["avg_effective_bits_dz"]
    ) / 3.0

    # 图 1：块长 vs 边际熵和（按表示）
    plt.figure(figsize=(8, 5))
    for rt, sub in g.groupby("repr_type"):
        sub = sub.sort_values("block_size")
        plt.plot(
            sub["block_size"],
            sub["sum_Hxyz"],
            marker="o",
            linewidth=2,
            label=rt,
        )
    plt.xlabel("block_size")
    plt.ylabel("H(dx)+H(dy)+H(dz)  (mean over samples)")
    plt.title("Phase C: sum of 12-bit marginal entropies vs block size")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_c1_blocksize_vs_sum_Hxyz.png"), dpi=150)
    plt.close()

    # 图 2：热力图 块长 × 表示 → sum_H
    pivot = g.pivot(index="block_size", columns="repr_type", values="sum_Hxyz")
    plt.figure(figsize=(8, 5))
    im = plt.imshow(pivot.values, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="sum Hxyz")
    plt.yticks(range(len(pivot.index)), [str(int(x)) for x in pivot.index])
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    plt.xlabel("repr_type")
    plt.ylabel("block_size")
    plt.title("Phase C: heatmap of sum Hxyz")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_c2_heatmap_blocksize_repr.png"), dpi=150)
    plt.close()

    # 图 3：平均所需位宽（三轴平均）vs 块长
    plt.figure(figsize=(8, 5))
    for rt, sub in g.groupby("repr_type"):
        sub = sub.sort_values("block_size")
        plt.plot(sub["block_size"], sub["avg_req_bits_xyz"], marker="s", linewidth=2, label=rt)
    plt.xlabel("block_size")
    plt.ylabel("mean(avg_required_bits_x,y,z)")
    plt.title("Phase C: mean required signed bits (int16 quant scale)")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_c3_blocksize_vs_required_bits.png"), dpi=150)
    plt.close()

    # 图 4：B=32 时 exponent 熵和 vs 有效位对比（分组柱）
    ref_bs = 32
    b32 = g[g["block_size"] == ref_bs]
    if not b32.empty:
        fig, ax1 = plt.subplots(figsize=(9, 5))
        x = np.arange(len(b32))
        w = 0.35
        ax1.bar(x - w / 2, b32["sum_Hexp"], width=w, label="sum Hexp", color="steelblue")
        ax1.set_xticks(x)
        ax1.set_xticklabels(b32["repr_type"], rotation=20, ha="right")
        ax1.set_ylabel("sum Hexp (dx+dy+dz)")
        ax1.set_title("block_size=32: sum Hexp vs mean effective mantissa bits")
        ax2 = ax1.twinx()
        ax2.bar(x + w / 2, b32["avg_eff_bits_xyz"], width=w, label="avg eff bits", color="darkorange", alpha=0.85)
        ax2.set_ylabel("mean avg_effective_bits (xyz)")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "fig_c4_b32_Hexp_vs_effective_bits.png"), dpi=150)
        plt.close()

    # 图 5：zero_ratio / small_ratio vs 块长（centroid vs absolute）
    plt.figure(figsize=(8, 5))
    for rt in ["absolute", "block_centroid_delta"]:
        sub = g[g["repr_type"] == rt].sort_values("block_size")
        if sub.empty:
            continue
        plt.plot(sub["block_size"], sub["zero_ratio"], marker="o", label=f"{rt} zero_ratio")
    plt.xlabel("block_size")
    plt.ylabel("ratio")
    plt.title("zero_ratio vs block_size (subset)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_c5_zero_ratio_vs_blocksize.png"), dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=str,
        default=os.path.join(_ROOT, "data", "modelnet40_test.npz"),
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(_ROOT, "outputs", "phase_c_block"),
    )
    ap.add_argument("--block-sizes", nargs="*", type=int, default=None)
    ap.add_argument(
        "--repr-types",
        nargs="*",
        default=None,
        help="默认 absolute morton_delta block_centroid_delta block_firstpoint_delta",
    )
    ap.add_argument("--quant-scale", type=float, default=2048.0)
    ap.add_argument("--max-samples", type=int, default=0, help="0 表示全部")
    ap.add_argument("--no-plots", action="store_true", help="跳过 matplotlib 出图")
    args = ap.parse_args()

    block_sizes = args.block_sizes if args.block_sizes else DEFAULT_BLOCK_SIZES
    repr_types = args.repr_types if args.repr_types else DEFAULT_REPR

    points, _labels, _ids, id_strings = load_test_npz(args.input)
    n = points.shape[0]
    if args.max_samples > 0:
        n = min(n, args.max_samples)

    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = os.path.join(args.output_dir, "figures")
    rows: List[dict] = []

    for i in range(n):
        sid = id_strings[i] if i < len(id_strings) else str(int(_ids[i]))
        xyz = points[i]
        ord_m = sort_indices_morton(xyz, p=10)
        xm = apply_order(xyz, ord_m)
        for bsz in block_sizes:
            for rt in repr_types:
                m = pooled_block_metrics(xm, int(bsz), rt, quant_scale=args.quant_scale)
                row = {
                    "sample_id": sid,
                    "block_size": int(bsz),
                    "repr_type": rt,
                    **m,
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    per_path = os.path.join(args.output_dir, "block_repr_entropy_per_sample.csv")
    df.to_csv(per_path, index=False)

    num_cols = [c for c in df.columns if c not in ("sample_id", "block_size", "repr_type")]
    summ_rows = []
    for keys, grp in df.groupby(["block_size", "repr_type"]):
        bsz, rt = keys
        m = grp[num_cols].mean(numeric_only=True)
        s = grp[num_cols].std(numeric_only=True)
        one = {"block_size": int(bsz), "repr_type": rt}
        for c in num_cols:
            one[f"{c}_mean"] = float(m[c])
            one[f"{c}_std"] = float(s[c])
        summ_rows.append(one)
    summ_df = pd.DataFrame(summ_rows)
    summ_path = os.path.join(args.output_dir, "block_repr_entropy_summary.csv")
    summ_df.to_csv(summ_path, index=False)

    print("per_sample:", per_path)
    print("summary:", summ_path)
    if not args.no_plots:
        plot_phase_c(per_path, fig_dir)
        print("figures:", fig_dir)

    # 简短文本结论，便于与阶段 4 衔接
    takeaway_path = os.path.join(args.output_dir, "phase_c_takeaway.txt")
    g = df.groupby(["block_size", "repr_type"], as_index=False).mean(numeric_only=True)
    g["sum_Hxyz"] = g["Hdx"] + g["Hdy"] + g["Hdz"]
    g["avg_req"] = (g["avg_required_bits_x"] + g["avg_required_bits_y"] + g["avg_required_bits_z"]) / 3.0
    best_h = g.sort_values("sum_Hxyz").iloc[0]
    best_b = g.sort_values("avg_req").iloc[0]
    lines = [
        "阶段 C 自动摘要（全样本均值意义下）：",
        f"- 12-bit 边际熵和最低: block_size={int(best_h['block_size'])}, repr_type={best_h['repr_type']}, sum_Hxyz={best_h['sum_Hxyz']:.4f}",
        f"- 平均所需符号位宽最低: block_size={int(best_b['block_size'])}, repr_type={best_b['repr_type']}, avg_req={best_b['avg_req']:.4f}",
        "详见 figures/ 下 PNG 与 CSV。",
    ]
    with open(takeaway_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("takeaway:", takeaway_path)


if __name__ == "__main__":
    main()
