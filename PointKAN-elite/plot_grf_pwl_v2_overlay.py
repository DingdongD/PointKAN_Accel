#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在单层上绘制 Vanilla GRF 与 PWL v2（按组 + 组均值）对比曲线。

PWL v2 与训练脚本一致：每层内 **每个 group 一条独立标量曲线**（各自断点/斜率/截距或 zero/const/linear），
**组间不共享** PWL 参数（meta: group_specific_breakpoints）。

依赖：GQA-LUT 中的 grf_pwl_v2_map.GRFPWLV2Mapper

示例（部件分割）::

    python plot_grf_pwl_v2_overlay.py \\
      --grf_json part_segmentation/grf_export_out/grf_coefficients.json \\
      --pwl_json part_segmentation/grf_export_out/pwl_v2.json \\
      --layer_name LFP_blocks_list.0.operation1.0.act1 \\
      --out_png part_segmentation/grf_export_out/pwl_vs_grf_act1.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ELITE_ROOT = SCRIPT_DIR
DEFAULT_GQA = os.path.normpath(os.path.join(ELITE_ROOT, "..", "..", "GQA-LUT"))


def main() -> None:
    p = argparse.ArgumentParser(description="单层 GRF vanilla vs PWL v2 曲线对比图")
    p.add_argument("--grf_json", type=str, required=True, help="grf_coefficients.json")
    p.add_argument("--pwl_json", type=str, required=True, help="pwl_v2.json")
    p.add_argument("--layer_name", type=str, default=None, help="默认取 grf/pwl 共有第一层名")
    p.add_argument("--bit", type=str, default=None, help="bits 键，默认 meta.deploy_breakpoint_bits")
    p.add_argument("--n_pts", type=int, default=2048, help="x 上 linspace 点数")
    p.add_argument("--out_png", type=str, required=True, help="输出 PNG")
    p.add_argument("--gqa_lut", type=str, default=os.environ.get("GQA_LUT", DEFAULT_GQA), help="GQA-LUT 根目录")
    args = p.parse_args()

    gqa = os.path.abspath(args.gqa_lut)
    if gqa not in sys.path:
        sys.path.insert(0, gqa)

    from grf_pwl_v2_map import GRFPWLV2Mapper

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mapper = GRFPWLV2Mapper.from_paths(args.grf_json, args.pwl_json, bit_key=args.bit)
    names = mapper.layer_names()
    if not names:
        raise SystemExit("grf 与 pwl 无共有层名")
    name = args.layer_name or names[0]
    if name not in mapper.grf_by_name:
        raise SystemExit(f"未知层: {name}")

    lv = mapper.v2_by_name[name]
    x_min = float(lv["x_range"]["min"])
    x_max = float(lv["x_range"]["max"])
    x = np.linspace(x_min, x_max, max(2, int(args.n_pts)), dtype=np.float64)

    y_g = mapper.eval_grf_vanilla(x, name)
    y_p = mapper.eval_pwl(x, name)
    G = y_g.shape[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    y_g_m = y_g.mean(axis=0)
    y_p_m = y_p.mean(axis=0)
    ax1.plot(x, y_g_m, "k-", lw=2.0, label="GRF mean (vanilla)", zorder=5)
    ax1.plot(x, y_p_m, "r--", lw=2.0, label=f"PWL mean (bits[{mapper.bit_key}])", zorder=4)
    for g in range(G):
        ax1.plot(x, y_g[g], color="C0", alpha=0.25, lw=0.9)
        ax1.plot(x, y_p[g], color="C1", alpha=0.25, lw=0.9, linestyle="--")
    ax1.set_ylabel("f(x)")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"{name}  (G={G}, per-group PWL, no cross-group sharing)")

    err = np.mean(np.abs(y_p - y_g), axis=0)
    ax2.plot(x, err, color="purple", lw=1.5, label="mean_g |PWL_g - GRF_g|")
    ax2.set_xlabel("x (activation)")
    ax2.set_ylabel("|err|")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_png)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    mse = float(np.mean((y_p - y_g) ** 2))
    print(f"[OK] {args.out_png}  layer={name} bit={mapper.bit_key} mean_mse={mse:.6e}")


if __name__ == "__main__":
    main()
