#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成两张总览图（与 build_pwl_v2_calibrated 导出的 JSON 配套）：

  A 图：每一子图画一层，**Vanilla GRF** f(x)=P/Q，8 条组曲线（浅色）+ 组均值（黑色粗线）。
  B 图：**同一布局、同一层顺序**，**PWL v2**（当前 bit 下）各组曲线 + 组均值（黑色粗线）。

便于肉眼对比「整网所有 KAT 层」上 GRF 形状 vs PWL 近似。

示例（在 PointKAN-elite 根目录）::

    python plot_grf_pwl_all_layers_ab.py \\
      --grf_json part_segmentation/grf_export_out/grf_coefficients.json \\
      --pwl_json part_segmentation/grf_export_out/pwl_v2.json \\
      --out_a part_segmentation/grf_export_out/all_layers_vanilla_grf.png \\
      --out_b part_segmentation/grf_export_out/all_layers_pwl_v2.png

或在 part_segmentation 目录::

    python ../plot_grf_pwl_all_layers_ab.py \\
      --grf_json grf_export_out/grf_coefficients.json \\
      --pwl_json grf_export_out/pwl_v2.json \\
      --out_a grf_export_out/all_layers_vanilla_grf.png \\
      --out_b grf_export_out/all_layers_pwl_v2.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GQA = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "GQA-LUT"))


def _layer_order_from_grf(grf_path: str) -> list[str]:
    with open(grf_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [str(l["name"]) for l in data.get("layers", [])]


def _subplot_grid(n: int) -> tuple[int, int]:
    if n <= 0:
        return 1, 1
    c = int(math.ceil(math.sqrt(n)))
    r = int(math.ceil(n / c))
    return r, c


def _draw_one_figure(
    *,
    mapper,
    layer_names: list[str],
    mode: str,
    n_pts: int,
    out_png: str,
    title_prefix: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(layer_names)
    nrows, ncols = _subplot_grid(n)
    fig_w = max(14, 3.2 * ncols)
    fig_h = max(12, 2.8 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    for idx, name in enumerate(layer_names):
        i, j = divmod(idx, ncols)
        ax = axes[i][j]
        lv = mapper.v2_by_name[name]
        x_min = float(lv["x_range"]["min"])
        x_max = float(lv["x_range"]["max"])
        x = np.linspace(x_min, x_max, max(2, int(n_pts)), dtype=np.float64)

        if mode == "grf":
            y = mapper.eval_grf_vanilla(x, name)
        else:
            y = mapper.eval_pwl(x, name)

        g_count = y.shape[0]
        for g in range(g_count):
            ax.plot(
                x,
                y[g],
                color=f"C{g % 10}",
                alpha=0.45,
                lw=0.9,
                label=f"g{g}" if idx == 0 and g < 8 else None,
            )
        ax.plot(x, y.mean(axis=0), color="black", lw=2.0, zorder=10, label="mean" if idx == 0 else None)
        ax.set_title(name, fontsize=7, rotation=0)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=6)
        if x_max - x_min > 1e-6:
            ax.set_xlim(x_min, x_max)

    for k in range(n, nrows * ncols):
        i, j = divmod(k, ncols)
        axes[i][j].set_visible(False)

    fig.suptitle(f"{title_prefix}  (n_layers={n}, points={n_pts})", fontsize=12, y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_png}")


def main() -> None:
    p = argparse.ArgumentParser(description="全层 Vanilla GRF 总览图 + 全层 PWL v2 总览图（两张）")
    p.add_argument("--grf_json", type=str, required=True)
    p.add_argument("--pwl_json", type=str, required=True)
    p.add_argument("--bit", type=str, default=None)
    p.add_argument("--n_pts", type=int, default=1024, help="每层 x 上采样点数（总图较大时可略减）")
    p.add_argument("--out_a", type=str, required=True, help="A 图：Vanilla GRF")
    p.add_argument("--out_b", type=str, required=True, help="B 图：PWL v2")
    p.add_argument("--gqa_lut", type=str, default=os.environ.get("GQA_LUT", DEFAULT_GQA))
    args = p.parse_args()

    gqa = os.path.abspath(args.gqa_lut)
    if gqa not in sys.path:
        sys.path.insert(0, gqa)

    from grf_pwl_v2_map import GRFPWLV2Mapper

    grf_path = os.path.abspath(args.grf_json)
    layer_order = _layer_order_from_grf(grf_path)
    common = GRFPWLV2Mapper.from_paths(grf_path, os.path.abspath(args.pwl_json), bit_key=args.bit)
    names = common.layer_names()
    layer_names = [n for n in layer_order if n in names]
    missing = [n for n in layer_order if n not in names]
    if missing:
        print("[WARN] 下列层在 pwl 中无对应名，已跳过:", missing)
    if not layer_names:
        raise SystemExit("无可用共有层")

    print(f"layers={len(layer_names)} bit={common.bit_key}")

    _draw_one_figure(
        mapper=common,
        layer_names=layer_names,
        mode="grf",
        n_pts=args.n_pts,
        out_png=os.path.abspath(args.out_a),
        title_prefix="A — Vanilla GRF (per-group + mean)",
    )
    _draw_one_figure(
        mapper=common,
        layer_names=layer_names,
        mode="pwl",
        n_pts=args.n_pts,
        out_png=os.path.abspath(args.out_b),
        title_prefix=f"B — PWL v2 bits[{common.bit_key}] (per-group + mean)",
    )


if __name__ == "__main__":
    main()
