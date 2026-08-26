#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KAN 预测的 **radii 曲线** 与 **GT（真实 FPS radii）曲线** 对比图。

- 横轴：FPS 步索引 0 … K-1
- 纵轴：radius（与 spec_kan_fps 中 radii 定义一致）
- 蓝实线：GT；红虚线：KAN 预测（prefix 段与 GT 重合，suffix 为模型输出）

**不是** prefix_ratio–Test MSE 汇总图；那类图请用 plot_prefix_ratio_summary_combined.py。

实现上复用 visualize_fps_radii_curves.compare 逻辑；本脚本默认开启自动扫描 one-click 目录。

示例:
  python plot_kan_radii_vs_gt_curves.py \\
    --auto_from ./runs/fps_radii_oneclick_all --auto_pr 0.10 --auto_k 512 \\
    --out_dir ./runs/kan_radii_vs_gt_pr0.10

  # 只要总览大图、不写 detail 子目录
  python plot_kan_radii_vs_gt_curves.py --auto_from ... --auto_pr 0.10 --no_detail
"""
from __future__ import annotations

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from visualize_fps_radii_curves import (  # noqa: E402
    discover_auto_entries,
    visualize_compare,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description="GT radii 曲线 vs KAN 预测 radii 曲线（横轴 FPS 步，非 MSE 曲线）"
    )
    p.add_argument(
        "--auto_from",
        type=str,
        default=os.path.join(SCRIPT_DIR, "runs", "fps_radii_oneclick_all"),
        help="训练根目录（与 one-click / multidataset sweep 输出一致）",
    )
    p.add_argument("--auto_k", type=int, default=512)
    p.add_argument("--auto_model_type", type=str, default="kan")
    p.add_argument(
        "--auto_pr",
        type=float,
        default=0.10,
        help="与目录名 pr0.xx 一致的 prefix_ratio（默认 0.10）",
    )
    p.add_argument(
        "--entry",
        action="append",
        default=[],
        help="额外追加 dataset|save_dir|radii_dir（可与 auto 叠加）",
    )
    p.add_argument(
        "--prefix_ratio",
        type=float,
        default=None,
        help="前向可视化用的 prefix 切分；默认与 checkpoint cfg 一致",
    )
    p.add_argument("--num_vis", type=int, default=4, help="每个数据集展示的测试样本列数")
    p.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="输出目录；默认 auto_from/kan_radii_vs_gt_curves_pr{pr}",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_detail", action="store_true", help="只生成一张多数据集网格总图")
    p.add_argument("--pred_label", type=str, default="", help="预测曲线图例，默认按权重类型推断")
    args = p.parse_args()

    import torch

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    base = os.path.abspath(args.auto_from)
    entries = discover_auto_entries(base, args.auto_model_type, args.auto_k, args.auto_pr)
    entries.extend(args.entry)
    if not entries:
        print("未发现任何可用条目（best.pt + radii_test）。检查 auto_from / auto_pr / auto_k。", file=sys.stderr)
        sys.exit(1)

    out_dir = (args.out_dir or "").strip()
    if not out_dir:
        out_dir = os.path.join(base, f"kan_radii_vs_gt_curves_pr{args.auto_pr:g}".replace(".", "p"))

    _seen = set()
    dedup = []
    for e in entries:
        if e in _seen:
            continue
        _seen.add(e)
        dedup.append(e)

    visualize_compare(
        dedup,
        args.prefix_ratio,
        args.num_vis,
        out_dir,
        device,
        args.seed,
        pred_label=args.pred_label,
        skip_detail=args.no_detail,
    )
    print(f"输出目录: {out_dir}")


if __name__ == "__main__":
    main()
