#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计测试集上前向时，各 KAT_Group 的输入激活相对 pwl_v2.json 中该层 x_range 的越界比例。

PWL 推理里 KAT_Group_PWL 会对输入做 clamp([x_min, x_max])；若此处「低于 min / 高于 max」比例高，
往往对应整网指标明显掉点。

用法（在 part_segmentation 目录下）::

    python analyze_pwl_xrange_coverage.py \\
      --checkpoint checkpoints/pointKANElite_part_exp \\
      --pwl_v2_json grf_export_out/pwl_v2.json \\
      --device cuda \\
      --max_samples 2000

输出：每层 total 元素数、低于 min 的比例、高于 max 的比例、合计越界比例；按越界比例降序打印。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
for d in (SCRIPT_DIR, ROOT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

from util.data_util import PartNormalDataset
from util.util import to_categorical
from inference_and_kan_analysis import build_model_and_load_checkpoint, get_checkpoint_path

NUM_CLASSES = 16


def _load_pwl_x_ranges(pwl_path: str) -> dict:
    with open(pwl_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    out = {}
    for layer in payload.get("layers", []):
        name = str(layer["name"])
        xr = layer["x_range"]
        out[name] = (float(xr["min"]), float(xr["max"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="KAT 输入 vs PWL x_range 越界比例（Part Seg）")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pointKANElite_part_exp")
    parser.add_argument("--pwl_v2_json", type=str, default="grf_export_out/pwl_v2.json")
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=2000, help="最多统计多少个样本（累加 batch 内点数前的 batch 数）")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_json", type=str, default=None)
    args = parser.parse_args()

    pwl_path = os.path.abspath(args.pwl_v2_json)
    x_ranges = _load_pwl_x_ranges(pwl_path)

    device = torch.device(args.device)
    ckpt = get_checkpoint_path(args.checkpoint)
    print(f"检查点: {ckpt}")
    print(f"PWL JSON: {pwl_path}")
    net = build_model_and_load_checkpoint(ckpt, device)
    net.eval()

    # 每层累计：总元素、低于 min 元素、高于 max 元素
    total_el = {n: 0 for n in x_ranges}
    below_el = {n: 0 for n in x_ranges}
    above_el = {n: 0 for n in x_ranges}

    hooks = []

    def make_hook(layer_name: str, x_min: float, x_max: float):
        def hook(_module, inp):
            t = inp[0].detach()
            flat = t.reshape(-1)
            n = int(flat.numel())
            if n == 0:
                return
            total_el[layer_name] += n
            below_el[layer_name] += int((flat < x_min).sum().item())
            above_el[layer_name] += int((flat > x_max).sum().item())

        return hook

    for name, mod in net.named_modules():
        if hasattr(mod, "weight_numerator") and hasattr(mod, "weight_denominator"):
            if name not in x_ranges:
                print(f"[WARN] 模型有 KAT 层 {name!r} 但 pwl_v2.json 无对应层，已跳过")
                continue
            lo, hi = x_ranges[name]
            hooks.append(mod.register_forward_pre_hook(make_hook(name, lo, hi)))

    test_ds = PartNormalDataset(npoints=args.num_points, split="test", normalize=False)
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

    seen = 0
    with torch.no_grad():
        for points, label, _target, norm_plt in loader:
            if seen >= args.max_samples:
                break
            bs = points.size(0)
            points = points.float().to(device).transpose(2, 1)
            norm_plt = norm_plt.float().to(device).transpose(2, 1)
            label = label.squeeze(1).long().to(device)
            onehot = to_categorical(label, NUM_CLASSES)
            _ = net(points, norm_plt, onehot)
            seen += bs

    for h in hooks:
        h.remove()

    print(f"\n统计样本数（batch 累计）: {seen}  （--max_samples={args.max_samples}）\n")
    rows = []
    for name in sorted(x_ranges.keys()):
        te = total_el[name]
        if te == 0:
            rows.append((name, 0.0, 0.0, 0.0, 0, 0, 0, x_ranges[name][0], x_ranges[name][1]))
            continue
        fb = below_el[name] / te
        fa = above_el[name] / te
        fo = (below_el[name] + above_el[name]) / te
        lo, hi = x_ranges[name]
        rows.append((name, fo, fb, fa, below_el[name], above_el[name], te, lo, hi))

    rows.sort(key=lambda r: r[1], reverse=True)
    print(f"{'layer':<48} {'out%':>8} {'<min%':>8} {'>max%':>8} {'cnt_lo':>10} {'cnt_hi':>10} {'n_elem':>12}  [x_min, x_max]")
    print("-" * 120)
    for name, fo, fb, fa, cl, ch, te, lo, hi in rows:
        print(f"{name:<48} {100*fo:>7.3f}% {100*fb:>7.3f}% {100*fa:>7.3f}% {cl:>10} {ch:>10} {te:>12}  [{lo:.4f}, {hi:.4f}]")

    worst = rows[0] if rows else None
    if worst and worst[6] > 0:
        print("\n说明: out% = 会被 PWL clamp 改写的激活元素占比（<min 或 >max）。")
        print("若若干层 out% 明显 > 0，可扩大校准范围或校准用 test 集后再导出 grf/pwl。")

    if args.out_json:
        rep = {
            "checkpoint": ckpt,
            "pwl_v2_json": pwl_path,
            "samples_seen": seen,
            "layers": [
                {
                    "name": name,
                    "frac_out": fo,
                    "frac_below": fb,
                    "frac_above": fa,
                    "count_below": cl,
                    "count_above": ch,
                    "count_total": te,
                    "x_min": lo,
                    "x_max": hi,
                }
                for name, fo, fb, fa, cl, ch, te, lo, hi in rows
            ],
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2, ensure_ascii=False)
        print(f"\n已写入: {args.out_json}")


if __name__ == "__main__":
    main()
