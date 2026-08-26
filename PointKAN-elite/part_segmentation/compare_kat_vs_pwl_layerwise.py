#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在**相同 KAT 输入激活**下，逐层对比 Vanilla KAT_Group 与从 pwl_v2.json 构建的 KAT_Group_PWL 的输出误差。

做法：
1) 对 vanilla 模型注册 forward_pre_hook，记录每个 KAT 模块的 input[0]（与 ModelNet40 collect_activation_ranges 一致）。
2) 对同一 tensor x 分别执行：y_v = KAT_Group(x)，y_p = KAT_Group_PWL(x)。
3) 打印每层 MAE / RMSE / max|diff| / 相对 RMSE（相对 y_v 的 RMS 尺度）。

说明：这里度量的是「若该层看到的输入与 vanilla 完全一致时，单层的替换误差」；
整网串联时前面层输出会漂移，累计效应需结合 run_inference_pwl_compare 全图指标一起看。

用法::

    cd part_segmentation
    python compare_kat_vs_pwl_layerwise.py \\
      --checkpoint checkpoints/pointKANElite_part_exp \\
      --pwl_v2_json grf_export_out/pwl_v2.json \\
      --device cuda --batch_size 4 --num_batches 2 --bit 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
for d in (SCRIPT_DIR, ROOT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

from inject_grf_pwl_v2 import (
    _build_pwl_module_from_layer,
    _find_kat_group_modules,
    _resolve_bit_key,
)
from inference_and_kan_analysis import build_model_and_load_checkpoint, get_checkpoint_path
from util.data_util import PartNormalDataset
from util.util import to_categorical

NUM_CLASSES = 16


def _get_submodule(model: torch.nn.Module, name: str) -> torch.nn.Module:
    obj = model
    for p in name.split("."):
        obj = getattr(obj, p)
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="逐层对比同输入下 Vanilla KAT vs PWL KAT 输出误差")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pointKANElite_part_exp")
    parser.add_argument("--pwl_v2_json", type=str, default="grf_export_out/pwl_v2.json")
    parser.add_argument("--bit", type=str, default=None)
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_batches", type=int, default=1, help="取测试集前 K 个 batch 对指标做平均")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out_json", type=str, default=None)
    parser.add_argument(
        "--sort_by",
        type=str,
        default="model",
        choices=("model", "rmse"),
        help="model=按网络前向遍历顺序；rmse=按平均 RMSE 从大到小（便于找最差层）",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    pwl_path = os.path.abspath(args.pwl_v2_json)
    with open(pwl_path, "r", encoding="utf-8") as f:
        pwl_payload = json.load(f)
    bit = _resolve_bit_key(pwl_payload, args.bit)
    v2_by_name = {str(l["name"]): l for l in pwl_payload.get("layers", [])}

    ckpt = get_checkpoint_path(args.checkpoint)
    print(f"检查点: {ckpt}")
    print(f"PWL: {pwl_path}  bits[{bit}]  num_batches={args.num_batches}\n")

    net = build_model_and_load_checkpoint(ckpt, device)
    net.eval()
    kat_list = _find_kat_group_modules(net)
    if not kat_list:
        raise SystemExit("未找到 KAT_Group 模块")

    for name, _ in kat_list:
        if name not in v2_by_name:
            raise KeyError(f"pwl_v2.json 缺少层: {name}")

    test_ds = PartNormalDataset(npoints=args.num_points, split="test", normalize=False)
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
    )

    # 每层累计：sum mae, sum mse, max max_abs, sum rel_rmse 的分子分母可简化为每 batch 后 list
    sum_mae = defaultdict(float)
    sum_mse = defaultdict(float)
    max_abs = defaultdict(float)
    sum_rel_num = defaultdict(float)
    sum_rel_den = defaultdict(float)
    n_batches_done = 0

    for batch_idx, (points, label, _target, norm_plt) in enumerate(loader):
        if batch_idx >= args.num_batches:
            break

        saved: dict[str, torch.Tensor] = {}

        def make_hook(layer_name: str):
            def hook(_mod, inp):
                saved[layer_name] = inp[0].detach().clone()

            return hook

        hooks = []
        for name, mod in kat_list:
            hooks.append(mod.register_forward_pre_hook(make_hook(name)))

        with torch.no_grad():
            points = points.float().to(device).transpose(2, 1)
            norm_plt = norm_plt.float().to(device).transpose(2, 1)
            label = label.squeeze(1).long().to(device)
            onehot = to_categorical(label, NUM_CLASSES)
            _ = net(points, norm_plt, onehot)

        for h in hooks:
            h.remove()

        for name, mod in kat_list:
            x = saved[name].to(device)
            y_v = mod(x)
            pwl = _build_pwl_module_from_layer(v2_by_name[name], bit, mod.num_groups, device)
            pwl.eval()
            y_p = pwl(x)
            d = (y_v - y_p).float()
            mae = d.abs().mean().item()
            mse = (d**2).mean().item()
            mx = d.abs().max().item()
            rmse = mse**0.5
            denom = y_v.float().pow(2).mean().sqrt().item() + 1e-8
            rel = rmse / denom

            sum_mae[name] += mae
            sum_mse[name] += mse
            max_abs[name] = max(max_abs[name], mx)
            sum_rel_num[name] += rel
            sum_rel_den[name] += 1.0

        n_batches_done += 1

    if n_batches_done == 0:
        raise SystemExit("未跑任何 batch，检查 DataLoader")

    print(
        f"{'layer':<48} {'MAE':>12} {'RMSE':>12} {'max|diff|':>12} {'rel_RMSE':>12}  (对 {n_batches_done} batch 平均)"
    )
    print("-" * 110)

    rows = []
    for name, _mod in kat_list:
        k = n_batches_done
        mae = sum_mae[name] / k
        mse = sum_mse[name] / k
        rmse = mse**0.5
        rel = sum_rel_num[name] / sum_rel_den[name]
        mx = max_abs[name]
        rows.append((name, mae, rmse, mx, rel))

    if args.sort_by == "rmse":
        rows.sort(key=lambda r: r[2], reverse=True)
    for name, mae, rmse, mx, rel in rows:
        print(f"{name:<48} {mae:>12.6e} {rmse:>12.6e} {mx:>12.6e} {rel:>12.6f}")

    print(
        "\n列说明: MAE/RMSE/max|diff| 为输出张量上统计；rel_RMSE = RMSE / RMS(y_v)，便于看相对尺度。"
    )

    if args.out_json:
        out = {
            "checkpoint": ckpt,
            "pwl_v2_json": pwl_path,
            "bit": bit,
            "num_batches": n_batches_done,
            "layers": [
                {
                    "name": name,
                    "mae": mae,
                    "rmse": rmse,
                    "max_abs_diff": mx,
                    "rel_rmse": rel,
                }
                for name, mae, rmse, mx, rel in rows
            ],
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n已写入: {args.out_json}")


if __name__ == "__main__":
    main()
