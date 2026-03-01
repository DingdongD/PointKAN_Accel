#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比「原始浮点 GRF-KAN」与「LUT 查表 GRF-KAN」在 ModelNet40 测试集上的性能。

流程：
1. 加载 checkpoint，构建模型，在测试集上评估（vanilla）。
2. 注入 LUT：将模型中所有 KAT_Group 替换为 KAT_Group_LUT，再在测试集上评估（LUT）。
3. 汇报两次的 loss / acc / acc_avg，并判断性能是否有损失。

使用前需先运行 run_grf_lut_verify.py（或 run_grf_lut.sh）生成 LUT 目录（grf_lut_config.json + grf_lut_tables.npz）。
"""

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

from data import ModelNet40
import models as models
from utils import cal_loss
from inference_and_kan_analysis import (
    get_checkpoint_path,
    build_model_and_load_checkpoint,
    run_evaluation,
)
from inject_grf_lut import inject_grf_lut


def main():
    parser = argparse.ArgumentParser(description="对比 Vanilla GRF-KAN 与 LUT 查表推理性能")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pointKANElite-pointkan_elite_exp-42",
                        help="检查点目录或 best_checkpoint.pth 路径")
    parser.add_argument("--lut_dir", type=str, default="./grf_lut",
                        help="LUT 目录，内含 grf_lut_config.json 与 grf_lut_tables.npz")
    parser.add_argument("--lut_mode", type=str, default="vanilla", choices=("vanilla", "base_offset"),
                        help="查表模式：vanilla 单表 或 base_offset 基表+偏移表合并")
    parser.add_argument("--use_quantized", action="store_true",
                        help="使用量化表推理（需先 run_grf_lut_verify.py --quantize_bits 8 等生成）")
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_json", type=str, default=None, help="将对比结果写入该 JSON 文件")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint_path = get_checkpoint_path(args.checkpoint)
    print(f"加载检查点: {checkpoint_path}")
    net = build_model_and_load_checkpoint(checkpoint_path, device)

    test_dataset = ModelNet40(partition="test", num_points=args.num_points)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
    )

    # 1) Vanilla 推理
    print("\n--- Vanilla GRF-KAN 推理 ---")
    vanilla = run_evaluation(net, test_loader, cal_loss, device)
    print(f"  Loss: {vanilla['loss']:.4f}  Acc: {vanilla['acc']:.2f}%  Acc_avg: {vanilla['acc_avg']:.2f}%")

    # 2) 注入 LUT 后再推理
    print(f"\n注入 LUT: {args.lut_dir} (mode={args.lut_mode}, use_quantized={args.use_quantized})")
    n_replaced = inject_grf_lut(net, args.lut_dir, lut_mode=args.lut_mode, use_quantized=args.use_quantized, device=device)
    print(f"  已替换 {n_replaced} 个 KAT_Group 为 KAT_Group_LUT")
    print("\n--- LUT 查表 GRF-KAN 推理 ---")
    lut_result = run_evaluation(net, test_loader, cal_loss, device)
    print(f"  Loss: {lut_result['loss']:.4f}  Acc: {lut_result['acc']:.2f}%  Acc_avg: {lut_result['acc_avg']:.2f}%")

    # 3) 对比与结论
    print("\n" + "=" * 60)
    print("性能对比 (ModelNet40 Test)")
    print("=" * 60)
    print(f"  {'':20}  Loss      Acc(%)   Acc_avg(%)")
    print(f"  {'Vanilla':20}  {vanilla['loss']:.4f}   {vanilla['acc']:.2f}    {vanilla['acc_avg']:.2f}")
    print(f"  {'LUT':20}  {lut_result['loss']:.4f}   {lut_result['acc']:.2f}    {lut_result['acc_avg']:.2f}")
    print("=" * 60)
    acc_diff = lut_result["acc"] - vanilla["acc"]
    acc_avg_diff = lut_result["acc_avg"] - vanilla["acc_avg"]
    if abs(acc_diff) < 0.01 and abs(acc_avg_diff) < 0.01:
        print("结论: LUT 查表与浮点 GRF-KAN 性能基本一致，无明显损失。")
    elif acc_diff < 0 or acc_avg_diff < 0:
        print(f"结论: LUT 查表相对浮点有轻微差异 (Acc 差 {acc_diff:+.2f}%, Acc_avg 差 {acc_avg_diff:+.2f}%)，可尝试增大 n_bins 或使用校准 x 范围。")
    else:
        print("结论: LUT 查表与浮点性能相当或略优（数值波动范围内）。")
    print()

    out = {
        "checkpoint": checkpoint_path,
        "lut_dir": os.path.abspath(args.lut_dir),
        "lut_mode": args.lut_mode,
        "use_quantized": args.use_quantized,
        "vanilla": vanilla,
        "lut": lut_result,
        "acc_diff": acc_diff,
        "acc_avg_diff": acc_avg_diff,
    }
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"结果已写入: {args.out_json}")


if __name__ == "__main__":
    main()
