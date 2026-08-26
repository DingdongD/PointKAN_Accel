#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比「原始浮点 GRF-KAN」与「v2 PWL 近似」在 ModelNet40 测试集上的 loss / acc。

依赖：与 checkpoint 对应的 pwl_v2.json（grf_pwl_trainer_v2 导出），且 layers[].name 与模型中 KAT_Group 的模块名一致。
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
# 默认检查点：本目录下已存在的 ModelNet40 训练输出（不依赖当前工作目录）
DEFAULT_CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints", "pointKANElite-pointkan_elite_exp-42")
# 仓库布局: .../PointKAN-pytorch/PointKAN-elite/classification_ModelNet40 → 上三级为 workspace 根，与 GQA-LUT 并列
_REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
_DEFAULT_PWL_CANDIDATES = [
    os.path.join(_REPO_ROOT, "GQA-LUT", "my_out", "pwl_v2.json"),
    os.path.join(_REPO_ROOT, "GQA-LUT", "my_compare_out", "pwl_v2.json"),
]


def _default_pwl_v2_json() -> str:
    for p in _DEFAULT_PWL_CANDIDATES:
        if os.path.isfile(p):
            return p
    return _DEFAULT_PWL_CANDIDATES[0]


DEFAULT_PWL_V2_JSON = _default_pwl_v2_json()

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
from inject_grf_pwl_v2 import inject_grf_pwl_v2


def _reject_placeholder_checkpoint(path: str) -> None:
    """文档示例里的占位路径易误拷，提前给出中文说明。"""
    norm = path.replace("\\", "/").lower()
    if "/path/to" in norm or norm.startswith("path/to/"):
        script = os.path.basename(__file__)
        print(
            "错误: --checkpoint 仍是占位路径（例如 /path/to/...），请改为本机真实路径。\n"
            "本仓库默认示例:\n"
            f"  python {script} --checkpoint {DEFAULT_CHECKPOINT_DIR}\n"
            f"  --pwl_v2_json {DEFAULT_PWL_V2_JSON}\n"
            "或直接指向 .pth / .pt 文件。\n",
            file=sys.stderr,
        )
        sys.exit(2)


def main():
    epilog = f"""
本工作区默认（已写死为基于脚本路径的绝对路径，可在任意 cwd 下运行）:
  checkpoint 目录: {DEFAULT_CHECKPOINT_DIR}
  PWL JSON:        {DEFAULT_PWL_V2_JSON}

一行命令（使用上述默认，且与 grf_export / GQA-LUT 导出链一致）:
  python %(prog)s

也可显式指向 .pth 文件（需与 pointKANElite 结构一致）:
  python %(prog)s --checkpoint {os.path.join(DEFAULT_CHECKPOINT_DIR, "best_checkpoint.pth")}
"""
    parser = argparse.ArgumentParser(
        description="对比 Vanilla GRF-KAN 与 v2 PWL 推理性能 (ModelNet40)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT_DIR,
        help="检查点：目录（其下需有 best_checkpoint.pth）或 .pth / .pt 文件；默认为本目录下 pointKANElite-pointkan_elite_exp-42",
    )
    parser.add_argument(
        "--pwl_v2_json",
        type=str,
        default=DEFAULT_PWL_V2_JSON,
        help="pwl_v2.json；默认同级 GQA-LUT/my_out/pwl_v2.json（或 my_compare_out），与 ModelNet40 grf 导出配套",
    )
    parser.add_argument(
        "--bit",
        type=str,
        default=None,
        help="选用 layers[].bits[该字符串]；须与 pwl_v2.json 中已导出的档位一致（例如仅 0–6 时不可填 7）",
    )
    parser.add_argument(
        "--pwl_impl",
        type=str,
        default="torch",
        choices=("torch", "triton"),
        help="PWL 实现：torch=PyTorch；triton=Triton fused kernel",
    )
    parser.add_argument("--max_batches", type=int, default=None, help="最多评测多少个 batch（用于快速验证）")
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_json", type=str, default=None)
    args = parser.parse_args()

    _reject_placeholder_checkpoint(args.checkpoint)
    pj = os.path.abspath(os.path.expanduser(args.pwl_v2_json))
    if "/path/to" in pj.replace("\\", "/").lower():
        print("错误: --pwl_v2_json 仍为占位路径，请改为真实的 pwl_v2.json。", file=sys.stderr)
        sys.exit(2)
    if not os.path.isfile(pj):
        print(f"错误: 找不到 pwl_v2.json: {pj}", file=sys.stderr)
        sys.exit(2)
    args.pwl_v2_json = pj

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

    print("\n--- Vanilla GRF-KAN 推理 ---")
    vanilla = run_evaluation(net, test_loader, cal_loss, device, max_batches=args.max_batches)
    print(f"  Loss: {vanilla['loss']:.4f}  Acc: {vanilla['acc']:.2f}%  Acc_avg: {vanilla['acc_avg']:.2f}%")

    print(f"\n注入 PWL v2: {args.pwl_v2_json}")
    n_replaced, bit_used = inject_grf_pwl_v2(
        net,
        args.pwl_v2_json,
        bit_key=args.bit,
        device=device,
        impl=args.pwl_impl,
    )
    print(f"  bits[{bit_used}]，已替换 {n_replaced} 个 KAT_Group -> KAT_Group_PWL")

    print("\n--- PWL v2 推理 ---")
    pwl_result = run_evaluation(net, test_loader, cal_loss, device, max_batches=args.max_batches)
    print(f"  Loss: {pwl_result['loss']:.4f}  Acc: {pwl_result['acc']:.2f}%  Acc_avg: {pwl_result['acc_avg']:.2f}%")

    print("\n" + "=" * 60)
    print("性能对比 (ModelNet40 Test)")
    print("=" * 60)
    print(f"  {'':20}  Loss      Acc(%)   Acc_avg(%)")
    print(f"  {'Vanilla':20}  {vanilla['loss']:.4f}   {vanilla['acc']:.2f}    {vanilla['acc_avg']:.2f}")
    print(f"  {'PWL v2':20}  {pwl_result['loss']:.4f}   {pwl_result['acc']:.2f}    {pwl_result['acc_avg']:.2f}")
    print("=" * 60)
    acc_diff = pwl_result["acc"] - vanilla["acc"]
    acc_avg_diff = pwl_result["acc_avg"] - vanilla["acc_avg"]
    if abs(acc_diff) < 0.01 and abs(acc_avg_diff) < 0.01:
        print("结论: PWL v2 与浮点 GRF-KAN 在测试指标上基本一致（阈值 0.01%）。")
    elif acc_diff < 0 or acc_avg_diff < 0:
        print(f"结论: PWL 相对浮点有差异 (Acc 差 {acc_diff:+.2f}%, Acc_avg 差 {acc_avg_diff:+.2f}%)。")
    else:
        print("结论: PWL 与浮点相当或略优（数值波动范围内）。")
    print()

    out = {
        "checkpoint": checkpoint_path,
        "pwl_v2_json": os.path.abspath(args.pwl_v2_json),
        "bit_used": bit_used,
        "vanilla": vanilla,
        "pwl_v2": pwl_result,
        "acc_diff": acc_diff,
        "acc_avg_diff": acc_avg_diff,
    }
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"结果已写入: {args.out_json}")


if __name__ == "__main__":
    main()
