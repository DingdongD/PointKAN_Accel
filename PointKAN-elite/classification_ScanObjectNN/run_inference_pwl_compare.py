#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比「原始浮点 GRF-KAN」与「v2 PWL 近似」在 ScanObjectNN 测试集上的 loss / acc / balanced acc。

流程与 part_segmentation/run_inference_pwl_compare.py 一致：先整网 Vanilla 评测，再 inject PWL，再评测同一 DataLoader。

pwl_v2.json 须与本任务 checkpoint 导出的 GRF 系数一致（层名对齐），勿混用 ModelNet40 / part_segmentation 的 JSON。

在任务目录下执行::

    python run_inference_pwl_compare.py
    python run_inference_pwl_compare.py --pwl_v2_json grf_export_out/pwl_v2.json --out_json compare_out.json
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints", "pointKANElite-scan_exp-42")
_DEFAULT_PWL_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "grf_export_out", "pwl_v2.json"),
    os.path.join(SCRIPT_DIR, "pwl_v2.json"),
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

from ScanObjectNN import ScanObjectNN
import models as models
from utils import cal_loss
from inference_and_kan_analysis import (
    get_checkpoint_path,
    build_model_and_load_checkpoint,
    run_evaluation,
)
from inject_grf_pwl_v2 import inject_grf_pwl_v2


def _reject_placeholder_checkpoint(path: str) -> None:
    norm = path.replace("\\", "/").lower()
    if "/path/to" in norm or norm.startswith("path/to/"):
        script = os.path.basename(__file__)
        print(
            "错误: --checkpoint 仍是占位路径。\n"
            "示例:\n"
            f"  python {script} --checkpoint {DEFAULT_CHECKPOINT_DIR}\n"
            f"  --pwl_v2_json {DEFAULT_PWL_V2_JSON}\n",
            file=sys.stderr,
        )
        sys.exit(2)


def main():
    epilog = f"""
默认 checkpoint 目录: {DEFAULT_CHECKPOINT_DIR}
默认 PWL（若存在）:   {DEFAULT_PWL_V2_JSON}

一行命令:
  python %(prog)s

须使用**本任务**导出的 pwl_v2.json（与 ScanObjectNN checkpoint 对应），勿使用 ModelNet40 的 JSON。
"""
    parser = argparse.ArgumentParser(
        description="对比 Vanilla GRF-KAN 与 v2 PWL 推理性能 (ScanObjectNN)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT_DIR,
        help="检查点目录（含 best_checkpoint.pth）或 .pth / .pt 文件",
    )
    parser.add_argument(
        "--pwl_v2_json",
        type=str,
        default=DEFAULT_PWL_V2_JSON,
        help="本任务 pwl_v2.json；默认同目录 grf_export_out/pwl_v2.json（若存在）",
    )
    parser.add_argument(
        "--bit",
        type=str,
        default=None,
        help="layers[].bits[该键]；须与 JSON 中已导出档位一致",
    )
    parser.add_argument(
        "--pwl_impl",
        type=str,
        default="torch",
        choices=("torch", "triton"),
        help="PWL 实现：torch=PyTorch；triton=Triton fused kernel",
    )
    parser.add_argument("--max_batches", type=int, default=None, help="最多评测多少个 batch（用于快速验证）")
    parser.add_argument("--num_classes", type=int, default=15)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=None,
        help="测试 DataLoader 的 batch（默认与 --batch_size 相同；显存紧张时可单独调小）",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_json", type=str, default=None)
    args = parser.parse_args()

    _reject_placeholder_checkpoint(args.checkpoint)
    pj = os.path.abspath(os.path.expanduser(args.pwl_v2_json))
    if "/path/to" in pj.replace("\\", "/").lower():
        print("错误: --pwl_v2_json 仍为占位路径。", file=sys.stderr)
        sys.exit(2)
    if not os.path.isfile(pj):
        print(f"错误: 找不到 pwl_v2.json: {pj}", file=sys.stderr)
        print("请先从本任务 checkpoint 导出 GRF 并在 GQA-LUT 生成 pwl_v2.json，或 --pwl_v2_json 指定路径。", file=sys.stderr)
        sys.exit(2)
    args.pwl_v2_json = pj

    device = torch.device(args.device)
    checkpoint_path = get_checkpoint_path(args.checkpoint)
    print(f"加载检查点: {checkpoint_path}")
    net = build_model_and_load_checkpoint(checkpoint_path, device, num_classes=args.num_classes)

    test_dataset = ScanObjectNN(partition="test", num_points=args.num_points)
    test_batch = args.test_batch_size if args.test_batch_size is not None else args.batch_size
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch,
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
    if n_replaced == 0:
        print(
            "警告: 未替换任何 KAT_Group。请检查 pwl_v2.json 的 layers[].name 是否与当前模型模块名一致。",
            file=sys.stderr,
        )

    print("\n--- PWL v2 推理 ---")
    pwl_result = run_evaluation(net, test_loader, cal_loss, device, max_batches=args.max_batches)
    print(f"  Loss: {pwl_result['loss']:.4f}  Acc: {pwl_result['acc']:.2f}%  Acc_avg: {pwl_result['acc_avg']:.2f}%")

    print("\n" + "=" * 60)
    print("性能对比 (ScanObjectNN Test)")
    print("=" * 60)
    print(f"  {'':20}  Loss      Acc(%)   Acc_avg(%)")
    print(f"  {'Vanilla':20}  {vanilla['loss']:.4f}   {vanilla['acc']:.2f}    {vanilla['acc_avg']:.2f}")
    print(f"  {'PWL v2':20}  {pwl_result['loss']:.4f}   {pwl_result['acc']:.2f}    {pwl_result['acc_avg']:.2f}")
    loss_diff = pwl_result["loss"] - vanilla["loss"]
    print(f"  {'Δ (PWL−Vanilla)':20}  {loss_diff:+.4f}   {pwl_result['acc'] - vanilla['acc']:+.2f}    {pwl_result['acc_avg'] - vanilla['acc_avg']:+.2f}")
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
        "task": "ScanObjectNN",
        "checkpoint": checkpoint_path,
        "pwl_v2_json": os.path.abspath(args.pwl_v2_json),
        "bit_used": bit_used,
        "n_replaced": n_replaced,
        "test_batch_size": test_batch,
        "vanilla": vanilla,
        "pwl_v2": pwl_result,
        "loss_diff": loss_diff,
        "acc_diff": acc_diff,
        "acc_avg_diff": acc_avg_diff,
    }
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"结果已写入: {args.out_json}")


if __name__ == "__main__":
    main()
