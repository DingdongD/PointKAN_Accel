#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比「原始浮点 GRF-KAN」与「v2 PWL 近似」在 Part Segmentation 测试集上的 loss / acc / instance mIoU。

pwl_v2.json 必须与本任务、本 checkpoint 导出的 GRF 系数一致，不可混用分类任务的 JSON。
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints", "pointKANElite_part_exp")
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

from util.data_util import PartNormalDataset
import model as models
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

须使用**部件分割任务**导出的 pwl_v2.json，勿混用 ModelNet40 / ScanObjectNN 的 JSON。
"""
    parser = argparse.ArgumentParser(
        description="对比 Vanilla GRF-KAN 与 v2 PWL 推理性能 (Part Segmentation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT_DIR,
        help="检查点目录（含 best_insiou_model.pth 或 best_checkpoint.pth）或 .pth / .pt",
    )
    parser.add_argument(
        "--pwl_v2_json",
        type=str,
        default=DEFAULT_PWL_V2_JSON,
        help="本任务 pwl_v2.json；默认同目录 grf_export_out/pwl_v2.json（若存在）",
    )
    parser.add_argument("--bit", type=str, default=None, help="layers[].bits[该键]")
    parser.add_argument(
        "--pwl_impl",
        type=str,
        default="torch",
        choices=("torch", "triton"),
        help="PWL 实现：torch=PyTorch；triton=Triton fused kernel",
    )
    parser.add_argument("--max_batches", type=int, default=None, help="最多评测多少个 batch（用于快速验证）")
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=None)
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
        print("请先从本任务 checkpoint 导出 GRF 并生成 pwl_v2.json，或 --pwl_v2_json 指定路径。", file=sys.stderr)
        sys.exit(2)
    args.pwl_v2_json = pj

    device = torch.device(args.device)
    checkpoint_path = get_checkpoint_path(args.checkpoint)
    print(f"加载检查点: {checkpoint_path}")
    net = build_model_and_load_checkpoint(checkpoint_path, device)

    test_batch = args.test_batch_size or args.batch_size
    test_dataset = PartNormalDataset(npoints=args.num_points, split="test", normalize=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
    )

    print("\n--- Vanilla GRF-KAN 推理 ---")
    vanilla = run_evaluation(net, test_loader, device, max_batches=args.max_batches)
    print(
        f"  Loss: {vanilla['loss']:.4f}  Acc: {vanilla['acc']:.2f}%  Ins mIoU: {vanilla['shape_avg_iou']:.4f}"
    )

    print(f"\n注入 PWL v2: {args.pwl_v2_json}")
    n_replaced, bit_used = inject_grf_pwl_v2(
        net,
        args.pwl_v2_json,
        bit_key=args.bit,
        device=device,
        impl=args.pwl_impl,
    )
    print(f"  bits[{bit_used}]，已替换 {n_replaced} 个 KAT_Group -> PWL (impl={args.pwl_impl})")

    print("\n--- PWL v2 推理 ---")
    pwl_result = run_evaluation(net, test_loader, device, max_batches=args.max_batches)
    print(
        f"  Loss: {pwl_result['loss']:.4f}  Acc: {pwl_result['acc']:.2f}%  Ins mIoU: {pwl_result['shape_avg_iou']:.4f}"
    )

    print("\n" + "=" * 60)
    print("性能对比 (Part Segmentation Test)")
    print("=" * 60)
    print(f"  {'':16}  Loss      Acc(%)    Ins mIoU")
    print(f"  {'Vanilla':16}  {vanilla['loss']:.4f}   {vanilla['acc']:.2f}    {vanilla['shape_avg_iou']:.4f}")
    print(f"  {'PWL v2':16}  {pwl_result['loss']:.4f}   {pwl_result['acc']:.2f}    {pwl_result['shape_avg_iou']:.4f}")
    print("=" * 60)

    acc_diff = pwl_result["acc"] - vanilla["acc"]
    iou_diff = pwl_result["shape_avg_iou"] - vanilla["shape_avg_iou"]
    if abs(acc_diff) < 0.01 and abs(iou_diff) < 0.001:
        print("结论: PWL v2 与浮点 GRF-KAN 基本一致（Acc 阈值 0.01%%，mIoU 阈值 0.001）。")
    elif acc_diff < 0 or iou_diff < -0.0005:
        print(f"结论: PWL 相对浮点有差异 (Acc 差 {acc_diff:+.2f}%, Ins mIoU 差 {iou_diff:+.4f})。")
    else:
        print("结论: PWL 与浮点相当或略优（数值波动范围内）。")
    print()

    out = {
        "task": "part_segmentation",
        "checkpoint": checkpoint_path,
        "pwl_v2_json": os.path.abspath(args.pwl_v2_json),
        "bit_used": bit_used,
        "vanilla": vanilla,
        "pwl_v2": pwl_result,
        "acc_diff": acc_diff,
        "shape_avg_iou_diff": iou_diff,
    }
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"结果已写入: {args.out_json}")


if __name__ == "__main__":
    main()
