#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析不同量化精度（4/6/8 bit）对 LUT 查表推理性能的影响，并绘制柱状图与直方图。

流程：
1. 在 base LUT 目录下为 4/6/8 bit 生成量化 LUT 子目录（若不存在）。
2. 对每种配置（vanilla float, vanilla 4/6/8b, base_offset float, base_offset 4/6/8b）跑测试集推理，记录 acc/loss。
3. 绘制：(1) 精度-准确率/损失 柱状图；(2) 各量化比特下每层 LUT 与浮点误差（max_abs_err）直方图。
"""

import argparse
import json
import os
import subprocess
import sys
import numpy as np
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


def ensure_quantized_lut_dirs(lut_dir_base, out_analysis_dir, bits_list, grf_json, n_bins=256):
    """为每个 bits 生成 LUT 子目录 lut_b4, lut_b6, lut_b8（若不存在）。"""
    for b in bits_list:
        subdir = os.path.join(out_analysis_dir, f"lut_b{b}")
        if os.path.isfile(os.path.join(subdir, "grf_lut_config.json")) and os.path.isfile(os.path.join(subdir, "grf_lut_tables.npz")):
            continue
        os.makedirs(subdir, exist_ok=True)
        cmd = [
            sys.executable,
            os.path.join(SCRIPT_DIR, "run_grf_lut_verify.py"),
            "--grf_json", grf_json,
            "--out_dir", subdir,
            "--n_bins", str(n_bins),
            "--quantize_bits", str(b),
        ]
        subprocess.run(cmd, check=True, cwd=SCRIPT_DIR)
    return [os.path.join(out_analysis_dir, f"lut_b{b}") for b in bits_list]


def run_one_eval(net, test_loader, criterion, device, lut_dir, lut_mode, use_quantized):
    """加载模型权重、注入 LUT、跑一次评估。会修改 net 内部模块。"""
    inject_grf_lut(net, lut_dir, lut_mode=lut_mode, use_quantized=use_quantized, device=device)
    return run_evaluation(net, test_loader, criterion, device)


def main():
    parser = argparse.ArgumentParser(description="不同量化精度下 LUT 推理性能分析与绘图")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pointKANElite-pointkan_elite_exp-42")
    parser.add_argument("--lut_dir", type=str, default="./grf_lut",
                        help="基准 LUT 目录（浮点表），需含 grf_coefficients.json 或与 checkpoint 一致")
    parser.add_argument("--out_dir", type=str, default="./grf_lut_quantization_analysis",
                        help="分析结果与图输出目录，内含 lut_b4/lut_b6/lut_b8 子目录")
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 4, 6, 8],
                        help="Quantization bit widths to analyze (e.g. 2 4 6 8)")
    parser.add_argument("--n_bins", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_plot", action="store_true", help="不画图，只写 JSON")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    checkpoint_path = get_checkpoint_path(args.checkpoint)
    grf_json = os.path.join(args.lut_dir, "grf_coefficients.json")
    if not os.path.isfile(grf_json):
        grf_json = os.path.join(args.out_dir, "grf_coefficients.json")
        if not os.path.isfile(grf_json):
            print("正在从 checkpoint 导出 GRF 系数到", grf_json)
            subprocess.run([
                sys.executable, os.path.join(SCRIPT_DIR, "run_grf_lut_verify.py"),
                "--checkpoint", checkpoint_path,
                "--out_dir", args.out_dir,
                "--n_bins", str(args.n_bins),
            ], check=True, cwd=SCRIPT_DIR)
    bits_list = [b for b in args.bits if b in (2, 4, 6, 8)]
    if not bits_list:
        bits_list = [2, 4, 6, 8]

    # 1) 确保各比特数量化 LUT 目录存在
    lut_dirs_quant = ensure_quantized_lut_dirs(
        args.lut_dir, args.out_dir, bits_list, grf_json, args.n_bins
    )

    # 2) 测试集
    test_dataset = ModelNet40(partition="test", num_points=1024)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
    )

    # 3) 各配置推理
    results = []
    net = build_model_and_load_checkpoint(checkpoint_path, device)

    # vanilla float
    r = run_one_eval(net, test_loader, cal_loss, device, args.lut_dir, "vanilla", False)
    results.append({"mode": "vanilla", "bits": "float", "acc": r["acc"], "acc_avg": r["acc_avg"], "loss": r["loss"]})
    # vanilla 4/6/8b
    for b, lut_d in zip(bits_list, lut_dirs_quant):
        net = build_model_and_load_checkpoint(checkpoint_path, device)
        r = run_one_eval(net, test_loader, cal_loss, device, lut_d, "vanilla", True)
        results.append({"mode": "vanilla", "bits": f"{b}bit", "acc": r["acc"], "acc_avg": r["acc_avg"], "loss": r["loss"]})
    # base_offset float
    net = build_model_and_load_checkpoint(checkpoint_path, device)
    r = run_one_eval(net, test_loader, cal_loss, device, args.lut_dir, "base_offset", False)
    results.append({"mode": "base_offset", "bits": "float", "acc": r["acc"], "acc_avg": r["acc_avg"], "loss": r["loss"]})
    # base_offset 4/6/8b
    for b, lut_d in zip(bits_list, lut_dirs_quant):
        net = build_model_and_load_checkpoint(checkpoint_path, device)
        r = run_one_eval(net, test_loader, cal_loss, device, lut_d, "base_offset", True)
        results.append({"mode": "base_offset", "bits": f"{b}bit", "acc": r["acc"], "acc_avg": r["acc_avg"], "loss": r["loss"]})

    with open(os.path.join(args.out_dir, "quantization_inference_results.json"), "w", encoding="utf-8") as f:
        json.dump({"checkpoint": checkpoint_path, "results": results}, f, indent=2, ensure_ascii=False)
    print("推理结果已写入 quantization_inference_results.json")

    # 4) 收集每层量化误差（用于直方图）
    layer_errors = {}
    for b in bits_list:
        verify_path = os.path.join(args.out_dir, f"lut_b{b}", "grf_lut_verify_results.json")
        if not os.path.isfile(verify_path):
            continue
        with open(verify_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        van_q = data.get("verify_vanilla_quantized", [])
        bo_q = data.get("verify_base_offset_quantized", [])
        layer_errors[f"vanilla_{b}bit"] = [x["max_abs_err"] for x in van_q]
        layer_errors[f"base_offset_{b}bit"] = [x["max_abs_err"] for x in bo_q]
        if van_q:
            layer_errors[f"vanilla_{b}bit_names"] = [x["name"] for x in van_q]

    # 5) 绘图
    if args.no_plot:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过绘图")
        return

    fig1, axes = plt.subplots(1, 2, figsize=(10, 4))
    x_labels = ["float"] + [f"{b}bit" for b in bits_list]
    x_pos = np.arange(len(x_labels))
    width = 0.35

    # Acc
    vanilla_acc = [next((r["acc"] for r in results if r["mode"] == "vanilla" and r["bits"] == lb), None) for lb in x_labels]
    base_offset_acc = [next((r["acc"] for r in results if r["mode"] == "base_offset" and r["bits"] == lb), None) for lb in x_labels]
    axes[0].bar(x_pos - width / 2, vanilla_acc, width, label="vanilla")
    axes[0].bar(x_pos + width / 2, base_offset_acc, width, label="base_offset")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(x_labels)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Inference Accuracy vs Quantization Bits")
    axes[0].legend()

    # Loss
    vanilla_loss = [next((r["loss"] for r in results if r["mode"] == "vanilla" and r["bits"] == lb), None) for lb in x_labels]
    base_offset_loss = [next((r["loss"] for r in results if r["mode"] == "base_offset" and r["bits"] == lb), None) for lb in x_labels]
    axes[1].bar(x_pos - width / 2, vanilla_loss, width, label="vanilla")
    axes[1].bar(x_pos + width / 2, base_offset_loss, width, label="base_offset")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(x_labels)
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Inference Loss vs Quantization Bits")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "quantization_accuracy_loss_bars.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("已保存 quantization_accuracy_loss_bars.png")

    # 直方图：每层 max_abs_err 分布（按比特数分组）
    if layer_errors:
        n_bits = len(bits_list)
        fig2, axes2 = plt.subplots(2, max(1, (n_bits + 1) // 2), figsize=(5 * max(1, (n_bits + 1) // 2), 8))
        if n_bits == 1:
            axes2 = np.array([axes2])
        axes2 = axes2.flatten()
        for i, b in enumerate(bits_list):
            ax = axes2[i]
            v = layer_errors.get(f"vanilla_{b}bit", [])
            bo = layer_errors.get(f"base_offset_{b}bit", [])
            if v:
                ax.hist(v, bins=min(20, len(v)), alpha=0.6, label=f"vanilla {b}bit", color="C0")
            if bo:
                ax.hist(bo, bins=min(20, len(bo)), alpha=0.6, label=f"base_offset {b}bit", color="C1")
            ax.set_xlabel("max_abs_err (LUT vs float)")
            ax.set_ylabel("Layer count")
            ax.set_title(f"Per-layer quant error distribution ({b}bit)")
            ax.legend()
        for j in range(len(bits_list), len(axes2)):
            axes2[j].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "quantization_per_layer_error_hist.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("已保存 quantization_per_layer_error_hist.png")

    # 柱状图：各比特下每层 max_abs_err（每层一根柱，按层索引）
    first_bits = bits_list[0] if bits_list else 4
    names_key = f"vanilla_{first_bits}bit_names"
    if layer_errors and layer_errors.get(names_key):
        names = layer_errors[names_key]
        n_layers = len(names)
        fig3, ax3 = plt.subplots(figsize=(max(10, n_layers * 0.4), 5))
        x = np.arange(n_layers)
        w = 0.2
        for i, b in enumerate(bits_list):
            v = layer_errors.get(f"vanilla_{b}bit", [])
            if len(v) == n_layers:
                ax3.bar(x + (i - 1) * w, v, width=w, label=f"vanilla {b}bit")
        ax3.set_xticks(x)
        ax3.set_xticklabels([n.split(".")[-1] for n in names], rotation=45, ha="right")
        ax3.set_xlabel("Layer")
        ax3.set_ylabel("max_abs_err")
        ax3.set_title("Per-layer vanilla quant LUT vs float f(x) error")
        ax3.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "quantization_per_layer_maxerr_bars.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("已保存 quantization_per_layer_maxerr_bars.png")


if __name__ == "__main__":
    main()
