#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 value 空间的 GRF 残差分解验证 + f(x) 曲线量化分析

- 残差框架：以组间 GRF 的 mean / min / max curve 为 base，残差 = f_g(x) - base(x)；验证 base + 残差 == 原始 f(x)。
- 量化对象：对 f(x) curve 量化（每个 x 对应的 f(x) 值），分析并可视化原始 f(x) 与量化反量化后 f(x) 的差异。

用法:
  python run_grf_verify_and_quantize.py --checkpoint checkpoints/pointKANElite-xxx --out_dir ./grf_analysis
  python run_grf_verify_and_quantize.py --grf_json ./grf_analysis/grf_coefficients.json --out_dir ./grf_analysis
"""

import argparse
import json
import os
import sys
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
for d in (SCRIPT_DIR, ROOT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

from grf_kan_json_and_residual import (
    eval_grf_vanilla,
    load_grf_from_json,
    export_grf_coefficients_to_json,
    get_base_and_residual_curves,
    verify_base_plus_residual_equals_vanilla,
    residual_curve_dynamic_range,
    quantize_fx_curve,
    quantization_analysis_fx_curve,
    quantize_base_and_offset,
    analysis_base_offset_quantization,
)


def get_checkpoint_path(checkpoint_arg):
    path = os.path.abspath(checkpoint_arg)
    if os.path.isfile(path) and path.endswith(".pth"):
        return path
    if os.path.isdir(path):
        c = os.path.join(path, "best_checkpoint.pth")
        if os.path.isfile(c):
            return c
        raise FileNotFoundError(f"未找到 best_checkpoint.pth: {path}")
    raise FileNotFoundError(f"路径不存在: {path}")


def load_state_dict(checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("net", ckpt)
    if next(iter(state.keys())).startswith("module."):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def collect_kan_polynomial_weights(state_dict):
    result = []
    for key in sorted(state_dict.keys()):
        if "weight_numerator" not in key or "act" not in key:
            continue
        base_key = key.replace(".weight_numerator", "")
        denom_key = base_key + ".weight_denominator"
        if denom_key not in state_dict:
            continue
        result.append({
            "name": base_key,
            "weight_numerator": state_dict[key].detach().cpu().numpy(),
            "weight_denominator": state_dict[denom_key].detach().cpu().numpy(),
        })
    return result


def load_activation_ranges(analysis_json_path):
    if not analysis_json_path or not os.path.isfile(analysis_json_path):
        return {}
    with open(analysis_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("activation_ranges", {})


def plot_fx_original_vs_quantized(x, f_orig, f_quant, layer_name, bits, out_path):
    """绘制原始 f(x) 与量化反量化后 f(x) 的对比：左图原始，右图量化后，下图为误差。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    G = f_orig.shape[0]
    err = f_quant - f_orig
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    ax0, ax1, ax2 = axes[0], axes[1], axes[2]
    for g in range(G):
        ax0.plot(x, f_orig[g], "-", label=f"g{g}", alpha=0.85)
    ax0.set_ylabel("f(x)")
    ax0.set_title(f"Original f(x): {layer_name}")
    ax0.legend(loc="best", fontsize=8, ncol=4)
    ax0.grid(True, alpha=0.3)
    ax0.axhline(0, color="k", linewidth=0.5)
    for g in range(G):
        ax1.plot(x, f_quant[g], "-", label=f"g{g}", alpha=0.85)
    ax1.set_ylabel("f(x)")
    ax1.set_title(f"Quantized (bits={bits}) f(x)")
    ax1.legend(loc="best", fontsize=8, ncol=4)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color="k", linewidth=0.5)
    for g in range(G):
        ax2.plot(x, err[g], "-", label=f"g{g}", alpha=0.85)
    ax2.set_xlabel("x (input activation)")
    ax2.set_ylabel("error")
    ax2.set_title("Quantized − Original")
    ax2.legend(loc="best", fontsize=8, ncol=4)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color="k", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_base_offset_quantized(x, f_orig, base, offset, base_q, offset_q, layer_name, base_bits, offset_bits, out_path):
    """绘制 base+offset 分别量化：原始 f、base_q、offset_q、重建 base_q+offset_q、误差。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    f_recon = base_q + offset_q
    err = f_recon - f_orig
    G = f_orig.shape[0]
    fig, axes = plt.subplots(4, 1, figsize=(8, 11), sharex=True)
    ax0, ax1, ax2, ax3 = axes[0], axes[1], axes[2], axes[3]
    for g in range(G):
        ax0.plot(x, f_orig[g], "-", label=f"g{g}", alpha=0.85)
    ax0.set_ylabel("f(x)")
    ax0.set_title(f"Original f(x): {layer_name}")
    ax0.legend(loc="best", fontsize=8, ncol=4)
    ax0.grid(True, alpha=0.3)
    ax0.axhline(0, color="k", linewidth=0.5)
    ax1.plot(x, base, "-", color="gray", label="base (float)", alpha=0.9)
    ax1.plot(x, base_q, "--", color="black", label=f"base_q (bits={base_bits})", alpha=0.9)
    ax1.set_ylabel("base(x)")
    ax1.set_title("Base curve: float vs quantized")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    for g in range(G):
        ax2.plot(x, offset_q[g], "-", label=f"offset_q g{g}", alpha=0.85)
    ax2.set_ylabel("offset(x)")
    ax2.set_title(f"Offset (residual) quantized (bits={offset_bits})")
    ax2.legend(loc="best", fontsize=8, ncol=4)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color="k", linewidth=0.5)
    for g in range(G):
        ax3.plot(x, f_recon[g], "-", label=f"recon g{g}", alpha=0.85)
    ax3.set_xlabel("x (input activation)")
    ax3.set_ylabel("f_recon(x)")
    ax3.set_title(f"Reconstructed base_q + offset_q (vs original)")
    ax3.legend(loc="best", fontsize=8, ncol=4)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(0, color="k", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    # 误差图单独一张，更清晰
    fig2, ax = plt.subplots(figsize=(8, 3))
    for g in range(G):
        ax.plot(x, err[g], "-", label=f"g{g}", alpha=0.85)
    ax.set_xlabel("x")
    ax.set_ylabel("error (recon − original)")
    ax.set_title(f"{layer_name} — base{base_bits}+offset{offset_bits} quantization error")
    ax.legend(loc="best", fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", linewidth=0.5)
    err_path = out_path.replace(".png", "_err.png")
    fig2.savefig(err_path, dpi=120, bbox_inches="tight")
    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser(description="GRF value 空间残差验证 + f(x) 曲线量化分析与可视化")
    parser.add_argument("--checkpoint", type=str, default=None, help="检查点目录或 best_checkpoint.pth")
    parser.add_argument("--grf_json", type=str, default=None, help="已有 GRF 系数 JSON")
    parser.add_argument("--analysis_json", type=str, default=None, help="kan_analysis_results.json，用于每层 x 范围")
    parser.add_argument("--out_dir", type=str, default="./grf_analysis", help="输出目录")
    parser.add_argument("--bits", type=int, nargs="+", default=[8, 6, 4], help="f(x) 整曲线量化比特数")
    parser.add_argument("--base_bits", type=int, default=8, help="base+offset 中 base 曲线量化比特数")
    parser.add_argument("--offset_bits", type=int, nargs="+", default=[8, 6, 4], help="base+offset 中 offset 曲线量化比特数，用于探索 offset 低比特")
    parser.add_argument("--base_kind", type=str, default="mean", choices=("mean", "min", "max"), help="base 曲线取 mean/min/max")
    parser.add_argument("--n_pts", type=int, default=300, help="x 网格点数")
    parser.add_argument("--margin", type=float, default=0.05, help="x 范围相对校准范围的扩展比例")
    parser.add_argument("--no_vis", action="store_true", help="不生成量化对比图")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    kan_weights_list = None
    activation_ranges = {}

    if args.grf_json and os.path.isfile(args.grf_json):
        print(f"从 JSON 加载: {args.grf_json}")
        kan_weights_list = load_grf_from_json(args.grf_json)
    elif args.checkpoint:
        ckpt_path = get_checkpoint_path(args.checkpoint)
        state = load_state_dict(ckpt_path)
        kan_weights_list = collect_kan_polynomial_weights(state)
        if args.analysis_json:
            activation_ranges = load_activation_ranges(args.analysis_json)
        grf_path = os.path.join(args.out_dir, "grf_coefficients.json")
        export_grf_coefficients_to_json(kan_weights_list, activation_ranges or None, path=grf_path)
        print(f"已导出 GRF 系数: {grf_path}")
    else:
        print("请提供 --checkpoint 或 --grf_json")
        return

    if not kan_weights_list:
        print("无 GRF 系数")
        return

    n_pts = args.n_pts
    margin = args.margin
    verify_results = []
    residual_ranges = []
    quant_results = []
    base_offset_quant_results = []
    base_kinds = ("mean", "min", "max")

    for item in kan_weights_list:
        name = item["name"]
        num_w = item["weight_numerator"]
        den_w = item["weight_denominator"]
        x_min, x_max = -2.0, 2.0
        if name in activation_ranges:
            r = activation_ranges[name]
            lo, hi = float(r["min"]), float(r["max"])
            span = max(hi - lo, 1e-6)
            x_min = lo - margin * span
            x_max = hi + margin * span
        x = np.linspace(x_min, x_max, n_pts)
        f = eval_grf_vanilla(x, num_w, den_w)

        # 1) value 空间 base+残差 验证
        layer_verify = {"name": name, "base_kinds": {}}
        layer_ranges = {"name": name, "base_kinds": {}}
        for kind in base_kinds:
            passed, max_diff = verify_base_plus_residual_equals_vanilla(f, base_kind=kind)
            layer_verify["base_kinds"][kind] = {"passed": passed, "max_abs_diff": round(float(max_diff), 10)}
            layer_ranges["base_kinds"][kind] = residual_curve_dynamic_range(f, base_kind=kind)
        verify_results.append(layer_verify)
        residual_ranges.append(layer_ranges)

        # 2) f(x) 曲线量化分析与可视化
        qlist = quantization_analysis_fx_curve(f, bits_list=args.bits)
        quant_results.append({"name": name, "quantization": qlist})
        if not args.no_vis:
            safe_name = name.replace(".", "_")
            for b in args.bits:
                f_q = quantize_fx_curve(f, bits=b)
                out_path = os.path.join(args.out_dir, f"grf_fx_quant_compare_{safe_name}_bits{b}.png")
                plot_fx_original_vs_quantized(x, f, f_q, name, b, out_path)

        # 3) base + offset 分量分别量化：探索 offset 低比特，合并 base_q+offset_q 与原始 f 算误差
        bo_list, base, offset = analysis_base_offset_quantization(
            f, base_kind=args.base_kind, base_bits=args.base_bits,
            offset_bits_list=args.offset_bits, symmetric=True,
        )
        base_offset_quant_results.append({"name": name, "base_bits": args.base_bits, "base_kind": args.base_kind, "offset_results": bo_list})
        if not args.no_vis:
            safe_name = name.replace(".", "_")
            for ob in args.offset_bits:
                base_q, offset_q = quantize_base_and_offset(base, offset, base_bits=args.base_bits, offset_bits=ob)
                out_path = os.path.join(args.out_dir, f"grf_base_offset_quant_{safe_name}_base{args.base_bits}_offset{ob}.png")
                plot_base_offset_quantized(x, f, base, offset, base_q, offset_q, name, args.base_bits, ob, out_path)

    all_passed = all(
        all(v["base_kinds"][k]["passed"] for k in base_kinds)
        for v in verify_results
    )
    print("\n" + "=" * 60)
    print("Value 空间 Base+残差 验证（base + residual_g(x) == f_g(x)）")
    print("=" * 60)
    print(f"  全部通过: {all_passed}")
    for r in verify_results:
        print(f"  {r['name']}: mean={r['base_kinds']['mean']['passed']}, min={r['base_kinds']['min']['passed']}, max={r['base_kinds']['max']['passed']}")
    print("=" * 60)

    print("\nf(x) 曲线量化分析 (各层):")
    for layer in quant_results:
        print(f"  {layer['name']}:")
        for q in layer["quantization"]:
            print(f"    bits={q['bits']} mse={q['mse']:.6f} max_abs_err={q['max_abs_err']:.6f}")

    print("\nbase + offset 分别量化（合并 base_q+offset_q 与原始激活误差）:")
    for layer in base_offset_quant_results:
        print(f"  {layer['name']} (base_kind={layer['base_kind']}, base_bits={layer['base_bits']}):")
        for r in layer["offset_results"]:
            print(f"    offset_bits={r['offset_bits']} mse={r['mse']:.6f} max_abs_err={r['max_abs_err']:.6f}")

    out = {
        "verify_base_plus_residual": verify_results,
        "residual_dynamic_range": residual_ranges,
        "quantization_fx_curve": quant_results,
        "base_offset_quantization": base_offset_quant_results,
    }
    out_path = os.path.join(args.out_dir, "grf_verify_and_quantize_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n结果已写入: {out_path}")
    if not args.no_vis:
        print(f"量化对比图与 base+offset 分量量化图已保存到: {args.out_dir}")


if __name__ == "__main__":
    main()
