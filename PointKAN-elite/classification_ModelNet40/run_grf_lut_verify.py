#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建每层 GRF-KAN 的均匀分割 LUT，验证查表与浮点 f(x) 一致性，并保存 LUT 供模型推理使用。

- 先对原始浮点 f(x) 做均匀格点 LUT，查表+线性插值替代计算，验证误差。
- 再对 base+offset 分别建 LUT，查表合并后与原始 f(x) 比较。
- 保存 LUT 到 out_dir（npz + config json），供后续模型 LUT 推理加载。
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
)
from grf_kan_lut import (
    build_lut_vanilla,
    build_lut_base_offset,
    lookup_lut_numpy,
    verify_lut_vs_float,
    quantize_table_symmetric,
    verify_lut_quantized_vs_float,
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


def main():
    parser = argparse.ArgumentParser(description="GRF-KAN 均匀 LUT 构建、验证与导出")
    parser.add_argument("--checkpoint", type=str, default=None, help="检查点目录或 .pth")
    parser.add_argument("--grf_json", type=str, default=None, help="已有 GRF 系数 JSON")
    parser.add_argument("--analysis_json", type=str, default=None, help="kan_analysis_results.json 提供 x 范围")
    parser.add_argument("--out_dir", type=str, default="./grf_lut", help="LUT 与结果输出目录")
    parser.add_argument("--n_bins", type=int, default=256, help="均匀分割格点数（每段 n_bins+1 个格点）")
    parser.add_argument("--base_kind", type=str, default="mean", choices=("mean", "min", "max"), help="base+offset LUT 的 base 类型")
    parser.add_argument("--margin", type=float, default=0.05, help="x 范围相对校准的扩展比例")
    parser.add_argument("--no_vis", action="store_true", help="不画 LUT 与浮点对比图")
    parser.add_argument("--quantize_bits", type=int, default=0, choices=(0, 2, 4, 6, 8),
                        help="表值量化比特数（默认 0=不量化）。同时作用于 vanilla 与 base+offset，除非用下列参数覆盖")
    parser.add_argument("--vanilla_quantize_bits", type=int, default=None, metavar="N",
                        help="仅 vanilla 表量化比特数 (2/4/6/8)；未设时使用 quantize_bits")
    parser.add_argument("--base_offset_quantize_bits", type=int, default=None, metavar="N",
                        help="base+offset 时 base 与 offset 共用比特数 (2/4/6/8)；未设时使用 quantize_bits")
    parser.add_argument("--base_table_quantize_bits", type=int, default=None, metavar="N",
                        help="仅 base+offset 的 base 表量化比特数 (2/4/6/8)；未设时用 base_offset_quantize_bits")
    parser.add_argument("--offset_table_quantize_bits", type=int, default=None, metavar="N",
                        help="仅 base+offset 的 offset 表量化比特数 (2/4/6/8)；未设时用 base_offset_quantize_bits。可与 base 不同，如 base 8bit offset 4bit")
    parser.add_argument("--calibrate", action="store_true",
                        help="使用校准集收集每层输入激活范围作为 x_min/x_max（需 --checkpoint）")
    parser.add_argument("--calibration_partition", type=str, default="train", choices=("train", "test"),
                        help="校准集分区")
    parser.add_argument("--calibration_samples", type=int, default=500, help="校准集样本数")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    kan_weights_list = None
    activation_ranges = {}

    if args.grf_json and os.path.isfile(args.grf_json):
        kan_weights_list = load_grf_from_json(args.grf_json)
        with open(args.grf_json, "r", encoding="utf-8") as f:
            grf_data = json.load(f)
        for layer in grf_data.get("layers", []):
            if "input_range" in layer:
                activation_ranges[layer["name"]] = layer["input_range"]
    elif args.checkpoint:
        ckpt_path = get_checkpoint_path(args.checkpoint)
        state = load_state_dict(ckpt_path)
        kan_weights_list = collect_kan_polynomial_weights(state)
        if args.analysis_json:
            activation_ranges = load_activation_ranges(args.analysis_json)
        elif getattr(args, "calibrate", False):
            from torch.utils.data import DataLoader
            from inference_and_kan_analysis import build_model_and_load_checkpoint, collect_activation_ranges
            from data import ModelNet40
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            net = build_model_and_load_checkpoint(ckpt_path, device)
            cal_dataset = ModelNet40(partition=args.calibration_partition, num_points=1024)
            cal_loader = DataLoader(cal_dataset, batch_size=min(32, args.calibration_samples), shuffle=True, num_workers=0, drop_last=False)
            activation_ranges = collect_activation_ranges(net, cal_loader, device, max_samples=args.calibration_samples)
            print(f"Calibration ({args.calibration_partition}, n={args.calibration_samples}) activation ranges collected for {len(activation_ranges)} layers.")
            for name, r in sorted(activation_ranges.items()):
                print(f"  {name}: x in [{r['min']:.4f}, {r['max']:.4f}]")
        export_grf_coefficients_to_json(kan_weights_list, activation_ranges or None,
                                        path=os.path.join(args.out_dir, "grf_coefficients.json"))
    else:
        print("请提供 --checkpoint 或 --grf_json")
        return

    if not kan_weights_list:
        print("无 GRF 系数")
        return

    n_bins = args.n_bins
    margin = args.margin
    quantize_bits = args.quantize_bits
    _allowed_bits = (2, 4, 6, 8)
    vb = getattr(args, "vanilla_quantize_bits", None)
    ob = getattr(args, "base_offset_quantize_bits", None)
    base_bits = getattr(args, "base_table_quantize_bits", None)
    off_bits = getattr(args, "offset_table_quantize_bits", None)
    vanilla_quantize_bits = vb if vb is not None and vb in _allowed_bits else (quantize_bits if quantize_bits in _allowed_bits else 0)
    base_offset_quantize_bits = ob if ob is not None and ob in _allowed_bits else (quantize_bits if quantize_bits in _allowed_bits else 0)
    base_table_quantize_bits = base_bits if base_bits is not None and base_bits in _allowed_bits else base_offset_quantize_bits
    offset_table_quantize_bits = off_bits if off_bits is not None and off_bits in _allowed_bits else base_offset_quantize_bits
    if base_table_quantize_bits == 0 and offset_table_quantize_bits > 0:
        base_table_quantize_bits = offset_table_quantize_bits
    if offset_table_quantize_bits == 0 and base_table_quantize_bits > 0:
        offset_table_quantize_bits = base_table_quantize_bits
    lut_config = {"layers": [], "n_bins": n_bins, "quantize_bits": quantize_bits,
                  "vanilla_quantize_bits": vanilla_quantize_bits if vanilla_quantize_bits > 0 else 0,
                  "base_offset_quantize_bits": base_offset_quantize_bits if base_offset_quantize_bits > 0 else 0,
                  "base_table_quantize_bits": base_table_quantize_bits if base_table_quantize_bits > 0 else 0,
                  "offset_table_quantize_bits": offset_table_quantize_bits if offset_table_quantize_bits > 0 else 0}
    lut_arrays = {}
    verify_vanilla = []
    verify_base_offset = []
    verify_vanilla_q = []
    verify_base_offset_q = []

    for idx, item in enumerate(kan_weights_list):
        name = item["name"]
        num_w = item["weight_numerator"]
        den_w = item["weight_denominator"]
        x_min, x_max = -2.0, 2.0
        x_range_source = "default"
        if name in activation_ranges:
            r = activation_ranges[name]
            lo, hi = float(r["min"]), float(r["max"])
            span = max(hi - lo, 1e-6)
            x_min = lo - margin * span
            x_max = hi + margin * span
            x_range_source = "calibration"

        # 1) 原始 f(x) LUT（浮点）
        lut_v = build_lut_vanilla(num_w, den_w, x_min, x_max, n_bins)
        err_v = verify_lut_vs_float(lut_v, num_w, den_w, kind="vanilla")
        verify_vanilla.append({"name": name, **err_v})
        lut_config["layers"].append({
            "name": name, "x_min": x_min, "x_max": x_max, "n_bins": n_bins,
            "x_range_source": x_range_source,
        })
        lut_arrays[f"table_{idx}"] = lut_v["table"]

        # 2) base+offset LUT（浮点，同范围）
        x_orig = np.linspace(x_min, x_max, 500)
        f_curves = eval_grf_vanilla(x_orig, num_w, den_w)
        lut_bo, _ = build_lut_base_offset(f_curves, x_orig, x_min, x_max, n_bins, base_kind=args.base_kind)
        err_bo = verify_lut_vs_float(lut_bo, num_w, den_w, kind="base_offset")
        verify_base_offset.append({"name": name, "base_kind": args.base_kind, **err_bo})
        lut_arrays[f"base_table_{idx}"] = lut_bo["base_table"]
        lut_arrays[f"offset_table_{idx}"] = lut_bo["offset_table"]

        # 3) 量化表与误差（可选；vanilla 与 base+offset 可分别指定比特数）
        if vanilla_quantize_bits > 0:
            t_q, t_scale, t_zp = quantize_table_symmetric(lut_v["table"], vanilla_quantize_bits)
            lut_arrays[f"table_{idx}_q"] = t_q
            lut_arrays[f"table_{idx}_scale"] = np.array(t_scale, dtype=np.float32)
            lut_arrays[f"table_{idx}_zp"] = np.array(t_zp, dtype=np.int32)
            lut_dict_q = {**lut_v, "table_q": t_q, "table_scale": t_scale, "table_zp": t_zp, "num_groups": lut_v["num_groups"]}
            err_vq = verify_lut_quantized_vs_float(lut_dict_q, num_w, den_w, kind="vanilla")
            verify_vanilla_q.append({"name": name, "bits": vanilla_quantize_bits, **err_vq})
        if base_table_quantize_bits > 0 or offset_table_quantize_bits > 0:
            base_q, base_scale, base_zp = quantize_table_symmetric(lut_bo["base_table"], base_table_quantize_bits)
            offset_q, offset_scale, offset_zp = quantize_table_symmetric(lut_bo["offset_table"], offset_table_quantize_bits)
            lut_arrays[f"base_table_{idx}_q"] = base_q
            lut_arrays[f"base_table_{idx}_scale"] = np.array(base_scale, dtype=np.float32)
            lut_arrays[f"base_table_{idx}_zp"] = np.array(base_zp, dtype=np.int32)
            lut_arrays[f"offset_table_{idx}_q"] = offset_q
            lut_arrays[f"offset_table_{idx}_scale"] = np.array(offset_scale, dtype=np.float32)
            lut_arrays[f"offset_table_{idx}_zp"] = np.array(offset_zp, dtype=np.int32)
            lut_bo_q = {"x_min": lut_bo["x_min"], "x_max": lut_bo["x_max"], "n_bins": lut_bo["n_bins"], "num_groups": lut_bo["num_groups"],
                        "base_table_q": base_q, "base_table_scale": base_scale, "base_table_zp": base_zp,
                        "offset_table_q": offset_q, "offset_table_scale": offset_scale, "offset_table_zp": offset_zp}
            err_boq = verify_lut_quantized_vs_float(lut_bo_q, num_w, den_w, kind="base_offset")
            verify_base_offset_q.append({"name": name, "base_kind": args.base_kind, "base_bits": base_table_quantize_bits, "offset_bits": offset_table_quantize_bits, **err_boq})
    lut_config["base_kind"] = args.base_kind

    # 保存 LUT
    np.savez_compressed(os.path.join(args.out_dir, "grf_lut_tables.npz"), **lut_arrays)
    with open(os.path.join(args.out_dir, "grf_lut_config.json"), "w", encoding="utf-8") as f:
        json.dump(lut_config, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("x 范围说明: 每层 x_min/x_max 来自校准集时记为 x_range_source=calibration，否则为 default（[-2,2] 扩展）")
    print("=" * 60)
    print("LUT 与浮点 f(x) 误差（vanilla 浮点查表）")
    for r in verify_vanilla:
        print(f"  {r['name']}: mse={r['mse']:.2e} max_err={r['max_abs_err']:.2e}")
    print("\nLUT 与浮点 f(x) 误差（base+offset 浮点查表合并）")
    for r in verify_base_offset:
        print(f"  {r['name']}: mse={r['mse']:.2e} max_err={r['max_abs_err']:.2e}")
    if vanilla_quantize_bits > 0:
        print(f"\nLUT 与浮点 f(x) 误差（vanilla {vanilla_quantize_bits}bit 量化查表）")
        for r in verify_vanilla_q:
            print(f"  {r['name']}: mse={r['mse']:.2e} max_err={r['max_abs_err']:.2e}")
    if base_table_quantize_bits > 0 or offset_table_quantize_bits > 0:
        print(f"\nLUT 与浮点 f(x) 误差（base+offset 量化查表合并，base={base_table_quantize_bits}bit offset={offset_table_quantize_bits}bit）")
        for r in verify_base_offset_q:
            print(f"  {r['name']}: mse={r['mse']:.2e} max_err={r['max_abs_err']:.2e}")
    print("=" * 60)
    out_res = {
        "verify_vanilla": verify_vanilla,
        "verify_base_offset": verify_base_offset,
        "x_range_note": "x_min/x_max 来自校准集时 config 中 x_range_source=calibration，否则为 default",
    }
    if vanilla_quantize_bits > 0 or base_table_quantize_bits > 0 or offset_table_quantize_bits > 0:
        out_res["verify_vanilla_quantized"] = verify_vanilla_q
        out_res["verify_base_offset_quantized"] = verify_base_offset_q
        out_res["vanilla_quantize_bits"] = vanilla_quantize_bits
        out_res["base_table_quantize_bits"] = base_table_quantize_bits
        out_res["offset_table_quantize_bits"] = offset_table_quantize_bits
    with open(os.path.join(args.out_dir, "grf_lut_verify_results.json"), "w", encoding="utf-8") as f:
        json.dump(out_res, f, indent=2, ensure_ascii=False)
    print(f"LUT 已保存到: {args.out_dir} (grf_lut_config.json + grf_lut_tables.npz)")


if __name__ == "__main__":
    main()
