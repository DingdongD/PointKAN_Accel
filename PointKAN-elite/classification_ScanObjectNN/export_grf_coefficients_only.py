#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仅从 checkpoint 导出 grf_coefficients.json（不加载 ScanObjectNN 数据、不做校准）。

适合快速接 GQA-LUT PWL；更精确时请用 inference_and_kan_analysis.py --export_grf_json（含校准 input_range）。
"""

import argparse
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
for d in (SCRIPT_DIR, ROOT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

from grf_kan_json_and_residual import export_grf_coefficients_to_json
from inference_and_kan_analysis import collect_kan_polynomial_weights, get_checkpoint_path, load_state_dict_for_model


def main():
    parser = argparse.ArgumentParser(description="仅 checkpoint 导出 GRF 系数 JSON")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/pointKANElite-scan_exp-42",
        help="目录（best_checkpoint.pth）或 .pth/.pt",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default="grf_export_out/grf_coefficients.json",
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    out_path = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    device = torch.device(args.device)
    ckpt_path = get_checkpoint_path(args.checkpoint)
    print(f"加载: {ckpt_path}")
    state = load_state_dict_for_model(ckpt_path, device)
    kan = collect_kan_polynomial_weights(state)
    print(f"KAT_Group 层数: {len(kan)}")
    export_grf_coefficients_to_json(kan, {}, out_path)
    print(f"已写入: {out_path}")


if __name__ == "__main__":
    main()
