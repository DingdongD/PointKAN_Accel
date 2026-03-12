#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将模型中所有 KAT_Group 替换为 KAT_Group_LUT，并加载 grf_lut_config.json + grf_lut_tables.npz。
支持 vanilla / base_offset 两种表形式，以及浮点表或量化表。
推理前调用 inject_grf_lut(model, lut_dir, lut_mode=..., use_quantized=...) 即可。
"""

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

from kat_rational import KAT_Group, KAT_Group_LUT


def _find_kat_group_modules(model):
    """按 name 顺序返回 [(full_name, module), ...]，仅 KAT_Group（非 LUT）。"""
    out = []
    for name, m in model.named_modules():
        if isinstance(m, KAT_Group) and not isinstance(m, KAT_Group_LUT):
            out.append((name, m))
    return out


def inject_grf_lut(model, lut_dir, lut_mode="vanilla", use_quantized=False, device=None):
    """
    从 lut_dir 读取 grf_lut_config.json 与 grf_lut_tables.npz，将模型中所有 KAT_Group 替换为 KAT_Group_LUT。

    Args:
        model: pointKANElite 等包含 KAT_Group 的模型。
        lut_dir: 目录路径，内含 grf_lut_config.json 和 grf_lut_tables.npz。
        lut_mode: "vanilla" 使用单表 f(x)；"base_offset" 使用 base 表 + offset 表合并。
        use_quantized: 若 True 且目录下存在量化表（如 table_0_q），则使用量化表推理。
        device: 可选，LUT 张量所在设备；默认与 model 一致。

    Returns:
        替换的层数。
    """
    config_path = os.path.join(lut_dir, "grf_lut_config.json")
    npz_path = os.path.join(lut_dir, "grf_lut_tables.npz")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"未找到 LUT 配置: {config_path}")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"未找到 LUT 表: {npz_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    layers = config.get("layers", [])
    tables_npz = np.load(npz_path)

    kat_list = _find_kat_group_modules(model)
    if len(kat_list) != len(layers):
        raise ValueError(
            f"模型中 KAT_Group 数量 ({len(kat_list)}) 与 LUT 配置层数 ({len(layers)}) 不一致"
        )

    # 是否实际使用量化：请求了量化且存在量化表
    has_vanilla_q = f"table_0_q" in tables_npz
    has_base_offset_q = f"base_table_0_q" in tables_npz
    use_q = use_quantized and (has_vanilla_q if lut_mode == "vanilla" else has_base_offset_q)

    dev = device if device is not None else next(model.parameters()).device
    replaced = 0
    for idx, (name, old_module) in enumerate(kat_list):
        layer_cfg = layers[idx]
        num_groups = int(old_module.num_groups)
        x_min = float(layer_cfg["x_min"])
        x_max = float(layer_cfg["x_max"])
        n_bins = int(layer_cfg["n_bins"])

        if lut_mode == "base_offset":
            if use_q:
                base_q = np.asarray(tables_npz[f"base_table_{idx}_q"])
                base_scale = float(tables_npz[f"base_table_{idx}_scale"])
                base_zp = int(tables_npz[f"base_table_{idx}_zp"])
                offset_q = np.asarray(tables_npz[f"offset_table_{idx}_q"])
                offset_scale = float(tables_npz[f"offset_table_{idx}_scale"])
                offset_zp = int(tables_npz[f"offset_table_{idx}_zp"])
                lut_module = KAT_Group_LUT(
                    num_groups=num_groups,
                    x_min=x_min,
                    x_max=x_max,
                    n_bins=n_bins,
                    device=dev,
                    base_table_q=base_q,
                    base_table_scale=base_scale,
                    base_table_zp=base_zp,
                    offset_table_q=offset_q,
                    offset_table_scale=offset_scale,
                    offset_table_zp=offset_zp,
                )
            else:
                base_table = np.asarray(tables_npz[f"base_table_{idx}"], dtype=np.float32)
                offset_table = np.asarray(tables_npz[f"offset_table_{idx}"], dtype=np.float32)
                lut_module = KAT_Group_LUT(
                    num_groups=num_groups,
                    x_min=x_min,
                    x_max=x_max,
                    n_bins=n_bins,
                    device=dev,
                    base_table=base_table,
                    offset_table=offset_table,
                )
        else:
            if use_q:
                table_q = np.asarray(tables_npz[f"table_{idx}_q"])
                table_scale = float(tables_npz[f"table_{idx}_scale"])
                table_zp = int(tables_npz[f"table_{idx}_zp"])
                lut_module = KAT_Group_LUT(
                    num_groups=num_groups,
                    x_min=x_min,
                    x_max=x_max,
                    n_bins=n_bins,
                    device=dev,
                    table_q=table_q,
                    table_scale=table_scale,
                    table_zp=table_zp,
                )
            else:
                table = np.asarray(tables_npz[f"table_{idx}"], dtype=np.float32)
                lut_module = KAT_Group_LUT(
                    num_groups=num_groups,
                    x_min=x_min,
                    x_max=x_max,
                    n_bins=n_bins,
                    device=dev,
                    table=table,
                )
        lut_module.to(dev)
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], lut_module)
        replaced += 1
    return replaced
