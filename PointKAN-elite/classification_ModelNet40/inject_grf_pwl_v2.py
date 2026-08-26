#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将模型中每个 KAT_Group 按模块路径名替换为 KAT_Group_PWL，参数来自 grf_pwl_trainer_v2 导出的 pwl_v2.json。

层名需与 JSON 中 layers[].name 一致（与 export / collect_kan 使用的命名一致）。
bit 默认取 meta.deploy_breakpoint_bits，可用 inject 时传入 bit_key 覆盖。
"""

import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
for d in (SCRIPT_DIR, ROOT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

from kat_rational import KAT_Group, KAT_Group_LUT, KAT_Group_PWL, KAT_Group_PWL_Triton


def _find_kat_group_modules(model):
    """按遍历顺序返回 [(full_name, module), ...]，仅原始 KAT_Group（排除 LUT/PWL）。"""
    out = []
    for name, m in model.named_modules():
        if isinstance(m, KAT_Group) and not isinstance(m, (KAT_Group_LUT, KAT_Group_PWL)):
            out.append((name, m))
    return out


def _resolve_bit_key(v2_payload, bit_key):
    if bit_key is not None:
        return str(int(bit_key))
    meta = v2_payload.get("meta", {})
    if meta.get("deploy_breakpoint_bits") is not None:
        return str(int(meta["deploy_breakpoint_bits"]))
    br = meta.get("decimal_bit_range", [0, 6])
    return str(int(br[1]))


def _group_specs_from_json(grp, x_min, x_max, *, layer_name="?", group_index=None):
    """单组 JSON -> 与 KAT_Group_PWL 缓冲区一致的标量/向量。"""
    if not grp.get("valid", True):
        gid = grp.get("group_id")
        if gid is None and group_index is not None:
            gid = group_index
        reason = grp.get("reason", "")
        raise ValueError(
            "层 %s group %s: valid=False，无法部署 PWL（%s）。"
            "低比特量化后断点可能碰撞或不再严格递增；请改用更高 bit（如 meta.deploy_breakpoint_bits）或重新训练/导出 v2。"
            % (layer_name, gid, reason or "无 reason 字段")
        )
    mode = str(grp.get("mode", "pwl"))
    if mode == "zero":
        return {
            "mode": 0,
            "const": 0.0,
            "lin_m": 0.0,
            "lin_b": 0.0,
            "split": None,
            "slopes": None,
            "intercepts": None,
            "K": 0,
        }
    if mode == "const":
        b = float(grp["intercepts"][0])
        return {
            "mode": 1,
            "const": b,
            "lin_m": 0.0,
            "lin_b": 0.0,
            "split": None,
            "slopes": None,
            "intercepts": None,
            "K": 0,
        }
    if mode == "linear":
        m = float(grp["slopes"][0])
        b = float(grp["intercepts"][0])
        return {
            "mode": 2,
            "const": 0.0,
            "lin_m": m,
            "lin_b": b,
            "split": None,
            "slopes": None,
            "intercepts": None,
            "K": 0,
        }
    inner = [float(v) for v in grp.get("breakpoints", [])]
    inner_arr = np.asarray(inner, dtype=np.float64)
    inner_arr = np.clip(inner_arr, x_min, x_max)
    inner_arr.sort()
    split_points = np.concatenate(([x_min], inner_arr, [x_max])).astype(np.float64)
    split_points.sort()
    s = np.asarray(grp["slopes"], dtype=np.float64).reshape(-1)
    b = np.asarray(grp["intercepts"], dtype=np.float64).reshape(-1)
    K = int(s.size)
    if K != b.size:
        raise ValueError("slopes/intercepts 长度不一致")
    if K != len(split_points) - 1:
        raise ValueError("PWL 区间数与 split_points 不一致")
    return {
        "mode": 3,
        "const": 0.0,
        "lin_m": 0.0,
        "lin_b": 0.0,
        "split": split_points,
        "slopes": s,
        "intercepts": b,
        "K": K,
    }


def _build_pwl_module_from_layer(layer_v2, bit_key, num_groups, device, *, impl: str = "torch"):
    x_min = float(layer_v2["x_range"]["min"])
    x_max = float(layer_v2["x_range"]["max"])
    bits = layer_v2.get("bits", {})
    if bit_key not in bits:
        avail = sorted(bits.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)
        lname = str(layer_v2.get("name", "?"))
        raise ValueError(
            "层 %r 无 bits[%s]；本文件该层仅有: %s。"
            "说明导出 v2 时断点比特扫描未包含该档，请扩大训练/导出比特范围后重新生成 pwl_v2.json，或改用已有档位（如 6）。"
            % (lname, bit_key, avail)
        )
    groups = bits[bit_key].get("groups", [])
    if len(groups) != int(num_groups):
        raise ValueError("JSON groups 数 %s != 模块 num_groups %s" % (len(groups), num_groups))

    lname = str(layer_v2.get("name", "?"))
    specs = [
        _group_specs_from_json(g, x_min, x_max, layer_name=lname, group_index=i)
        for i, g in enumerate(groups)
    ]
    max_k = max((s["K"] for s in specs), default=0)
    max_split_len = max_k + 1 if max_k > 0 else 1

    G = int(num_groups)
    modes = np.zeros(G, dtype=np.int64)
    const_vals = np.zeros(G, dtype=np.float32)
    lin_m = np.zeros(G, dtype=np.float32)
    lin_b = np.zeros(G, dtype=np.float32)
    n_intervals = np.zeros(G, dtype=np.int64)
    split_stacked = np.full((G, max_split_len), x_max, dtype=np.float32)
    slope_stacked = np.zeros((G, max_k), dtype=np.float32)
    intercept_stacked = np.zeros((G, max_k), dtype=np.float32)

    for g, s in enumerate(specs):
        modes[g] = s["mode"]
        const_vals[g] = s["const"]
        lin_m[g] = s["lin_m"]
        lin_b[g] = s["lin_b"]
        n_intervals[g] = s["K"]
        if s["mode"] == 3 and s["K"] > 0:
            sp = s["split"].astype(np.float32)
            split_stacked[g, : sp.size] = sp
            slope_stacked[g, : s["K"]] = s["slopes"].astype(np.float32)
            intercept_stacked[g, : s["K"]] = s["intercepts"].astype(np.float32)

    if impl == "torch":
        mod = KAT_Group_PWL(
            num_groups=G,
            x_min=x_min,
            x_max=x_max,
            modes=modes,
            const_vals=const_vals,
            lin_m=lin_m,
            lin_b=lin_b,
            split_stacked=split_stacked,
            slope_stacked=slope_stacked,
            intercept_stacked=intercept_stacked,
            n_intervals=n_intervals,
        )
    elif impl == "triton":
        mod = KAT_Group_PWL_Triton(
            num_groups=G,
            x_min=x_min,
            x_max=x_max,
            modes=modes,
            const_vals=const_vals,
            lin_m=lin_m,
            lin_b=lin_b,
            split_stacked=split_stacked,
            slope_stacked=slope_stacked,
            intercept_stacked=intercept_stacked,
            n_intervals=n_intervals,
        )
    else:
        raise ValueError("impl must be 'torch' or 'triton'")
    return mod.to(device)


def inject_grf_pwl_v2(model, v2_json_path, bit_key=None, device=None, *, impl: str = "torch"):
    """
    读取 pwl_v2.json，按 named_modules 中 KAT_Group 的路径名在 JSON 中查找对应层并替换为 KAT_Group_PWL。

    Returns:
        (替换层数, 实际使用的 bit 字符串).
    """
    if not os.path.isfile(v2_json_path):
        raise FileNotFoundError(v2_json_path)
    with open(v2_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    bit = _resolve_bit_key(payload, bit_key)
    v2_by_name = {str(l["name"]): l for l in payload.get("layers", [])}

    dev = device if device is not None else next(model.parameters()).device
    kat_list = _find_kat_group_modules(model)
    # 注入前统一检查，避免先跑完 vanilla 再在某一層才报错
    for name, _old in kat_list:
        if name not in v2_by_name:
            raise KeyError(
                "模块路径 %r 在 pwl_v2.json 中无对应层。JSON 中含 %s 层。"
                % (name, len(v2_by_name))
            )
        lv = v2_by_name[name]
        bmap = lv.get("bits", {})
        if bit not in bmap:
            avail = sorted(bmap.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)
            raise ValueError(
                "全局使用 bits[%s]，但层 %r 中不存在该键；该层仅有: %s。"
                "请改用 --bit 为上述之一，或重新导出含更高断点比特的 pwl_v2.json。"
                % (bit, name, avail)
            )
    if impl not in ("torch", "triton"):
        raise ValueError("impl must be 'torch' or 'triton'")
    replaced = 0
    for name, old_module in kat_list:
        if name not in v2_by_name:
            raise KeyError(
                "模块路径 %r 在 pwl_v2.json 中无对应层。JSON 中含 %s 层。"
                % (name, len(v2_by_name))
            )
        layer_v2 = v2_by_name[name]
        num_groups = int(old_module.num_groups)
        pwl_mod = _build_pwl_module_from_layer(layer_v2, bit, num_groups, dev, impl=impl)
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], pwl_mod)
        replaced += 1
    return replaced, bit
