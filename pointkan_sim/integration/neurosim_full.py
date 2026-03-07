"""
NeuroSim 全网络仿真（单次 main 调用，逐层 config + weight/input CSV）
=====================================================================

根据 conv_linear_list（真实前向收集的 Conv1d/Linear 层信息）生成 NeuroSim config CSV 与每层
weight/input CSV，调用 NeuroSIM/main 一次，解析 stdout 得到每层 latency(ns) 与 energy(nJ)，
供 run_sim_with_real_eval 注入 mnsim_context，与 MNSIM 全 NetStruct 并列使用。
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..operators.neurosim_adapter import (
    _parse_layer_results,
    _run_neurosim_main,
    parse_clock_period_ns,
    _write_matrix_activation_conv,
    _write_matrix_weight,
)
from .real_data_eval import ConvLinearLayerInfo


def _stretch_input_conv1d(in_c: int, in_w: int, k_w: int, stride: int = 1) -> np.ndarray:
    num_patches = max(1, (in_w - k_w) // stride + 1)
    K = in_c * k_w
    return np.random.randn(1, num_patches, K).astype(np.float32)


@dataclass
class NeuroSimFullResult:
    """全网络 NeuroSim 逐层结果，与 MNSIM 全 NetStruct 用法兼容。"""
    layer_latency_ns: List[float]
    layer_energy_nj: List[float]
    success: bool
    error: Optional[str] = None
    clock_period_ns: Optional[float] = None  # NeuroSim 实际工作时钟周期(ns)，用于 latency_ns -> cycles 换算


def run_neurosim_full(
    conv_linear_list: List[ConvLinearLayerInfo],
    neurosim_root: str = "/workspace/DNN_NeuroSim_V1.4/Inference_pytorch",
    wl_weight: int = 8,
    wl_activation: int = 8,
    subarray: int = 128,
    parallel_read: int = 128,
    timeout_s: int = 300,
    clk_freq_hz: Optional[float] = None,
    technode: Optional[int] = None,
) -> NeuroSimFullResult:
    """
    对 conv_linear_list 中每一层生成 NeuroSim config 行与 weight/input CSV，
    调用 NeuroSIM/main 一次，解析每层 readLatency + buffer latency、readDynamicEnergy + buffer energy，
    转为 latency_ns 与 energy_nj 列表，供注入 mnsim_context。
    clk_freq_hz: 可选，工作频率上限(Hz)，如 100e6=100MHz；传入 main 实现频率控制。
    technode: 可选，工艺节点(nm)；NeuroSim Technology 仅支持 130/90/65/45/32/22/14/10/7，28 会自动改为 32。
    """
    if not conv_linear_list:
        return NeuroSimFullResult(
            layer_latency_ns=[],
            layer_energy_nj=[],
            success=True,
        )

    # NeuroSim Technology.cpp 仅支持 130,90,65,45,32,22,14,10,7 nm；28nm 会触发 exit(-1)
    if technode is not None and technode == 28:
        technode = 32

    main_dir = os.path.join(neurosim_root, "NeuroSIM")
    if not os.path.isdir(main_dir):
        return NeuroSimFullResult(
            layer_latency_ns=[],
            layer_energy_nj=[],
            success=False,
            error=f"NeuroSIM 目录不存在: {main_dir}",
        )

    config_rows: List[str] = []
    weight_input_pairs: List[Tuple[str, str]] = []
    # 在 NeuroSIM 目录下创建临时目录，便于 main 用相对路径读取（cwd=main_dir）
    tmpdir = tempfile.mkdtemp(dir=main_dir, prefix="pointkan_neurosim_")
    rel_prefix = os.path.basename(tmpdir)

    try:
        for idx, info in enumerate(conv_linear_list):
            if info.layer_type == "conv":
                in_h, in_w = 1, max(1, info.length)
                in_c = max(1, info.in_channels)
                out_c = max(1, info.out_channels)
                k_h, k_w = 1, max(1, info.kernel_size)
                stride = 1
                pool = 0
                num_patches = (in_w - k_w) // stride + 1
                K = in_c * k_h * k_w
                config_rows.append(
                    f"{in_h},{in_w},{in_c},{k_h},{k_w},{out_c},{pool},{stride}"
                )
                weight_path = os.path.join(tmpdir, f"weight{idx}.csv")
                input_path = os.path.join(tmpdir, f"input{idx}.csv")
                weight_data = np.random.randn(out_c, in_c, k_h, k_w).astype(np.float32)
                _write_matrix_weight(weight_data, weight_path)
                input_stretched = _stretch_input_conv1d(in_c, in_w, k_w, stride)
                _write_matrix_activation_conv(input_stretched, wl_activation, input_path)
                weight_input_pairs.append((os.path.join(rel_prefix, f"weight{idx}.csv"), os.path.join(rel_prefix, f"input{idx}.csv")))
            else:
                # fc/linear: in_h=1, in_w=tokens, in_c=in_f, k_h=1, k_w=1, out_c=out_f
                in_f = max(1, info.in_features)
                out_f = max(1, info.out_features)
                M = max(1, info.tokens)
                in_h, in_w = 1, M
                in_c, out_c = in_f, out_f
                k_h, k_w = 1, 1
                pool, stride = 0, 1
                config_rows.append(
                    f"{in_h},{in_w},{in_c},{k_h},{k_w},{out_c},{pool},{stride}"
                )
                weight_path = os.path.join(tmpdir, f"weight{idx}.csv")
                input_path = os.path.join(tmpdir, f"input{idx}.csv")
                num_patches = (in_w - k_w) // stride + 1
                K = in_c * k_h * k_w
                weight_data = np.random.randn(out_c, in_c, k_h, k_w).astype(np.float32)
                _write_matrix_weight(weight_data, weight_path)
                input_stretched = np.random.randn(1, num_patches, K).astype(np.float32)
                _write_matrix_activation_conv(input_stretched, wl_activation, input_path)
                weight_input_pairs.append((os.path.join(rel_prefix, f"weight{idx}.csv"), os.path.join(rel_prefix, f"input{idx}.csv")))

        config_path = os.path.join(tmpdir, "config.csv")
        with open(config_path, "w") as f:
            f.write("\n".join(config_rows) + "\n")

        rel_config = os.path.join(rel_prefix, "config.csv")
        ok, out, err = _run_neurosim_main(
            rel_config,
            weight_input_pairs,
            neurosim_root,
            wl_weight=wl_weight,
            wl_activation=wl_activation,
            subarray=subarray,
            parallel_read=parallel_read,
            timeout_s=timeout_s,
            clk_freq_hz=clk_freq_hz,
            technode=technode,
        )

        if not ok or not out:
            return NeuroSimFullResult(
                layer_latency_ns=[],
                layer_energy_nj=[],
                success=False,
                error=err or "NeuroSim main failed or no stdout",
            )

        parsed = _parse_layer_results(out)
        if len(parsed) != len(conv_linear_list):
            return NeuroSimFullResult(
                layer_latency_ns=[],
                layer_energy_nj=[],
                success=False,
                error=f"NeuroSim parsed {len(parsed)} layers, expected {len(conv_linear_list)}",
            )

        layer_latency_ns = []
        layer_energy_nj = []
        for (read_ns, read_pj, buf_ns, buf_pj) in parsed:
            layer_latency_ns.append(read_ns + buf_ns)
            layer_energy_nj.append((read_pj + buf_pj) / 1000.0)

        clock_period_ns = parse_clock_period_ns(out)

        return NeuroSimFullResult(
            layer_latency_ns=layer_latency_ns,
            layer_energy_nj=layer_energy_nj,
            success=True,
            clock_period_ns=clock_period_ns,
        )
    finally:
        if os.path.isdir(tmpdir):
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass
