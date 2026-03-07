#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroSim 全网络调用 — 独立测试脚本
====================================

用于单独运行、排查 NeuroSim main 崩溃（如 returncode=-11/SIGSEGV）。
不依赖 pointkan_sim 包，可直接：python test_neurosim_full_standalone.py [选项]

功能：
  - 生成与 NeuroSim main 兼容的 config.csv 及每层 weight/input CSV；
  - 在 NeuroSIM 目录下以相对路径调用 main，解析 stdout 得到每层 latency(ns) 与 energy(pJ)；
  - 支持「简化层列表」「PointKAN-elite 完整 36 层」或「从 checkpoint + 真实数据抓取每层输入/权重」三种模式。

数据来源说明：
  - 未指定 --checkpoint 时：权重与输入均为随机数，仅用于结构/格式测试。
  - 指定 --checkpoint 时：加载 ckpt 与 ModelNet40 一个 batch，前向一次并用 hook 抓取每层真实
    输入激活与权重，写入 CSV，实现基于真实数据与权重的 NeuroSim 仿真。

tokens 含义（本工程）：
  - 对 Linear 层：表示该层输入中「位置维」长度。LFP 中为当前 stage 的点数（512/256/128/64）；
    Classifier 为 1（全局池化后每样本一个向量）。
  - 对 Conv1d 层不使用 tokens，使用 length 表示空间长度 L。

输出 CSV（--output_dir 指定目录，默认 sim_results）：
  - neurosim_layer_detail.csv：逐层 read/buffer/ic、breakdown、total_latency_ns、total_energy_nj、num_tiles、speedup；若提供层列表则含 buffer_data_bits、buffer_data_bytes（该层输入数据量，近似层间 buffer 用量）
  - neurosim_summary.csv：总延迟、总能耗、总 tile 数、clock、breakdown 汇总；NeuroSim buffer 规模（buffer_size_bits、num_buffer_core、total_buffer_capacity_bits/bytes）
  - neurosim_breakdown_by_operator.csv：按算子类型（conv / fc）的端到端延迟与能耗构成（层数、总延迟、总能耗及 ADC/Buffer/Accum/Other/IC 分项）
工作频率：从 stdout 解析 clkPeriod 或 Chip clock period，输出周期(ns)与频率(MHz)。调节频率需改 NeuroSim 源码 Param.cpp 的 clkFreq（Hz）。

用法:
  python test_neurosim_full_standalone.py
  python test_neurosim_full_standalone.py --num_layers 2 --output_dir sim_results
  python test_neurosim_full_standalone.py --pointkan_elite_full   # 完整 36 层仿真（随机数据）
  python test_neurosim_full_standalone.py --checkpoint PointKAN-elite/classification_ModelNet40/checkpoints/pointKANElite-pointkan_elite_exp-42  # 真实 ckpt+数据
  python test_neurosim_full_standalone.py --pointkan_elite_full --keep_tmp
  python test_neurosim_full_standalone.py --neurosim_root /path/to/DNN_NeuroSim_V1.4/Inference_pytorch
    
SubArray 尺寸：main 的 argv[4]/argv[5] 对应 numRowSubArray、numRowParallel；numColSubArray 由 NeuroSim Param 固定为 128。
  默认 --subarray 128 --parallel_read 128 即得到 128×128。若 stdout 报 "SubArray Size is too large"，可改为 --subarray 32 --parallel_read 32。
输出 latency / energy：NeuroSim 按层汇总，每层的 readLatency 与 readDynamicEnergy 为该层**整层**的乘加（全部 tile、完整激活×权重）的延迟与能耗，不是单 subarray。
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    print("需要 numpy: pip install numpy")
    sys.exit(1)


# =============================================================================
# 层信息数据结构（仅用于生成 config/weight/input CSV，与 pointkan_sim 的 ConvLinearLayerInfo 字段对齐）
# =============================================================================

@dataclass
class LayerSpec:
    """
    单层规格：Conv1d 或 Linear（FC）。
    用于本脚本内生成 NeuroSim config 行与 weight/input CSV，不依赖真实权重/激活。
    """
    name: str
    layer_type: str  # "conv" | "fc"
    # Conv1d 用
    in_channels: int = 0
    out_channels: int = 0
    kernel_size: int = 1
    length: int = 0   # 空间长度 L（输入序列长度）
    # Linear 用
    in_features: int = 0
    out_features: int = 0
    tokens: int = 0    # Linear 的「位置数」：LFP 中为当前点数(512/256/128/64)，Classifier 为 1


# =============================================================================
# 与 neurosim_adapter / neurosim_full 一致的 CSV 生成（与 DNN_NeuroSim VMM_CIM 格式兼容）
# =============================================================================

def _dec2bin_vec(x, n: int) -> Tuple[list, list]:
    """
    将浮点数组量化为 n 位有符号定点，返回各 bit 平面（MSB 符号位 + 后续数值位）及 scale。
    用于激活的多 bit 展开，与 NeuroSim 读入的 '0'/'1' 字符串一致。
    """
    out, scale = [], []
    delta = 1.0 / (2 ** (n - 1))
    x_int = np.floor(x / delta).astype(np.int64)
    base = 2 ** (n - 1)
    y = (x_int < 0).astype(np.float64)
    rest = x_int + base * y
    out.append(y.copy())
    scale.append(-base * delta)
    for _ in range(n - 1):
        base /= 2
        y = (rest >= base).astype(np.float64)
        rest -= base * y
        out.append(y.copy())
        scale.append(base * delta)
    return out, scale


def _write_matrix_weight(weight_matrix: np.ndarray, filepath: str) -> None:
    """
    权重重排为 (K, C_out)：每行一个「输入通道×卷积核」向量，共 C_out 行。
    NeuroSim 期望 weight CSV 形状与 config 中 in_c*k_h*k_w × out_c 一致（列主序展开）。
    """
    w = weight_matrix.reshape(weight_matrix.shape[0], -1).T
    np.savetxt(filepath, w, delimiter=",", fmt="%.6f")


def _write_matrix_activation_conv(input_matrix: np.ndarray, bit_length: int, filepath: str) -> None:
    """
    将 stretch 后的激活 (1, num_patches, K) 量化为 bit_length 位后写入 CSV。
    NeuroSim Chip.cpp CopyInput 期望：行数 = K（与权重行一致），列数 = num_patches*bit_length；
    列顺序为：patch0 的 bit0..bit7，patch1 的 bit0..bit7，…（与 VMM_CIM.ipynb write_matrix_activation_conv 一致）。
    """
    data = input_matrix[0, :, :]
    num_patches, K = data.shape
    filled_bin, _ = _dec2bin_vec(data.ravel(), bit_length)
    # 中间矩阵 (num_patches, K*bit_length)
    filled_b = np.zeros((num_patches, K * bit_length), dtype="U1")
    for i, b in enumerate(filled_bin):
        b_2d = b.reshape(num_patches, K)
        filled_b[:, i::bit_length] = b_2d
    # 转为 (K, num_patches*bit_length) 以匹配 NeuroSim 读入布局
    out = np.empty((K, num_patches * bit_length), dtype="U1")
    for k in range(K):
        for p in range(num_patches):
            out[k, p * bit_length : (p + 1) * bit_length] = filled_b[p, k * bit_length : (k + 1) * bit_length]
    np.savetxt(filepath, out, delimiter=",", fmt="%s")


def _stretch_input_conv1d(in_c: int, in_w: int, k_w: int, stride: int = 1) -> np.ndarray:
    """
    构造 Conv1d 的「滑窗展开」输入：(1, num_patches, K)，K = in_c * k_w。
    num_patches = (in_w - k_w) // stride + 1。本脚本用随机数填充，仅做维度与格式测试。
    """
    num_patches = max(1, (in_w - k_w) // stride + 1)
    K = in_c * k_w
    return np.random.randn(1, num_patches, K).astype(np.float32)


def _stretch_conv_input_from_array(x: np.ndarray, k_w: int, stride: int = 1) -> np.ndarray:
    """
    将真实 Conv1d 输入从 (B, C_in, L) 按滑窗展开为 (1, num_patches, K)，K = C_in * k_w。
    使用第一个样本 (x[0])，与 NeuroSim 单 batch 输入一致。
    """
    # x: (B, C_in, L)
    if x.ndim != 3:
        raise ValueError("Conv input must be (B, C_in, L)")
    L = x.shape[2]
    C_in = x.shape[1]
    num_patches = max(1, (L - k_w) // stride + 1)
    K = C_in * k_w
    out = np.zeros((1, num_patches, K), dtype=np.float32)
    for i in range(num_patches):
        start = i * stride
        patch = x[0, :, start : start + k_w]  # (C_in, k_w)
        out[0, i, :] = patch.ravel()
    return out


# =============================================================================
# 解析 NeuroSim main 的 stdout（含 breakdown：ADC / Accumulation / Other 及每层 tile 数）
# =============================================================================

PATTERN_READ_LATENCY = re.compile(r"layer(\d+)'s readLatency is:\s*([\d.eE+-]+)\s*ns")
PATTERN_READ_ENERGY = re.compile(r"layer\d+'s readDynamicEnergy is:\s*([\d.eE+-]+)\s*pJ")
PATTERN_BUFFER_LATENCY = re.compile(r"layer\d+'s buffer latency is:\s*([\d.eE+-]+)\s*ns")
PATTERN_BUFFER_ENERGY = re.compile(r"layer\d+'s buffer readDynamicEnergy is:\s*([\d.eE+-]+)\s*pJ")
PATTERN_IC_ENERGY = re.compile(r"layer\d+'s ic readDynamicEnergy is:\s*([\d.eE+-]+)\s*pJ")
PATTERN_ADC_LATENCY = re.compile(r"ADC \(or S/As and precharger for SRAM\) readLatency is\s*:\s*([\d.eE+-]+)\s*ns")
PATTERN_ADC_ENERGY = re.compile(r"ADC \(or S/As and precharger for SRAM\) readDynamicEnergy is\s*:\s*([\d.eE+-]+)\s*pJ")
PATTERN_ACCUM_LATENCY = re.compile(r"Accumulation Circuits.*readLatency is\s*:\s*([\d.eE+-]+)\s*ns")
PATTERN_ACCUM_ENERGY = re.compile(r"Accumulation Circuits.*readDynamicEnergy is\s*:\s*([\d.eE+-]+)\s*pJ")
PATTERN_OTHER_LATENCY = re.compile(r"Other Peripheries.*readLatency is\s*:\s*([\d.eE+-]+)\s*ns")
PATTERN_OTHER_ENERGY = re.compile(r"Other Peripheries.*readDynamicEnergy is\s*:\s*([\d.eE+-]+)\s*pJ")
PATTERN_TILE_LAYER = re.compile(r"layer(\d+):\s*(\d+)")
PATTERN_CLK_PERIOD = re.compile(r"clkPeriod:\s*([\d.eE+-]+)")
PATTERN_CHIP_CLK_PERIOD = re.compile(r"Chip clock period is:\s*([\d.eE+-]+)\s*ns")


@dataclass
class NeuroSimLayerResult:
    """单层 NeuroSim 解析结果（含 breakdown、tile 数、speedup）。"""
    layer_index: int
    read_latency_ns: float
    read_energy_pj: float
    buffer_latency_ns: float
    buffer_energy_pj: float
    ic_energy_pj: float
    adc_latency_ns: float
    adc_energy_pj: float
    accum_latency_ns: float
    accum_energy_pj: float
    other_latency_ns: float
    other_energy_pj: float
    num_tiles: int = 0
    speedup: int = 0

    @property
    def total_latency_ns(self) -> float:
        return self.read_latency_ns + self.buffer_latency_ns

    @property
    def total_energy_nj(self) -> float:
        return (self.read_energy_pj + self.buffer_energy_pj + self.ic_energy_pj) / 1000.0


def parse_tile_count_per_layer(stdout: str) -> List[int]:
    """解析 '# of tile used for each layer' 下 layerN: M，返回每层 tile 数列表（tiles[i] 对应 layer i+1）。"""
    lines = stdout.splitlines()
    tiles: List[int] = []
    in_section = False
    for line in lines:
        if "# of tile used for each layer" in line:
            in_section = True
            continue
        if in_section:
            if line.strip().startswith("---") or "Speed-up" in line or "Utilization" in line:
                break
            m = PATTERN_TILE_LAYER.match(line.strip())
            if m:
                idx, count = int(m.group(1)), int(m.group(2))
                while len(tiles) < idx:
                    tiles.append(0)
                tiles[idx - 1] = count
    return tiles


def parse_speedup_per_layer(stdout: str) -> List[int]:
    """解析 'Speed-up of each layer' 下 layerN: M，返回每层 speedup 列表。"""
    lines = stdout.splitlines()
    speedups: List[int] = []
    in_section = False
    for line in lines:
        if "Speed-up of each layer" in line:
            in_section = True
            continue
        if in_section:
            if line.strip().startswith("---") or "Utilization" in line:
                break
            m = PATTERN_TILE_LAYER.match(line.strip())
            if m:
                idx, val = int(m.group(1)), int(m.group(2))
                while len(speedups) < idx:
                    speedups.append(0)
                speedups[idx - 1] = val
    return speedups


def parse_clock_period_ns(stdout: str) -> Optional[float]:
    """解析时钟周期(ns)。优先用 'Chip clock period is: X ns'（用户设定/最终周期），否则用 'clkPeriod: X'（秒，关键路径）。"""
    chip_period_ns: Optional[float] = None
    clk_period_sec: Optional[float] = None
    for line in stdout.splitlines():
        m = PATTERN_CHIP_CLK_PERIOD.search(line)
        if m:
            chip_period_ns = float(m.group(1))
        m = PATTERN_CLK_PERIOD.search(line)
        if m:
            clk_period_sec = float(m.group(1))
    if chip_period_ns is not None:
        return chip_period_ns
    if clk_period_sec is not None:
        return clk_period_sec * 1e9
    return None


def parse_layer_results_full(stdout: str) -> List[NeuroSimLayerResult]:
    """
    从 main 的 stdout 按层解析完整结果：read/buffer/ic、breakdown（ADC/Accum/Other）、tile 数、speedup。
    返回 NeuroSimLayerResult 列表。
    """
    tile_counts = parse_tile_count_per_layer(stdout)
    speedups = parse_speedup_per_layer(stdout)
    lines = stdout.splitlines()
    results: List[NeuroSimLayerResult] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        read_m = PATTERN_READ_LATENCY.search(line)
        if read_m and "buffer" not in line:
            layer_idx = int(read_m.group(1))
            read_ns = float(read_m.group(2))
            read_pj = buf_ns = buf_pj = ic_pj = 0.0
            adc_ns = adc_pj = accum_ns = accum_pj = other_ns = other_pj = 0.0
            for j in range(i + 1, min(i + 25, len(lines))):
                l = lines[j]
                if "readDynamicEnergy is:" in l and "buffer" not in l and "leakage" not in l and "ic " not in l:
                    m = PATTERN_READ_ENERGY.search(l)
                    if m:
                        read_pj = float(m.group(1))
                if "buffer latency" in l:
                    m = PATTERN_BUFFER_LATENCY.search(l)
                    if m:
                        buf_ns = float(m.group(1))
                if "buffer readDynamicEnergy" in l:
                    m = PATTERN_BUFFER_ENERGY.search(l)
                    if m:
                        buf_pj = float(m.group(1))
                if "ic readDynamicEnergy" in l:
                    m = PATTERN_IC_ENERGY.search(l)
                    if m:
                        ic_pj = float(m.group(1))
                if "ADC " in l and "readLatency" in l:
                    m = PATTERN_ADC_LATENCY.search(l)
                    if m:
                        adc_ns = float(m.group(1))
                if "ADC " in l and "readDynamicEnergy" in l:
                    m = PATTERN_ADC_ENERGY.search(l)
                    if m:
                        adc_pj = float(m.group(1))
                if "Accumulation" in l and "readLatency" in l:
                    m = PATTERN_ACCUM_LATENCY.search(l)
                    if m:
                        accum_ns = float(m.group(1))
                if "Accumulation" in l and "readDynamicEnergy" in l:
                    m = PATTERN_ACCUM_ENERGY.search(l)
                    if m:
                        accum_pj = float(m.group(1))
                if "Other Peripheries" in l and "readLatency" in l:
                    m = PATTERN_OTHER_LATENCY.search(l)
                    if m:
                        other_ns = float(m.group(1))
                if "Other Peripheries" in l and "readDynamicEnergy" in l:
                    m = PATTERN_OTHER_ENERGY.search(l)
                    if m:
                        other_pj = float(m.group(1))
                        break
            num_tiles = tile_counts[layer_idx - 1] if layer_idx <= len(tile_counts) else 0
            speedup = speedups[layer_idx - 1] if layer_idx <= len(speedups) else 0
            results.append(NeuroSimLayerResult(
                layer_index=layer_idx,
                read_latency_ns=read_ns,
                read_energy_pj=read_pj,
                buffer_latency_ns=buf_ns,
                buffer_energy_pj=buf_pj,
                ic_energy_pj=ic_pj,
                adc_latency_ns=adc_ns,
                adc_energy_pj=adc_pj,
                accum_latency_ns=accum_ns,
                accum_energy_pj=accum_pj,
                other_latency_ns=other_ns,
                other_energy_pj=other_pj,
                num_tiles=num_tiles,
                speedup=speedup,
            ))
        i += 1
    return results


def parse_layer_results(stdout: str) -> List[Tuple[float, float, float, float]]:
    """
    从 main 的 stdout 按层解析：每层返回 (read_ns, read_pj, buffer_ns, buffer_pj)。
    用于汇总每层 latency = read_ns + buffer_ns，energy_nj = (read_pj + buffer_pj)/1000。
    """
    full = parse_layer_results_full(stdout)
    return [
        (r.read_latency_ns, r.read_energy_pj, r.buffer_latency_ns, r.buffer_energy_pj)
        for r in full
    ]


# =============================================================================
# 层列表构造：简化版（前 N 层）与 PointKAN-elite 完整 36 层
# =============================================================================

def make_pointkan_like_layers(num_layers: int) -> List[LayerSpec]:
    """
    构造「与 PointKAN-elite 前几层顺序/类型相近」的简化层列表（小维度，便于快速排查）。
    交替 conv + fc，维度偏小以避免 NeuroSim SubArray/资源超限。
    """
    layers = []
    for i in range(num_layers):
        if i % 2 == 0:
            layers.append(LayerSpec(
                name=f"conv_{i}",
                layer_type="conv",
                in_channels=min(256, 128),
                out_channels=min(128, 256),
                kernel_size=1,
                length=min(512, 64),
            ))
        else:
            layers.append(LayerSpec(
                name=f"fc_{i}",
                layer_type="fc",
                in_features=min(256, 128),
                out_features=min(128, 256),
                tokens=1,
            ))
    return layers


def get_pointkan_elite_full_layers() -> List[LayerSpec]:
    """
    返回 PointKAN-elite 分类网络（ModelNet40）的完整 conv/linear 序列，共 36 层。
    与 pointkan_sim/network/pointkan_elite_net.py 及真实前向 collect_conv_linear_from_forward 顺序一致：
      - 输入 [B,3,1024]，embed_dim=32，4 stage，reducers=[2,2,2,2]，dim_expansion=[2,2,2,2]。
      - Embedding: 1×Conv1d(3→32, k=1, L=1024)
      - 每 stage: LFP 内 2×Conv1d + 4×Linear，GFP 内 2×Conv1d（L 随 stage 减半）
      - Classifier: 3×Linear(512→512, 512→256, 256→40)，tokens=1
    """
    # 默认配置与 create_pointkan_elite_classification 一致
    input_points = 1024
    embed_dim = 32
    num_classes = 40
    dim_expansion = [2, 2, 2, 2]
    reducers = [2, 2, 2, 2]

    layers: List[LayerSpec] = []
    current_points = input_points
    current_channels = embed_dim

    # 1. Embedding: Conv1d(3→32, k=1), L=1024
    layers.append(LayerSpec(
        name="embedding",
        layer_type="conv",
        in_channels=3,
        out_channels=embed_dim,
        kernel_size=1,
        length=input_points,
    ))

    for s in range(4):
        out_points = current_points // reducers[s]
        out_channels = current_channels * dim_expansion[s]

        # LFP-stage: transfer1 Conv → op1 fc1, fc2 → transfer2 Conv → op2 fc1, fc2
        layers.append(LayerSpec(
            name=f"LFP-stage{s}-transfer1",
            layer_type="conv",
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            length=out_points,
        ))
        layers.append(LayerSpec(
            name=f"LFP-stage{s}-op1-fc1",
            layer_type="fc",
            in_features=out_channels,
            out_features=out_channels,
            tokens=out_points,
        ))
        layers.append(LayerSpec(
            name=f"LFP-stage{s}-op1-fc2",
            layer_type="fc",
            in_features=out_channels,
            out_features=out_channels,
            tokens=out_points,
        ))
        layers.append(LayerSpec(
            name=f"LFP-stage{s}-transfer2",
            layer_type="conv",
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            length=out_points,
        ))
        layers.append(LayerSpec(
            name=f"LFP-stage{s}-op2-fc1",
            layer_type="fc",
            in_features=out_channels,
            out_features=out_channels,
            tokens=out_points,
        ))
        layers.append(LayerSpec(
            name=f"LFP-stage{s}-op2-fc2",
            layer_type="fc",
            in_features=out_channels,
            out_features=out_channels,
            tokens=out_points,
        ))

        # GFP (stage-s): 2×Conv1d
        layers.append(LayerSpec(
            name=f"stage-{s}-gfp1",
            layer_type="conv",
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            length=out_points,
        ))
        layers.append(LayerSpec(
            name=f"stage-{s}-gfp2",
            layer_type="conv",
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            length=out_points,
        ))

        current_points = out_points
        current_channels = out_channels

    # Classifier: FC1(512→512), FC2(512→256), FC3(256→40)，tokens=1
    fc_dim = current_channels  # 512
    mid_dim = fc_dim // 2     # 256
    layers.append(LayerSpec(
        name="classifier-fc1",
        layer_type="fc",
        in_features=fc_dim,
        out_features=fc_dim,
        tokens=1,
    ))
    layers.append(LayerSpec(
        name="classifier-fc2",
        layer_type="fc",
        in_features=fc_dim,
        out_features=mid_dim,
        tokens=1,
    ))
    layers.append(LayerSpec(
        name="classifier-fc3",
        layer_type="fc",
        in_features=mid_dim,
        out_features=num_classes,
        tokens=1,
    ))

    return layers


# =============================================================================
# 调用 NeuroSIM/main
# =============================================================================

def _buffer_data_bits_for_layer(spec: LayerSpec, num_bits_input: int) -> int:
    """从层规格计算该层输入激活的 bit 数（即经层间 buffer 的数据量）。"""
    if spec.layer_type == "conv":
        return num_bits_input * max(1, spec.in_channels) * 1 * max(1, spec.length)
    return num_bits_input * max(1, spec.in_features) * max(1, spec.tokens)


def compute_neurosim_buffer_size(
    layers: List[LayerSpec],
    num_bits_input: int,
    sync_data_transfer: bool = True,
    global_buffer_core_row: int = 128,
    global_buffer_core_col: int = 128,
) -> dict:
    """
    按 NeuroSim Chip 逻辑计算 global buffer 规模（与 Param.cpp / Chip.cpp 一致）。
    - bufferSize = numBitInput * maxLayerInput * (2 if sync_data_transfer else 1)
    - numBufferCore = ceil(bufferSize / (globalBufferCoreSizeRow * globalBufferCoreSizeCol))
    - 总容量(bits) = numBufferCore * core_row * core_col
    返回 dict: buffer_size_bits, max_layer_input_elements, num_buffer_core, total_buffer_capacity_bits, total_buffer_capacity_bytes
    """
    if not layers:
        return {
            "buffer_size_bits": 0,
            "max_layer_input_elements": 0,
            "num_buffer_core": 0,
            "total_buffer_capacity_bits": 0,
            "total_buffer_capacity_bytes": 0,
        }
    max_elems = 0
    for spec in layers:
        if spec.layer_type == "conv":
            n = max(1, spec.in_channels) * 1 * max(1, spec.length)
        else:
            n = max(1, spec.in_features) * max(1, spec.tokens)
        if n > max_elems:
            max_elems = n
    factor = 2 if sync_data_transfer else 1
    buffer_size_bits = num_bits_input * max_elems * factor
    core_capacity = global_buffer_core_row * global_buffer_core_col
    num_buffer_core = max(0, math.ceil(buffer_size_bits / core_capacity))
    total_cap_bits = num_buffer_core * core_capacity
    return {
        "buffer_size_bits": buffer_size_bits,
        "max_layer_input_elements": max_elems,
        "num_buffer_core": num_buffer_core,
        "total_buffer_capacity_bits": total_cap_bits,
        "total_buffer_capacity_bytes": (total_cap_bits + 7) // 8,
    }


def compute_operator_breakdown(
    full_results: List[NeuroSimLayerResult],
    layer_specs: List[LayerSpec],
) -> List[Tuple[str, int, float, float, float, float, float, float, float, float, float, float, float]]:
    """
    按算子类型（conv / fc）汇总端到端延迟与能耗构成。
    返回列表：每项为 (operator_type, num_layers, total_latency_ns, total_energy_nj,
              adc_lat_ns, adc_en_nj, buf_lat_ns, buf_en_nj, accum_lat_ns, accum_en_nj, other_lat_ns, other_en_nj, ic_en_nj)。
    """
    agg: dict = {}
    for r in full_results:
        if r.layer_index < 1 or r.layer_index > len(layer_specs):
            continue
        t = layer_specs[r.layer_index - 1].layer_type
        if t not in agg:
            agg[t] = (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        n, tl, te, al, ae, bl, be, cl, ce, ol, oe, ie = agg[t]
        n += 1
        tl += r.total_latency_ns
        te += r.total_energy_nj
        al += r.adc_latency_ns
        ae += r.adc_energy_pj / 1000.0
        bl += r.buffer_latency_ns
        be += r.buffer_energy_pj / 1000.0
        cl += r.accum_latency_ns
        ce += r.accum_energy_pj / 1000.0
        ol += r.other_latency_ns
        oe += r.other_energy_pj / 1000.0
        ie += r.ic_energy_pj / 1000.0
        agg[t] = (n, tl, te, al, ae, bl, be, cl, ce, ol, oe, ie)
    out_rows = []
    for op_type in ("conv", "fc"):
        if op_type not in agg:
            continue
        n, tl, te, al, ae, bl, be, cl, ce, ol, oe, ie = agg[op_type]
        out_rows.append((op_type, n, tl, te, al, ae, bl, be, cl, ce, ol, oe, ie))
    return out_rows


def write_operator_breakdown_csv(
    operator_rows: List[Tuple[str, int, float, float, float, float, float, float, float, float, float, float, float]],
    output_dir: str,
) -> str:
    """将按算子类型的端到端延迟与能耗构成写入 neurosim_breakdown_by_operator.csv。返回文件路径。"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "neurosim_breakdown_by_operator.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "operator_type", "num_layers", "total_latency_ns", "total_energy_nj",
            "adc_latency_ns", "adc_energy_nj", "buffer_latency_ns", "buffer_energy_nj",
            "accum_latency_ns", "accum_energy_nj", "other_latency_ns", "other_energy_nj", "ic_energy_nj",
        ])
        for row in operator_rows:
            w.writerow([row[0], row[1], round(row[2], 2), round(row[3], 6)] + [round(x, 4) for x in row[4:]])
    return path


def write_neurosim_csvs(
    full_results: List[NeuroSimLayerResult],
    layer_names: List[str],
    output_dir: str,
    clock_period_ns: Optional[float] = None,
    layer_specs: Optional[List[LayerSpec]] = None,
    num_bits_input: int = 8,
    buffer_metrics: Optional[dict] = None,
) -> Tuple[str, str]:
    """
    将逐层明细与汇总写入 CSV。返回 (明细路径, 汇总路径)。
    total_energy = ADC + Buffer + Accum + Other + IC（breakdown 合计与总能耗一致）；total_latency = ADC + Buffer + Accum + Other。
    若提供 layer_specs，则逐层写入 buffer_data_bits / buffer_data_bytes（该层输入激活数据量，近似层间 buffer 用量）。
    若提供 buffer_metrics（由 compute_neurosim_buffer_size 返回），则汇总 CSV 中写入 NeuroSim buffer 规模。
    """
    os.makedirs(output_dir, exist_ok=True)
    detail_path = os.path.join(output_dir, "neurosim_layer_detail.csv")
    summary_path = os.path.join(output_dir, "neurosim_summary.csv")

    # 表头与可选列
    detail_headers = [
        "layer_index", "layer_name", "read_latency_ns", "read_energy_pj", "buffer_latency_ns", "buffer_energy_pj",
        "ic_energy_pj", "adc_latency_ns", "adc_energy_pj", "accum_latency_ns", "accum_energy_pj",
        "other_latency_ns", "other_energy_pj", "total_latency_ns", "total_energy_nj", "num_tiles", "speedup",
    ]
    if layer_specs is not None:
        detail_headers.extend(["buffer_data_bits", "buffer_data_bytes"])

    # 逐层明细（含 speedup，可选 buffer 数据量）
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(detail_headers)
        for r in full_results:
            name = layer_names[r.layer_index - 1] if r.layer_index <= len(layer_names) else f"layer{r.layer_index}"
            row = [
                r.layer_index, name, r.read_latency_ns, r.read_energy_pj, r.buffer_latency_ns, r.buffer_energy_pj,
                r.ic_energy_pj, r.adc_latency_ns, r.adc_energy_pj, r.accum_latency_ns, r.accum_energy_pj,
                r.other_latency_ns, r.other_energy_pj, r.total_latency_ns, r.total_energy_nj, r.num_tiles, r.speedup,
            ]
            if layer_specs is not None and 1 <= r.layer_index <= len(layer_specs):
                bits = _buffer_data_bits_for_layer(layer_specs[r.layer_index - 1], num_bits_input)
                row.extend([bits, (bits + 7) // 8])
            elif layer_specs is not None:
                row.extend(["", ""])
            w.writerow(row)

    # 汇总（总 latency、总 energy、总 tile、breakdown 合计、时钟、频率）
    total_latency_ns = sum(r.total_latency_ns for r in full_results)
    total_energy_nj = sum(r.total_energy_nj for r in full_results)
    total_tiles = sum(r.num_tiles for r in full_results)
    sum_adc_lat = sum(r.adc_latency_ns for r in full_results)
    sum_adc_en = sum(r.adc_energy_pj for r in full_results) / 1000.0
    sum_buf_lat = sum(r.buffer_latency_ns for r in full_results)
    sum_buf_en = sum(r.buffer_energy_pj for r in full_results) / 1000.0
    sum_accum_lat = sum(r.accum_latency_ns for r in full_results)
    sum_accum_en = sum(r.accum_energy_pj for r in full_results) / 1000.0
    sum_other_lat = sum(r.other_latency_ns for r in full_results)
    sum_other_en = sum(r.other_energy_pj for r in full_results) / 1000.0
    sum_ic_en = sum(r.ic_energy_pj for r in full_results) / 1000.0
    breakdown_latency_sum = sum_adc_lat + sum_buf_lat + sum_accum_lat + sum_other_lat
    breakdown_energy_sum = sum_adc_en + sum_buf_en + sum_accum_en + sum_other_en + sum_ic_en

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "unit_or_note"])
        w.writerow(["total_latency", total_latency_ns, "ns"])
        w.writerow(["total_energy", total_energy_nj, "nJ"])
        w.writerow(["total_tiles", total_tiles, "tiles"])
        if clock_period_ns is not None:
            w.writerow(["clock_period_ns", clock_period_ns, "ns"])
            w.writerow(["clock_frequency_MHz", 1000.0 / clock_period_ns, "MHz"])
        w.writerow(["breakdown_adc_latency", sum_adc_lat, "ns"])
        w.writerow(["breakdown_adc_energy", sum_adc_en, "nJ"])
        w.writerow(["breakdown_buffer_latency", sum_buf_lat, "ns"])
        w.writerow(["breakdown_buffer_energy", sum_buf_en, "nJ"])
        w.writerow(["breakdown_accumulation_latency", sum_accum_lat, "ns"])
        w.writerow(["breakdown_accumulation_energy", sum_accum_en, "nJ"])
        w.writerow(["breakdown_other_peripheries_latency", sum_other_lat, "ns"])
        w.writerow(["breakdown_other_peripheries_energy", sum_other_en, "nJ"])
        w.writerow(["breakdown_ic_energy", sum_ic_en, "nJ"])
        w.writerow(["breakdown_latency_sum", breakdown_latency_sum, "ns; 应=total_latency"])
        w.writerow(["breakdown_energy_sum", breakdown_energy_sum, "nJ; 应=total_energy"])
        if buffer_metrics:
            w.writerow(["neurosim_buffer_size_bits", buffer_metrics["buffer_size_bits"], "numBitInput*maxLayerInput*(2 if sync)"])
            w.writerow(["neurosim_max_layer_input_elements", buffer_metrics["max_layer_input_elements"], "elements"])
            w.writerow(["neurosim_num_buffer_core", buffer_metrics["num_buffer_core"], "cores"])
            w.writerow(["neurosim_total_buffer_capacity_bits", buffer_metrics["total_buffer_capacity_bits"], "bits"])
            w.writerow(["neurosim_total_buffer_capacity_bytes", buffer_metrics["total_buffer_capacity_bytes"], "bytes"])

    return detail_path, summary_path


def run_neurosim_main(
    config_path: str,
    weight_input_pairs: List[Tuple[str, str]],
    main_dir: str,
    wl_weight: int = 8,
    wl_activation: int = 8,
    subarray: int = 128,
    parallel_read: int = 128,
    timeout_s: int = 120,
    clk_freq_hz: Optional[float] = None,
    technode: Optional[int] = None,
) -> Tuple[bool, str, str, int]:
    """
    在 main_dir 下执行 NeuroSIM/main，传入 config 与每层 weight/input CSV 相对路径。
    若提供 clk_freq_hz（Hz），作为可选参数传入 main 覆盖时钟频率上限；若提供 technode（nm），再传入工艺节点。
    返回 (成功与否, stdout, stderr/错误信息, returncode)。
    """
    main_exe = os.path.join(main_dir, "main")
    if not os.path.isfile(main_exe):
        return False, "", f"main 不存在: {main_exe}", -1
    args = [
        main_exe,
        config_path,
        str(wl_weight),
        str(wl_activation),
        str(subarray),
        str(parallel_read),
    ]
    for wf, inf in weight_input_pairs:
        args.append(wf)
        args.append(inf)
    if technode is not None and technode > 0:
        args.append(str(clk_freq_hz if (clk_freq_hz is not None and clk_freq_hz > 0) else 0))
        args.append(str(technode))
    elif clk_freq_hz is not None and clk_freq_hz > 0:
        args.append(str(clk_freq_hz))
    try:
        proc = subprocess.run(
            args,
            cwd=main_dir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if proc.returncode != 0:
            err = f"returncode={proc.returncode}"
            if proc.stderr:
                err += "\nstderr:\n" + (proc.stderr.strip()[:2000] or "")
            return False, proc.stdout or "", err, proc.returncode
        return True, proc.stdout or "", proc.stderr or "", proc.returncode
    except subprocess.TimeoutExpired:
        return False, "", "timeout", -1
    except Exception as e:
        return False, "", str(e), -1


# =============================================================================
# 主流程：生成 config + 每层 weight/input CSV → 调用 main → 解析并打印结果
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="NeuroSim 全网络独立测试（便于排查 main 崩溃或做 PointKAN-elite 完整仿真）",
    )
    parser.add_argument(
        "--neurosim_root",
        type=str,
        default="/workspace/DNN_NeuroSim_V1.4/Inference_pytorch",
        help="DNN_NeuroSim Inference_pytorch 根目录",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=4,
        help="简化模式下参与测试的层数（仅当未使用 --pointkan_elite_full 时生效）",
    )
    parser.add_argument(
        "--pointkan_elite_full",
        action="store_true",
        help="使用 PointKAN-elite 完整 36 层结构进行仿真（与真实前向 conv/linear 顺序一致）；未与 --checkpoint 同用时为随机数据",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="若指定，则加载该 ckpt 并对 ModelNet40 一个 batch 前向，用真实每层输入激活与权重写 CSV（需安装 torch 与 PointKAN-elite 环境）",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=1024,
        help="点云点数（与 checkpoint 训练时一致）；仅 --checkpoint 时有效",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="前向抓取激活时使用的 batch 大小；仅 --checkpoint 时有效",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="加载模型与前向时的设备；仅 --checkpoint 时有效",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sim_results",
        help="写入 NeuroSim 逐层明细与汇总 CSV 的目录（默认 sim_results）；为空则不写 CSV",
    )
    parser.add_argument("--keep_tmp", action="store_true", help="保留临时目录不删除，便于查看生成的 CSV")
    parser.add_argument("--timeout", type=int, default=800, help="main 进程超时秒数")
    parser.add_argument("--wl_weight", type=int, default=8, help="权重量化位宽")
    parser.add_argument("--wl_activation", type=int, default=8, help="激活量化位宽")
    parser.add_argument(
        "--subarray",
        type=int,
        default=128,
        help="NeuroSim numRowSubArray，与 Param 的 numColSubArray(128) 组成 SubArray 尺寸；128 即 128×128；若报 SubArray Size is too large 可改为 32",
    )
    parser.add_argument(
        "--parallel_read",
        type=int,
        default=128,
        help="NeuroSim numRowParallel（并行读行数）；通常与 --subarray 同设为 128",
    )
    parser.add_argument(
        "--clk_freq",
        type=float,
        default=None,
        help="可选。时钟频率上限（Hz），传入 NeuroSim main；例如 500e6=500MHz、1e9=1GHz。不传则使用 Param 默认；实际频率仍为 min(clk_freq, 1/clkPeriod)。",
    )
    parser.add_argument(
        "--technode",
        type=int,
        default=None,
        help="可选。工艺节点（nm），传入 NeuroSim main。NeuroSim 仅支持 130/90/65/45/32/22/14/10/7 nm；若传 28 将自动改为 32。",
    )
    args = parser.parse_args()

    # NeuroSim Technology.cpp 仅支持 130,90,65,45,32,22,14,10,7 nm；28nm 会触发 exit(-1) -> returncode 255
    if args.technode == 28:
        print("注意: NeuroSim 不支持 28 nm，已自动改为 32 nm（否则会报错 exit(-1)/returncode 255）。")
        args.technode = 32

    main_dir = os.path.join(args.neurosim_root, "NeuroSIM")
    if not os.path.isdir(main_dir):
        print(f"错误: NeuroSIM 目录不存在: {main_dir}")
        return 1

    # 若指定 checkpoint：从真实前向抓取每层输入激活与权重，用于写 CSV
    real_layers_tensors: Optional[List[Tuple[Any, Any, Any]]] = None
    if args.checkpoint:
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from pointkan_sim.integration.real_data_eval import (
                load_model_and_dataset,
                collect_conv_linear_tensors_from_forward,
            )
        except ImportError as e:
            print(f"使用 --checkpoint 需要 pointkan_sim 与 PointKAN-elite 环境: {e}")
            return 1
        try:
            net, test_loader, _ = load_model_and_dataset(
                args.checkpoint,
                num_points=args.num_points,
                batch_size=args.batch_size,
                device=args.device,
            )
            batch = next(iter(test_loader))
            batch_x = batch[0] if isinstance(batch, (list, tuple)) else batch
            real_layers_tensors = collect_conv_linear_tensors_from_forward(
                net, args.device, batch_x.to(args.device),
            )
            layers = [info for info, _, _ in real_layers_tensors]
            print(f"模式: 真实 ckpt + 数据，层数: {len(layers)}（来自 {args.checkpoint}）")
        except Exception as e:
            print(f"加载 checkpoint 或抓取层数据失败: {e}")
            import traceback
            traceback.print_exc()
            return 1
    elif args.pointkan_elite_full:
        layers = get_pointkan_elite_full_layers()
        print(f"模式: PointKAN-elite 完整结构（随机数据），层数: {len(layers)}")
    else:
        layers = make_pointkan_like_layers(args.num_layers)
        print(f"模式: 简化层列表（随机数据），层数: {len(layers)}")

    for i, L in enumerate(layers):
        if L.layer_type == "conv":
            print(f"  [{i}] {L.name} conv in_ch={L.in_channels} out_ch={L.out_channels} L={L.length} K={L.kernel_size}")
        else:
            print(f"  [{i}] {L.name} fc in_f={L.in_features} out_f={L.out_features} tokens={L.tokens}")

    # 在 NeuroSIM 目录下建临时目录，使用相对路径供 main 读入
    tmpdir = tempfile.mkdtemp(dir=main_dir, prefix="pointkan_test_")
    rel_prefix = os.path.basename(tmpdir)
    print(f"\n临时目录: {tmpdir}")
    print(f"相对前缀: {rel_prefix} (cwd={main_dir})")

    config_rows = []
    weight_input_pairs: List[Tuple[str, str]] = []

    try:
        for idx, info in enumerate(layers):
            if info.layer_type == "conv":
                in_h, in_w = 1, max(1, info.length)
                in_c, out_c = max(1, info.in_channels), max(1, info.out_channels)
                k_h, k_w = 1, max(1, info.kernel_size)
                stride, pool = 1, 0
                config_rows.append(f"{in_h},{in_w},{in_c},{k_h},{k_w},{out_c},{pool},{stride}")
                weight_path = os.path.join(tmpdir, f"weight{idx}.csv")
                input_path = os.path.join(tmpdir, f"input{idx}.csv")
                if real_layers_tensors is not None:
                    _, input_t, weight_t = real_layers_tensors[idx]
                    w_np = weight_t.cpu().numpy()
                    x_np = input_t.cpu().numpy()
                    _write_matrix_weight(w_np, weight_path)
                    input_stretched = _stretch_conv_input_from_array(x_np, k_w, stride)
                    _write_matrix_activation_conv(input_stretched, args.wl_activation, input_path)
                else:
                    weight_data = np.random.randn(out_c, in_c, k_h, k_w).astype(np.float32)
                    _write_matrix_weight(weight_data, weight_path)
                    input_stretched = _stretch_input_conv1d(in_c, in_w, k_w, stride)
                    _write_matrix_activation_conv(input_stretched, args.wl_activation, input_path)
                weight_input_pairs.append((os.path.join(rel_prefix, f"weight{idx}.csv"), os.path.join(rel_prefix, f"input{idx}.csv")))
            else:
                in_f, out_f = max(1, info.in_features), max(1, info.out_features)
                M = max(1, info.tokens)
                in_h, in_w = 1, M
                in_c, out_c = in_f, out_f
                k_h, k_w = 1, 1
                stride, pool = 1, 0
                config_rows.append(f"{in_h},{in_w},{in_c},{k_h},{k_w},{out_c},{pool},{stride}")
                weight_path = os.path.join(tmpdir, f"weight{idx}.csv")
                input_path = os.path.join(tmpdir, f"input{idx}.csv")
                num_patches = (in_w - k_w) // stride + 1
                K = in_c * k_h * k_w
                if real_layers_tensors is not None:
                    _, input_t, weight_t = real_layers_tensors[idx]
                    w_np = weight_t.cpu().numpy()
                    x_np = input_t.cpu().numpy()
                    _write_matrix_weight(w_np, weight_path)
                    if x_np.ndim == 2:
                        x_np = x_np[0:1, np.newaxis, :]
                    else:
                        x_np = x_np[0:1]
                    if x_np.shape[1] != num_patches or x_np.shape[2] != K:
                        x_np = np.reshape(x_np, (1, -1, K))[:, :num_patches, :].astype(np.float32)
                    _write_matrix_activation_conv(x_np.astype(np.float32), args.wl_activation, input_path)
                else:
                    weight_data = np.random.randn(out_c, in_c, k_h, k_w).astype(np.float32)
                    _write_matrix_weight(weight_data, weight_path)
                    input_stretched = np.random.randn(1, num_patches, K).astype(np.float32)
                    _write_matrix_activation_conv(input_stretched, args.wl_activation, input_path)
                weight_input_pairs.append((os.path.join(rel_prefix, f"weight{idx}.csv"), os.path.join(rel_prefix, f"input{idx}.csv")))

        config_path = os.path.join(tmpdir, "config.csv")
        with open(config_path, "w") as f:
            f.write("\n".join(config_rows) + "\n")

        rel_config = os.path.join(rel_prefix, "config.csv")
        cmd = [
            os.path.join(main_dir, "main"),
            rel_config,
            str(args.wl_weight),
            str(args.wl_activation),
            str(args.subarray),
            str(args.parallel_read),
        ]
        for wf, inf in weight_input_pairs:
            cmd.extend([wf, inf])
        print("\n执行的命令 (cwd=%s):" % main_dir)
        print("  " + " \\\n    ".join(cmd[:8] + ["..."] if len(cmd) > 8 else cmd))

        ok, out, err, returncode = run_neurosim_main(
            rel_config,
            weight_input_pairs,
            main_dir,
            wl_weight=args.wl_weight,
            wl_activation=args.wl_activation,
            subarray=args.subarray,
            parallel_read=args.parallel_read,
            timeout_s=args.timeout,
            clk_freq_hz=args.clk_freq,
            technode=args.technode,
        )

        print("\n---------- 结果 ----------")
        print(f"  returncode: {returncode}")
        print(f"  ok: {ok}")
        if not ok:
            print("  【仿真失败】NeuroSim 未正常退出，请查看上方 stderr 与 stdout 中的 ERROR 行；returncode != 0 时不会输出汇总与 CSV。")
        print(f"  stdout 长度: {len(out)} 字符")
        if err:
            print(f"  stderr:\n{err}")
        if out:
            lines = out.splitlines()
            max_show = 300 if not ok else 80  # 失败时多打印以便看到 ERROR 行
            print(f"\n---------- stdout 前 {min(len(lines), max_show)} 行 ----------")
            for i, line in enumerate(lines[:max_show]):
                print(f"  {i+1}: {line}")
            if len(lines) > max_show:
                print("  ...")

        if ok and out:
            full_results = parse_layer_results_full(out)
            parsed = [(r.read_latency_ns, r.read_energy_pj, r.buffer_latency_ns, r.buffer_energy_pj) for r in full_results]
            print(f"\n解析到 {len(parsed)} 层（每层 latency/energy 为该层整层乘加，即完整激活×权重）：")
            for i, (r_ns, r_pj, b_ns, b_pj) in enumerate(parsed):
                lat = r_ns + b_ns
                en = (r_pj + b_pj) / 1000.0
                print(f"  layer{i+1}: latency_ns={lat:.2f}, energy_nj={en:.4f}")
            # 汇总与 breakdown 打印，并写入 CSV（breakdown 含 IC，合计=总能耗/总延迟）
            if full_results:
                total_lat = sum(r.total_latency_ns for r in full_results)
                total_en = sum(r.total_energy_nj for r in full_results)
                total_tiles = sum(r.num_tiles for r in full_results)
                sum_adc_lat = sum(r.adc_latency_ns for r in full_results)
                sum_adc_en = sum(r.adc_energy_pj for r in full_results) / 1000.0
                sum_buf_lat = sum(r.buffer_latency_ns for r in full_results)
                sum_buf_en = sum(r.buffer_energy_pj for r in full_results) / 1000.0
                sum_accum_lat = sum(r.accum_latency_ns for r in full_results)
                sum_accum_en = sum(r.accum_energy_pj for r in full_results) / 1000.0
                sum_other_lat = sum(r.other_latency_ns for r in full_results)
                sum_other_en = sum(r.other_energy_pj for r in full_results) / 1000.0
                sum_ic_en = sum(r.ic_energy_pj for r in full_results) / 1000.0
                breakdown_lat = sum_adc_lat + sum_buf_lat + sum_accum_lat + sum_other_lat
                breakdown_en = sum_adc_en + sum_buf_en + sum_accum_en + sum_other_en + sum_ic_en
                clock_period_ns = parse_clock_period_ns(out)
                freq_mhz = (1000.0 / clock_period_ns) if clock_period_ns and clock_period_ns > 0 else None

                print(f"\n---------- 汇总 ----------")
                print(f"  总延迟 total_latency_ns: {total_lat:.2f}")
                print(f"  总能耗 total_energy_nj: {total_en:.4f}")
                print(f"  总 tile 数 total_tiles: {total_tiles}")
                if clock_period_ns is not None:
                    print(f"  工作时钟 clock_period_ns: {clock_period_ns:.4f}  (频率: {freq_mhz:.2f} MHz)" if freq_mhz else f"  工作时钟 clock_period_ns: {clock_period_ns:.4f}")
                print(f"  Breakdown 延迟(ns): ADC={sum_adc_lat:.2f}, Buffer={sum_buf_lat:.2f}, Accum={sum_accum_lat:.2f}, Other={sum_other_lat:.2f}  (合计={breakdown_lat:.2f})")
                print(f"  Breakdown 能耗(nJ): ADC={sum_adc_en:.4f}, Buffer={sum_buf_en:.4f}, Accum={sum_accum_en:.4f}, Other={sum_other_en:.4f}, IC={sum_ic_en:.4f}  (合计={breakdown_en:.4f})")
                # NeuroSim buffer 规模（与 Param/Chip 一致：sync_data_transfer=1 时 bufferSize = numBitInput*maxLayerInput*2）
                buf_metrics = compute_neurosim_buffer_size(layers, args.wl_activation, sync_data_transfer=True)
                print(f"\n---------- NeuroSim Buffer 规模 ----------")
                print(f"  buffer_size_bits (数据需求): {buf_metrics['buffer_size_bits']}  (numBitInput*maxLayerInput*2)")
                print(f"  max_layer_input_elements: {buf_metrics['max_layer_input_elements']}")
                print(f"  num_buffer_core: {buf_metrics['num_buffer_core']}")
                print(f"  total_buffer_capacity_bits: {buf_metrics['total_buffer_capacity_bits']}")
                print(f"  total_buffer_capacity_bytes: {buf_metrics['total_buffer_capacity_bytes']}")
                if args.output_dir:
                    layer_names_list = [L.name for L in layers]
                    detail_path, summary_path = write_neurosim_csvs(
                        full_results, layer_names_list, args.output_dir, clock_period_ns=clock_period_ns,
                        layer_specs=layers, num_bits_input=args.wl_activation, buffer_metrics=buf_metrics,
                    )
                    print(f"\n  逐层明细已写: {detail_path}")
                    print(f"  汇总与 breakdown 已写: {summary_path}")
                    # 按算子类型（conv/fc）的端到端延迟与能耗构成
                    op_breakdown = compute_operator_breakdown(full_results, layers)
                    if op_breakdown:
                        by_op_path = write_operator_breakdown_csv(op_breakdown, args.output_dir)
                        print(f"  按算子类型构成已写: {by_op_path}")
                        print("\n---------- 按算子类型 端到端延迟与能耗构成 ----------")
                        for row in op_breakdown:
                            op_type, n, tl, te = row[0], row[1], row[2], row[3]
                            al, ae, bl, be = row[4], row[5], row[6], row[7]
                            cl, ce, ol, oe, ie = row[8], row[9], row[10], row[11], row[12]
                            print(f"  [{op_type}] 层数={n}, 总延迟={tl:.2f} ns, 总能耗={te:.4f} nJ")
                            print(f"       ADC: {al:.2f} ns, {ae:.4f} nJ  |  Buffer: {bl:.2f} ns, {be:.4f} nJ  |  Accum: {cl:.2f} ns, {ce:.4f} nJ  |  Other: {ol:.2f} ns, {oe:.4f} nJ  |  IC: {ie:.4f} nJ")
                print(f"\n  调节 NeuroSim 仿真频率: 修改 DNN_NeuroSim 中 Param.cpp 的 clkFreq（单位 Hz，如 1e9=1GHz）；实际频率取 min(clkFreq, 1/clkPeriod)。")
    finally:
        if not args.keep_tmp and os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)
            print(f"\n已删除临时目录: {tmpdir}")
        elif args.keep_tmp:
            print(f"\n已保留临时目录: {tmpdir}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
