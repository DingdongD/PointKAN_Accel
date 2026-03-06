"""
NeuroSim 适配器（Conv1d/Linear 的 cycle 与 energy 估算）
==========================================================

通过调用 DNN_NeuroSim 的 main 可执行文件，为单层或全网络生成 config + weight/input CSV，
运行仿真并解析 stdout，得到每层的 readLatency(ns)、readDynamicEnergy(pJ)、buffer 等，
转为 pointkan_sim 使用的 cycles 与 energy_nj。与 mnsim_adapter 并列，由 cim_backend 选择。
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class NeuroSimEstimate:
    """单层 NeuroSim 或回退估算结果，与 MNSIMEstimate 字段兼容。"""
    cycles: int
    latency_ns: float
    energy_nj: float
    avg_power_mw: float
    source: str


# NeuroSim 每层 stdout 格式（main.cpp 打印）
_PATTERN_READ_LATENCY = re.compile(r"layer\d+'s readLatency is:\s*([\d.eE+-]+)\s*ns")
_PATTERN_READ_ENERGY = re.compile(r"layer\d+'s readDynamicEnergy is:\s*([\d.eE+-]+)\s*pJ")
_PATTERN_BUFFER_LATENCY = re.compile(r"layer\d+'s buffer latency is:\s*([\d.eE+-]+)\s*ns")
_PATTERN_BUFFER_ENERGY = re.compile(r"layer\d+'s buffer readDynamicEnergy is:\s*([\d.eE+-]+)\s*pJ")
_PATTERN_CLK_PERIOD = re.compile(r"clkPeriod:\s*([\d.eE+-]+)")


def _write_matrix_weight(weight_matrix: np.ndarray, filepath: str) -> None:
    """Weight: (out_c, in_c*k_h*k_w) -> 存为 (K, N) = (in_c*k_h*k_w, out_c)。"""
    # weight_matrix 为 (out_c, K)，存成 (K, out_c)
    w = weight_matrix.reshape(weight_matrix.shape[0], -1).T
    np.savetxt(filepath, w, delimiter=",", fmt="%.6f")


def _dec2bin_vec(x: np.ndarray, n: int) -> Tuple[List[np.ndarray], List[float]]:
    """将浮点向量量化为 n-bit 有符号，返回各 bit 平面与 scale。"""
    y = np.zeros_like(x)
    out: List[np.ndarray] = []
    scale: List[float] = []
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


def _write_matrix_activation_conv(
    input_matrix: np.ndarray,
    bit_length: int,
    filepath: str,
) -> None:
    """
    input_matrix: (1, num_patches, K)，逐 patch 量化为 bit_length bit 后写入。
    NeuroSim CopyInput 期望 input 行数=K、列数=num_patches*bit_length；写入 (K, num_patches*bit_length)，列序 patch*bit_length+bit。
    """
    # input_matrix[0] shape (num_patches, K)
    data = input_matrix[0, :, :]
    num_patches, K = data.shape
    filled_bin, _ = _dec2bin_vec(data.ravel(), bit_length)
    # filled_b 中间 (num_patches, K*bit_length)；NeuroSim 期望 (K, num_patches*bit_length)
    filled_b = np.zeros((num_patches, K * bit_length), dtype="U1")
    for i, b in enumerate(filled_bin):
        b_2d = b.reshape(num_patches, K)
        filled_b[:, i::bit_length] = b_2d
    out = np.empty((K, num_patches * bit_length), dtype="U1")
    for k in range(K):
        for p in range(num_patches):
            out[k, p * bit_length : (p + 1) * bit_length] = filled_b[p, k * bit_length : (k + 1) * bit_length]
    np.savetxt(filepath, out, delimiter=",", fmt="%s")


def _stretch_input_conv1d(
    in_c: int,
    in_w: int,
    k_w: int,
    stride: int = 1,
) -> np.ndarray:
    """
    模拟 Conv1d 的 patch 展开：空间长 in_w，核长 k_w，得到 num_patches 个 patch，每个长 K=in_c*k_w。
    返回 (1, num_patches, K)，随机填充用于 NeuroSim 维度匹配。
    """
    num_patches = (in_w - k_w) // stride + 1
    K = in_c * k_w
    return np.random.randn(1, num_patches, K).astype(np.float32)


def _run_neurosim_main(
    config_csv_path: str,
    weight_input_pairs: List[Tuple[str, str]],
    neurosim_root: str,
    wl_weight: int = 8,
    wl_activation: int = 8,
    subarray: int = 128,
    parallel_read: int = 128,
    timeout_s: int = 120,
) -> Tuple[bool, str, str]:
    """
    调用 NeuroSIM/main，返回 (成功?, stdout, stderr)。
    neurosim_root: Inference_pytorch 目录，main 在 NeuroSIM 子目录下。
    """
    main_dir = os.path.join(neurosim_root, "NeuroSIM")
    main_exe = os.path.join(main_dir, "main")
    if not os.path.isfile(main_exe):
        return False, "", f"NeuroSim main not found: {main_exe}"

    args = [
        main_exe,
        config_csv_path,
        str(wl_weight),
        str(wl_activation),
        str(subarray),
        str(parallel_read),
    ]
    for wf, inf in weight_input_pairs:
        args.append(wf)
        args.append(inf)

    try:
        proc = subprocess.run(
            args,
            cwd=main_dir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if proc.returncode != 0:
            err_msg = f"returncode={proc.returncode}"
            if proc.stderr:
                err_msg += "; stderr: " + (proc.stderr.strip()[:1500] or "")
            return False, proc.stdout or "", err_msg
        return True, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired:
        return False, "", "NeuroSim main timeout"
    except Exception as e:
        return False, "", str(e)


def _parse_layer_results(stdout: str) -> List[Tuple[float, float, float, float]]:
    """
    解析 stdout，按层顺序返回 (read_latency_ns, read_energy_pJ, buffer_latency_ns, buffer_energy_pJ)。
    main 打印顺序：layer i's readLatency -> readDynamicEnergy -> ... -> buffer latency -> buffer readDynamicEnergy
    """
    lines = stdout.splitlines()
    results: List[Tuple[float, float, float, float]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "layer" in line and "readLatency is:" in line and "buffer" not in line:
            read_lat = _PATTERN_READ_LATENCY.search(line)
            read_ns = float(read_lat.group(1)) if read_lat else 0.0
            read_pj = 0.0
            buf_ns = 0.0
            buf_pj = 0.0
            for j in range(i + 1, min(i + 8, len(lines))):
                l = lines[j]
                if "readDynamicEnergy is:" in l and "buffer" not in l and "leakage" not in l:
                    m = _PATTERN_READ_ENERGY.search(l)
                    if m:
                        read_pj = float(m.group(1))
                    break
            for j in range(i + 1, min(i + 8, len(lines))):
                l = lines[j]
                if "buffer latency" in l:
                    m = _PATTERN_BUFFER_LATENCY.search(l)
                    if m:
                        buf_ns = float(m.group(1))
                    break
            for j in range(i + 1, min(i + 8, len(lines))):
                l = lines[j]
                if "buffer readDynamicEnergy" in l:
                    m = _PATTERN_BUFFER_ENERGY.search(l)
                    if m:
                        buf_pj = float(m.group(1))
                    break
            results.append((read_ns, read_pj, buf_ns, buf_pj))
        i += 1
    return results


def estimate_conv1d_neurosim(
    *,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    num_points: int,
    clock_period_ns: float,
    neurosim_root: str = "/workspace/DNN_NeuroSim_V1.4/Inference_pytorch",
    wl_weight: int = 8,
    wl_activation: int = 8,
    subarray: int = 128,
    parallel_read: int = 128,
    fallback_power_mw: float = 25.0,
) -> NeuroSimEstimate:
    """
    Conv1d 单层：生成 1 行 config + weight/input CSV，调 NeuroSim main，解析 layer1 结果。
    Conv1d 映射为 in_h=1, in_w=num_points, in_c=in_channels, k_h=1, k_w=kernel_size, out_c=out_channels.
    """
    in_ch = max(1, in_channels)
    out_ch = max(1, out_channels)
    k_w = max(1, kernel_size)
    L = max(k_w, num_points)
    stride = 1
    # NeuroSim config 一行: in_h, in_w, in_c, k_h, k_w, out_c, pool, stride
    in_h, in_w = 1, L
    in_c, out_c = in_ch, out_ch
    k_h = 1
    pool = 0
    num_patches = (in_w - k_w) // stride + 1
    K = in_c * k_h * k_w

    with tempfile.TemporaryDirectory(prefix="pointkan_neurosim_") as tmpdir:
        config_path = os.path.join(tmpdir, "config.csv")
        with open(config_path, "w") as f:
            f.write(f"{in_h},{in_w},{in_c},{k_h},{k_w},{out_c},{pool},{stride}\n")

        weight_path = os.path.join(tmpdir, "weight0.csv")
        input_path = os.path.join(tmpdir, "input0.csv")
        weight_data = np.random.randn(out_c, in_c, k_h, k_w).astype(np.float32)
        _write_matrix_weight(weight_data, weight_path)
        input_stretched = _stretch_input_conv1d(in_c, in_w, k_w, stride)
        _write_matrix_activation_conv(input_stretched, wl_activation, input_path)

        ok, out, err = _run_neurosim_main(
            config_path,
            [(weight_path, input_path)],
            neurosim_root,
            wl_weight=wl_weight,
            wl_activation=wl_activation,
            subarray=subarray,
            parallel_read=parallel_read,
        )
        if not ok or not out:
            cycles = max(1, int(math.ceil(num_patches * K * out_ch / 128)))
            latency_ns = cycles * clock_period_ns
            energy_nj = fallback_power_mw * latency_ns / 1e3
            return NeuroSimEstimate(
                cycles=cycles,
                latency_ns=latency_ns,
                energy_nj=energy_nj,
                avg_power_mw=fallback_power_mw,
                source="neurosim_fallback",
            )

        parsed = _parse_layer_results(out)
        if not parsed:
            cycles = max(1, int(math.ceil(num_patches * K * out_ch / 128)))
            latency_ns = cycles * clock_period_ns
            energy_nj = fallback_power_mw * latency_ns / 1e3
            return NeuroSimEstimate(
                cycles=cycles,
                latency_ns=latency_ns,
                energy_nj=energy_nj,
                avg_power_mw=fallback_power_mw,
                source="neurosim_parse_fallback",
            )

        read_ns, read_pj, buf_ns, buf_pj = parsed[0]
        total_latency_ns = read_ns + buf_ns
        total_energy_pj = read_pj + buf_pj
        energy_nj = total_energy_pj / 1000.0
        cycles = max(1, math.ceil(total_latency_ns / clock_period_ns))
        avg_power_mw = (energy_nj * 1e3 / total_latency_ns) if total_latency_ns > 0 else 0.0

        return NeuroSimEstimate(
            cycles=cycles,
            latency_ns=total_latency_ns,
            energy_nj=energy_nj,
            avg_power_mw=avg_power_mw,
            source="neurosim",
        )


def estimate_linear_neurosim(
    *,
    in_features: int,
    out_features: int,
    tokens: int = 1,
    clock_period_ns: float = 1.0,
    neurosim_root: str = "/workspace/DNN_NeuroSim_V1.4/Inference_pytorch",
    wl_weight: int = 8,
    wl_activation: int = 8,
    subarray: int = 128,
    parallel_read: int = 128,
    fallback_power_mw: float = 25.0,
) -> NeuroSimEstimate:
    """
    Linear 单层：映射为 1x1 conv (in_h=1, in_w=1, in_c=in_f, k_h=1, k_w=1, out_c=out_f)。
    tokens 对应 num_patches：NeuroSim 用 in_w 表示“宽度”，全连接时 in_w=1 则 numInVector=1；
    若希望多 token，需 in_w=tokens，此时 in_h=1, in_w=tokens, in_c=in_f, k_h=1, k_w=1 -> num_patches = tokens.
    """
    in_f = max(1, in_features)
    out_f = max(1, out_features)
    M = max(1, tokens)
    # 全连接多 token：in_h=1, in_w=M, in_c=in_f, k_h=1, k_w=1 -> num_patches = (1-1+1)*(M-1+1)=M
    in_h, in_w = 1, M
    in_c, out_c = in_f, out_f
    k_h, k_w = 1, 1
    pool, stride = 0, 1
    num_patches = (in_w - k_w) // stride + 1
    K = in_c * k_h * k_w

    with tempfile.TemporaryDirectory(prefix="pointkan_neurosim_") as tmpdir:
        config_path = os.path.join(tmpdir, "config.csv")
        with open(config_path, "w") as f:
            f.write(f"{in_h},{in_w},{in_c},{k_h},{k_w},{out_c},{pool},{stride}\n")

        weight_path = os.path.join(tmpdir, "weight0.csv")
        input_path = os.path.join(tmpdir, "input0.csv")
        weight_data = np.random.randn(out_c, in_c, k_h, k_w).astype(np.float32)
        _write_matrix_weight(weight_data, weight_path)
        input_stretched = np.random.randn(1, num_patches, K).astype(np.float32)
        _write_matrix_activation_conv(input_stretched, wl_activation, input_path)

        ok, out, err = _run_neurosim_main(
            config_path,
            [(weight_path, input_path)],
            neurosim_root,
            wl_weight=wl_weight,
            wl_activation=wl_activation,
            subarray=subarray,
            parallel_read=parallel_read,
        )
        if not ok or not out:
            cycles = max(1, math.ceil(M * in_f * out_f / 128))
            latency_ns = cycles * clock_period_ns
            energy_nj = fallback_power_mw * latency_ns / 1e3
            return NeuroSimEstimate(
                cycles=cycles,
                latency_ns=latency_ns,
                energy_nj=energy_nj,
                avg_power_mw=fallback_power_mw,
                source="neurosim_fallback",
            )

        parsed = _parse_layer_results(out)
        if not parsed:
            cycles = max(1, math.ceil(M * in_f * out_f / 128))
            latency_ns = cycles * clock_period_ns
            energy_nj = fallback_power_mw * latency_ns / 1e3
            return NeuroSimEstimate(
                cycles=cycles,
                latency_ns=latency_ns,
                energy_nj=energy_nj,
                avg_power_mw=fallback_power_mw,
                source="neurosim_parse_fallback",
            )

        read_ns, read_pj, buf_ns, buf_pj = parsed[0]
        total_latency_ns = read_ns + buf_ns
        total_energy_pj = read_pj + buf_pj
        energy_nj = total_energy_pj / 1000.0
        cycles = max(1, math.ceil(total_latency_ns / clock_period_ns))
        avg_power_mw = (energy_nj * 1e3 / total_latency_ns) if total_latency_ns > 0 else 0.0

        return NeuroSimEstimate(
            cycles=cycles,
            latency_ns=total_latency_ns,
            energy_nj=energy_nj,
            avg_power_mw=avg_power_mw,
            source="neurosim",
        )


