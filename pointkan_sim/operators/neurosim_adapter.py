"""
NeuroSim 适配器（Conv1d/Linear 的 cycle 与 energy 估算）
==========================================================

通过调用 DNN_NeuroSim 的 main 可执行文件，为单层或全网络生成 config + weight/input CSV，
运行仿真并解析 stdout，得到每层的 readLatency(ns)、readDynamicEnergy(pJ)、buffer 等，
转为 pointkan_sim 使用的 cycles 与 energy_nj；供 PE 级 Conv1d/Linear 与 Predictor FC 估算使用。
"""
from __future__ import annotations

import math
import os
import re
import json
import hashlib
import subprocess
import sys
import tempfile
from functools import lru_cache
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from ..utils.shared_helpers import stretch_input_conv1d as _stretch_input_conv1d



@dataclass
class NeuroSimEstimate:
    """单层 NeuroSim 或回退估算结果。"""
    cycles: int
    latency_ns: float
    energy_nj: float
    avg_power_mw: float
    source: str


def _fallback_estimate(
    *,
    work_items: int,
    clock_period_ns: float,
    fallback_power_mw: float,
    source: str,
) -> NeuroSimEstimate:
    cycles = max(1, int(work_items))
    latency_ns = cycles * clock_period_ns
    energy_nj = fallback_power_mw * latency_ns / 1e3
    return NeuroSimEstimate(
        cycles=cycles,
        latency_ns=latency_ns,
        energy_nj=energy_nj,
        avg_power_mw=fallback_power_mw,
        source=source,
    )


def _unique_positive_pairs(pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    seen = set()
    for subarray, parallel_read in pairs:
        sub = max(1, int(subarray))
        par = max(1, int(parallel_read))
        key = (sub, par)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _single_layer_retry_candidates(
    *,
    in_dim: int,
    out_dim: int,
    requested_subarray: int,
    requested_parallel_read: int,
) -> List[Tuple[int, int]]:
    """
    Small single-layer NeuroSim runs are sensitive to an oversized subarray.
    Try the requested config first, then progressively smaller powers-of-two
    bounded by the smaller matrix dimension. This keeps predictor FC layers
    from spuriously falling back when 128x128 is too large for a tiny layer.
    """
    min_dim = max(1, min(int(in_dim), int(out_dim)))
    pow2_cap = 1 << max(0, min_dim.bit_length() - 1)
    pow2_cap = max(16, min(128, pow2_cap))
    sizes = [
        int(requested_subarray),
        min(128, pow2_cap),
        min(64, pow2_cap),
        min(32, pow2_cap),
        min(16, pow2_cap),
    ]
    pairs = [(s, min(int(requested_parallel_read), s)) for s in sizes if s > 0]
    return _unique_positive_pairs(pairs)


def _normalize_single_layer_mode(mode: Optional[str]) -> str:
    value = str(mode or "subprocess").strip().lower()
    if value not in {"subprocess", "pybind", "auto"}:
        return "subprocess"
    return value


def _stable_rng(*parts: object) -> np.random.Generator:
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(seed)


def _build_single_layer_weight_data(
    *,
    layer_type: str,
    out_channels: int,
    in_channels: int,
    kernel_size: int,
) -> np.ndarray:
    rng = _stable_rng("neurosim_weight", layer_type, out_channels, in_channels, kernel_size)
    return rng.standard_normal((out_channels, in_channels, 1, kernel_size), dtype=np.float32)


def _build_single_layer_input_stretched(
    *,
    layer_type: str,
    in_channels: int,
    kernel_size: int,
    length_or_tokens: int,
) -> np.ndarray:
    if layer_type == "conv":
        # Preserve Conv1d im2col semantics but make the synthetic tensor deterministic.
        rng = _stable_rng("neurosim_conv_input", in_channels, kernel_size, length_or_tokens)
        num_patches = max(1, (int(length_or_tokens) - int(kernel_size)) + 1)
        k_dim = max(1, int(in_channels) * int(kernel_size))
        return rng.standard_normal((1, num_patches, k_dim), dtype=np.float32)
    rng = _stable_rng("neurosim_fc_input", in_channels, length_or_tokens)
    return rng.standard_normal((1, max(1, int(length_or_tokens)), max(1, int(in_channels))), dtype=np.float32)


def _estimate_activation_row_activity(input_matrix: np.ndarray, bit_length: int) -> float:
    data = np.asarray(input_matrix, dtype=np.float32)
    if data.size <= 0:
        return 0.5
    filled_bin, _ = _dec2bin_vec(data.ravel(), max(1, int(bit_length)))
    if not filled_bin:
        return 0.5
    total = 0.0
    count = 0
    for plane in filled_bin:
        total += float(np.count_nonzero(plane))
        count += int(plane.size)
    if count <= 0:
        return 0.5
    return max(0.0, min(1.0, total / float(count)))


# NeuroSim 每层 stdout 格式（main.cpp 打印）
_PATTERN_READ_LATENCY = re.compile(r"layer\d+'s readLatency is:\s*([\d.eE+-]+)\s*ns")
_PATTERN_READ_ENERGY = re.compile(r"layer\d+'s readDynamicEnergy is:\s*([\d.eE+-]+)\s*pJ")
_PATTERN_BUFFER_LATENCY = re.compile(r"layer\d+'s buffer latency is:\s*([\d.eE+-]+)\s*ns")
_PATTERN_BUFFER_ENERGY = re.compile(r"layer\d+'s buffer readDynamicEnergy is:\s*([\d.eE+-]+)\s*pJ")
_PATTERN_CLK_PERIOD = re.compile(r"clkPeriod:\s*([\d.eE+-]+)")
_PATTERN_CHIP_CLK_PERIOD = re.compile(r"Chip clock period is:\s*([\d.eE+-]+)\s*ns")


def parse_clock_period_ns(stdout: str) -> Optional[float]:
    """
    从 NeuroSim main 的 stdout 解析实际工作时钟周期（ns）。
    优先使用 'Chip clock period is: X ns'（用户设定/最终周期），否则用 'clkPeriod: X'（秒，关键路径）。
    未找到返回 None，调用方应用 hw_config 默认（如 1 ns = 1 GHz）并建议核对。
    """
    chip_period_ns: Optional[float] = None
    clk_period_sec: Optional[float] = None
    for line in stdout.splitlines():
        m = _PATTERN_CHIP_CLK_PERIOD.search(line)
        if m:
            chip_period_ns = float(m.group(1))
        m = _PATTERN_CLK_PERIOD.search(line)
        if m:
            clk_period_sec = float(m.group(1))
    if chip_period_ns is not None:
        return chip_period_ns
    if clk_period_sec is not None:
        return clk_period_sec * 1e9
    return None


def parse_critical_clk_period_ns(stdout: str) -> Optional[float]:
    """
    解析 NeuroSim 关键路径时钟周期（ns）。
    对应 stdout 中的 'clkPeriod: X'（单位：秒）。
    """
    clk_period_sec: Optional[float] = None
    for line in stdout.splitlines():
        m = _PATTERN_CLK_PERIOD.search(line)
        if m:
            clk_period_sec = float(m.group(1))
    if clk_period_sec is None:
        return None
    return float(clk_period_sec) * 1e9


def parse_chip_clk_period_ns(stdout: str) -> Optional[float]:
    """
    解析 NeuroSim 最终/用户设定的 Chip clock period（ns）。
    对应 stdout 中的 'Chip clock period is: X ns'。
    """
    chip_period_ns: Optional[float] = None
    for line in stdout.splitlines():
        m = _PATTERN_CHIP_CLK_PERIOD.search(line)
        if m:
            chip_period_ns = float(m.group(1))
    return chip_period_ns


def _write_matrix_weight(weight_matrix: np.ndarray, filepath: str) -> None:
    """Weight CSV for DNN main: raw algorithm weights in (K, N) layout."""
    w = _build_weight_matrix_for_neurosim(weight_matrix)
    np.savetxt(filepath, w, delimiter=",", fmt="%.6f")


def _build_weight_matrix_for_neurosim(weight_matrix: np.ndarray) -> np.ndarray:
    """Return NeuroSim weight matrix in (K, N) layout used by CSV + pybind exact path."""
    return weight_matrix.reshape(weight_matrix.shape[0], -1).T.astype(np.float64, copy=False)


def _round_like_cpp(value: float) -> int:
    adjusted = value + 0.5 if value >= 0 else value - 0.5
    return math.trunc(adjusted)


def _build_conductance_weight_matrix_for_pybind(
    weight_matrix: np.ndarray,
    *,
    synapse_bit: int,
    cell_bit: int,
    max_conductance: float,
    min_conductance: float,
    algo_weight_max: float = 1.0,
    algo_weight_min: float = -1.0,
) -> np.ndarray:
    """
    Mirror Chip.cpp::LoadInWeightData() for the pybind exact path.

    Input:
      raw algorithm weights in (K, N) layout.
    Output:
      conductance-expanded matrix in (K, N * num_col_per_synapse) layout.
    """
    raw = np.asarray(weight_matrix, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError("weight_matrix must be 2D")
    if synapse_bit <= 0 or cell_bit <= 0:
        raise ValueError("synapse_bit and cell_bit must be positive")
    num_col_per_synapse = max(1, int(math.ceil(float(synapse_bit) / float(cell_bit))))
    cellrange = int(2 ** int(cell_bit))
    normalized_min = 0.0
    normalized_max = float(2 ** int(synapse_bit))
    denom = float(algo_weight_max - algo_weight_min)
    if abs(denom) < 1e-12:
        raise ValueError("algo_weight_max and algo_weight_min must differ")

    out = np.empty((raw.shape[0], raw.shape[1] * num_col_per_synapse), dtype=np.float64)
    for r in range(raw.shape[0]):
        for c in range(raw.shape[1]):
            f = float(raw[r, c])
            newdata = ((normalized_max - normalized_min) / denom * (f - algo_weight_max) + normalized_max)
            value = _round_like_cpp(newdata)
            synapse_values: List[int] = [0] * num_col_per_synapse
            for z in range(num_col_per_synapse):
                remainder = math.ceil(float(value % cellrange))
                value = math.ceil(float(value / cellrange))
                synapse_values.insert(0, int(remainder))
            for u in range(num_col_per_synapse):
                cellvalue = float(synapse_values[u])
                conductance = cellvalue / float(cellrange - 1) * (max_conductance - min_conductance) + min_conductance
                out[r, c * num_col_per_synapse + u] = conductance
    return out


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
    out = _build_activation_matrix_for_neurosim(input_matrix, bit_length).astype(np.int64)
    np.savetxt(filepath, out, delimiter=",", fmt="%d")


def _build_activation_matrix_for_neurosim(
    input_matrix: np.ndarray,
    bit_length: int,
) -> np.ndarray:
    """
    Return NeuroSim activation matrix in (K, num_patches*bit_length) layout.
    The matrix exactly matches what _write_matrix_activation_conv writes to CSV.
    """
    data = input_matrix[0, :, :]
    num_patches, K = data.shape
    filled_bin, _ = _dec2bin_vec(data.ravel(), bit_length)
    filled_b = np.zeros((num_patches, K * bit_length), dtype=np.float64)
    for i, b in enumerate(filled_bin):
        b_2d = b.reshape(num_patches, K)
        filled_b[:, i::bit_length] = b_2d
    out = np.empty((K, num_patches * bit_length), dtype=np.float64)
    for k in range(K):
        for p in range(num_patches):
            out[k, p * bit_length : (p + 1) * bit_length] = filled_b[p, k * bit_length : (k + 1) * bit_length]
    return out


def _run_neurosim_main(
    config_csv_path: str,
    weight_input_pairs: List[Tuple[str, str]],
    neurosim_root: str,
    wl_weight: int = 8,
    wl_activation: int = 8,
    subarray: int = 128,
    subarray_cols: Optional[int] = None,
    parallel_read: int = 128,
    timeout_s: Optional[int] = None,
    clk_freq_hz: Optional[float] = None,
    technode: Optional[int] = None,
    adc_clk_freq_hz: Optional[float] = None,
    sar_column_power_w: Optional[float] = None,
    num_col_muxed: Optional[int] = None,
    tile_buffer_core_size_row: Optional[int] = None,
    tile_buffer_core_size_col: Optional[int] = None,
    force_tile_size_cm: Optional[int] = None,
    force_pe_size_cm: Optional[int] = None,
    force_num_tiles_cm: Optional[int] = None,
    neurosim_relu_mode: Optional[int] = None,
    neurosim_chip_activation_mode: Optional[int] = None,
    neurosim_enable_activation: Optional[int] = None,
    progress_interval_s: Optional[float] = None,
    progress_log_info: Optional[Any] = None,
) -> Tuple[bool, str, str]:
    """
    调用 NeuroSIM/main，返回 (成功?, stdout, stderr)。
    neurosim_root: Inference_pytorch 目录，main 在 NeuroSIM 子目录下。
    clk_freq_hz: 可选，芯片/数字外围时钟(Hz)，传入 main。
    technode: 可选，工艺节点(nm)；28 会由调用方映射为 32。
    adc_clk_freq_hz: 可选，SAR ADC 专用时钟(Hz)，默认由 main 使用 1GHz；与 clk_freq_hz 独立。
    sar_column_power_w: 可选，用户 Column_Power (W)，与 SarADC 经验式同量纲；>0 时仍用 P×(bit+1)/f_adc 得能量。
    num_col_muxed: 可选，numColMuxed（多少列共享 1 个 ADC/SenseAmp）；>0 时传入 main 覆盖默认值。
    force_tile_size_cm: 可选，强制 Conventional Mapping 的 tile size（如 1024）。
    force_pe_size_cm: 可选，强制 Conventional Mapping 的 PE size（如 512）。
    force_num_tiles_cm: 可选，强制 Conventional Mapping 的 tile 总数预算（如 60）。
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
    # 与 main 一致：可选参数按顺序追加；未用填 0
    #
    # main.cpp 目前支持（按顺序）：
    #   clkFreqHz, technode, adcClkHz, sarColumnPowerW, resistanceOn, resistanceOff,
    #   cellBit, memcelltype, numColMuxed, numColSubArray, forceTileSizeCM, forcePESizeCM,
    #   forceNumTileCM, reLu(0/1), chipActivation(0/1), enableActivation(0/1),
    #   tileBufferCoreSizeRow, tileBufferCoreSizeCol
    clk_v = float(clk_freq_hz) if (clk_freq_hz is not None and clk_freq_hz > 0) else 0.0
    tn_v = int(technode) if (technode is not None and technode > 0) else 0
    adc_v = float(adc_clk_freq_hz) if (adc_clk_freq_hz is not None and adc_clk_freq_hz > 0) else 0.0
    sar_v = float(sar_column_power_w) if (sar_column_power_w is not None and sar_column_power_w > 0) else 0.0
    mux_v = int(num_col_muxed) if (num_col_muxed is not None and int(num_col_muxed) > 0) else 0
    suba_col_v = int(subarray_cols) if (subarray_cols is not None and int(subarray_cols) > 0) else 0
    f_tile_v = int(force_tile_size_cm) if (force_tile_size_cm is not None and int(force_tile_size_cm) > 0) else 0
    f_pe_v = int(force_pe_size_cm) if (force_pe_size_cm is not None and int(force_pe_size_cm) > 0) else 0
    f_num_v = int(force_num_tiles_cm) if (force_num_tiles_cm is not None and int(force_num_tiles_cm) > 0) else 0
    relu_v = int(neurosim_relu_mode) if neurosim_relu_mode in (0, 1) else -1
    chipact_v = int(neurosim_chip_activation_mode) if neurosim_chip_activation_mode in (0, 1) else -1
    enact_v = int(neurosim_enable_activation) if neurosim_enable_activation in (0, 1) else -1
    tbuf_row_v = int(tile_buffer_core_size_row) if (tile_buffer_core_size_row is not None and int(tile_buffer_core_size_row) > 0) else 0
    tbuf_col_v = int(tile_buffer_core_size_col) if (tile_buffer_core_size_col is not None and int(tile_buffer_core_size_col) > 0) else 0

    if clk_v or tn_v or adc_v or sar_v or mux_v or suba_col_v or f_tile_v or f_pe_v or f_num_v or relu_v >= 0 or chipact_v >= 0 or enact_v >= 0 or tbuf_row_v or tbuf_col_v:
        # Always append optional params in fixed order so main.cpp can parse by position.
        # Use 0 for legacy placeholders and -1 for tri-state switches (unset).
        args.extend([
            str(clk_v), str(tn_v), str(adc_v), str(sar_v),
            "0", "0", "0", "0",
            str(mux_v), str(suba_col_v), str(f_tile_v), str(f_pe_v), str(f_num_v),
            str(relu_v), str(chipact_v), str(enact_v), str(tbuf_row_v), str(tbuf_col_v),
        ])

    # 默认行为保持不变：不启用心跳时，用 subprocess.run 直接阻塞等待。
    hb = None
    try:
        if progress_interval_s is not None:
            hb = float(progress_interval_s)
    except Exception:
        hb = None
    hb = hb if (hb is not None and hb > 0) else None

    try:
        if hb is None and progress_log_info is None:
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

        import selectors
        import time

        log_fn = progress_log_info
        start = time.perf_counter()
        last_hb = start
        stdout_lines: List[str] = []
        stderr_lines: List[str] = []
        last_nonempty = ""

        proc2 = subprocess.Popen(
            args,
            cwd=main_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        sel = selectors.DefaultSelector()
        if proc2.stdout is not None:
            sel.register(proc2.stdout, selectors.EVENT_READ, data="stdout")
        if proc2.stderr is not None:
            sel.register(proc2.stderr, selectors.EVENT_READ, data="stderr")

        def _maybe_heartbeat(now: float) -> None:
            if hb is None or log_fn is None:
                return
            if now - last_hb < hb:
                return
            msg = f"NeuroSim main still running... elapsed={now - start:.1f}s"
            if last_nonempty:
                msg += f" | last='{last_nonempty[:180]}'"
            msg += f" | stdout_lines={len(stdout_lines)} stderr_lines={len(stderr_lines)}"
            try:
                log_fn(msg)
            except Exception:
                pass

        while True:
            now = time.perf_counter()
            if timeout_s is not None and timeout_s > 0 and (now - start) > float(timeout_s):
                try:
                    proc2.kill()
                except Exception:
                    pass
                return False, "\n".join(stdout_lines), f"NeuroSim main timeout ({timeout_s}s)"

            if proc2.poll() is not None:
                break

            _maybe_heartbeat(now)
            # 以小步 select 等待输出或心跳点
            wait_s = 1.0
            if hb is not None:
                wait_s = max(0.2, min(1.0, (last_hb + hb) - now))
            events = sel.select(timeout=wait_s)
            if not events:
                continue
            for key, _ in events:
                stream = key.fileobj
                tag = key.data
                try:
                    line = stream.readline()
                except Exception:
                    line = ""
                if not line:
                    try:
                        sel.unregister(stream)
                    except Exception:
                        pass
                    continue
                s = line.rstrip("\n")
                if s.strip():
                    last_nonempty = s.strip()
                if tag == "stdout":
                    stdout_lines.append(s)
                else:
                    stderr_lines.append(s)
            if hb is not None and (time.perf_counter() - last_hb) >= hb:
                last_hb = time.perf_counter()

        # drain remaining buffered output
        try:
            if proc2.stdout is not None:
                rest = proc2.stdout.read()
                if rest:
                    stdout_lines.extend(rest.splitlines())
            if proc2.stderr is not None:
                rest = proc2.stderr.read()
                if rest:
                    stderr_lines.extend(rest.splitlines())
        except Exception:
            pass

        rc = proc2.returncode or 0
        out_s = "\n".join(stdout_lines)
        err_s = "\n".join(stderr_lines)
        if rc != 0:
            err_msg = f"returncode={rc}"
            if err_s.strip():
                err_msg += "; stderr: " + (err_s.strip()[:1500] or "")
            return False, out_s, err_msg
        return True, out_s, err_s
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


def _estimate_single_layer_pybind(
    *,
    layer_type: str,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    length_or_tokens: int,
    neurosim_root: str,
    wl_weight: int,
    wl_activation: int,
    subarray: int,
    parallel_read: int,
    clk_freq_hz: Optional[float],
    technode: Optional[int],
    adc_clk_freq_hz: Optional[float],
    sar_column_power_w: Optional[float],
    num_col_muxed: int,
    clk_mode: str,
    activity_row_read: float,
    weight_matrix: np.ndarray,
    input_matrix: np.ndarray,
) -> Tuple[float, float]:
    with tempfile.TemporaryDirectory(prefix="pointkan_neurosim_pybind_") as tmpdir:
        npz_path = os.path.join(tmpdir, "single_layer_exact.npz")
        np.savez_compressed(
            npz_path,
            weight_matrix=np.asarray(weight_matrix, dtype=np.float64),
            input_matrix=np.asarray(input_matrix, dtype=np.float64),
        )
        payload = {
            "layer_type": str(layer_type),
            "in_channels": int(in_channels),
            "out_channels": int(out_channels),
            "kernel_size": int(kernel_size),
            "length_or_tokens": int(length_or_tokens),
            "neurosim_root": str(neurosim_root),
            "wl_weight": int(wl_weight),
            "wl_activation": int(wl_activation),
            "subarray": int(subarray),
            "parallel_read": int(parallel_read),
            "clk_freq_hz": float(clk_freq_hz) if clk_freq_hz else None,
            "technode": int(technode) if technode else None,
            "adc_clk_freq_hz": float(adc_clk_freq_hz) if adc_clk_freq_hz else None,
            "sar_column_power_w": float(sar_column_power_w) if sar_column_power_w else None,
            "num_col_muxed": int(num_col_muxed),
            "clk_mode": str(clk_mode or "capped"),
            "activity_row_read": float(activity_row_read),
            "npz_path": npz_path,
        }
        child_code = r"""
import json, os, sys
import numpy as np

payload = json.loads(sys.argv[1])
so_dir = os.path.join(payload["neurosim_root"], "NeuroSIM")
sys.path.insert(0, so_dir)
import neurosim_core as nc

weight_matrix = np.load(payload["npz_path"])["weight_matrix"]
input_matrix = np.load(payload["npz_path"])["input_matrix"]

if payload["layer_type"] == "conv":
    net = [[
        1.0,
        float(max(int(payload["kernel_size"]), int(payload["length_or_tokens"]))),
        float(max(1, int(payload["in_channels"]))),
        1.0,
        float(max(1, int(payload["kernel_size"]))),
        float(max(1, int(payload["out_channels"]))),
        0.0,
        1.0,
    ]]
else:
    net = [[
        1.0,
        float(max(1, int(payload["length_or_tokens"]))),
        float(max(1, int(payload["in_channels"]))),
        1.0,
        1.0,
        float(max(1, int(payload["out_channels"]))),
        0.0,
        1.0,
    ]]

cell_bit = 1
num_col_per_synapse = max(1, int(np.ceil(int(payload["wl_weight"]) / float(cell_bit))))
nc.configure_param(
    int(payload["wl_weight"]),
    int(payload["wl_activation"]),
    int(payload["subarray"]),
    128,
    int(payload["parallel_read"]),
    int(num_col_per_synapse),
    int(payload.get("num_col_muxed") or 8),
    32,
    cell_bit,
    float(payload["clk_freq_hz"]) if payload["clk_freq_hz"] else 100e6,
    float(payload["adc_clk_freq_hz"]) if payload["adc_clk_freq_hz"] else 1e9,
    int(payload["technode"]) if payload["technode"] else 32,
    100e3,
    100e3 * 17,
    False,
    True,
)
if payload["sar_column_power_w"]:
    nc.set_sar_column_power_w(float(payload["sar_column_power_w"]))
nc.set_synchronous(True)
nc.chip_setup(net)
layer = nc.chip_run_layer_exact(
    0,
    weight_matrix.tolist(),
    input_matrix.tolist(),
    False,
)
gcp_s = float(layer.get("clk_period_s", 0.0) or 0.0)
eff_clk_hz = float(payload["clk_freq_hz"]) if payload["clk_freq_hz"] else 100e6
user_period_s = 1.0 / eff_clk_hz
mode = (payload.get("clk_mode") or "capped").strip().lower()
if mode == "critical":
    actual_period_s = gcp_s if gcp_s > 0 else user_period_s
else:
    actual_period_s = max(gcp_s, user_period_s) if gcp_s > 0 else user_period_s
latency_ns = float(layer["read_latency_s"]) * actual_period_s * 1e9
print(json.dumps({
    "latency_ns": latency_ns,
    "energy_nj": float(layer["read_energy_nj"]),
}))
"""
        env = os.environ.copy()
        repo_root = "/workspace/PointKAN-pytorch"
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = repo_root if not existing else (repo_root + os.pathsep + existing)
        proc = subprocess.run(
            [sys.executable, "-c", child_code, json.dumps(payload)],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            raise RuntimeError(stderr or f"pybind child exited with {proc.returncode}")
        try:
            parsed = json.loads((proc.stdout or "").strip().splitlines()[-1])
        except Exception as exc:
            raise RuntimeError(f"failed to parse pybind child output: {exc}; stdout={proc.stdout!r}")
        return float(parsed["latency_ns"]), float(parsed["energy_nj"])


@lru_cache(maxsize=512)
def _estimate_conv1d_neurosim_cached(
    *,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    num_points: int,
    clock_period_ns: float,
    neurosim_root: str,
    wl_weight: int,
    wl_activation: int,
    subarray: int,
    parallel_read: int,
    fallback_power_mw: float,
    clk_freq_hz: Optional[float],
    technode: Optional[int],
    adc_clk_freq_hz: Optional[float],
    sar_column_power_w: Optional[float],
    num_col_muxed: int,
    clk_mode: str,
    mode: str,
    allow_subarray_retry: bool,
) -> Tuple[int, float, float, float, str]:
    in_ch = max(1, int(in_channels))
    out_ch = max(1, int(out_channels))
    k_w = max(1, int(kernel_size))
    L = max(k_w, int(num_points))
    num_patches = (L - k_w) + 1
    K = in_ch * k_w
    normalized_mode = _normalize_single_layer_mode(mode)
    input_stretched = _build_single_layer_input_stretched(
        layer_type="conv",
        in_channels=in_ch,
        kernel_size=k_w,
        length_or_tokens=L,
    )
    activity_row_read = _estimate_activation_row_activity(input_stretched, wl_activation)
    weight_data = _build_single_layer_weight_data(
        layer_type="conv",
        out_channels=out_ch,
        in_channels=in_ch,
        kernel_size=k_w,
    )
    weight_matrix = _build_weight_matrix_for_neurosim(weight_data)
    conductance_weight_matrix = _build_conductance_weight_matrix_for_pybind(
        weight_matrix,
        synapse_bit=wl_weight,
        cell_bit=1,
        max_conductance=1.0 / 100e3,
        min_conductance=1.0 / (100e3 * 17.0),
    )
    input_matrix = _build_activation_matrix_for_neurosim(input_stretched, wl_activation)

    if normalized_mode in {"pybind", "auto"}:
        candidates = (
            _single_layer_retry_candidates(
                in_dim=K,
                out_dim=out_ch,
                requested_subarray=subarray,
                requested_parallel_read=parallel_read,
            ) if allow_subarray_retry else [(max(1, int(subarray)), max(1, int(parallel_read)))]
        )
        for cand_subarray, cand_parallel_read in candidates:
            try:
                latency_ns, energy_nj = _estimate_single_layer_pybind(
                    layer_type="conv",
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=k_w,
                    length_or_tokens=L,
                    neurosim_root=neurosim_root,
                    wl_weight=wl_weight,
                    wl_activation=wl_activation,
                    subarray=cand_subarray,
                    parallel_read=cand_parallel_read,
                    clk_freq_hz=clk_freq_hz,
                    technode=technode,
                    adc_clk_freq_hz=adc_clk_freq_hz,
                    sar_column_power_w=sar_column_power_w,
                    num_col_muxed=num_col_muxed,
                    clk_mode=clk_mode,
                    activity_row_read=activity_row_read,
                    weight_matrix=conductance_weight_matrix,
                    input_matrix=input_matrix,
                )
                cycles = max(1, math.ceil(latency_ns / clock_period_ns))
                avg_power_mw = (energy_nj * 1e3 / latency_ns) if latency_ns > 0 else 0.0
                source = "neurosim_pybind"
                if cand_subarray != subarray or cand_parallel_read != parallel_read:
                    source = f"neurosim_pybind_retry_s{cand_subarray}_p{cand_parallel_read}"
                return cycles, latency_ns, energy_nj, avg_power_mw, source
            except Exception:
                continue
        normalized_mode = "subprocess"

    stride = 1
    in_h, in_w = 1, L
    in_c, out_c = in_ch, out_ch
    k_h = 1
    pool = 0

    with tempfile.TemporaryDirectory(prefix="pointkan_neurosim_") as tmpdir:
        config_path = os.path.join(tmpdir, "config.csv")
        with open(config_path, "w") as f:
            f.write(f"{in_h},{in_w},{in_c},{k_h},{k_w},{out_c},{pool},{stride}\n")

        weight_path = os.path.join(tmpdir, "weight0.csv")
        input_path = os.path.join(tmpdir, "input0.csv")
        _write_matrix_weight(weight_data, weight_path)
        _write_matrix_activation_conv(input_stretched, wl_activation, input_path)

        candidates = (
            _single_layer_retry_candidates(
                in_dim=K,
                out_dim=out_ch,
                requested_subarray=subarray,
                requested_parallel_read=parallel_read,
            ) if allow_subarray_retry else [(max(1, int(subarray)), max(1, int(parallel_read)))]
        )
        for cand_subarray, cand_parallel_read in candidates:
            ok, out, err = _run_neurosim_main(
                config_path,
                [(weight_path, input_path)],
                neurosim_root,
                wl_weight=wl_weight,
                wl_activation=wl_activation,
                subarray=cand_subarray,
                parallel_read=cand_parallel_read,
                clk_freq_hz=clk_freq_hz,
                technode=technode,
                adc_clk_freq_hz=adc_clk_freq_hz,
                sar_column_power_w=sar_column_power_w,
                num_col_muxed=num_col_muxed,
            )
            if not ok or not out:
                continue
            parsed = _parse_layer_results(out)
            if not parsed:
                continue
            read_ns, read_pj, _buf_ns, _buf_pj = parsed[0]
            total_latency_ns = read_ns
            energy_nj = read_pj / 1000.0
            cycles = max(1, math.ceil(total_latency_ns / clock_period_ns))
            avg_power_mw = (energy_nj * 1e3 / total_latency_ns) if total_latency_ns > 0 else 0.0
            source = "neurosim"
            if cand_subarray != subarray or cand_parallel_read != parallel_read:
                source = f"neurosim_retry_s{cand_subarray}_p{cand_parallel_read}"
            return cycles, total_latency_ns, energy_nj, avg_power_mw, source

    fallback = _fallback_estimate(
        work_items=int(math.ceil(num_patches * K * out_ch / 128)),
        clock_period_ns=clock_period_ns,
        fallback_power_mw=fallback_power_mw,
        source="neurosim_fallback",
    )
    return fallback.cycles, fallback.latency_ns, fallback.energy_nj, fallback.avg_power_mw, fallback.source


@lru_cache(maxsize=512)
def _estimate_linear_neurosim_cached(
    *,
    in_features: int,
    out_features: int,
    tokens: int,
    clock_period_ns: float,
    neurosim_root: str,
    wl_weight: int,
    wl_activation: int,
    subarray: int,
    parallel_read: int,
    fallback_power_mw: float,
    clk_freq_hz: Optional[float],
    technode: Optional[int],
    adc_clk_freq_hz: Optional[float],
    sar_column_power_w: Optional[float],
    num_col_muxed: int,
    clk_mode: str,
    mode: str,
    allow_subarray_retry: bool,
) -> Tuple[int, float, float, float, str]:
    in_f = max(1, int(in_features))
    out_f = max(1, int(out_features))
    M = max(1, int(tokens))
    normalized_mode = _normalize_single_layer_mode(mode)
    input_stretched = _build_single_layer_input_stretched(
        layer_type="fc",
        in_channels=in_f,
        kernel_size=1,
        length_or_tokens=M,
    )
    activity_row_read = _estimate_activation_row_activity(input_stretched, wl_activation)
    weight_data = _build_single_layer_weight_data(
        layer_type="fc",
        out_channels=out_f,
        in_channels=in_f,
        kernel_size=1,
    )
    weight_matrix = _build_weight_matrix_for_neurosim(weight_data)
    conductance_weight_matrix = _build_conductance_weight_matrix_for_pybind(
        weight_matrix,
        synapse_bit=wl_weight,
        cell_bit=1,
        max_conductance=1.0 / 100e3,
        min_conductance=1.0 / (100e3 * 17.0),
    )
    input_matrix = _build_activation_matrix_for_neurosim(input_stretched, wl_activation)

    if normalized_mode in {"pybind", "auto"}:
        candidates = (
            _single_layer_retry_candidates(
                in_dim=in_f,
                out_dim=out_f,
                requested_subarray=subarray,
                requested_parallel_read=parallel_read,
            ) if allow_subarray_retry else [(max(1, int(subarray)), max(1, int(parallel_read)))]
        )
        for cand_subarray, cand_parallel_read in candidates:
            try:
                latency_ns, energy_nj = _estimate_single_layer_pybind(
                    layer_type="fc",
                    in_channels=in_f,
                    out_channels=out_f,
                    kernel_size=1,
                    length_or_tokens=M,
                    neurosim_root=neurosim_root,
                    wl_weight=wl_weight,
                    wl_activation=wl_activation,
                    subarray=cand_subarray,
                    parallel_read=cand_parallel_read,
                    clk_freq_hz=clk_freq_hz,
                    technode=technode,
                    adc_clk_freq_hz=adc_clk_freq_hz,
                    sar_column_power_w=sar_column_power_w,
                    num_col_muxed=num_col_muxed,
                    clk_mode=clk_mode,
                    activity_row_read=activity_row_read,
                    weight_matrix=conductance_weight_matrix,
                    input_matrix=input_matrix,
                )
                cycles = max(1, math.ceil(latency_ns / clock_period_ns))
                avg_power_mw = (energy_nj * 1e3 / latency_ns) if latency_ns > 0 else 0.0
                source = "neurosim_pybind"
                if cand_subarray != subarray or cand_parallel_read != parallel_read:
                    source = f"neurosim_pybind_retry_s{cand_subarray}_p{cand_parallel_read}"
                return cycles, latency_ns, energy_nj, avg_power_mw, source
            except Exception:
                continue
        normalized_mode = "subprocess"

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
        _write_matrix_weight(weight_data, weight_path)
        _write_matrix_activation_conv(input_stretched, wl_activation, input_path)

        candidates = (
            _single_layer_retry_candidates(
                in_dim=in_f,
                out_dim=out_f,
                requested_subarray=subarray,
                requested_parallel_read=parallel_read,
            ) if allow_subarray_retry else [(max(1, int(subarray)), max(1, int(parallel_read)))]
        )
        for cand_subarray, cand_parallel_read in candidates:
            ok, out, err = _run_neurosim_main(
                config_path,
                [(weight_path, input_path)],
                neurosim_root,
                wl_weight=wl_weight,
                wl_activation=wl_activation,
                subarray=cand_subarray,
                parallel_read=cand_parallel_read,
                clk_freq_hz=clk_freq_hz,
                technode=technode,
                adc_clk_freq_hz=adc_clk_freq_hz,
                sar_column_power_w=sar_column_power_w,
                num_col_muxed=num_col_muxed,
            )
            if not ok or not out:
                continue
            parsed = _parse_layer_results(out)
            if not parsed:
                continue
            read_ns, read_pj, _buf_ns, _buf_pj = parsed[0]
            total_latency_ns = read_ns
            energy_nj = read_pj / 1000.0
            cycles = max(1, math.ceil(total_latency_ns / clock_period_ns))
            avg_power_mw = (energy_nj * 1e3 / total_latency_ns) if total_latency_ns > 0 else 0.0
            source = "neurosim"
            if cand_subarray != subarray or cand_parallel_read != parallel_read:
                source = f"neurosim_retry_s{cand_subarray}_p{cand_parallel_read}"
            return cycles, total_latency_ns, energy_nj, avg_power_mw, source

    fallback = _fallback_estimate(
        work_items=int(math.ceil(M * in_f * out_f / 128)),
        clock_period_ns=clock_period_ns,
        fallback_power_mw=fallback_power_mw,
        source="neurosim_fallback",
    )
    return fallback.cycles, fallback.latency_ns, fallback.energy_nj, fallback.avg_power_mw, fallback.source


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
    clk_freq_hz: Optional[float] = None,
    technode: Optional[int] = None,
    adc_clk_freq_hz: Optional[float] = None,
    sar_column_power_w: Optional[float] = None,
    num_col_muxed: int = 8,
    clk_mode: str = "capped",
    mode: str = "subprocess",
    allow_subarray_retry: bool = True,
) -> NeuroSimEstimate:
    """
    Conv1d 单层：生成 1 行 config + weight/input CSV，调 NeuroSim main，解析 layer1 结果。
    Conv1d 映射为 in_h=1, in_w=num_points, in_c=in_channels, k_h=1, k_w=kernel_size, out_c=out_channels.
    """
    cycles, latency_ns, energy_nj, avg_power_mw, source = _estimate_conv1d_neurosim_cached(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        num_points=num_points,
        clock_period_ns=clock_period_ns,
        neurosim_root=neurosim_root,
        wl_weight=wl_weight,
        wl_activation=wl_activation,
        subarray=subarray,
        parallel_read=parallel_read,
        fallback_power_mw=fallback_power_mw,
        clk_freq_hz=clk_freq_hz,
        technode=technode,
        adc_clk_freq_hz=adc_clk_freq_hz,
        sar_column_power_w=sar_column_power_w,
        num_col_muxed=int(num_col_muxed),
        clk_mode=clk_mode,
        mode=mode,
        allow_subarray_retry=allow_subarray_retry,
    )
    return NeuroSimEstimate(
        cycles=cycles,
        latency_ns=latency_ns,
        energy_nj=energy_nj,
        avg_power_mw=avg_power_mw,
        source=source,
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
    clk_freq_hz: Optional[float] = None,
    technode: Optional[int] = None,
    adc_clk_freq_hz: Optional[float] = None,
    sar_column_power_w: Optional[float] = None,
    num_col_muxed: int = 8,
    clk_mode: str = "capped",
    mode: str = "subprocess",
    allow_subarray_retry: bool = True,
) -> NeuroSimEstimate:
    """
    Linear 单层：映射为 1x1 conv (in_h=1, in_w=1, in_c=in_f, k_h=1, k_w=1, out_c=out_f)。
    tokens 对应 num_patches：NeuroSim 用 in_w 表示“宽度”，全连接时 in_w=1 则 numInVector=1；
    若希望多 token，需 in_w=tokens，此时 in_h=1, in_w=tokens, in_c=in_f, k_h=1, k_w=1 -> num_patches = tokens.
    """
    cycles, latency_ns, energy_nj, avg_power_mw, source = _estimate_linear_neurosim_cached(
        in_features=in_features,
        out_features=out_features,
        tokens=tokens,
        clock_period_ns=clock_period_ns,
        neurosim_root=neurosim_root,
        wl_weight=wl_weight,
        wl_activation=wl_activation,
        subarray=subarray,
        parallel_read=parallel_read,
        fallback_power_mw=fallback_power_mw,
        clk_freq_hz=clk_freq_hz,
        technode=technode,
        adc_clk_freq_hz=adc_clk_freq_hz,
        sar_column_power_w=sar_column_power_w,
        num_col_muxed=int(num_col_muxed),
        clk_mode=clk_mode,
        mode=mode,
        allow_subarray_retry=allow_subarray_retry,
    )
    return NeuroSimEstimate(
        cycles=cycles,
        latency_ns=latency_ns,
        energy_nj=energy_nj,
        avg_power_mw=avg_power_mw,
        source=source,
    )


# NOTE: estimate_rram_dist_vmm_neurosim 已移除。
# PointCIM RRAM VMM 建模改用 MICSim（matrix_power_estimator._run_micsim_single_fc），
# 由 rram_matrix_estimator.compute_rram_dist_primitives 统一调度。
# 请勿使用 DNN_NeuroSim_V1.4 main 子进程路径建模 RRAM 距离阵列。

def _estimate_rram_dist_vmm_neurosim_REMOVED(
    *,
    coord_dims: int = 3,
    num_points: int = 1024,
    coord_bits: int = 12,
    input_bits: int = 12,
    array_rows: int = 128,
    array_cols: int = 128,
    clock_period_ns: float = 1.0,
    neurosim_root: str = "/workspace/DNN_NeuroSim_V1.4/Inference_pytorch",
    fallback_power_mw: float = 25.0,
    clk_freq_hz: Optional[float] = None,
    technode: int = 32,
) -> NeuroSimEstimate:
    """
    用 NeuroSim 估算 RRAM 距离阵列的 VMM 阶段（PointCIM 模式 Stage 2）。

    距离公式 ||qi - pj||² 的交叉项 -2qi·pj 可映射为一个 Linear 层：
      in_features  = coord_dims  (= 3, 查询坐标 x/y/z)
      out_features = num_points  (= N, 存储点数)
      wl_weight    = coord_bits  (RRAM 权重精度，点云坐标量化位宽)
      wl_activation= input_bits  (输入精度，查询坐标量化位宽)
      subarray     = min(array_rows, array_cols)

    NeuroSim 建模 RRAM crossbar 的 VMM 推理阶段（read latency + energy + 外围），
    不包含写入（编程）阶段——写入阶段由 `compute_rram_dist_primitives` 中的解析模型覆盖。

    Args:
        coord_dims:      坐标维度（默认 3）
        num_points:      存储点数 N（作为 out_features）
        coord_bits:      RRAM 权重位宽
        input_bits:      输入激活位宽
        array_rows:      RRAM array 行数（影响 subarray 选取上限）
        array_cols:      RRAM array 列数
        clock_period_ns: 时钟周期（ns）
        neurosim_root:   DNN_NeuroSim_V1.4/Inference_pytorch 路径
        fallback_power_mw: NeuroSim 失败时的回退功耗
        clk_freq_hz:     可选时钟频率（Hz），传给 NeuroSim main
        technode:        可选工艺节点（nm）

    Returns:
        NeuroSimEstimate，source 字段标注 "neurosim_rram_vmm" 或 "neurosim_rram_vmm_fallback"
    """
    # 28nm 不受支持，自动映射为 32nm（与 neurosim_full.py 保持一致）
    if technode == 28:
        technode = 32

    in_f = max(1, coord_dims)
    out_f = max(1, num_points)
    # NeuroSim 要求 subarray 行数 ≤ K（= in_c）。
    # RRAM 距离 VMM: K = coord_dims = 3，subarray 必须 ≤ 3，取不超过 in_f 的最大 2 次幂。
    max_subarray = 1 << max(0, in_f.bit_length() - 1)   # 最大不超过 in_f 的 2 次幂
    max_subarray = max(1, min(max_subarray, array_rows, 128))
    subarray = max_subarray
    parallel_read = min(array_cols, 128)

    # 映射为 tokens=1 的 Linear 推理
    in_h, in_w = 1, 1
    in_c, out_c = in_f, out_f
    k_h, k_w = 1, 1
    pool, stride = 0, 1

    with tempfile.TemporaryDirectory(prefix="pointkan_rram_dist_") as tmpdir:
        config_path = os.path.join(tmpdir, "config.csv")
        with open(config_path, "w") as f:
            f.write(f"{in_h},{in_w},{in_c},{k_h},{k_w},{out_c},{pool},{stride}\n")

        weight_path = os.path.join(tmpdir, "weight0.csv")
        input_path = os.path.join(tmpdir, "input0.csv")
        weight_data = np.random.randn(out_c, in_c, k_h, k_w).astype(np.float32)
        _write_matrix_weight(weight_data, weight_path)
        # _write_matrix_activation_conv 期望 (1, num_patches, K)
        # 对于 Linear 映射（1 query，in_c 个坐标特征）：num_patches=1，K=in_c
        input_data = np.random.rand(1, 1, in_c).astype(np.float32) * (2 ** input_bits - 1)
        _write_matrix_activation_conv(input_data, input_bits, input_path)

        # RRAM VMM 候选集：subarray 必须 ≤ in_f（K=coord_dims），从最大 2 次幂向下逐步缩小
        rram_candidates = _unique_positive_pairs([
            (s, min(parallel_read, s))
            for s in [subarray, subarray // 2, max(1, subarray // 4), 1]
            if s > 0
        ])
        for cand_subarray, cand_parallel_read in rram_candidates:
            ok, out_str, err = _run_neurosim_main(
                config_path,
                [(weight_path, input_path)],
                neurosim_root,
                wl_weight=coord_bits,
                wl_activation=input_bits,
                subarray=cand_subarray,
                parallel_read=cand_parallel_read,
                clk_freq_hz=clk_freq_hz,
                technode=technode,
            )
            if not ok or not out_str:
                continue
            parsed = _parse_layer_results(out_str)
            if not parsed:
                continue
            read_ns, read_pj, buf_ns, buf_pj = parsed[0]
            # NeuroSim main's read totals already include buffer + IC on the
            # active DNN path; keep buffer values for diagnostics only.
            total_latency_ns = read_ns
            total_energy_pj = read_pj
            energy_nj = total_energy_pj / 1000.0
            cycles = max(1, math.ceil(total_latency_ns / clock_period_ns))
            avg_power_mw = (energy_nj * 1e3 / total_latency_ns) if total_latency_ns > 0 else 0.0
            source = "neurosim_rram_vmm"
            if cand_subarray != subarray or cand_parallel_read != parallel_read:
                source = f"neurosim_rram_vmm_retry_s{cand_subarray}_p{cand_parallel_read}"
            return NeuroSimEstimate(
                cycles=cycles,
                latency_ns=total_latency_ns,
                energy_nj=energy_nj,
                avg_power_mw=avg_power_mw,
                source=source,
            )

    # NeuroSim 全部 candidate 失败 → 解析回退
    work_items = max(1, math.ceil(in_f * out_f / 128))
    return _fallback_estimate(
        work_items=work_items,
        clock_period_ns=clock_period_ns,
        fallback_power_mw=fallback_power_mw,
        source="neurosim_rram_vmm_fallback",
    )
