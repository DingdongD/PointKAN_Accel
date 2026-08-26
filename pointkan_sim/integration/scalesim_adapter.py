"""
SCALE-Sim 适配器：用 SCALE-Sim 为 baseline 脉动阵列提供周期、利用率、MAC/访问统计与能耗。

- **周期**：从 COMPUTE_REPORT 的 Total Cycles 取用。
- **利用率**：从 COMPUTE_REPORT 的 Overall Util %、Mapping Efficiency %、Compute Util % 取用。
- **阵列尺寸**：由 hw_config.systolic_array.rows/cols 传入 SCALE-Sim（生成临时 config 的 ArrayHeight/ArrayWidth）。
- **能耗**：与 hw_config 联动（与 pointkan_sim.integration.memory_model_adapter 的 SRAM/DRAM 访问级模型一致）。
  - MAC：来自 SystolicArrayConfig.mac_power_mw。
  - SRAM/DRAM：(1) 访问次数×单次能耗（CACTI/DRAMSim3 回填）；(2) 或 trace→DRAMSim3 得 DRAM 能耗（见下）。

**Trace 调度与 SCALE-Sim/demo_with_accelergy.py 对齐**：
  - 本 adapter 在单层跑 SCALE-Sim 时若 DRAMConfig 提供 dramsim3_bin 与 dramsim3_config_ini，会保存 trace 并调用
    DRAMSim3，用其 read_energy+write_energy 作为该层 DRAM 能耗（_run_dramsim3_for_layer_trace）。
  - 完整两遍调度（更精确的 stall 与延迟）可参考 demo_with_accelergy.py：第一次 SCALE-Sim save_trace=True →
    合并 trace → DRAMSim3 → dram_latency_dramsim3.py 生成 .npy → 第二次 SCALE-Sim UseRamulatorTrace=True 读 .npy。
    本工程当前采用单遍 SCALE-Sim + 可选 DRAMSim3 能耗，周期仍来自 SCALE-Sim Total Cycles。
"""

from __future__ import annotations

import configparser
import csv
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..configs.hw_config import HardwareConfig

from .dramsim_utils import parse_dramsim3_burst_bytes as _parse_dramsim3_burst_bytes
from .dramsim_utils import read_dramsim3_ini as _read_dramsim3_ini

# 默认 SCALE-Sim 根目录（可被环境变量覆盖）
DEFAULT_SCALE_SIM_ROOT = os.environ.get("SCALE_SIM_ROOT", "/workspace/SCALE-Sim")
DEFAULT_RUN_NAME = "pointkan_systolic"

# 已打印过的 (rows, cols)，避免每层都打日志
_scalesim_logged_array: set = set()


def _log_scalesim_array_size_once(rows: int, cols: int, op_kind: str, layer_name: str) -> None:
    key = (rows, cols)
    if key in _scalesim_logged_array:
        return
    _scalesim_logged_array.add(key)
    try:
        from ..utils.logger import log_info
        log_info(f"SCALE-Sim 阵列尺寸: {rows}×{cols}（{op_kind} 首层 {layer_name}，后续层同）")
    except Exception:
        pass


def _build_scalesim_progress_prefix(
    op_kind: str,
    layer_name: str,
    *,
    progress_index: Optional[int] = None,
    progress_total: Optional[int] = None,
) -> str:
    op_label = str(op_kind or "layer").strip() or "layer"
    layer_label = str(layer_name or "").strip() or "unnamed"
    if progress_index is not None and progress_total is not None:
        return f"SCALE-Sim progress [{op_label} {int(progress_index)}/{int(progress_total)} {layer_label}]"
    return f"SCALE-Sim progress [{op_label} {layer_label}]"


def _log_scalesim_progress(
    op_kind: str,
    layer_name: str,
    message: str,
    *,
    progress_index: Optional[int] = None,
    progress_total: Optional[int] = None,
) -> None:
    try:
        from ..utils.logger import log_info

        prefix = _build_scalesim_progress_prefix(
            op_kind,
            layer_name,
            progress_index=progress_index,
            progress_total=progress_total,
        )
        log_info(f"{prefix} {message}")
    except Exception:
        pass


def _format_elapsed_seconds(elapsed_s: float) -> str:
    if elapsed_s < 1.0:
        return f"{elapsed_s:.2f}s"
    return f"{elapsed_s:.1f}s"


def _strip_keys(d: dict) -> dict:
    return {k.strip(): v for k, v in d.items()}


def _float_util(key: str, row: dict, default: Optional[float] = None):
    """解析 COMPUTE_REPORT 中的利用率字段（可能为数字或 N/A）。"""
    v = row.get(key, default)
    if v == "" or v is None:
        return default
    s = str(v).strip()
    if s.upper() == "N/A":
        return default
    try:
        return float(s)
    except ValueError:
        return default


@dataclass
class ScalesimLayerResult:
    """单层 SCALE-Sim 运行结果：周期、利用率、访问统计与能耗（操作×单次能耗）。"""
    total_cycles: int = 0
    stall_cycles: int = 0
    sram_ifmap_reads: int = 0
    sram_filter_reads: int = 0
    sram_ofmap_writes: int = 0
    dram_ifmap_reads: int = 0
    dram_filter_reads: int = 0
    dram_ofmap_writes: int = 0
    macs: int = 0
    overall_util_pct: Optional[float] = None
    mapping_eff_pct: Optional[float] = None
    compute_util_pct: Optional[float] = None
    energy_nj: float = 0.0
    mac_energy_nj: float = 0.0
    sram_energy_nj: float = 0.0
    dram_energy_nj: float = 0.0
    sram_cycles: int = 0
    dram_cycles: int = 0
    sram_clock_period_ns: float = 1.0
    dram_clock_period_ns: float = 1.0
    dram_trace_dir: Optional[str] = None
    feature_input_dram_trace_dir: Optional[str] = None
    weight_dram_trace_dir: Optional[str] = None
    feature_output_dram_trace_dir: Optional[str] = None
    feature_input_dram_bytes: int = 0
    feature_output_dram_bytes: int = 0
    weight_dram_bytes: int = 0
    weight_tensor_bytes: int = 0
    weight_dram_trace_bytes: int = 0
    feature_local_sram_trace_bytes: int = 0
    weight_sram_trace_bytes: int = 0
    feature_input_dram_energy_nj: float = 0.0
    weight_dram_energy_nj: float = 0.0
    feature_output_dram_energy_nj: float = 0.0
    feature_input_dram_cycles: int = 0
    weight_dram_cycles: int = 0
    feature_output_dram_cycles: int = 0
    weight_sram_energy_nj: float = 0.0
    weight_sram_cycles: int = 0
    weight_sram_bytes_read: int = 0
    weight_sram_bytes_write: int = 0
    weight_refill_tiles: int = 1
    weight_refill_tile_bytes: int = 0
    feature_local_sram_energy_nj: float = 0.0
    feature_local_sram_cycles: int = 0
    feature_local_sram_bytes_read: int = 0
    feature_local_sram_bytes_write: int = 0
    success: bool = False
    message: str = ""
    systolic_array_rows: int = 0  # 本次 SCALE-Sim 使用的 ArrayHeight，便于 CSV 核对
    systolic_array_cols: int = 0  # 本次 SCALE-Sim 使用的 ArrayWidth

    def to_metadata(self) -> Dict[str, Any]:
        """转为事件 metadata，供 breakdown 与逐层 CSV 使用。"""
        return {
            "scalesim_total_cycles": self.total_cycles,
            "scalesim_stall_cycles": self.stall_cycles,
            "systolic_array_rows": self.systolic_array_rows,
            "systolic_array_cols": self.systolic_array_cols,
            "sram_ifmap_reads": self.sram_ifmap_reads,
            "sram_filter_reads": self.sram_filter_reads,
            "sram_ofmap_writes": self.sram_ofmap_writes,
            "dram_ifmap_reads": self.dram_ifmap_reads,
            "dram_filter_reads": self.dram_filter_reads,
            "dram_ofmap_writes": self.dram_ofmap_writes,
            "macs": self.macs,
            "scalesim_energy_nj": self.energy_nj,
            "scalesim_mac_energy_nj": self.mac_energy_nj,
            "scalesim_sram_energy_nj": self.sram_energy_nj,
            "scalesim_dram_energy_nj": self.dram_energy_nj,
            "scalesim_sram_cycles": self.sram_cycles,
            "scalesim_dram_cycles": self.dram_cycles,
            "scalesim_sram_clock_period_ns": self.sram_clock_period_ns,
            "scalesim_dram_clock_period_ns": self.dram_clock_period_ns,
            "scalesim_dram_trace_dir": self.dram_trace_dir,
            "feature_input_dram_trace_dir": getattr(self, "feature_input_dram_trace_dir", None),
            "weight_dram_trace_dir": getattr(self, "weight_dram_trace_dir", None),
            "feature_output_dram_trace_dir": getattr(self, "feature_output_dram_trace_dir", None),
            "feature_input_dram_bytes": getattr(self, "feature_input_dram_bytes", 0),
            "feature_output_dram_bytes": getattr(self, "feature_output_dram_bytes", 0),
            "weight_dram_bytes": getattr(self, "weight_dram_bytes", 0),
            "weight_tensor_bytes": getattr(self, "weight_tensor_bytes", 0),
            "weight_dram_trace_bytes": getattr(self, "weight_dram_trace_bytes", 0),
            "feature_local_sram_trace_bytes": getattr(self, "feature_local_sram_trace_bytes", 0),
            "weight_sram_trace_bytes": getattr(self, "weight_sram_trace_bytes", 0),
            "feature_input_dram_energy_nj": getattr(self, "feature_input_dram_energy_nj", 0.0),
            "weight_dram_energy_nj": getattr(self, "weight_dram_energy_nj", 0.0),
            "feature_output_dram_energy_nj": getattr(self, "feature_output_dram_energy_nj", 0.0),
            "feature_input_dram_cycles": getattr(self, "feature_input_dram_cycles", 0),
            "weight_dram_cycles": getattr(self, "weight_dram_cycles", 0),
            "feature_output_dram_cycles": getattr(self, "feature_output_dram_cycles", 0),
            "weight_sram_energy_nj": getattr(self, "weight_sram_energy_nj", 0.0),
            "weight_sram_cycles": getattr(self, "weight_sram_cycles", 0),
            "weight_sram_bytes_read": getattr(self, "weight_sram_bytes_read", 0),
            "weight_sram_bytes_write": getattr(self, "weight_sram_bytes_write", 0),
            "weight_refill_tiles": getattr(self, "weight_refill_tiles", 1),
            "weight_refill_tile_bytes": getattr(self, "weight_refill_tile_bytes", 0),
            "feature_local_sram_energy_nj": getattr(self, "feature_local_sram_energy_nj", 0.0),
            "feature_local_sram_cycles": getattr(self, "feature_local_sram_cycles", 0),
            "feature_local_sram_bytes_read": getattr(self, "feature_local_sram_bytes_read", 0),
            "feature_local_sram_bytes_write": getattr(self, "feature_local_sram_bytes_write", 0),
            "overall_util_pct": self.overall_util_pct,
            "mapping_eff_pct": self.mapping_eff_pct,
            "compute_util_pct": self.compute_util_pct,
        }


def _write_conv_topology_csv(path: str, layer_name: str, ifmap_h: int, ifmap_w: int,
                             filter_h: int, filter_w: int, channels: int, num_filter: int,
                             stride: int = 1) -> None:
    with open(path, "w") as f:
        f.write("Layer name, IFMAP Height, IFMAP Width, Filter Height, Filter Width, Channels, Num Filter, Strides,\n")
        f.write("{}, {}, {}, {}, {}, {}, {}, {},\n".format(
            layer_name, ifmap_h, ifmap_w, filter_h, filter_w, channels, num_filter, stride))


def _write_layout_csv(path: str, layer_names: list) -> None:
    header = ("Layer name, IFMAP Height Intraline Factor, IFMAP Width Intraline Factor, "
              "Filter Height Intraline Factor, Filter Width Intraline Factor, Channel Intraline Factor, "
              "Num Filter Intraline Factor, IFMAP Height Intraline Order, IFMAP Width Intraline Order, "
              "Channel Intraline Order, IFMAP Height Interline Order, IFMAP Width Interline Order, "
              "Channel Interline Order, Num Filter Intraline Order, Channel Intraline Order, "
              "Filter Height Intraline Order, Filter Width Intraline Order, Num Filter Interline Order, "
              "Channel Intraline Order, Filter Height Interline Order, Filter Width Interline Order, \n")
    row = "2, 2, 1, 1, 16, 4,     0, 1, 2,     4, 5, 3,     3, 2, 1, 0,     4, 5, 6, 7,\n"
    with open(path, "w") as f:
        f.write(header)
        for name in layer_names:
            f.write(name + ", " + row.strip() + "\n")


def _int(key: str, row: dict, default: int = 0) -> int:
    v = row.get(key, default)
    if v == "" or v is None:
        return default
    try:
        return int(float(str(v).strip()))
    except ValueError:
        return default


def _energy_breakdown_from_hw_config(
    hw_config: Optional["HardwareConfig"],
    macs: int,
    sram_ifmap_reads: int,
    sram_filter_reads: int,
    sram_ofmap_writes: int,
    dram_ifmap_reads: int,
    dram_filter_reads: int,
    dram_ofmap_writes: int,
    dram_nj_override: Optional[float] = None,
    sram_nj_override: Optional[float] = None,
) -> Dict[str, float]:
    """Return MAC/SRAM/DRAM energy components for one SCALE-Sim layer."""
    if hw_config is None:
        return {"mac_energy_nj": 0.0, "sram_energy_nj": 0.0, "dram_energy_nj": 0.0, "total_energy_nj": 0.0}

    clock_ns = float(getattr(hw_config, "clock_period", 1.0) or 1.0)
    sa = getattr(hw_config, "systolic_array", None)
    mac_power_mw = float(getattr(sa, "mac_power_mw", 7.144) or 7.144) if sa else 7.144
    mac_nj = macs * mac_power_mw * clock_ns / 1000.0

    if sram_nj_override is not None:
        sram_nj = float(sram_nj_override)
    else:
        sram_nj = 0.0
        buf = getattr(hw_config, "inter_layer_buffer", None)  # 通过CACTI会调用回填？
        if buf is not None:
            read_nj = getattr(buf, "read_energy_nj_per_access", None)
            write_nj = getattr(buf, "write_energy_nj_per_access", None)
            if read_nj is not None:
                sram_nj += (sram_ifmap_reads + sram_filter_reads) * float(read_nj)
            if write_nj is not None:
                sram_nj += sram_ofmap_writes * float(write_nj)

    if dram_nj_override is not None:
        dram_nj = float(dram_nj_override)
    else:
        dram_nj = 0.0
        dram = getattr(hw_config, "dram", None)
        if dram is not None:
            read_nj = getattr(dram, "read_energy_nj_per_access", None)
            write_nj = getattr(dram, "write_energy_nj_per_access", None)
            if read_nj is not None:
                dram_nj += (dram_ifmap_reads + dram_filter_reads) * float(read_nj)
            if write_nj is not None:
                dram_nj += dram_ofmap_writes * float(write_nj)

    return {
        "mac_energy_nj": mac_nj,
        "sram_energy_nj": sram_nj,
        "dram_energy_nj": dram_nj,
        "total_energy_nj": mac_nj + sram_nj + dram_nj,
    }



_scalesim_logged_mem_config: set = set()


def _get_scalesim_data_bytes(hw_config: Optional["HardwareConfig"]) -> int:
    if hw_config is not None:
        sa = getattr(hw_config, "systolic_array", None)
        if sa is not None:
            v = getattr(sa, "data_bytes", None)
            if v:
                try:
                    return max(1, int(v))
                except Exception:
                    pass
    return 2  # default FXP16


def _get_scalesim_sram_cfg(hw_config: Optional["HardwareConfig"]) -> Dict[str, int]:
    size_bytes = 32768
    ifmap_size_bytes = 32768
    filter_size_bytes = 32768
    ofmap_size_bytes = 32768
    block_bytes = 64
    banks = 1
    bus_bits = 128
    if hw_config is not None:
        sa = getattr(hw_config, "systolic_array", None)
        if sa is not None:
            size_bytes = int(getattr(sa, "sram_size_bytes", size_bytes) or size_bytes)
            ifmap_size_bytes = int(getattr(sa, "ifmap_sram_size_bytes", None) or size_bytes)
            filter_size_bytes = int(getattr(sa, "filter_sram_size_bytes", None) or size_bytes)
            ofmap_size_bytes = int(getattr(sa, "ofmap_sram_size_bytes", None) or size_bytes)
            block_bytes = int(getattr(sa, "sram_block_bytes", block_bytes) or block_bytes)
            banks = int(getattr(sa, "sram_num_banks", banks) or banks)
            bus_bits = int(getattr(sa, "sram_bus_width_bits", bus_bits) or bus_bits)
    return {
        "size_bytes": max(1, size_bytes),
        "ifmap_size_bytes": max(1, ifmap_size_bytes),
        "filter_size_bytes": max(1, filter_size_bytes),
        "ofmap_size_bytes": max(1, ofmap_size_bytes),
        "block_bytes": max(1, block_bytes),
        "banks": max(1, banks),
        "bus_bits": max(1, bus_bits),
    }


def _get_scalesim_sram_freq_hz(hw_config: Optional["HardwareConfig"]) -> int:
    if hw_config is not None:
        sa = getattr(hw_config, "systolic_array", None)
        if sa is not None:
            freq_hz = getattr(sa, "sram_freq_hz", None)
            if freq_hz:
                try:
                    return max(1, int(float(freq_hz)))
                except Exception:
                    pass
        clock_ns = float(getattr(hw_config, "clock_period", 1.0) or 1.0)
        return max(1, int(round(1e9 / max(1e-6, clock_ns))))
    return 1_000_000_000


def _get_feature_residency_entry(hw_config: Optional["HardwareConfig"], layer_name: str):
    if hw_config is None:
        return None
    plan = getattr(hw_config, "feature_residency_plan_by_layer", None) or {}
    return plan.get(layer_name)


def _estimate_scalesim_filter_bytes(op_kind: str, *, in_channels: int = 0, out_channels: int = 0, kernel_size: int = 1, in_features: int = 0, out_features: int = 0, bytes_per_access: int = 2) -> int:
    if op_kind == "conv1d":
        return max(1, in_channels) * max(1, out_channels) * max(1, kernel_size) * max(1, bytes_per_access)
    return max(1, in_features) * max(1, out_features) * max(1, bytes_per_access)


def _parse_dramsim3_tck_ns(config_path: str) -> float:
    parser = _read_dramsim3_ini(config_path)
    if parser is None:
        return 1.0
    for section in parser.sections():
        if parser.has_option(section, "tCK"):
            try:
                return max(1e-6, float(parser.get(section, "tCK")))
            except ValueError:
                return 1.0
    return 1.0


def _log_scalesim_memory_config_once(tag: str, info: Dict[str, Any]) -> None:
    if tag in _scalesim_logged_mem_config:
        return
    _scalesim_logged_mem_config.add(tag)
    try:
        from ..utils.logger import log_info
        log_info("SCALE-Sim memory-model config:")
        for k, v in info.items():
            log_info(f"  {k} = {v}")
    except Exception:
        pass


def _scalesim_named_sram_runtime_stats(
    hw_config: Optional["HardwareConfig"],
    read_bytes: int,
    write_bytes: int,
    layer_name: str,
    buffer_tag: str,
) -> Optional[Dict[str, Any]]:
    if hw_config is None or (read_bytes + write_bytes) <= 0:
        return None
    repo_root = getattr(hw_config, "repo_root", None)
    try:
        from .memory_model_adapter import get_memory_model_roots
    except Exception:
        return None
    roots = get_memory_model_roots(repo_root)
    if roots is None:
        return None
    memory_model_repo, cycle_sim_abs = roots
    try:
        from memory_model.cacti_runner import CactiRunner
        from memory_model.sram_buffer import SRAMBuffer
        from memory_model.types import CactiInput
    except Exception:
        return None

    cfg = _get_scalesim_sram_cfg(hw_config)
    freq_hz = _get_scalesim_sram_freq_hz(hw_config)
    clock_period_ns = 1e9 / float(freq_hz)

    cacti_bin = os.path.join(cycle_sim_abs, "cacti", "cacti")
    template_cfg = os.path.join(cycle_sim_abs, "cacti", "cache.cfg")
    generated_cfg_dir = os.path.join(memory_model_repo, "configs", "cacti", "generated")
    output_dir = os.path.join(memory_model_repo, "outputs", "cacti")
    if not os.path.isfile(cacti_bin) or not os.path.isfile(template_cfg):
        return None

    _log_scalesim_memory_config_once(
        f"cacti_{buffer_tag}",
        {
            "cacti_bin": cacti_bin,
            "template_cfg": template_cfg,
            "generated_cfg_dir": generated_cfg_dir,
            "output_dir": output_dir,
            "systolic_sram_size_bytes": size_bytes,
            "systolic_sram_block_bytes": cfg["block_bytes"],
            "systolic_sram_banks": cfg["banks"],
            "systolic_sram_bus_bits": cfg["bus_bits"],
            "systolic_sram_freq_hz": freq_hz,
            "systolic_sram_clock_period_ns": clock_period_ns,
            "buffer_tag": buffer_tag,
        },
    )

    runner = CactiRunner(
        cacti_bin=cacti_bin,
        template_cfg=template_cfg,
        generated_cfg_dir=generated_cfg_dir,
        output_dir=output_dir,
        reuse_if_exists=True,
    )

    sram = SRAMBuffer(
        name=f"scalesim_{buffer_tag}_{layer_name}",
        cacti_runner=runner,
        cacti_cfg=CactiInput(
            size_bytes=size_bytes,
            block_size_bytes=cfg["block_bytes"],
            output_bus_width_bits=cfg["bus_bits"],
            cache_type="ram",
            technology_u=0.028,
            associativity=1,
            read_write_port=1,
            exclusive_read_port=0,
            exclusive_write_port=0,
            single_ended_read_ports=0,
            uca_bank_count=cfg["banks"],
        ),
        freq_hz=freq_hz,
        num_banks=cfg["banks"],
    )

    cur = 0
    energy_nj = 0.0
    cycles = 0
    if read_bytes > 0:
        stat = sram.read(int(read_bytes), cur)
        energy_nj += float(stat.dynamic_energy_nj)
        cur += int(stat.service_cycles)
        cycles += int(stat.service_cycles)
    if write_bytes > 0:
        stat = sram.write(int(write_bytes), cur)
        energy_nj += float(stat.dynamic_energy_nj)
        cycles += int(stat.service_cycles)
    return {
        "energy_nj": float(energy_nj),
        "cycles": max(1, int(cycles)) if (read_bytes + write_bytes) > 0 else 0,
        "clock_period_ns": float(clock_period_ns),
        "bytes_read": int(read_bytes),
        "bytes_write": int(write_bytes),
        "cfg": cfg,
    }


def _scalesim_sram_runtime_stats(
    hw_config: Optional["HardwareConfig"],
    sram_reads: int,
    sram_writes: int,
    bytes_per_access: int,
    layer_name: str,
) -> Optional[Dict[str, Any]]:
    if hw_config is None or (sram_reads + sram_writes) <= 0:
        return None
    repo_root = getattr(hw_config, "repo_root", None)
    try:
        from .memory_model_adapter import get_memory_model_roots
    except Exception:
        return None
    roots = get_memory_model_roots(repo_root)
    if roots is None:
        return None
    memory_model_repo, cycle_sim_abs = roots
    try:
        from memory_model.cacti_runner import CactiRunner
        from memory_model.sram_buffer import SRAMBuffer
        from memory_model.types import CactiInput
    except Exception:
        return None

    cfg = _get_scalesim_sram_cfg(hw_config)
    freq_hz = _get_scalesim_sram_freq_hz(hw_config)
    clock_period_ns = 1e9 / float(freq_hz)

    cacti_bin = os.path.join(cycle_sim_abs, "cacti", "cacti")
    template_cfg = os.path.join(cycle_sim_abs, "cacti", "cache.cfg")
    generated_cfg_dir = os.path.join(memory_model_repo, "configs", "cacti", "generated")
    output_dir = os.path.join(memory_model_repo, "outputs", "cacti")
    if not os.path.isfile(cacti_bin) or not os.path.isfile(template_cfg):
        return None

    _log_scalesim_memory_config_once(
        "cacti",
        {
            "cacti_bin": cacti_bin,
            "template_cfg": template_cfg,
            "generated_cfg_dir": generated_cfg_dir,
            "output_dir": output_dir,
            "systolic_sram_size_bytes": size_bytes,
            "systolic_sram_block_bytes": cfg["block_bytes"],
            "systolic_sram_banks": cfg["banks"],
            "systolic_sram_bus_bits": cfg["bus_bits"],
            "systolic_sram_freq_hz": freq_hz,
            "systolic_sram_clock_period_ns": clock_period_ns,
        },
    )

    runner = CactiRunner(
        cacti_bin=cacti_bin,
        template_cfg=template_cfg,
        generated_cfg_dir=generated_cfg_dir,
        output_dir=output_dir,
        reuse_if_exists=True,
    )

    sram = SRAMBuffer(
        name=f"scalesim_sram_{layer_name}",
        cacti_runner=runner,
        cacti_cfg=CactiInput(
            size_bytes=size_bytes,
            block_size_bytes=cfg["block_bytes"],
            output_bus_width_bits=cfg["bus_bits"],
            cache_type="ram",
            technology_u=0.028,
            associativity=1,
            read_write_port=1,
            exclusive_read_port=0,
            exclusive_write_port=0,
            single_ended_read_ports=0,
            uca_bank_count=cfg["banks"],
        ),
        freq_hz=freq_hz,
        num_banks=cfg["banks"],
    )

    read_bytes = int(bytes_per_access) * max(0, int(sram_reads))
    write_bytes = int(bytes_per_access) * max(0, int(sram_writes))
    return _scalesim_named_sram_runtime_stats(
        hw_config=hw_config,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        layer_name=layer_name,
        buffer_tag="local_sram",
    )


def _iter_scalesim_trace_csv(trace_path: str) -> Iterable[Tuple[int, List[int]]]:
    """Yield (cycle, addresses) from a SCALE-Sim *_TRACE.csv file."""
    if not trace_path or not os.path.isfile(trace_path):
        return
    with open(trace_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            try:
                cycle = int(float(str(row[0]).strip()))
            except Exception:
                continue
            addrs: List[int] = []
            for token in row[1:]:
                s = str(token).strip()
                if not s:
                    continue
                try:
                    val = int(float(s))
                except Exception:
                    continue
                if val < 0:
                    continue
                addrs.append(val)
            if addrs:
                yield cycle, addrs


def _normalize_trace_cycle(cycle: int, min_cycle: int) -> int:
    return max(0, int(cycle - min_cycle))


def _scan_min_cycle(trace_csvs: List[Tuple[str, bool]]) -> Optional[int]:
    min_cycle: Optional[int] = None
    for trace_path, _ in trace_csvs:
        for cycle, _ in _iter_scalesim_trace_csv(trace_path):
            if min_cycle is None or cycle < min_cycle:
                min_cycle = cycle
    return min_cycle


def _run_dramsim3_for_trace_csvs(
    hw_config: Optional["HardwareConfig"],
    trace_csvs: List[Tuple[str, bool]],
    layer_name: str,
    access_tag: str,
) -> Optional[Dict[str, Any]]:
    """Replay native SCALE-Sim DRAM trace CSVs directly into DRAMSim3."""
    if hw_config is None or not trace_csvs:
        return None
    repo_root = getattr(hw_config, "repo_root", None)
    try:
        from .memory_model_adapter import get_memory_model_roots, resolve_dramsim3_config_path
    except Exception:
        return None
    roots = get_memory_model_roots(repo_root)
    if roots is None:
        return None
    memory_model_repo, cycle_sim_abs = roots
    dram = getattr(hw_config, "dram", None)
    if dram is None:
        return None
    dramsim3_bin = getattr(dram, "dramsim3_bin", None)
    dramsim3_ini = getattr(dram, "dramsim3_config_ini", None)
    if not dramsim3_bin or not dramsim3_ini:
        return None
    if not (os.path.isfile(dramsim3_bin) and os.path.isfile(dramsim3_ini)):
        return None
    try:
        from memory_model.dramsim3_runner import DRAMSim3Runner
        from memory_model.memory_system import MemorySystem
    except Exception:
        return None

    bytes_per_access = _get_scalesim_data_bytes(hw_config)
    burst_bytes = _parse_dramsim3_burst_bytes(dramsim3_ini)
    min_cycle = _scan_min_cycle(trace_csvs)
    if min_cycle is None:
        return None

    out_root = os.path.join(memory_model_repo, "outputs", "dramsim3", "scalesim_access")
    os.makedirs(out_root, exist_ok=True)
    safe_tag = str(access_tag or "scalesim_trace").replace("/", "_")
    trace_dir = tempfile.mkdtemp(prefix=f"{layer_name}_{safe_tag}_", dir=out_root)
    trace_path = os.path.join(trace_dir, f"{safe_tag}.trace")
    max_cycle = 0
    event_count = 0
    with open(trace_path, "w", encoding="utf-8") as f:
        for trace_path_csv, is_write in trace_csvs:
            for cycle, addrs in _iter_scalesim_trace_csv(trace_path_csv):
                norm_cycle = _normalize_trace_cycle(cycle, min_cycle)
                if norm_cycle > max_cycle:
                    max_cycle = norm_cycle
                burst_addrs: set = set()
                for addr in addrs:
                    byte_addr = int(addr) * max(1, bytes_per_access)
                    burst_addr = (byte_addr // burst_bytes) * burst_bytes
                    burst_addrs.add(burst_addr)
                if not burst_addrs:
                    continue
                op = "WRITE" if is_write else "READ"
                for burst_addr in sorted(burst_addrs):
                    f.write(f"0x{burst_addr:x} {op} {norm_cycle}\n")
                    event_count += 1

    config_path = resolve_dramsim3_config_path(hw_config, trace_dir) or dramsim3_ini
    dram_clock_period_ns = _parse_dramsim3_tck_ns(config_path)
    burst_bytes = _parse_dramsim3_burst_bytes(config_path)
    _log_scalesim_memory_config_once(
        f"dramsim3_{safe_tag}",
        {
            "dramsim3_bin": dramsim3_bin,
            "dramsim3_config": config_path,
            "dramsim3_tck_ns": dram_clock_period_ns,
            "scalesim_trace_dir": trace_dir,
            "access_tag": safe_tag,
            "native_trace_files": ";".join(path for path, _ in trace_csvs),
            "burst_bytes": burst_bytes,
            "bytes_per_access": bytes_per_access,
        },
    )

    runner = DRAMSim3Runner(dramsim3_bin=dramsim3_bin, work_root=os.path.join(cycle_sim_abs, "DRAMsim3"))
    memsys = MemorySystem({}, runner)
    num_cycles = max(100000, max_cycle + 10000)
    result = memsys.run_dram_trace(config_path=config_path, trace_path=trace_path, num_cycles=num_cycles, output_dir=trace_dir)
    metrics = memsys._extract_dram_metrics(result)

    js = result.json_stats if result.json_stats else {}
    if isinstance(js, dict) and "0" in js and isinstance(js["0"], dict):
        js = js["0"]
    read_pj = float(js.get("read_energy") or 0.0)
    write_pj = float(js.get("write_energy") or 0.0)
    energy_nj = (read_pj + write_pj) / 1000.0
    done_cycle = memsys._estimate_refill_done_cycle(max_cycle, result)
    bytes_total = event_count * burst_bytes

    return {
        "energy_nj": float(energy_nj),
        "cycles": max(1, int(done_cycle)),
        "clock_period_ns": float(dram_clock_period_ns),
        "trace_dir": trace_dir,
        "metrics": metrics,
        "config_path": config_path,
        "bytes_total": int(bytes_total),
    }


def _run_sram_runtime_for_trace_csvs(
    hw_config: Optional["HardwareConfig"],
    trace_csvs: List[Tuple[str, str]],
    layer_name: str,
    buffer_tag: str,
) -> Optional[Dict[str, Any]]:
    """Replay native SCALE-Sim SRAM trace CSVs into a CACTI-backed SRAMBuffer."""
    if hw_config is None or not trace_csvs:
        return None
    repo_root = getattr(hw_config, "repo_root", None)
    try:
        from .memory_model_adapter import get_memory_model_roots
    except Exception:
        return None
    roots = get_memory_model_roots(repo_root)
    if roots is None:
        return None
    memory_model_repo, cycle_sim_abs = roots
    try:
        from memory_model.cacti_runner import CactiRunner
        from memory_model.sram_buffer import SRAMBuffer
        from memory_model.types import CactiInput
    except Exception:
        return None

    cfg = _get_scalesim_sram_cfg(hw_config)
    freq_hz = _get_scalesim_sram_freq_hz(hw_config)
    clock_period_ns = 1e9 / float(freq_hz)
    bytes_per_access = _get_scalesim_data_bytes(hw_config)

    cacti_bin = os.path.join(cycle_sim_abs, "cacti", "cacti")
    template_cfg = os.path.join(cycle_sim_abs, "cacti", "cache.cfg")
    generated_cfg_dir = os.path.join(memory_model_repo, "configs", "cacti", "generated")
    output_dir = os.path.join(memory_model_repo, "outputs", "cacti")
    if not os.path.isfile(cacti_bin) or not os.path.isfile(template_cfg):
        return None

    _log_scalesim_memory_config_once(
        f"cacti_{buffer_tag}_trace",
        {
            "cacti_bin": cacti_bin,
            "template_cfg": template_cfg,
            "generated_cfg_dir": generated_cfg_dir,
            "output_dir": output_dir,
            "systolic_sram_size_bytes": cfg["size_bytes"],
            "systolic_sram_block_bytes": cfg["block_bytes"],
            "systolic_sram_banks": cfg["banks"],
            "systolic_sram_bus_bits": cfg["bus_bits"],
            "systolic_sram_freq_hz": freq_hz,
            "systolic_sram_clock_period_ns": clock_period_ns,
            "buffer_tag": buffer_tag,
            "native_trace_files": ";".join(path for path, _ in trace_csvs),
        },
    )

    runner = CactiRunner(
        cacti_bin=cacti_bin,
        template_cfg=template_cfg,
        generated_cfg_dir=generated_cfg_dir,
        output_dir=output_dir,
        reuse_if_exists=True,
    )
    sram = SRAMBuffer(
        name=f"scalesim_{buffer_tag}_{layer_name}",
        cacti_runner=runner,
        cacti_cfg=CactiInput(
            size_bytes=cfg["size_bytes"],
            block_size_bytes=cfg["block_bytes"],
            output_bus_width_bits=cfg["bus_bits"],
            cache_type="ram",
            technology_u=0.028,
            associativity=1,
            read_write_port=1,
            exclusive_read_port=0,
            exclusive_write_port=0,
            single_ended_read_ports=0,
            uca_bank_count=cfg["banks"],
        ),
        freq_hz=freq_hz,
        num_banks=cfg["banks"],
    )

    events: List[Tuple[int, str, int]] = []
    for trace_path, access_kind in trace_csvs:
        for cycle, addrs in _iter_scalesim_trace_csv(trace_path):
            events.append((cycle, access_kind, len(addrs) * bytes_per_access))
    if not events:
        return None
    min_cycle = min(ev[0] for ev in events)
    energy_nj = 0.0
    total_cycles = 0
    bytes_read = 0
    bytes_write = 0
    for cycle, access_kind, num_bytes in sorted(events, key=lambda x: x[0]):
        cur_cycle = _normalize_trace_cycle(cycle, min_cycle)
        if access_kind == "write":
            stat = sram.write(int(num_bytes), cur_cycle)
            bytes_write += int(num_bytes)
        else:
            stat = sram.read(int(num_bytes), cur_cycle)
            bytes_read += int(num_bytes)
        energy_nj += float(stat.dynamic_energy_nj)
        total_cycles = max(total_cycles, int(stat.end_cycle))
    return {
        "energy_nj": float(energy_nj),
        "cycles": max(1, int(total_cycles)),
        "clock_period_ns": float(clock_period_ns),
        "bytes_read": int(bytes_read),
        "bytes_write": int(bytes_write),
        "cfg": cfg,
    }


def _run_dramsim3_for_access_bytes(
    hw_config: Optional["HardwareConfig"],
    read_bytes: int,
    write_bytes: int,
    layer_name: str,
    access_tag: str = "scalesim_access",
) -> Optional[Dict[str, Any]]:
    if hw_config is None or (read_bytes + write_bytes) <= 0:
        return None
    repo_root = getattr(hw_config, "repo_root", None)
    try:
        from .memory_model_adapter import get_memory_model_roots, resolve_dramsim3_config_path
    except Exception:
        return None
    roots = get_memory_model_roots(repo_root)
    if roots is None:
        return None
    memory_model_repo, cycle_sim_abs = roots
    dram = getattr(hw_config, "dram", None)
    if dram is None:
        return None
    dramsim3_bin = getattr(dram, "dramsim3_bin", None)
    dramsim3_ini = getattr(dram, "dramsim3_config_ini", None)
    if not dramsim3_bin or not dramsim3_ini:
        return None
    if not (os.path.isfile(dramsim3_bin) and os.path.isfile(dramsim3_ini)):
        return None

    try:
        from memory_model.dramsim3_runner import DRAMSim3Runner
        from memory_model.memory_system import MemorySystem
        from memory_model.dram_trace import DRAMTrace
    except Exception:
        return None

    out_root = os.path.join(memory_model_repo, "outputs", "dramsim3", "scalesim_access")
    os.makedirs(out_root, exist_ok=True)
    safe_tag = str(access_tag or "scalesim_access").replace("/", "_")
    trace_dir = tempfile.mkdtemp(prefix=f"{layer_name}_{safe_tag}_", dir=out_root)
    trace_path = os.path.join(trace_dir, f"{safe_tag}.trace")

    trace = DRAMTrace()
    burst_bytes = 64
    issue_gap = 1
    base_addr = 0x20000000
    start_cycle = 0
    if read_bytes > 0:
        trace.extend_burst(
            base_addr=base_addr,
            num_bytes=int(read_bytes),
            burst_bytes=burst_bytes,
            is_write=False,
            start_cycle=start_cycle,
            issue_gap=issue_gap,
        )
        num_txn = (int(read_bytes) + burst_bytes - 1) // burst_bytes
        start_cycle += max(0, num_txn) * issue_gap
        base_addr += max(int(read_bytes), burst_bytes)
    if write_bytes > 0:
        trace.extend_burst(
            base_addr=base_addr,
            num_bytes=int(write_bytes),
            burst_bytes=burst_bytes,
            is_write=True,
            start_cycle=start_cycle,
            issue_gap=issue_gap,
        )
        num_txn = (int(write_bytes) + burst_bytes - 1) // burst_bytes
        start_cycle += max(0, num_txn) * issue_gap

    trace.dump(trace_path)

    config_path = resolve_dramsim3_config_path(hw_config, trace_dir) or dramsim3_ini
    dram_clock_period_ns = _parse_dramsim3_tck_ns(config_path)
    _log_scalesim_memory_config_once(
        "dramsim3",
        {
            "dramsim3_bin": dramsim3_bin,
            "dramsim3_config": config_path,
            "dramsim3_tck_ns": dram_clock_period_ns,
            "scalesim_access_trace_dir": trace_dir,
            "access_tag": safe_tag,
        },
    )

    runner = DRAMSim3Runner(dramsim3_bin=dramsim3_bin, work_root=os.path.join(cycle_sim_abs, "DRAMsim3"))
    memsys = MemorySystem({}, runner)
    num_cycles = max(100000, start_cycle + 10000)
    result = memsys.run_dram_trace(config_path=config_path, trace_path=trace_path, num_cycles=num_cycles, output_dir=trace_dir)
    metrics = memsys._extract_dram_metrics(result)

    js = result.json_stats if result.json_stats else {}
    if isinstance(js, dict) and "0" in js and isinstance(js["0"], dict):
        js = js["0"]
    read_pj = float(js.get("read_energy") or 0.0)
    write_pj = float(js.get("write_energy") or 0.0)
    energy_nj = (read_pj + write_pj) / 1000.0

    done_cycle = memsys._estimate_refill_done_cycle(max(0, start_cycle - 1), result)
    cycles = max(1, int(done_cycle))

    return {
        "energy_nj": float(energy_nj),
        "cycles": cycles,
        "clock_period_ns": float(dram_clock_period_ns),
        "trace_dir": trace_dir,
        "metrics": metrics,
        "config_path": config_path,
    }

def _run_dramsim3_for_layer_trace(
    layer_trace_dir: str,
    scalesim_root: str,
    run_name: str,
    dramsim3_bin: str,
    dramsim3_config_ini: str,
) -> Optional[float]:
    """
    用 SCALE-Sim 单层 trace（IFMAP/FILTER/OFMAP_DRAM_TRACE.csv）跑 DRAMSim3，返回该层 DRAM 能耗 (nJ)。
    参考 SCALE-Sim/demo_with_accelergy.py：trace 写入 results/<topology>/<run_name>/layer0/ 后调用 run_dramsim3.py。
    """
    import json
    import shutil
    import subprocess
    topo_name = "pointkan_scalesim_topo"
    results_dir = os.path.join(scalesim_root, "results")
    topo_run_dir = os.path.join(results_dir, topo_name, run_name)
    layer0_dir = os.path.join(topo_run_dir, "layer0")
    os.makedirs(layer0_dir, exist_ok=True)
    for fname in ("IFMAP_DRAM_TRACE.csv", "FILTER_DRAM_TRACE.csv", "OFMAP_DRAM_TRACE.csv"):
        src = os.path.join(layer_trace_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(layer0_dir, fname))
    out_dramsim3 = os.path.join(results_dir, topo_name + "_dramsim3")
    cmd = [
        sys.executable,
        os.path.join(scalesim_root, "scripts", "run_dramsim3.py"),
        "-topology", topo_name,
        "-run_name", run_name,
        "-dramsim3_bin", dramsim3_bin,
        "-config_ini", dramsim3_config_ini,
        "-output_dir", out_dramsim3,
    ]
    try:
        subprocess.run(cmd, check=True, cwd=scalesim_root, capture_output=True, timeout=120)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    json_path = os.path.join(out_dramsim3, "layer0", "dramsim3.json")
    if not os.path.isfile(json_path):
        return None
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception:
        return None
    # 结构可能为 {"0": {"read_energy": pj, "write_energy": pj}, ...} 或单层
    total_pj = 0.0
    for v in (data.values() if isinstance(data, dict) else [data]):
        if isinstance(v, dict):
            total_pj += float(v.get("read_energy") or 0) + float(v.get("write_energy") or 0)
    return total_pj / 1000.0  # pJ → nJ


def _scalesim_config_with_array_size(
    base_cfg_path: str,
    rows: int,
    cols: int,
    tmp_dir: str,
    hw_config: Optional["HardwareConfig"] = None,
) -> str:
    """
    根据 hw_config.systolic_array 生成临时 config，供 SCALE-Sim 控制阵列尺寸与局部 SRAM 容量。
    采用「读原文 + 正则替换关键字段」保证 SCALE-Sim 读到的与 demo 格式一致，避免 ConfigParser 写回格式/大小写问题。
    """
    import re
    out_path = os.path.join(tmp_dir, "scalesim_array_{}x{}.cfg".format(rows, cols))
    with open(base_cfg_path, "r") as f:
        content = f.read()
    sram_cfg = _get_scalesim_sram_cfg(hw_config)
    ifmap_sram_kb = max(1, int(math.ceil(float(sram_cfg["ifmap_size_bytes"]) / 1024.0)))
    filter_sram_kb = max(1, int(math.ceil(float(sram_cfg["filter_size_bytes"]) / 1024.0)))
    ofmap_sram_kb = max(1, int(math.ceil(float(sram_cfg["ofmap_size_bytes"]) / 1024.0)))
    # 替换 [architecture_presets] 中的 ArrayHeight / ArrayWidth（兼容 "ArrayHeight:    64" 或 "ArrayHeight = 64" 等）
    content = re.sub(r"^(ArrayHeight)\s*[=:]\s*\d+\s*$", r"\1:    {}".format(rows), content, flags=re.MULTILINE | re.IGNORECASE)
    content = re.sub(r"^(ArrayWidth)\s*[=:]\s*\d+\s*$", r"\1:    {}".format(cols), content, flags=re.MULTILINE | re.IGNORECASE)
    # 当前先用同一容量统一驱动 IFMAP/FILTER/OFMAP SRAM，这样增大局部 SRAM 会真实影响 SCALE-Sim 原生 trace。
    content = re.sub(r"^(IfmapSramSzkB)\s*[=:]\s*\d+\s*$", r"\1:    {}".format(ifmap_sram_kb), content, flags=re.MULTILINE | re.IGNORECASE)
    content = re.sub(r"^(FilterSramSzkB)\s*[=:]\s*\d+\s*$", r"\1:    {}".format(filter_sram_kb), content, flags=re.MULTILINE | re.IGNORECASE)
    content = re.sub(r"^(OfmapSramSzkB)\s*[=:]\s*\d+\s*$", r"\1:    {}".format(ofmap_sram_kb), content, flags=re.MULTILINE | re.IGNORECASE)
    with open(out_path, "w") as f:
        f.write(content)
    # 读回校验，确保写入的阵列尺寸和局部 SRAM 正确（用正则避免 ConfigParser 对冒号格式的兼容问题）
    try:
        with open(out_path) as f:
            text = f.read()
        m_h = re.search(r"^ArrayHeight\s*[=:]\s*(\d+)", text, re.MULTILINE | re.IGNORECASE)
        m_w = re.search(r"^ArrayWidth\s*[=:]\s*(\d+)", text, re.MULTILINE | re.IGNORECASE)
        m_i = re.search(r"^IfmapSramSzkB\s*[=:]\s*(\d+)", text, re.MULTILINE | re.IGNORECASE)
        m_f = re.search(r"^FilterSramSzkB\s*[=:]\s*(\d+)", text, re.MULTILINE | re.IGNORECASE)
        m_o = re.search(r"^OfmapSramSzkB\s*[=:]\s*(\d+)", text, re.MULTILINE | re.IGNORECASE)
        if m_h and m_w and m_i and m_f and m_o:
            r, c = int(m_h.group(1)), int(m_w.group(1))
            i_kb, f_kb, o_kb = int(m_i.group(1)), int(m_f.group(1)), int(m_o.group(1))
            if r != rows or c != cols or i_kb != ifmap_sram_kb or f_kb != filter_sram_kb or o_kb != ofmap_sram_kb:
                try:
                    from ..utils.logger import log_info
                    log_info(
                        "WARNING: 临时 cfg 校验不一致:"
                        f" ArrayHeight={r} ArrayWidth={c}"
                        f" IfmapSramSzkB={i_kb} FilterSramSzkB={f_kb} OfmapSramSzkB={o_kb}"
                        f"，与目标 {rows}x{cols}, IFMAP={ifmap_sram_kb}KB FILTER={filter_sram_kb}KB OFMAP={ofmap_sram_kb}KB 不一致"
                    )
                except Exception:
                    pass
    except Exception:
        pass
    return out_path


def run_scalesim_conv1d_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    num_points: int,
    layer_name: str = "conv1d",
    progress_index: Optional[int] = None,
    progress_total: Optional[int] = None,
    scalesim_root: Optional[str] = None,
    config_path: Optional[str] = None,
    run_name: Optional[str] = None,
    hw_config: Optional["HardwareConfig"] = None,
) -> ScalesimLayerResult:
    """
    对单层 Conv1d 跑 SCALE-Sim，返回周期与访问统计。
    Conv1d 映射：ifmap (1, L)，filter (1, k)，channels=in_ch, num_filter=out_ch。
    """
    scalesim_root = scalesim_root or DEFAULT_SCALE_SIM_ROOT
    run_name = run_name or DEFAULT_RUN_NAME
    if not os.path.isdir(scalesim_root):
        return ScalesimLayerResult(success=False, message=f"SCALE-Sim root not found: {scalesim_root}")
    base_cfg = config_path or os.path.join(scalesim_root, "configs", "demo_64x64_32kb_ws.cfg")
    if not os.path.isfile(base_cfg):
        return ScalesimLayerResult(success=False, message=f"Config not found: {base_cfg}")

    ifmap_h, ifmap_w = 1, max(1, num_points)
    filter_h, filter_w = 1, max(1, kernel_size)
    channels, num_filter = max(1, in_channels), max(1, out_channels)
    macs = (ifmap_w - filter_w + 1) * num_filter * (filter_h * filter_w * channels)

    try:
        if scalesim_root not in sys.path:
            sys.path.insert(0, scalesim_root)
        from scalesim.scale_sim import scalesim
    except ImportError:
        return ScalesimLayerResult(success=False, message="scalesim not importable (SCALE-Sim not installed)")

    layer_t0 = time.perf_counter()
    _log_scalesim_progress(
        "conv1d",
        layer_name,
        f"start (in_ch={channels}, out_ch={num_filter}, kernel={filter_w}, points={ifmap_w})",
        progress_index=progress_index,
        progress_total=progress_total,
    )
    with tempfile.TemporaryDirectory(prefix="pointkan_scalesim_") as tmp:
        # 阵列尺寸由 hw_config.systolic_array.rows/cols 传入 SCALE-Sim（ArrayHeight/ArrayWidth）
        rows, cols = 64, 64
        if hw_config and getattr(hw_config, "systolic_array", None):
            sa = hw_config.systolic_array
            rows = max(1, int(getattr(sa, "rows", 64) or 64))
            cols = max(1, int(getattr(sa, "cols", 64) or 64))
        # 首次调用时打一条日志，便于确认阵列尺寸已传入（避免重复刷屏，仅用 layer_name 占位区分）
        _log_scalesim_array_size_once(rows, cols, "conv1d", layer_name)
        prep_t0 = time.perf_counter()
        cfg_path = _scalesim_config_with_array_size(base_cfg, rows, cols, tmp, hw_config)

        topo_path = os.path.join(tmp, "topology.csv")
        layout_path = os.path.join(tmp, "layout.csv")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir, exist_ok=True)
        _write_conv_topology_csv(
            topo_path, layer_name,
            ifmap_h, ifmap_w, filter_h, filter_w, channels, num_filter, stride=1,
        )
        _write_layout_csv(layout_path, [layer_name])
        _log_scalesim_progress(
            "conv1d",
            layer_name,
            f"prepare done in {_format_elapsed_seconds(time.perf_counter() - prep_t0)}",
            progress_index=progress_index,
            progress_total=progress_total,
        )
        use_dramsim3_trace = False
        if hw_config and getattr(hw_config, "dram", None):
            d = hw_config.dram
            if getattr(d, "dramsim3_bin", None) and getattr(d, "dramsim3_config_ini", None):
                if os.path.isfile(d.dramsim3_bin) and os.path.isfile(d.dramsim3_config_ini):
                    use_dramsim3_trace = True
        s = scalesim(
            save_disk_space=not use_dramsim3_trace,
            verbose=False,
            config=cfg_path,
            topology=topo_path,
            layout=layout_path,
            input_type_gemm=False,
        )
        core_t0 = time.perf_counter()
        _log_scalesim_progress(
            "conv1d",
            layer_name,
            "run_scale start",
            progress_index=progress_index,
            progress_total=progress_total,
        )
        s.run_scale(top_path=out_dir)
        _log_scalesim_progress(
            "conv1d",
            layer_name,
            f"run_scale done in {_format_elapsed_seconds(time.perf_counter() - core_t0)}",
            progress_index=progress_index,
            progress_total=progress_total,
        )
        subdirs = [d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))]
        report_dir = os.path.join(out_dir, subdirs[0]) if subdirs else out_dir
        layer_trace_dir = os.path.join(report_dir, "layer0")
        native_ifmap_dram_trace = os.path.join(layer_trace_dir, "IFMAP_DRAM_TRACE.csv")
        native_filter_dram_trace = os.path.join(layer_trace_dir, "FILTER_DRAM_TRACE.csv")
        native_ofmap_dram_trace = os.path.join(layer_trace_dir, "OFMAP_DRAM_TRACE.csv")
        native_ifmap_sram_trace = os.path.join(layer_trace_dir, "IFMAP_SRAM_TRACE.csv")
        native_filter_sram_trace = os.path.join(layer_trace_dir, "FILTER_SRAM_TRACE.csv")
        native_ofmap_sram_trace = os.path.join(layer_trace_dir, "OFMAP_SRAM_TRACE.csv")
        compute_path = os.path.join(report_dir, "COMPUTE_REPORT.csv")
        detail_path = os.path.join(report_dir, "DETAILED_ACCESS_REPORT.csv")
        if not os.path.isfile(compute_path) or not os.path.isfile(detail_path):
            return ScalesimLayerResult(success=False, message="COMPUTE or DETAILED report missing", macs=macs)
        with open(compute_path) as f:
            compute_rows = list(csv.DictReader(f, skipinitialspace=True))
        with open(detail_path) as f:
            detail_rows = list(csv.DictReader(f, skipinitialspace=True))
        if not compute_rows or not detail_rows:
            return ScalesimLayerResult(success=False, message="Empty reports", macs=macs)
        comp = _strip_keys(compute_rows[0])
        det = _strip_keys(detail_rows[0])
        total_cycles = _int("Total Cycles", comp, 0)
        stall_cycles = _int("Stall Cycles", comp, 0)
        sram_i = _int("SRAM IFMAP Reads", det)
        sram_f = _int("SRAM Filter Reads", det)
        sram_o = _int("SRAM OFMAP Writes", det)
        dram_i = _int("DRAM IFMAP Reads", det)
        dram_f = _int("DRAM Filter Reads", det)
        dram_o = _int("DRAM OFMAP Writes", det)
        bytes_per_access = _get_scalesim_data_bytes(hw_config)
        sram_reads = sram_i + sram_f
        sram_writes = sram_o
        residency = _get_feature_residency_entry(hw_config, layer_name)
        # Feature tensors are moved by the persistent feature SRAM path (layer_buffer_ops).
        # ScaleSim's internal Conv/Linear runtime only keeps local feature-buffer accesses on-chip.
        feature_ifmap_dram_bytes = 0
        feature_ofmap_dram_bytes = 0
        filter_dram_bytes = _estimate_scalesim_filter_bytes(
            "conv1d",
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bytes_per_access=bytes_per_access,
        )
        dram_reads = max(0, int(feature_ifmap_dram_bytes // max(1, bytes_per_access))) + max(0, int(filter_dram_bytes // max(1, bytes_per_access)))
        dram_writes = max(0, int(feature_ofmap_dram_bytes // max(1, bytes_per_access)))

        sram_nj_override = None
        sram_cycles = 0
        sram_clock_period_ns = float(getattr(hw_config, "clock_period", 1.0) or 1.0)
        dram_nj_override = None
        dram_cycles = 0
        dram_clock_period_ns = 1.0
        dram_trace_dir = None
        feature_input_dram_stats = None
        weight_dram_stats = None
        feature_output_dram_stats = None
        weight_sram_stats = None

        use_native_trace_replay = all(
            os.path.isfile(p)
            for p in (native_ifmap_sram_trace, native_filter_sram_trace, native_ofmap_sram_trace, native_filter_dram_trace)
        )

        feature_local_sram_stats = None
        if use_native_trace_replay:
            feature_local_sram_stats = _run_sram_runtime_for_trace_csvs(
                hw_config=hw_config,
                trace_csvs=[
                    (native_ifmap_sram_trace, "read"),
                    (native_ofmap_sram_trace, "write"),
                ],
                layer_name=layer_name,
                buffer_tag="feature_local_sram",
            )
        if feature_local_sram_stats is None:
            feature_local_sram_stats = _scalesim_named_sram_runtime_stats(
                hw_config=hw_config,
                read_bytes=int(bytes_per_access) * max(0, int(sram_i)),
                write_bytes=int(bytes_per_access) * max(0, int(sram_o)),
                layer_name=layer_name,
                buffer_tag="feature_local_sram",
            )

        weight_refill_tiles = 1
        weight_refill_tile_bytes = int(filter_dram_bytes)
        weight_sram_stats = None
        if use_native_trace_replay:
            weight_dram_stats = _run_dramsim3_for_trace_csvs(
                hw_config,
                trace_csvs=[(native_filter_dram_trace, False)],
                layer_name=layer_name,
                access_tag="weight_read",
            )
            weight_sram_stats = _run_sram_runtime_for_trace_csvs(
                hw_config=hw_config,
                trace_csvs=[
                    (native_filter_dram_trace, "write"),
                    (native_filter_sram_trace, "read"),
                ],
                layer_name=layer_name,
                buffer_tag="weight_sram",
            )
            if weight_dram_stats is not None:
                weight_refill_tile_bytes = min(int(filter_dram_bytes), _get_scalesim_sram_cfg(hw_config).get("size_bytes", 32768))
                weight_refill_tiles = max(1, int(math.ceil(int(filter_dram_bytes) / float(max(1, weight_refill_tile_bytes)))))
        elif filter_dram_bytes > 0:
            weight_sram_capacity = _get_scalesim_sram_cfg(hw_config).get("size_bytes", 32768)
            weight_refill_tile_bytes = max(1, min(int(filter_dram_bytes), int(weight_sram_capacity)))
            weight_refill_tiles = max(1, int(math.ceil(int(filter_dram_bytes) / float(weight_refill_tile_bytes))))
            total_weight_energy = 0.0
            total_weight_cycles = 0
            weight_trace_dirs = []
            remaining_weight_bytes = int(filter_dram_bytes)
            tile_idx = 0
            last_stat = None
            while remaining_weight_bytes > 0:
                tile_bytes = min(weight_refill_tile_bytes, remaining_weight_bytes)
                stat = _run_dramsim3_for_access_bytes(
                    hw_config,
                    read_bytes=tile_bytes,
                    write_bytes=0,
                    layer_name=layer_name,
                    access_tag=f"weight_read_tile{tile_idx:02d}",
                )
                if stat is not None:
                    last_stat = stat
                    total_weight_energy += float(stat.get("energy_nj") or 0.0)
                    total_weight_cycles += int(stat.get("cycles") or 0)
                    if stat.get("trace_dir"):
                        weight_trace_dirs.append(stat.get("trace_dir"))
                remaining_weight_bytes -= tile_bytes
                tile_idx += 1
            if total_weight_cycles > 0 or total_weight_energy > 0.0:
                weight_dram_stats = {
                    "energy_nj": total_weight_energy,
                    "cycles": total_weight_cycles,
                    "clock_period_ns": float((last_stat or {}).get("clock_period_ns") or dram_clock_period_ns),
                    "trace_dir": ";".join(weight_trace_dirs),
                }
            weight_sram_stats = _scalesim_named_sram_runtime_stats(
                hw_config=hw_config,
                read_bytes=int(bytes_per_access) * max(0, int(sram_f)),
                write_bytes=filter_dram_bytes,
                layer_name=layer_name,
                buffer_tag="weight_sram",
            )

        sram_nj_override = float((feature_local_sram_stats or {}).get("energy_nj", 0.0) or 0.0) + float((weight_sram_stats or {}).get("energy_nj", 0.0) or 0.0)
        sram_cycles = int((feature_local_sram_stats or {}).get("cycles", 0) or 0) + int((weight_sram_stats or {}).get("cycles", 0) or 0)
        for _stat in (feature_local_sram_stats, weight_sram_stats):
            if _stat is not None and _stat.get("clock_period_ns"):
                sram_clock_period_ns = float(_stat.get("clock_period_ns"))
                break

        if feature_ifmap_dram_bytes > 0:
            feature_input_dram_stats = _run_dramsim3_for_access_bytes(
                hw_config,
                read_bytes=feature_ifmap_dram_bytes,
                write_bytes=0,
                layer_name=layer_name,
                access_tag="feature_input",
            )
        if feature_ofmap_dram_bytes > 0:
            feature_output_dram_stats = _run_dramsim3_for_access_bytes(
                hw_config,
                read_bytes=0,
                write_bytes=feature_ofmap_dram_bytes,
                layer_name=layer_name,
                access_tag="feature_output",
            )

        if feature_input_dram_stats or weight_dram_stats or feature_output_dram_stats:
            dram_nj_override = float((feature_input_dram_stats or {}).get("energy_nj", 0.0) or 0.0) + float((weight_dram_stats or {}).get("energy_nj", 0.0) or 0.0) + float((feature_output_dram_stats or {}).get("energy_nj", 0.0) or 0.0)
            dram_cycles = int((feature_input_dram_stats or {}).get("cycles", 0) or 0) + int((weight_dram_stats or {}).get("cycles", 0) or 0) + int((feature_output_dram_stats or {}).get("cycles", 0) or 0)
            for _stat in (feature_input_dram_stats, weight_dram_stats, feature_output_dram_stats):
                if _stat is not None and _stat.get("clock_period_ns"):
                    dram_clock_period_ns = float(_stat.get("clock_period_ns"))
                    break
            dram_trace_dir = ";".join([p for p in [
                (feature_input_dram_stats or {}).get("trace_dir"),
                (weight_dram_stats or {}).get("trace_dir"),
                (feature_output_dram_stats or {}).get("trace_dir"),
            ] if p]) or None
        elif use_dramsim3_trace:
            dram_nj_override = _run_dramsim3_for_layer_trace(
                layer_trace_dir, scalesim_root, run_name,
                hw_config.dram.dramsim3_bin, hw_config.dram.dramsim3_config_ini,
            )

        memory_t0 = time.perf_counter()
        energy_parts = _energy_breakdown_from_hw_config(
            hw_config, macs, sram_i, sram_f, sram_o, dram_i, dram_f, dram_o,
            dram_nj_override=dram_nj_override,
            sram_nj_override=sram_nj_override,
        )
        energy_nj = energy_parts["total_energy_nj"]
        _log_scalesim_progress(
            "conv1d",
            layer_name,
            (
                f"memory-model done in {_format_elapsed_seconds(time.perf_counter() - memory_t0)} "
                f"(cycles={total_cycles}, energy={energy_nj:.2f} nJ)"
            ),
            progress_index=progress_index,
            progress_total=progress_total,
        )
        _log_scalesim_progress(
            "conv1d",
            layer_name,
            (
                f"done in {_format_elapsed_seconds(time.perf_counter() - layer_t0)} "
                f"(cycles={total_cycles}, sram={energy_parts['sram_energy_nj']:.2f} nJ, "
                f"dram={energy_parts['dram_energy_nj']:.2f} nJ)"
            ),
            progress_index=progress_index,
            progress_total=progress_total,
        )
        return ScalesimLayerResult(
            total_cycles=total_cycles,
            stall_cycles=stall_cycles,
            sram_ifmap_reads=sram_i,
            sram_filter_reads=sram_f,
            sram_ofmap_writes=sram_o,
            dram_ifmap_reads=dram_i,
            dram_filter_reads=dram_f,
            dram_ofmap_writes=dram_o,
            macs=macs,
            overall_util_pct=_float_util("Overall Util %", comp),
            mapping_eff_pct=_float_util("Mapping Efficiency %", comp),
            compute_util_pct=_float_util("Compute Util %", comp),
            energy_nj=energy_nj,
            mac_energy_nj=energy_parts["mac_energy_nj"],
            sram_energy_nj=energy_parts["sram_energy_nj"],
            dram_energy_nj=energy_parts["dram_energy_nj"],
            sram_cycles=sram_cycles,
            dram_cycles=dram_cycles,
            sram_clock_period_ns=sram_clock_period_ns,
            dram_clock_period_ns=dram_clock_period_ns,
            dram_trace_dir=dram_trace_dir,
            feature_input_dram_trace_dir=(feature_input_dram_stats or {}).get("trace_dir"),
            weight_dram_trace_dir=(weight_dram_stats or {}).get("trace_dir"),
            feature_output_dram_trace_dir=(feature_output_dram_stats or {}).get("trace_dir"),
            success=True,
            feature_input_dram_bytes=feature_ifmap_dram_bytes,
            feature_output_dram_bytes=feature_ofmap_dram_bytes,
            weight_dram_bytes=filter_dram_bytes,
            weight_tensor_bytes=filter_dram_bytes,
            weight_dram_trace_bytes=int((weight_dram_stats or {}).get("bytes_total", 0) or 0),
            feature_local_sram_trace_bytes=int((feature_local_sram_stats or {}).get("bytes_read", 0) or 0) + int((feature_local_sram_stats or {}).get("bytes_write", 0) or 0),
            weight_sram_trace_bytes=int((weight_sram_stats or {}).get("bytes_read", 0) or 0) + int((weight_sram_stats or {}).get("bytes_write", 0) or 0),
            feature_input_dram_energy_nj=float((feature_input_dram_stats or {}).get("energy_nj", 0.0) or 0.0),
            weight_dram_energy_nj=float((weight_dram_stats or {}).get("energy_nj", 0.0) or 0.0),
            feature_output_dram_energy_nj=float((feature_output_dram_stats or {}).get("energy_nj", 0.0) or 0.0),
            feature_input_dram_cycles=int((feature_input_dram_stats or {}).get("cycles", 0) or 0),
            weight_dram_cycles=int((weight_dram_stats or {}).get("cycles", 0) or 0),
            feature_output_dram_cycles=int((feature_output_dram_stats or {}).get("cycles", 0) or 0),
            weight_sram_energy_nj=float((weight_sram_stats or {}).get("energy_nj", 0.0) or 0.0),
            weight_sram_cycles=int((weight_sram_stats or {}).get("cycles", 0) or 0),
            weight_sram_bytes_read=int((weight_sram_stats or {}).get("bytes_read", 0) or 0),
            weight_sram_bytes_write=int((weight_sram_stats or {}).get("bytes_write", 0) or 0),
            weight_refill_tiles=int(weight_refill_tiles),
            weight_refill_tile_bytes=int(weight_refill_tile_bytes),
            feature_local_sram_energy_nj=float((feature_local_sram_stats or {}).get("energy_nj", 0.0) or 0.0),
            feature_local_sram_cycles=int((feature_local_sram_stats or {}).get("cycles", 0) or 0),
            feature_local_sram_bytes_read=int((feature_local_sram_stats or {}).get("bytes_read", 0) or 0),
            feature_local_sram_bytes_write=int((feature_local_sram_stats or {}).get("bytes_write", 0) or 0),
            systolic_array_rows=rows,
            systolic_array_cols=cols,
        )


def run_scalesim_linear_layer(
    in_features: int,
    out_features: int,
    tokens: int,
    layer_name: str = "linear",
    progress_index: Optional[int] = None,
    progress_total: Optional[int] = None,
    scalesim_root: Optional[str] = None,
    config_path: Optional[str] = None,
    run_name: Optional[str] = None,
    hw_config: Optional["HardwareConfig"] = None,
) -> ScalesimLayerResult:
    """
    对单层 Linear 跑 SCALE-Sim，返回周期与访问统计。
    Linear 映射为 1×1 conv：ifmap (tokens, 1) channels=in_f，filter (1,1) num_filter=out_f，
    满足 SCALE-Sim 约束 filter_h<=ifmap_h 且 filter_w<=ifmap_w。MACs = tokens * in_f * out_f。
    """
    scalesim_root = scalesim_root or DEFAULT_SCALE_SIM_ROOT
    run_name = run_name or DEFAULT_RUN_NAME
    if not os.path.isdir(scalesim_root):
        return ScalesimLayerResult(success=False, message=f"SCALE-Sim root not found: {scalesim_root}")
    base_cfg = config_path or os.path.join(scalesim_root, "configs", "demo_64x64_32kb_ws.cfg")
    if not os.path.isfile(base_cfg):
        return ScalesimLayerResult(success=False, message=f"Config not found: {base_cfg}")

    # 1x1 conv: ifmap (tokens, 1), channels=in_features; filter (1,1), num_filter=out_features
    ifmap_h, ifmap_w = max(1, tokens), 1
    filter_h, filter_w = 1, 1
    channels, num_filter = max(1, in_features), max(1, out_features)
    macs = ifmap_h * ifmap_w * num_filter * (filter_h * filter_w * channels)

    try:
        if scalesim_root not in sys.path:
            sys.path.insert(0, scalesim_root)
        from scalesim.scale_sim import scalesim
    except ImportError:
        return ScalesimLayerResult(success=False, message="scalesim not importable", macs=macs)

    layer_t0 = time.perf_counter()
    _log_scalesim_progress(
        "linear",
        layer_name,
        f"start (tokens={ifmap_h}, in_features={channels}, out_features={num_filter})",
        progress_index=progress_index,
        progress_total=progress_total,
    )
    with tempfile.TemporaryDirectory(prefix="pointkan_scalesim_") as tmp:
        rows, cols = 64, 64
        if hw_config and getattr(hw_config, "systolic_array", None):
            sa = hw_config.systolic_array
            rows = max(1, int(getattr(sa, "rows", 64) or 64))
            cols = max(1, int(getattr(sa, "cols", 64) or 64))
        _log_scalesim_array_size_once(rows, cols, "linear", layer_name)
        prep_t0 = time.perf_counter()
        cfg_path = _scalesim_config_with_array_size(base_cfg, rows, cols, tmp, hw_config)

        topo_path = os.path.join(tmp, "topology.csv")
        layout_path = os.path.join(tmp, "layout.csv")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir, exist_ok=True)
        _write_conv_topology_csv(
            topo_path, layer_name,
            ifmap_h, ifmap_w, filter_h, filter_w, channels, num_filter, stride=1,
        )
        _write_layout_csv(layout_path, [layer_name])
        _log_scalesim_progress(
            "linear",
            layer_name,
            f"prepare done in {_format_elapsed_seconds(time.perf_counter() - prep_t0)}",
            progress_index=progress_index,
            progress_total=progress_total,
        )
        use_dramsim3_trace = False
        if hw_config and getattr(hw_config, "dram", None):
            d = hw_config.dram
            if getattr(d, "dramsim3_bin", None) and getattr(d, "dramsim3_config_ini", None):
                if os.path.isfile(d.dramsim3_bin) and os.path.isfile(d.dramsim3_config_ini):
                    use_dramsim3_trace = True
        s = scalesim(
            save_disk_space=not use_dramsim3_trace,
            verbose=False,
            config=cfg_path,
            topology=topo_path,
            layout=layout_path,
            input_type_gemm=False,
        )
        core_t0 = time.perf_counter()
        _log_scalesim_progress(
            "linear",
            layer_name,
            "run_scale start",
            progress_index=progress_index,
            progress_total=progress_total,
        )
        s.run_scale(top_path=out_dir)
        _log_scalesim_progress(
            "linear",
            layer_name,
            f"run_scale done in {_format_elapsed_seconds(time.perf_counter() - core_t0)}",
            progress_index=progress_index,
            progress_total=progress_total,
        )
        subdirs = [d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))]
        report_dir = os.path.join(out_dir, subdirs[0]) if subdirs else out_dir
        layer_trace_dir = os.path.join(report_dir, "layer0")
        native_ifmap_dram_trace = os.path.join(layer_trace_dir, "IFMAP_DRAM_TRACE.csv")
        native_filter_dram_trace = os.path.join(layer_trace_dir, "FILTER_DRAM_TRACE.csv")
        native_ofmap_dram_trace = os.path.join(layer_trace_dir, "OFMAP_DRAM_TRACE.csv")
        native_ifmap_sram_trace = os.path.join(layer_trace_dir, "IFMAP_SRAM_TRACE.csv")
        native_filter_sram_trace = os.path.join(layer_trace_dir, "FILTER_SRAM_TRACE.csv")
        native_ofmap_sram_trace = os.path.join(layer_trace_dir, "OFMAP_SRAM_TRACE.csv")
        compute_path = os.path.join(report_dir, "COMPUTE_REPORT.csv")
        detail_path = os.path.join(report_dir, "DETAILED_ACCESS_REPORT.csv")
        if not os.path.isfile(compute_path) or not os.path.isfile(detail_path):
            return ScalesimLayerResult(success=False, message="COMPUTE or DETAILED report missing", macs=macs)
        with open(compute_path) as f:
            compute_rows = list(csv.DictReader(f, skipinitialspace=True))
        with open(detail_path) as f:
            detail_rows = list(csv.DictReader(f, skipinitialspace=True))
        if not compute_rows or not detail_rows:
            return ScalesimLayerResult(success=False, message="Empty reports", macs=macs)
        comp = _strip_keys(compute_rows[0])
        det = _strip_keys(detail_rows[0])
        total_cycles = _int("Total Cycles", comp, 0)
        stall_cycles = _int("Stall Cycles", comp, 0)
        sram_i = _int("SRAM IFMAP Reads", det)
        sram_f = _int("SRAM Filter Reads", det)
        sram_o = _int("SRAM OFMAP Writes", det)
        dram_i = _int("DRAM IFMAP Reads", det)
        dram_f = _int("DRAM Filter Reads", det)
        dram_o = _int("DRAM OFMAP Writes", det)
        bytes_per_access = _get_scalesim_data_bytes(hw_config)
        sram_reads = sram_i + sram_f
        sram_writes = sram_o
        residency = _get_feature_residency_entry(hw_config, layer_name)
        # Feature tensors are moved by the persistent feature SRAM path (layer_buffer_ops).
        # ScaleSim's internal Conv/Linear runtime only keeps local feature-buffer accesses on-chip.
        feature_ifmap_dram_bytes = 0
        feature_ofmap_dram_bytes = 0
        filter_dram_bytes = _estimate_scalesim_filter_bytes(
            "linear",
            in_features=in_features,
            out_features=out_features,
            bytes_per_access=bytes_per_access,
        )
        dram_reads = max(0, int(feature_ifmap_dram_bytes // max(1, bytes_per_access))) + max(0, int(filter_dram_bytes // max(1, bytes_per_access)))
        dram_writes = max(0, int(feature_ofmap_dram_bytes // max(1, bytes_per_access)))

        sram_nj_override = None
        sram_cycles = 0
        sram_clock_period_ns = float(getattr(hw_config, "clock_period", 1.0) or 1.0)
        dram_nj_override = None
        dram_cycles = 0
        dram_clock_period_ns = 1.0
        dram_trace_dir = None
        feature_input_dram_stats = None
        weight_dram_stats = None
        feature_output_dram_stats = None
        weight_sram_stats = None

        use_native_trace_replay = all(
            os.path.isfile(p)
            for p in (native_ifmap_sram_trace, native_filter_sram_trace, native_ofmap_sram_trace, native_filter_dram_trace)
        )

        feature_local_sram_stats = None
        if use_native_trace_replay:
            feature_local_sram_stats = _run_sram_runtime_for_trace_csvs(
                hw_config=hw_config,
                trace_csvs=[
                    (native_ifmap_sram_trace, "read"),
                    (native_ofmap_sram_trace, "write"),
                ],
                layer_name=layer_name,
                buffer_tag="feature_local_sram",
            )
        if feature_local_sram_stats is None:
            feature_local_sram_stats = _scalesim_named_sram_runtime_stats(
                hw_config=hw_config,
                read_bytes=int(bytes_per_access) * max(0, int(sram_i)),
                write_bytes=int(bytes_per_access) * max(0, int(sram_o)),
                layer_name=layer_name,
                buffer_tag="feature_local_sram",
            )

        weight_refill_tiles = 1
        weight_refill_tile_bytes = int(filter_dram_bytes)
        weight_sram_stats = None
        if use_native_trace_replay:
            weight_dram_stats = _run_dramsim3_for_trace_csvs(
                hw_config,
                trace_csvs=[(native_filter_dram_trace, False)],
                layer_name=layer_name,
                access_tag="weight_read",
            )
            weight_sram_stats = _run_sram_runtime_for_trace_csvs(
                hw_config=hw_config,
                trace_csvs=[
                    (native_filter_dram_trace, "write"),
                    (native_filter_sram_trace, "read"),
                ],
                layer_name=layer_name,
                buffer_tag="weight_sram",
            )
            if weight_dram_stats is not None:
                weight_refill_tile_bytes = min(int(filter_dram_bytes), _get_scalesim_sram_cfg(hw_config).get("size_bytes", 32768))
                weight_refill_tiles = max(1, int(math.ceil(int(filter_dram_bytes) / float(max(1, weight_refill_tile_bytes)))))
        elif filter_dram_bytes > 0:
            weight_sram_capacity = _get_scalesim_sram_cfg(hw_config).get("size_bytes", 32768)
            weight_refill_tile_bytes = max(1, min(int(filter_dram_bytes), int(weight_sram_capacity)))
            weight_refill_tiles = max(1, int(math.ceil(int(filter_dram_bytes) / float(weight_refill_tile_bytes))))
            total_weight_energy = 0.0
            total_weight_cycles = 0
            weight_trace_dirs = []
            remaining_weight_bytes = int(filter_dram_bytes)
            tile_idx = 0
            last_stat = None
            while remaining_weight_bytes > 0:
                tile_bytes = min(weight_refill_tile_bytes, remaining_weight_bytes)
                stat = _run_dramsim3_for_access_bytes(
                    hw_config,
                    read_bytes=tile_bytes,
                    write_bytes=0,
                    layer_name=layer_name,
                    access_tag=f"weight_read_tile{tile_idx:02d}",
                )
                if stat is not None:
                    last_stat = stat
                    total_weight_energy += float(stat.get("energy_nj") or 0.0)
                    total_weight_cycles += int(stat.get("cycles") or 0)
                    if stat.get("trace_dir"):
                        weight_trace_dirs.append(stat.get("trace_dir"))
                remaining_weight_bytes -= tile_bytes
                tile_idx += 1
            if total_weight_cycles > 0 or total_weight_energy > 0.0:
                weight_dram_stats = {
                    "energy_nj": total_weight_energy,
                    "cycles": total_weight_cycles,
                    "clock_period_ns": float((last_stat or {}).get("clock_period_ns") or dram_clock_period_ns),
                    "trace_dir": ";".join(weight_trace_dirs),
                }
            weight_sram_stats = _scalesim_named_sram_runtime_stats(
                hw_config=hw_config,
                read_bytes=int(bytes_per_access) * max(0, int(sram_f)),
                write_bytes=filter_dram_bytes,
                layer_name=layer_name,
                buffer_tag="weight_sram",
            )

        sram_nj_override = float((feature_local_sram_stats or {}).get("energy_nj", 0.0) or 0.0) + float((weight_sram_stats or {}).get("energy_nj", 0.0) or 0.0)
        sram_cycles = int((feature_local_sram_stats or {}).get("cycles", 0) or 0) + int((weight_sram_stats or {}).get("cycles", 0) or 0)
        for _stat in (feature_local_sram_stats, weight_sram_stats):
            if _stat is not None and _stat.get("clock_period_ns"):
                sram_clock_period_ns = float(_stat.get("clock_period_ns"))
                break

        if feature_ifmap_dram_bytes > 0:
            feature_input_dram_stats = _run_dramsim3_for_access_bytes(
                hw_config,
                read_bytes=feature_ifmap_dram_bytes,
                write_bytes=0,
                layer_name=layer_name,
                access_tag="feature_input",
            )
        if feature_ofmap_dram_bytes > 0:
            feature_output_dram_stats = _run_dramsim3_for_access_bytes(
                hw_config,
                read_bytes=0,
                write_bytes=feature_ofmap_dram_bytes,
                layer_name=layer_name,
                access_tag="feature_output",
            )

        if feature_input_dram_stats or weight_dram_stats or feature_output_dram_stats:
            dram_nj_override = float((feature_input_dram_stats or {}).get("energy_nj", 0.0) or 0.0) + float((weight_dram_stats or {}).get("energy_nj", 0.0) or 0.0) + float((feature_output_dram_stats or {}).get("energy_nj", 0.0) or 0.0)
            dram_cycles = int((feature_input_dram_stats or {}).get("cycles", 0) or 0) + int((weight_dram_stats or {}).get("cycles", 0) or 0) + int((feature_output_dram_stats or {}).get("cycles", 0) or 0)
            for _stat in (feature_input_dram_stats, weight_dram_stats, feature_output_dram_stats):
                if _stat is not None and _stat.get("clock_period_ns"):
                    dram_clock_period_ns = float(_stat.get("clock_period_ns"))
                    break
            dram_trace_dir = ";".join([p for p in [
                (feature_input_dram_stats or {}).get("trace_dir"),
                (weight_dram_stats or {}).get("trace_dir"),
                (feature_output_dram_stats or {}).get("trace_dir"),
            ] if p]) or None
        elif use_dramsim3_trace:
            dram_nj_override = _run_dramsim3_for_layer_trace(
                layer_trace_dir, scalesim_root, run_name,
                hw_config.dram.dramsim3_bin, hw_config.dram.dramsim3_config_ini,
            )

        memory_t0 = time.perf_counter()
        energy_parts = _energy_breakdown_from_hw_config(
            hw_config, macs, sram_i, sram_f, sram_o, dram_i, dram_f, dram_o,
            dram_nj_override=dram_nj_override,
            sram_nj_override=sram_nj_override,
        )
        energy_nj = energy_parts["total_energy_nj"]
        _log_scalesim_progress(
            "linear",
            layer_name,
            (
                f"memory-model done in {_format_elapsed_seconds(time.perf_counter() - memory_t0)} "
                f"(cycles={total_cycles}, energy={energy_nj:.2f} nJ)"
            ),
            progress_index=progress_index,
            progress_total=progress_total,
        )
        _log_scalesim_progress(
            "linear",
            layer_name,
            (
                f"done in {_format_elapsed_seconds(time.perf_counter() - layer_t0)} "
                f"(cycles={total_cycles}, sram={energy_parts['sram_energy_nj']:.2f} nJ, "
                f"dram={energy_parts['dram_energy_nj']:.2f} nJ)"
            ),
            progress_index=progress_index,
            progress_total=progress_total,
        )
        return ScalesimLayerResult(
            total_cycles=total_cycles,
            stall_cycles=stall_cycles,
            sram_ifmap_reads=sram_i,
            sram_filter_reads=sram_f,
            sram_ofmap_writes=sram_o,
            dram_ifmap_reads=dram_i,
            dram_filter_reads=dram_f,
            dram_ofmap_writes=dram_o,
            macs=macs,
            overall_util_pct=_float_util("Overall Util %", comp),
            mapping_eff_pct=_float_util("Mapping Efficiency %", comp),
            compute_util_pct=_float_util("Compute Util %", comp),
            energy_nj=energy_nj,
            mac_energy_nj=energy_parts["mac_energy_nj"],
            sram_energy_nj=energy_parts["sram_energy_nj"],
            dram_energy_nj=energy_parts["dram_energy_nj"],
            sram_cycles=sram_cycles,
            dram_cycles=dram_cycles,
            sram_clock_period_ns=sram_clock_period_ns,
            dram_clock_period_ns=dram_clock_period_ns,
            dram_trace_dir=dram_trace_dir,
            feature_input_dram_trace_dir=(feature_input_dram_stats or {}).get("trace_dir"),
            weight_dram_trace_dir=(weight_dram_stats or {}).get("trace_dir"),
            feature_output_dram_trace_dir=(feature_output_dram_stats or {}).get("trace_dir"),
            success=True,
            feature_input_dram_bytes=feature_ifmap_dram_bytes,
            feature_output_dram_bytes=feature_ofmap_dram_bytes,
            weight_dram_bytes=filter_dram_bytes,
            weight_tensor_bytes=filter_dram_bytes,
            weight_dram_trace_bytes=int((weight_dram_stats or {}).get("bytes_total", 0) or 0),
            feature_local_sram_trace_bytes=int((feature_local_sram_stats or {}).get("bytes_read", 0) or 0) + int((feature_local_sram_stats or {}).get("bytes_write", 0) or 0),
            weight_sram_trace_bytes=int((weight_sram_stats or {}).get("bytes_read", 0) or 0) + int((weight_sram_stats or {}).get("bytes_write", 0) or 0),
            feature_input_dram_energy_nj=float((feature_input_dram_stats or {}).get("energy_nj", 0.0) or 0.0),
            weight_dram_energy_nj=float((weight_dram_stats or {}).get("energy_nj", 0.0) or 0.0),
            feature_output_dram_energy_nj=float((feature_output_dram_stats or {}).get("energy_nj", 0.0) or 0.0),
            feature_input_dram_cycles=int((feature_input_dram_stats or {}).get("cycles", 0) or 0),
            weight_dram_cycles=int((weight_dram_stats or {}).get("cycles", 0) or 0),
            feature_output_dram_cycles=int((feature_output_dram_stats or {}).get("cycles", 0) or 0),
            weight_sram_energy_nj=float((weight_sram_stats or {}).get("energy_nj", 0.0) or 0.0),
            weight_sram_cycles=int((weight_sram_stats or {}).get("cycles", 0) or 0),
            weight_sram_bytes_read=int((weight_sram_stats or {}).get("bytes_read", 0) or 0),
            weight_sram_bytes_write=int((weight_sram_stats or {}).get("bytes_write", 0) or 0),
            weight_refill_tiles=int(weight_refill_tiles),
            weight_refill_tile_bytes=int(weight_refill_tile_bytes),
            feature_local_sram_energy_nj=float((feature_local_sram_stats or {}).get("energy_nj", 0.0) or 0.0),
            feature_local_sram_cycles=int((feature_local_sram_stats or {}).get("cycles", 0) or 0),
            feature_local_sram_bytes_read=int((feature_local_sram_stats or {}).get("bytes_read", 0) or 0),
            feature_local_sram_bytes_write=int((feature_local_sram_stats or {}).get("bytes_write", 0) or 0),
            systolic_array_rows=rows,
            systolic_array_cols=cols,
        )
