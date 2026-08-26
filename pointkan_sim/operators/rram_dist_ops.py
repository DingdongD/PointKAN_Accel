"""
RRAM VMM 距离计算事件生成（PointCIM 模式专用）
=================================================

PointCIM 模式下，FPS/KNN 的距离计算 ||qi - pj||² = qi² - 2qi·pj + pj²
按以下 Producer-Consumer 流水线分为两类事件：

  ── One-time events（per SA layer，由 FPS/KNN operator 初始化时生成）──
  Stage 0  rram_write     —— 将 N 个点的坐标编程到 RRAM（DESTINY 写入模型）
  Stage 1  rram_self_k2   —— self-mode CIM 计算 k²=pj²，结果写入 SRAM cache
                             Producer：rram_write；Consumer：rram_self_k2

  ── Per-query events（每次 FPS 迭代 / KNN 查询生成）──
  Stage 2  rram_vmm       —— cross-mode VMM 计算 -2qi·pj（MICSim cross-mode）
                             Producer：rram_write（RRAM 内容）
                             Consumer：rram_vmm
  Stage 3  rram_self_q    —— 数字逻辑计算 qi²（3 次 multiply-add）
  Stage 4  rram_sum       —— 累加 qi² + rram_output + k²_cache → final d²
                             Consumer：rram_self_k2（k² cache），rram_vmm（cross-term）

与 acc SRAM-CIM 的对应关系：
  SRAM-CIM Stage 1 (k²)      ↔ PointCIM Stage 0+1 (rram_write + rram_self_k2, one-time)
  SRAM-CIM Stage 2 (-2qk)    ↔ PointCIM Stage 2   (rram_vmm, per query)
  SRAM-CIM Stage 3 (merge)   ↔ PointCIM Stage 3+4 (rram_self_q + rram_sum, per query)

MICSim 口径：与 matrix_power_estimator.py --distance_backend micsim 完全一致。
  cross-mode: _run_micsim_single_fc(cin=M, cout=N)  → layer_read_latency_ns / energy_pj
  self-mode:  _run_micsim_single_fc(cin=M, cout=1)  × self_rounds
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Any, List, Tuple

if TYPE_CHECKING:
    from ..configs.hw_config import HardwareConfig

from ..simulator.event import Event
from .common_ops import _normalized_scheduler_timing, _neurosim_clock_ns

try:
    from ..integration.rram_matrix_estimator import compute_rram_dist_primitives
except ImportError:
    def compute_rram_dist_primitives(rram_cfg, num_points):
        raise RuntimeError(
            "rram_matrix_estimator 导入失败，请确认 pointkan_sim/integration/ 已正确安装。"
        )


def _safe_ceil_div(a: int, b: int) -> int:
    return max(1, math.ceil(a / max(b, 1)))


def _rram_event(
    event_id: int,
    cycle: int,
    operator: str,
    stage: str,
    sub_stage: str,
    circuit_module: str,
    operation: str,
    power_mw: float,
    duration: int,
    clock_period_ns: float,
    metadata: Dict[str, Any],
) -> Event:
    return Event(
        event_id=event_id,
        cycle=cycle,
        operator=operator,
        stage=stage,
        sub_stage=sub_stage,
        circuit_module=circuit_module,
        operation=operation,
        power=power_mw,
        duration=duration,
        clock_period_ns=clock_period_ns,
        metadata=dict(metadata),
    )


# ---------------------------------------------------------------------------
# One-time Stage 0: RRAM 写入（per SA layer）
# ---------------------------------------------------------------------------

def gen_rram_dist_write_events(
    operator_name: str,
    hw_config: "HardwareConfig",
    start_event_id: int,
    start_cycle: int,
    num_points: int,
    metadata: Dict[str, Any],
) -> Tuple[List[Event], int, int]:
    """
    生成 RRAM 点云坐标编程事件（one-time per SA layer）。
    Producer：本事件写入 RRAM 内容，后续 rram_self_k2 和 rram_vmm 均依赖它。
    """
    rram_cfg = getattr(hw_config, "rram_dist", None)
    if rram_cfg is None:
        return [], start_event_id, start_cycle

    clock_period = float(getattr(hw_config, "clock_period", 1.0))
    neurosim_clock_period = float(_neurosim_clock_ns(hw_config))
    rram_cfg._clock_period_ns = neurosim_clock_period

    prims = compute_rram_dist_primitives(rram_cfg, num_points)
    write_energy_nj = prims["write_energy_nj"]
    write_latency_ns = float(prims.get("write_latency_ns", max(1, int(prims["write_cycles"])) * neurosim_clock_period))
    write_timing = _normalized_scheduler_timing(
        latency_ns=write_latency_ns,
        energy_nj=write_energy_nj,
        base_clock_ns=clock_period,
    )

    meta = dict(metadata)
    meta.update({
        "rram_write_num_points": num_points,
        "rram_write_cycles": int(write_timing["duration_cycles"]),
        "rram_write_native_cycles": max(1, int(prims["write_cycles"])),
        "rram_write_latency_ns": write_latency_ns,
        "rram_write_energy_nj": write_energy_nj,
        "rram_num_arrays": prims["num_arrays_total"],
        "rram_points_per_array": prims["points_per_array"],
        "rram_vmm_source": prims["vmm_source"],
        "native_clock_period_ns": neurosim_clock_period,
        # Producer-Consumer 标记
        "produces": "rram_content",
        "scheduler_normalized": True,
    })

    ev = _rram_event(
        event_id=start_event_id, cycle=start_cycle,
        operator=operator_name, stage="rram_write", sub_stage="rram_program_coords",
        circuit_module="rram_dist_array", operation="rram_program",
        power_mw=float(write_timing["power_mw"]), duration=int(write_timing["duration_cycles"]),
        clock_period_ns=clock_period, metadata=meta,
    )
    return [ev], start_event_id + 1, start_cycle + int(write_timing["duration_cycles"])


# ---------------------------------------------------------------------------
# One-time Stage 1: self-mode k² 计算（per SA layer）
# ---------------------------------------------------------------------------

def gen_rram_dist_self_k2_events(
    operator_name: str,
    hw_config: "HardwareConfig",
    start_event_id: int,
    start_cycle: int,
    num_points: int,
    metadata: Dict[str, Any],
) -> Tuple[List[Event], int, int]:
    """
    生成 self-mode CIM 计算 k²（pj²）的事件（one-time per SA layer）。
    Consumer：消耗 RRAM 内容（rram_write 产生）
    Producer：写入 k² 到 SRAM cache，供 per-query rram_sum 使用。

    MICSim 口径：_run_micsim_single_fc(cin=M, cout=1) × self_rounds
    与 matrix_power_estimator.py self-mode 结果一致。
    """
    rram_cfg = getattr(hw_config, "rram_dist", None)
    if rram_cfg is None:
        return [], start_event_id, start_cycle

    clock_period = float(getattr(hw_config, "clock_period", 1.0))
    neurosim_clock_period = float(_neurosim_clock_ns(hw_config))
    rram_cfg._clock_period_ns = neurosim_clock_period

    prims = compute_rram_dist_primitives(rram_cfg, num_points)
    self_k2_energy_nj = prims["self_k2_energy_nj"]
    self_k2_latency_ns = prims["self_k2_latency_ns"]
    self_k2_timing = _normalized_scheduler_timing(
        latency_ns=self_k2_latency_ns,
        energy_nj=self_k2_energy_nj,
        base_clock_ns=clock_period,
    )

    meta = dict(metadata)
    meta.update({
        "rram_self_k2_num_points": num_points,
        "rram_self_k2_cycles": int(self_k2_timing["duration_cycles"]),
        "rram_self_k2_native_cycles": max(1, int(prims["self_k2_cycles"])),
        "rram_self_k2_latency_ns": self_k2_latency_ns,
        "rram_self_k2_energy_nj": self_k2_energy_nj,
        "rram_vmm_source": prims["vmm_source"],
        "native_clock_period_ns": neurosim_clock_period,
        # Producer-Consumer 标记
        "consumes": "rram_content",
        "produces": "k2_sram_cache",
        "scheduler_normalized": True,
    })

    ev = _rram_event(
        event_id=start_event_id, cycle=start_cycle,
        operator=operator_name, stage="rram_self_k2", sub_stage="self_mode_k2_compute",
        circuit_module="rram_dist_array", operation="rram_self_mode_k2",
        power_mw=float(self_k2_timing["power_mw"]), duration=int(self_k2_timing["duration_cycles"]),
        clock_period_ns=clock_period, metadata=meta,
    )
    return [ev], start_event_id + 1, start_cycle + int(self_k2_timing["duration_cycles"])


# ---------------------------------------------------------------------------
# Per-query Stages 2-4：VMM + q² + accumulate（每次 FPS 迭代 / KNN 查询）
# ---------------------------------------------------------------------------

def gen_rram_dist_events_one_batch(
    operator_name: str,
    hw_config: "HardwareConfig",
    start_event_id: int,
    start_cycle: int,
    metadata: Dict[str, Any],
    *,
    num_points: int = 0,
) -> Tuple[List[Event], int, int]:
    """
    生成单批 RRAM 距离计算事件（per query）：Stage 2–4（vmm / self_q / sum），
    供 FPS/KNN 每次迭代调用。

    Stage 2  rram_vmm     : cross-mode VMM (-2q·K)，MICSim cross-mode 建模
                            Consumer: rram_content（RRAM write 产生）
                            Producer: cross_term
    Stage 3  rram_self_q  : 数字逻辑 q²=x²+y²+z²
    Stage 4  rram_sum     : d² = q² + cross_term + k²_cache
                            Consumer: cross_term, k2_sram_cache

    与 acc SRAM-CIM 的事件数相同（3 条），事件语义对齐：
      vmm ↔ k²+cross_term CIM
      sum ↔ dist2_merge_adder
    """
    rram_cfg = getattr(hw_config, "rram_dist", None)
    if rram_cfg is None:
        return [], start_event_id, start_cycle

    clock_period = float(getattr(hw_config, "clock_period", 1.0))
    neurosim_clock_period = float(_neurosim_clock_ns(hw_config))
    rram_cfg._clock_period_ns = neurosim_clock_period

    if num_points <= 0:
        num_points = int(metadata.get("num_points", metadata.get("input_points", 1024)))

    prims = compute_rram_dist_primitives(rram_cfg, num_points)
    vmm_energy_nj = prims["vmm_energy_nj"]
    vmm_latency_ns = prims["vmm_latency_ns"]
    vmm_timing = _normalized_scheduler_timing(
        latency_ns=vmm_latency_ns,
        energy_nj=vmm_energy_nj,
        base_clock_ns=clock_period,
    )
    self_term_cycles = prims["self_term_cycles"]
    sum_cycles = prims["sum_cycles"]

    power_self_q = float(getattr(rram_cfg, "self_term_power_mw", 0.0))
    power_sum = float(getattr(rram_cfg, "sum_power_mw", 0.0))

    events: List[Event] = []
    eid = start_event_id
    cycle = start_cycle

    meta_base = dict(metadata)
    meta_base.update({
        "rram_num_points": num_points,
        "rram_vmm_cycles": int(vmm_timing["duration_cycles"]),
        "rram_vmm_native_cycles": max(1, int(prims["vmm_cycles"])),
        "rram_vmm_energy_nj": vmm_energy_nj,
        "rram_vmm_latency_ns": vmm_latency_ns,
        "rram_num_arrays": prims["num_arrays_total"],
        "rram_vmm_source": prims["vmm_source"],
        "native_clock_period_ns": neurosim_clock_period,
        "scheduler_normalized": True,
    })

    # Stage 2: cross-mode VMM (-2q·K)
    # Consumer: rram_content; Producer: cross_term
    events.append(_rram_event(
        event_id=eid, cycle=cycle,
        operator=operator_name, stage="rram_vmm", sub_stage="cross_term_vmm",
        circuit_module="rram_dist_array", operation="rram_vmm_cross_term",
        power_mw=float(vmm_timing["power_mw"]), duration=int(vmm_timing["duration_cycles"]),
        clock_period_ns=clock_period,
        metadata={**meta_base, "consumes": "rram_content", "produces": "cross_term"},
    ))
    eid += 1
    cycle += int(vmm_timing["duration_cycles"])

    # Stage 3: q² = x²+y²+z²（数字逻辑）
    events.append(_rram_event(
        event_id=eid, cycle=cycle,
        operator=operator_name, stage="rram_self_q", sub_stage="self_term_query",
        circuit_module="rram_dist_digital", operation="self_term_q_squared",
        power_mw=power_self_q, duration=max(1, self_term_cycles),
        clock_period_ns=clock_period, metadata=meta_base,
    ))
    eid += 1
    cycle += max(1, self_term_cycles)

    # Stage 4: d² = q² + cross_term + k²_cache
    # Consumer: cross_term + k2_sram_cache
    events.append(_rram_event(
        event_id=eid, cycle=cycle,
        operator=operator_name, stage="rram_sum", sub_stage="dist_accumulate",
        circuit_module="rram_dist_digital", operation="dist_q2_minus2qk_k2_sum",
        power_mw=power_sum, duration=max(1, sum_cycles),
        clock_period_ns=clock_period,
        metadata={**meta_base, "consumes": "cross_term,k2_sram_cache"},
    ))
    eid += 1
    cycle += max(1, sum_cycles)

    return events, eid, cycle
