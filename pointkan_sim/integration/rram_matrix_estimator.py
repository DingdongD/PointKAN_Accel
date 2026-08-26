"""
RRAM 矩阵写入 & VMM 功耗/延迟估算（pointkan_sim 内置版本）
=============================================================

从 hw_sim/cycle_simulator/destiny_3d_cache/matrix_power_estimator.py 移植而来。

设计原则：
  - VMM（读取）阶段：仅通过 neurosim_dist 建模
  - 写入（编程）阶段：解析模型（NeuroSim 不覆盖写入路径）
  - 若 neurosim_dist 不可用或执行失败，显式报错而不是静默回退
  - compute_mapping / estimate_write_cost / estimate_vmm_cost 与原版等价

NeuroSim 接入路径：
  RRAMDistArrayConfig.neurosim_root = "/workspace/DNN_NeuroSim_V1.4/Inference_pytorch"
  → compute_rram_dist_primitives 自动调用 estimate_rram_dist_vmm_neurosim
  → VMM latency/energy 来自 NeuroSim crossbar 建模（read + 外围），写入能耗仍用静态值

用法（在仿真器内部）：
  from pointkan_sim.integration.rram_matrix_estimator import (
      compute_mapping, estimate_write_cost, estimate_vmm_cost,
      compute_rram_dist_primitives,
  )
"""
from __future__ import annotations

import math
import os
import sys
from functools import lru_cache
from typing import Dict, Any
import numbers


_NEUROSIM_DIST_MODULE_CACHE: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# 1. 矩阵映射
# ---------------------------------------------------------------------------

def compute_mapping(
    M: int,
    K: int,
    weight_bits: int,
    array_rows: int,
    array_cols: int,
    mapping_mode: str = "shared_rows",
) -> Dict[str, Any]:
    """
    计算点云坐标映射到 RRAM array 的布局信息（等价于原版 compute_mapping）。

    参数：
      M           每点坐标维度数（x/y/z=3）
      K           点数 N
      weight_bits 坐标量化位宽（列方向 bit-slice 宽度）
      array_rows  RRAM array 行数
      array_cols  RRAM array 列数
      mapping_mode  "diagonal"（对角 block）或 "shared_rows"（仅前 M 行沿列展开）

    Returns:
      dict with mapping info（字段与原版完全一致）
    """
    num_points = K
    fixed_duplicate = 1
    num_point_blocks = num_points * fixed_duplicate
    rows_per_point = M
    cols_per_point = weight_bits
    blocks_per_array_row = array_rows // rows_per_point
    blocks_per_array_col = array_cols // cols_per_point
    blocks_per_array_diag = min(blocks_per_array_row, blocks_per_array_col)

    if mapping_mode == "diagonal":
        points_per_array = blocks_per_array_diag
        if points_per_array < 1:
            raise ValueError(
                f"array {array_rows}x{array_cols} 无法容纳 1 个点 block "
                f"({rows_per_point}x{cols_per_point})，请增大 array 或减小 bits/M")
        num_arrays_total = math.ceil(num_point_blocks / points_per_array)
        num_arrays_row = 1
        num_arrays_col = num_arrays_total
        used_rows_per_full_array = points_per_array * rows_per_point
        total_rows_needed = num_point_blocks * rows_per_point
    elif mapping_mode == "shared_rows":
        points_per_array = blocks_per_array_col
        if points_per_array < 1:
            raise ValueError(
                f"array {array_rows}x{array_cols} 无法容纳 1 个点 block "
                f"({rows_per_point}x{cols_per_point})，请增大 array 或减小 bits/M")
        num_arrays_total = math.ceil(num_point_blocks / points_per_array)
        num_arrays_row = 1
        num_arrays_col = num_arrays_total
        used_rows_per_full_array = rows_per_point
        total_rows_needed = rows_per_point
    else:
        raise ValueError("mapping_mode 必须是 'diagonal' 或 'shared_rows'")

    full_points_per_array = points_per_array
    points_in_last_array = num_point_blocks - (num_arrays_total - 1) * full_points_per_array
    used_cols_per_full_array = full_points_per_array * cols_per_point
    used_rows_last_array = used_rows_per_full_array
    used_cols_last_array = points_in_last_array * cols_per_point
    total_cols_needed = num_point_blocks * cols_per_point

    return {
        "M": M,
        "K": K,
        "num_points": num_points,
        "fixed_duplicate": fixed_duplicate,
        "num_point_blocks": num_point_blocks,
        "weight_bits": weight_bits,
        "array_rows": array_rows,
        "array_cols": array_cols,
        "rows_per_point": rows_per_point,
        "cols_per_point": cols_per_point,
        "mapping_mode": mapping_mode,
        "points_per_array": points_per_array,
        "blocks_per_array_row": blocks_per_array_row,
        "blocks_per_array_col": blocks_per_array_col,
        "blocks_per_array_diag": blocks_per_array_diag,
        "full_points_per_array": full_points_per_array,
        "points_in_last_array": points_in_last_array,
        "total_rows_needed": total_rows_needed,
        "total_cols_needed": total_cols_needed,
        "num_arrays_row": num_arrays_row,
        "num_arrays_col": num_arrays_col,
        "num_arrays_total": num_arrays_total,
        "used_rows_per_array": used_rows_per_full_array,
        "used_cols_per_array": used_cols_per_full_array,
        "used_rows_last_array": used_rows_last_array,
        "used_cols_last_array": used_cols_last_array,
    }


# ---------------------------------------------------------------------------
# 2. 写入代价（RRAM 编程）
# ---------------------------------------------------------------------------

def estimate_write_cost(
    mapping: Dict[str, Any],
    write_energy_pj: float,
    write_latency_ns: float,
    word_width: int,
) -> Dict[str, Any]:
    """
    估算将整个矩阵写入 RRAM 的功耗与延迟（等价于原版 estimate_write_cost）。

    写入模型：逐行写入，每行一次写操作（word_width bits）。
    若 array 列数 > word_width，则需多次写操作完成一行。
    """
    total_write_ops = 0
    for a_idx in range(mapping["num_arrays_total"]):
        if a_idx < mapping["num_arrays_total"] - 1:
            points = mapping["full_points_per_array"]
        else:
            points = mapping["points_in_last_array"]

        if mapping["mapping_mode"] == "diagonal":
            rows = points * mapping["rows_per_point"]
        else:
            rows = mapping["rows_per_point"]
        cols = points * mapping["cols_per_point"]
        writes_per_row = math.ceil(cols / word_width)
        total_write_ops += rows * writes_per_row

    total_write_energy_pj = total_write_ops * write_energy_pj
    total_write_latency_ns = total_write_ops * write_latency_ns

    return {
        "total_write_ops": total_write_ops,
        "write_energy_per_op_pj": write_energy_pj,
        "total_write_energy_pj": total_write_energy_pj,
        "total_write_energy_nj": total_write_energy_pj / 1000,
        "total_write_energy_uj": total_write_energy_pj / 1e6,
        "total_write_latency_ns": total_write_latency_ns,
        "total_write_latency_us": total_write_latency_ns / 1000,
    }


def array_point_counts(mapping: Dict[str, Any]) -> list[int]:
    counts: list[int] = []
    total_arrays = int(mapping["num_arrays_total"])
    full_points = int(mapping["full_points_per_array"])
    last_points = int(mapping["points_in_last_array"])
    for idx in range(total_arrays):
        pts = full_points if idx < total_arrays - 1 else last_points
        if pts > 0:
            counts.append(int(pts))
    return counts


@lru_cache(maxsize=64)
def _destiny_single_array_rw(
    array_rows: int,
    array_cols: int,
    word_width: int,
    optimization_target: str = "WriteEDP",
) -> Dict[str, float]:
    destiny_dir = "/workspace/PointKAN-pytorch/hw_sim/cycle_simulator/destiny_3d_cache"
    if destiny_dir not in sys.path:
        sys.path.insert(0, destiny_dir)
    from destiny_runner import DestinyRunner, DestinyInput  # type: ignore

    array_capacity_bits = int(array_rows) * int(array_cols)
    array_capacity_kb = max(1, math.ceil(array_capacity_bits / (8 * 1024)))
    runner = DestinyRunner()
    result = runner.run(
        DestinyInput(
            capacity=array_capacity_kb,
            capacity_unit="KB",
            word_width_bits=int(word_width),
            optimization_target=str(optimization_target),
        )
    )
    if result.read_energy_pj is None or result.write_energy_pj is None:
        raise RuntimeError("DESTINY did not return valid read/write energy for RRAM array")
    return {
        "read_latency_ns": float(result.read_latency_ns),
        "write_latency_ns": float(result.write_latency_ns),
        "read_energy_pj": float(result.read_energy_pj),
        "write_energy_pj": float(result.write_energy_pj),
        "area_mm2": float(result.area_mm2),
        "leakage_power_uw": float(result.leakage_power_uw),
    }


def _estimate_destiny_write_cost(
    mapping: Dict[str, Any],
    array_rows: int,
    array_cols: int,
    word_width: int,
    optimization_target: str = "WriteEDP",
) -> Dict[str, Any]:
    rw = _destiny_single_array_rw(array_rows, array_cols, word_width, optimization_target)
    write_cost = estimate_write_cost(
        mapping,
        rw["write_energy_pj"],
        rw["write_latency_ns"],
        word_width,
    )
    write_cost["destiny_single_array"] = rw
    return write_cost


# ---------------------------------------------------------------------------
# 3. VMM 读取代价（self-mode / cross-mode）
# ---------------------------------------------------------------------------

def estimate_vmm_cost(
    mapping: Dict[str, Any],
    read_energy_pj: float,
    read_latency_ns: float,
    input_bits: int = 8,
) -> Dict[str, Any]:
    """
    估算 self-mode 与 cross-mode 的读取功耗与延迟（等价于原版 estimate_vmm_cost）。

    定义：
      self-mode:  每个点块使用各自的 (xi,yi,zi) 作为输入，按块串行处理
      cross-mode: 所有块广播同一组 (xi,yi,zi)，关键路径按单块统计（并行执行）

    参数：
      read_energy_pj  每次 VMM 读操作能耗 (pJ)，由调用方从 DESTINY/MICSim 预取
      read_latency_ns 每次 VMM 读操作延迟 (ns)
    """
    reads_per_point_block = mapping["rows_per_point"] * input_bits
    num_point_blocks = mapping["num_point_blocks"]

    if mapping["mapping_mode"] == "diagonal":
        self_rounds = 1
        self_points_per_round = num_point_blocks
    else:
        self_rounds = mapping["points_per_array"]
        self_points_per_round = mapping["num_arrays_total"]

    self_total_ops = num_point_blocks * reads_per_point_block
    self_energy_pj = self_total_ops * read_energy_pj
    self_latency_ns = self_rounds * reads_per_point_block * read_latency_ns

    cross_ops_energy = num_point_blocks * reads_per_point_block
    cross_energy_pj = cross_ops_energy * read_energy_pj
    cross_latency_ns = reads_per_point_block * read_latency_ns   # single block is critical path

    return {
        "input_bits": input_bits,
        "backend": "static",
        "fixed_duplicate": mapping["fixed_duplicate"],
        "self_mode": {
            "reads_per_eval": self_total_ops,
            "point_rounds": self_rounds,
            "points_per_round": self_points_per_round,
            "total_energy_pj": self_energy_pj,
            "total_energy_nj": self_energy_pj / 1000,
            "critical_path_latency_ns": self_latency_ns,
            "critical_path_latency_us": self_latency_ns / 1000,
        },
        "cross_mode": {
            "reads_per_eval": cross_ops_energy,
            "total_energy_pj": cross_energy_pj,
            "total_energy_nj": cross_energy_pj / 1000,
            "critical_path_latency_ns": cross_latency_ns,
            "critical_path_latency_us": cross_latency_ns / 1000,
        },
    }


# ---------------------------------------------------------------------------
# 4. VMM 建模后端
#    仅使用 neurosim_dist pybind
# ---------------------------------------------------------------------------

# ---- neurosim_dist pybind 加载 ----

def _get_neurosim_dist(so_dir: str = ""):
    """
    动态加载 neurosim_dist Python 包装层，返回模块对象。
    加载顺序：
      1. 用户指定的 so_dir（若包含 neurosim_dist.py）
      2. DNN_NeuroSim_V1.4 构建目录中的 neurosim_dist.py

    注意：不再优先加载 pointkan_sim 包目录中的旧 neurosim_dist .so，
    以免绕过 DNN_NeuroSim_V1.4 的 Python 包装层。
    """
    import sys
    import importlib.util

    candidates = []
    if so_dir:
        candidates.append(so_dir)
    # DNN_NeuroSim_V1.4 构建目录
    candidates.append("/workspace/DNN_NeuroSim_V1.4/Inference_pytorch/NeuroSIM")

    for d in candidates:
        if not os.path.isdir(d):
            continue
        wrapper_path = os.path.join(d, "neurosim_dist.py")
        if not os.path.isfile(wrapper_path):
            continue
        wrapper_path = os.path.abspath(wrapper_path)
        cached = _NEUROSIM_DIST_MODULE_CACHE.get(wrapper_path)
        if cached is not None:
            return cached
        if d not in sys.path:
            sys.path.insert(0, d)
        spec = importlib.util.spec_from_file_location("pointkan_neurosim_dist_wrapper", wrapper_path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _NEUROSIM_DIST_MODULE_CACHE[wrapper_path] = module
        return module
    raise ImportError("neurosim_dist.py not found in any candidate directory")


def compute_rram_dist_primitives(
    rram_cfg: Any,
    num_points: int,
) -> Dict[str, Any]:
    """
    根据 RRAMDistArrayConfig 和点数 num_points，计算 RRAM 距离原语。
    与 matrix_power_estimator.py --distance_backend micsim 结果保持数值一致。

    neurosim_dist 调用口径：
      cross-mode (-2qk, per query):
        run_neurosim_fc(cin=M, cout=num_blocks, ...)
        → layer_read_latency_ns, layer_read_energy_pj
      self-mode  (k², one-time per layer):
        run_neurosim_fc(cin=M, cout=1, ...) × self_rounds
        → self_k2_latency_ns, self_k2_energy_pj

    返回：
      vmm_cycles         cross-mode VMM 周期数（per query，系统时钟口径）
      vmm_latency_ns     cross-mode VMM 延迟 (ns)
      vmm_energy_nj      cross-mode VMM 能耗 (nJ)
      vmm_power_mw       cross-mode VMM 平均功耗 (mW)
      vmm_source         建模来源：'neurosim_dist'
      self_k2_cycles     self-mode k² 周期数（one-time per layer）
      self_k2_latency_ns self-mode k² 延迟 (ns)
      self_k2_energy_nj  self-mode k² 能耗 (nJ)
      write_cycles       RRAM 编程周期数（one-time per layer，静态模型）
      write_energy_nj    RRAM 编程能耗 (nJ)
      self_term_cycles   q² 数字计算周期数（per query）
      sum_cycles         d² 累加周期数（per query）
      num_arrays_total   RRAM array 数量
      points_per_array   每 array 容纳点数
    """
    M = int(getattr(rram_cfg, "coord_dims", 3))
    weight_bits = int(getattr(rram_cfg, "coord_bits", 12))
    input_bits = int(getattr(rram_cfg, "input_bits", 12))
    array_rows = int(getattr(rram_cfg, "array_rows", 128))
    array_cols = int(getattr(rram_cfg, "array_cols", 128))
    mapping_mode = str(getattr(rram_cfg, "mapping_mode", "diagonal"))
    word_width = int(getattr(rram_cfg, "word_width", 128))
    clock_period = float(getattr(rram_cfg, "_clock_period_ns", 1.0))
    write_energy_pj = float(getattr(rram_cfg, "write_energy_pj_per_op", 0.0))
    write_latency_ns = float(getattr(rram_cfg, "write_latency_ns_per_op", 0.0))
    destiny_opt_target = str(getattr(rram_cfg, "destiny_optimization_target", "WriteEDP"))
    self_term_cycles = int(getattr(rram_cfg, "self_term_cycles", 3))
    sum_cycles = int(getattr(rram_cfg, "sum_cycles", 1))

    # neurosim_dist 参数（保留 micsim_* 配置名以兼容现有 pointcim 配置）
    micsim_root = str(getattr(rram_cfg, "micsim_root", ""))
    micsim_clk_freq_hz = float(getattr(rram_cfg, "micsim_clk_freq_hz", 1e9))
    micsim_adc_clk_freq_hz = float(getattr(rram_cfg, "micsim_adc_clk_freq_hz", 2e8))
    micsim_sar_col_puw = float(getattr(rram_cfg, "micsim_sar_column_power_uw", 128.0))
    micsim_sar_col_w = micsim_sar_col_puw * 1e-6
    micsim_level_output = int(getattr(rram_cfg, "micsim_level_output", 64))
    micsim_mem_type = str(getattr(rram_cfg, "micsim_static_mem_type", "rram"))
    micsim_ron = float(getattr(rram_cfg, "micsim_static_ron_ohm", 240000.0))
    micsim_roff = float(getattr(rram_cfg, "micsim_static_roff_ohm", 24000000.0))
    micsim_cell_bit = int(getattr(rram_cfg, "micsim_static_cell_bit", 1))
    micsim_avg_act = float(getattr(rram_cfg, "micsim_avg_activity", 0.5))
    micsim_avg_cond = float(getattr(rram_cfg, "micsim_avg_conductance", 1e-6))
    micsim_num_col_muxed = int(getattr(rram_cfg, "micsim_num_col_muxed", 0))
    if micsim_num_col_muxed <= 0:
        micsim_num_col_muxed = weight_bits   # 跟随权重位宽，与 matrix_power_estimator.py 默认一致
    micsim_saradc = int(getattr(rram_cfg, "micsim_saradc", 1))
    micsim_current_mode = int(getattr(rram_cfg, "micsim_current_mode", 1))
    micsim_adc_mode = int(getattr(rram_cfg, "micsim_adc_mode", 0))

    mapping = compute_mapping(M, num_points, weight_bits, array_rows, array_cols, mapping_mode)
    if write_energy_pj > 0.0 and write_latency_ns > 0.0:
        write_cost = estimate_write_cost(mapping, write_energy_pj, write_latency_ns, word_width)
        write_source = "static_config"
    else:
        write_cost = _estimate_destiny_write_cost(
            mapping=mapping,
            array_rows=array_rows,
            array_cols=array_cols,
            word_width=word_width,
            optimization_target=destiny_opt_target,
        )
        write_source = "destiny_helper"
    num_blocks = mapping["num_point_blocks"]

    # self-mode 调度轮数（与 estimate_vmm_cost_micsim / estimate_vmm_cost 口径一致）
    if mapping_mode == "diagonal":
        self_rounds = 1
    else:
        self_rounds = mapping["points_per_array"]

    # -------------------------------------------------------------------
    # VMM 阶段：仅使用 neurosim_dist pybind（依赖 neurosim_core）
    # -------------------------------------------------------------------
    _ = micsim_root  # 保留配置字段兼容性；运行时不再使用 MICSim backend
    _ = micsim_mem_type
    _nd_so_dir = str(getattr(rram_cfg, "neurosim_dist_so_dir", ""))
    nd = _get_neurosim_dist(_nd_so_dir)

    _nd_common = dict(
        weight_bits=weight_bits,
        input_bits=input_bits,
        array_rows=array_rows,
        array_cols=array_cols,
        clk_freq_hz=micsim_clk_freq_hz,
        technode=22,
        adc_clk_freq_hz=micsim_adc_clk_freq_hz,
        sar_col_power_w=micsim_sar_col_w,
        r_on_ohm=micsim_ron,
        r_off_ohm=micsim_roff,
        cell_bit=micsim_cell_bit,
        memcell_type=2,                # RRAM
        avg_activity=micsim_avg_act,
        avg_conductance=micsim_avg_cond,
        level_output=micsim_level_output,
        num_col_muxed=micsim_num_col_muxed,
        saradc=micsim_saradc,
        current_mode=micsim_current_mode,
        adc_mode=micsim_adc_mode,
    )

    # PointCIM: all RRAM arrays operate in parallel with independent ADCs.
    # Model a SINGLE physical array (cout=points_per_array) for latency,
    # then scale energy by num_arrays_total.
    points_per_array = mapping["points_per_array"]
    num_arrays_total = mapping["num_arrays_total"]

    _dist_debug(f"cross-term start cin={M} cout={points_per_array} array={array_rows}x{array_cols} num_arrays_total={num_arrays_total}")
    prim_single_array = nd.run_neurosim_fc(cin=M, cout=points_per_array, **_nd_common)
    _assert_finite_neurosim_fc_result(prim_single_array, label=f"cross(cin={M}, cout={points_per_array})")
    vmm_latency_ns = float(prim_single_array["layer_read_latency_ns"])
    single_array_energy_nj = float(prim_single_array["layer_read_energy_pj"]) / 1000.0
    vmm_energy_nj = single_array_energy_nj * num_arrays_total
    _dist_debug(f"cross-term done cin={M} cout={points_per_array} latency_ns={vmm_latency_ns:.3f} energy_nj={vmm_energy_nj:.6f}")

    _dist_debug(f"self-term start cin={M} cout=1 array={array_rows}x{array_cols} self_rounds={self_rounds}")
    prim_self = nd.run_neurosim_fc(cin=M, cout=1, **_nd_common)
    _assert_finite_neurosim_fc_result(prim_self, label=f"self(cin={M}, cout=1)")
    self_k2_latency_ns = self_rounds * float(prim_self["layer_read_latency_ns"])
    self_k2_energy_nj = num_blocks * float(prim_self["layer_read_energy_pj"]) / 1000.0
    _dist_debug(f"self-term done cin={M} cout=1 latency_ns={self_k2_latency_ns:.3f} energy_nj={self_k2_energy_nj:.6f}")
    vmm_source = "neurosim_dist"

    vmm_cycles = max(1, math.ceil(vmm_latency_ns / max(clock_period, 1e-9))) if vmm_latency_ns > 0 else 1
    vmm_power_mw = (vmm_energy_nj * 1e3 / vmm_latency_ns) if vmm_latency_ns > 0 else 0.0
    self_k2_cycles = max(1, math.ceil(self_k2_latency_ns / max(clock_period, 1e-9))) if self_k2_latency_ns > 0 else 1

    write_latency_total_ns = write_cost["total_write_latency_ns"]
    write_energy_nj = write_cost["total_write_energy_nj"]
    write_cycles = (max(1, math.ceil(write_latency_total_ns / max(clock_period, 1e-9)))
                    if write_latency_total_ns > 0 else 1)

    return {
        "mapping": mapping,
        "array_point_counts": array_point_counts(mapping),
        # cross-mode VMM (per query)
        "vmm_cycles": vmm_cycles,
        "vmm_latency_ns": vmm_latency_ns,
        "vmm_energy_nj": vmm_energy_nj,
        "vmm_power_mw": vmm_power_mw,
        "vmm_source": vmm_source,
        # self-mode k² (one-time per layer)
        "self_k2_cycles": self_k2_cycles,
        "self_k2_latency_ns": self_k2_latency_ns,
        "self_k2_energy_nj": self_k2_energy_nj,
        # RRAM write (one-time per layer, static model)
        "write_cycles": write_cycles,
        "write_latency_ns": write_latency_total_ns,
        "write_energy_nj": write_energy_nj,
        "write_source": write_source,
        # per-query digital stages
        "self_term_cycles": self_term_cycles,
        "sum_cycles": sum_cycles,
        "num_arrays_total": mapping["num_arrays_total"],
        "points_per_array": mapping["points_per_array"],
        "write_cost": write_cost,
    }


def _assert_finite_neurosim_fc_result(result: Dict[str, Any], label: str) -> None:
    required = (
        "layer_read_latency_ns",
        "layer_read_energy_pj",
    )
    bad_fields = []
    for key in required:
        value = result.get(key, None)
        if not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
            bad_fields.append(f"{key}={value!r}")
    if bad_fields:
        raise RuntimeError(
            f"neurosim_dist.run_neurosim_fc produced non-finite outputs for {label}: "
            + ", ".join(bad_fields)
        )


def _dist_debug_enabled() -> bool:
    return str(os.environ.get("POINTKAN_RRAM_DIST_DEBUG", "") or "").strip().lower() in {"1", "true", "yes", "on"}


def _dist_debug(msg: str) -> None:
    if _dist_debug_enabled():
        print(f"[rram_dist] {msg}", flush=True)
