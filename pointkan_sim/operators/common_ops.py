"""
通用算子事件生成（Common Ops）
===============================

Conv1d/Linear 的量化/反量化/BN/LayerNorm/ReLU/SPool 等逻辑电路事件生成，
以及 PE 级 Conv1d/Linear 的 RRAM-CIM 事件（无 MNSIM 全 NetStruct 时的回退）。
算子名统一用 PascalCase（Conv1d, Linear, BN 等），与 constants.OP_NAME_* 一致。
"""
from __future__ import annotations

import math
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from ..simulator.event import Event
from ..utils.math_utils import safe_div_ceil
from .mnsim_adapter import estimate_rram_cim_op
from .neurosim_adapter import (
    NeuroSimEstimate,
    estimate_conv1d_neurosim,
    estimate_linear_neurosim,
)

if TYPE_CHECKING:
    from ..configs.hw_config import HardwareConfig


def _clock(hw_config: "HardwareConfig") -> float:
    """从 hw_config 取时钟周期 (ns)。"""
    return float(getattr(hw_config, "clock_period", 1.0))


def _logic_power(hw_config: "HardwareConfig", key: str, default_power: float) -> float:
    """从 hw_config.logic_circuit.<key> 取 power (mW)。"""
    logic = getattr(hw_config, "logic_circuit", None)
    sub = getattr(logic, key, None) if logic is not None else None
    p = getattr(sub, "power", default_power) if sub is not None else default_power
    return float(p if p is not None else default_power)


def _logic_cycles(hw_config: "HardwareConfig", key: str, default_cycles: int) -> int:
    """从 hw_config.logic_circuit.<key> 取 cycles。"""
    logic = getattr(hw_config, "logic_circuit", None)
    sub = getattr(logic, key, None) if logic is not None else None
    c = getattr(sub, "cycles", default_cycles) if sub is not None else default_cycles
    return int(c if c is not None else default_cycles)


def _logic_parallelism(hw_config: "HardwareConfig", key: str, default_parallelism: int = 1) -> int:
    """从 hw_config.logic_circuit.<key> 取并行度。"""
    logic = getattr(hw_config, "logic_circuit", None)
    sub = getattr(logic, key, None) if logic is not None else None
    par = getattr(sub, "parallelism", default_parallelism) if sub is not None else default_parallelism
    return max(1, int(par if par is not None else default_parallelism))


def _logic_cycles_per_element(hw_config: "HardwareConfig", key: str, default: int = 1) -> int:
    """从 hw_config.logic_circuit.<key> 取每元素周期数（如 fp32_add/fp32_mul=1，fp32_div=10）。"""
    logic = getattr(hw_config, "logic_circuit", None)
    sub = getattr(logic, key, None) if logic is not None else None
    c = getattr(sub, "cycles_per_element", default) if sub is not None else default
    return max(1, int(c if c is not None else default))


def _prod(values: List[int]) -> int:
    out = 1
    for v in values:
        out *= max(1, int(v))
    return out


def _event(
    event_id: int,
    cycle: int,
    operator: str,
    stage: str,
    sub_stage: str,
    circuit_module: str,
    operation: str,
    power: float,
    duration: int,
    clock_period_ns: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> Event:
    """构造一条仿真事件（operator 建议用 PascalCase，如 Conv1d, BN）。"""
    return Event(
        event_id=event_id,
        cycle=cycle,
        operator=operator,
        stage=stage,
        sub_stage=sub_stage,
        circuit_module=circuit_module,
        operation=operation,
        power=float(power),
        duration=max(1, int(duration)),
        clock_period_ns=clock_period_ns,
        metadata=metadata or {},
    )


def _energy_to_power_mw(energy_nj: float, duration_cycles: int, clock_period_ns: float) -> float:
    dur = max(1, int(duration_cycles))
    clk = float(clock_period_ns) if clock_period_ns else 1.0
    if energy_nj <= 0 or dur <= 0 or clk <= 0:
        return 0.0
    return float(energy_nj) * 1000.0 / (dur * clk)


def _scalesim_memory_events(
    base_cycle: int,
    start_event_id: int,
    op_name: str,
    clk: float,
    hw_config: "HardwareConfig",
    meta: Dict[str, Any],
) -> List[Event]:
    events: List[Event] = []

    def _append_mem_event(
        event_id: int,
        operator: str,
        sub_stage: str,
        circuit_module: str,
        operation: str,
        energy_nj: float,
        cycles: int,
        clock_period_ns: float,
        extra_meta: Dict[str, Any],
    ) -> int:
        if energy_nj <= 0.0 or cycles <= 0:
            return event_id
        events.append(_event(
            event_id,
            base_cycle,
            operator,
            "buffer_io",
            sub_stage,
            circuit_module,
            operation,
            _energy_to_power_mw(energy_nj, cycles, clock_period_ns),
            cycles,
            clock_period_ns,
            {**meta, **extra_meta, "source": "scalesim_memory"},
        ))
        return event_id + 1

    event_id = start_event_id

    feature_local_sram_energy = float(meta.get("feature_local_sram_energy_nj") or 0.0)
    feature_local_sram_cycles = int(meta.get("feature_local_sram_cycles") or 0)
    feature_local_sram_clk = float(meta.get("scalesim_sram_clock_period_ns") or clk)
    feature_local_sram_bytes = int(meta.get("feature_local_sram_bytes_read", 0) or 0) + int(meta.get("feature_local_sram_bytes_write", 0) or 0)
    event_id = _append_mem_event(
        event_id,
        "LayerBuffer",
        "scalesim_feature_local_sram_access",
        "systolic_feature_sram",
        f"{op_name.lower()}_scalesim_feature_local_sram_access",
        feature_local_sram_energy,
        max(1, feature_local_sram_cycles) if feature_local_sram_energy > 0 else 0,
        feature_local_sram_clk,
        {
            "memory_kind": "sram",
            "buffer_name": "systolic_feature_sram_buffer",
            "buffer_phase": "compute_local",
            "bytes": feature_local_sram_bytes,
            "owner_operator": op_name,
            "owner_operator": op_name,
            "resource_id": "systolic_feature_sram_shared",
            "exclusive": True,
            "reset_scope": "operator_boundary",
        },
    )

    weight_sram_energy = float(meta.get("weight_sram_energy_nj") or 0.0)
    weight_sram_cycles = int(meta.get("weight_sram_cycles") or 0)
    weight_sram_clk = float(meta.get("scalesim_sram_clock_period_ns") or clk)
    weight_sram_bytes_read = int(meta.get("weight_sram_bytes_read", 0) or 0)
    weight_sram_bytes_write = int(meta.get("weight_sram_bytes_write", 0) or 0)
    weight_refill_tiles = max(1, int(meta.get("weight_refill_tiles", 1) or 1))
    weight_refill_tile_bytes = int(meta.get("weight_refill_tile_bytes", 0) or 0)
    if weight_sram_energy > 0.0 and weight_sram_cycles > 0:
        per_tile_write = weight_refill_tile_bytes if weight_refill_tile_bytes > 0 else int(math.ceil(weight_sram_bytes_write / float(weight_refill_tiles)))
        per_tile_energy = weight_sram_energy / float(weight_refill_tiles)
        per_tile_cycles = max(1, int(math.ceil(weight_sram_cycles / float(weight_refill_tiles))))
        remaining_read = weight_sram_bytes_read
        base_read = int(weight_sram_bytes_read // weight_refill_tiles)
        read_rem = int(weight_sram_bytes_read % weight_refill_tiles)
        for tile_idx in range(weight_refill_tiles):
            tile_read = base_read + (1 if tile_idx < read_rem else 0)
            tile_write = min(per_tile_write, max(0, weight_sram_bytes_write - tile_idx * per_tile_write))
            event_id = _append_mem_event(
                event_id,
                "LayerBuffer",
                "scalesim_weight_sram_access",
                "weight_sram_buffer",
                f"{op_name.lower()}_scalesim_weight_sram_access",
                per_tile_energy,
                per_tile_cycles,
                weight_sram_clk,
                {
                    "memory_kind": "sram",
                    "buffer_name": "weight_sram_buffer",
                    "buffer_phase": "weight_stage",
                    "bytes": tile_read + tile_write,
                    "owner_operator": op_name,
                    "owner_operator": op_name,
                    "bytes_read": tile_read,
                    "bytes_write": tile_write,
                    "tile_index": tile_idx,
                    "tile_count": weight_refill_tiles,
                    "resource_id": "weight_sram_shared",
                    "exclusive": True,
                    "reset_scope": "weight_tile_boundary",
                },
            )

    dram_clk = float(meta.get("scalesim_dram_clock_period_ns") or clk)
    dram_parts = [
        (
            "feature_input_read",
            float(meta.get("feature_input_dram_energy_nj") or 0.0),
            int(meta.get("feature_input_dram_cycles") or 0),
            int(meta.get("feature_input_dram_bytes", 0) or 0),
            meta.get("feature_input_dram_trace_dir", ""),
            f"{op_name.lower()}_scalesim_feature_input_dram_access",
            1,
            int(meta.get("feature_input_dram_bytes", 0) or 0),
        ),
        (
            "weight_read",
            float(meta.get("weight_dram_energy_nj") or 0.0),
            int(meta.get("weight_dram_cycles") or 0),
            int(meta.get("weight_dram_bytes", 0) or 0),
            meta.get("weight_dram_trace_dir", ""),
            f"{op_name.lower()}_scalesim_weight_dram_access",
            weight_refill_tiles,
            weight_refill_tile_bytes,
        ),
        (
            "feature_output_write",
            float(meta.get("feature_output_dram_energy_nj") or 0.0),
            int(meta.get("feature_output_dram_cycles") or 0),
            int(meta.get("feature_output_dram_bytes", 0) or 0),
            meta.get("feature_output_dram_trace_dir", ""),
            f"{op_name.lower()}_scalesim_feature_output_dram_access",
            1,
            int(meta.get("feature_output_dram_bytes", 0) or 0),
        ),
    ]
    for phase, energy_nj, cycles, num_bytes, trace_dir, operation, tile_count, tile_bytes in dram_parts:
        if energy_nj <= 0.0 or cycles <= 0 or num_bytes <= 0:
            continue
        per_tile_energy = energy_nj / float(max(1, tile_count))
        per_tile_cycles = max(1, int(math.ceil(cycles / float(max(1, tile_count)))))
        for tile_idx in range(max(1, tile_count)):
            this_tile_bytes = tile_bytes if tile_bytes > 0 else int(math.ceil(num_bytes / float(max(1, tile_count))))
            if tile_count > 1:
                remaining = max(0, num_bytes - tile_idx * this_tile_bytes)
                this_tile_bytes = min(this_tile_bytes, remaining)
            event_id = _append_mem_event(
                event_id,
                "DRAM",
                "scalesim_dram_access",
                "dram",
                operation,
                per_tile_energy,
                per_tile_cycles,
                dram_clk,
                {
                    "memory_kind": "dram",
                    "owner_operator": op_name,
                    "resource_id": "offchip_dram",
                    "buffer_name": "offchip_dram",
                    "buffer_phase": phase,
                    "bytes": this_tile_bytes,
                    "scalesim_dram_trace_dir": trace_dir,
                    "tile_index": tile_idx,
                    "tile_count": max(1, tile_count),
                },
            )

    return events


def _resolve_numel(params: Dict[str, Any], default_features: int = 1, default_points: int = 1) -> int:
    """从 params 解析元素总数（num_elements 或 num_features * num_points）。"""
    num_elements = int(params.get("num_elements", 0))
    if num_elements > 0:
        return num_elements
    num_features = int(params.get("num_features", params.get("normalized_shape", default_features)))
    num_points = int(params.get("num_points", default_points))
    return max(1, num_features * num_points)


def _quant_pipeline_power_cycles(
    hw_config: "HardwareConfig", numel: int, mode: str = "quant_8bit"
) -> tuple:
    """
    量化/反量化流水线：T = ceil(N/P) + pipeline_latency，功耗按并行度线性放大。
    mode='quant_8bit' 用 quant_pipeline_8bit（latency 6），'dequant' 用 dequant_pipeline（latency 4）。
    """
    lc = getattr(hw_config, "logic_circuit", None)
    if mode == "dequant":
        cfg = getattr(lc, "dequant_pipeline", None) if lc else None
        default_power, default_latency = 2.30, 4
    else:
        cfg = getattr(lc, "quant_pipeline_8bit", None) if lc else None
        default_power, default_latency = 2.234, 6
    power_base = float(getattr(cfg, "power", default_power) or default_power) if cfg else default_power
    par = max(1, int(getattr(cfg, "parallelism", 1) or 1)) if cfg else 1
    latency = int(getattr(cfg, "pipeline_latency", default_latency) or default_latency) if cfg else default_latency
    power = power_base * par
    cycles = max(1, safe_div_ceil(numel, par) + latency)
    return power, cycles


def gen_quant_events(
    numel: int,
    hw_config: "HardwareConfig",
    layer_name: str = "",
    op_name: str = "Conv1d",
) -> List[Event]:
    """W8A8：激活 FP32→INT8 量化事件（act_quant），在 MNSIM INT 计算前插入；op_name 用于 operator_sequence 显示（PascalCase）。"""
    clk = _clock(hw_config)
    power, cycles = _quant_pipeline_power_cycles(hw_config, numel, mode="quant_8bit")
    return [
        _event(
            0, 0,
            op_name, "quant", "quantize",
            "quant_pipeline_8bit", "activation_fp32_to_int8",
            power, cycles, clk,
            {"layer_name": layer_name, "numel": numel, "w8a8": True},
        )
    ]


def gen_dequant_events(
    numel: int,
    hw_config: "HardwareConfig",
    layer_name: str = "",
    op_name: str = "Conv1d",
) -> List[Event]:
    """W8A8：输出 INT8→FP32 反量化事件，供后续 BN/ReLU 等浮点运算；op_name 用于 operator_sequence 显示（PascalCase）。"""
    clk = _clock(hw_config)
    power, cycles = _quant_pipeline_power_cycles(hw_config, numel, mode="dequant")
    return [
        _event(
            0, 0,
            op_name, "dequant", "dequantize",
            "dequant_pipeline", "output_int8_to_fp32",
            power, cycles, clk,
            {"layer_name": layer_name, "numel": numel, "w8a8": True},
        )
    ]


def gen_bn_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    """BatchNorm1d：按向量宽度与 logic_circuit.batchnorm1d 的 cycles/power 生成归一化事件；BN 融合时可由 dispatcher 跳过。
    vec（vector_width）：未传时默认用 hw_config.cim_parallelism，表示本逻辑电路一次处理的元素数（与 CIM 阵列并行度同名，此处仅作 BN 的“每向量”粒度）。
    总周期 = ceil(numel/vec) * per_vector_cycles，其中 per_vector_cycles 来自 logic_circuit.batchnorm1d.cycles，默认 1。"""
    batch_size = int(params.get("batch_size", params.get("batch", 1)))
    channels = int(params.get("num_channels", params.get("num_features", params.get("in_channels", 1))))
    features = int(params.get("num_features", params.get("num_points", 1)))
    # BN 支持 B,N 和 B,N,S：2D 情况视为 S=1
    s_dim = int(params.get("s", params.get("seq_len", params.get("num_points", 1))))
    if "input_shape" in params and isinstance(params["input_shape"], (list, tuple)):
        shp = params["input_shape"]
        if len(shp) >= 2:
            channels = int(shp[1])
        s_dim = int(shp[2]) if len(shp) >= 3 else 1

    par = _logic_parallelism(hw_config, "batchnorm1d", 1)
    cycles = max(1, batch_size * (1 + safe_div_ceil(channels * max(1, s_dim), par)))
    power = _logic_power(hw_config, "batchnorm1d", 3.1718) * par
    clk = _clock(hw_config)
    return [_event(0, 0, "BN", "normalize", "batchnorm1d", "batchnorm1d", "batchnorm1d_affine", power, cycles, clk, {"batch": batch_size, "N": channels, "S": s_dim, "parallelism": par})]


def gen_layernorm_events(params: Dict[str, Any], hw_config: "HardwareConfig", op_name: str = "LayerNorm") -> List[Event]:
    """LayerNorm 周期公式：Nvec * (3*C/8 + 17)，C=features（归一化维度）。"""
    batch_size = int(params.get("batch_size", params.get("batch", 1)))
    num_points = int(params.get("num_points", params.get("tokens", 1)))
    features = int(params.get("num_features", params.get("normalized_shape", 1)))
    n_vec = max(1, batch_size * num_points)
    # 每向量周期 = 3*C/8 + 17（3*C/8 向上取整为整数）
    per_vec_cycles = (3 * features + 7) // 8 + 17
    base_cycles = n_vec * per_vec_cycles
    if params.get("num_elements", 0):
        numel = int(params["num_elements"])
    else:
        numel = max(1, batch_size * num_points * features)

    par = _logic_parallelism(hw_config, "layernorm", 1)
    cycles = max(1, safe_div_ceil(base_cycles, par))
    power = _logic_power(hw_config, "layernorm", 56.0) * par
    clk = _clock(hw_config)
    return [_event(0, 0, op_name, "normalize", "layernorm", "layernorm", "layernorm_affine", power, cycles, clk, {"numel": numel, "parallelism": par})]


def gen_relu_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    numel = _resolve_numel(params)
    par = _logic_parallelism(hw_config, "relu", 1)
    cycles = max(1, safe_div_ceil(numel, par))
    power = _logic_power(hw_config, "relu", 0.45) * par
    clk = _clock(hw_config)
    return [_event(0, 0, "ReLU", "activation", "relu", "relu", "relu", power, cycles, clk, {"numel": numel, "parallelism": par})]


def gen_spool_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    batch_size = int(params.get("batch_size", params.get("batch", 1)))
    channels = int(params.get("in_channels", params.get("num_features", 1)))
    num_points = int(params.get("num_points", 1))
    k_neighbors = int(params.get("k_neighbors", 1))

    # 更新公式：T = 56 + (N-1)*47, N=B*D*Ns
    n_total = max(1, batch_size * channels * num_points)
    cycles = max(1, 23 + (n_total - 1) * 12) # max(1, 56 + (n_total - 1) * 47)
    power = _logic_power(hw_config, "spool", 26.82)
    clk = _clock(hw_config)
    return [
        _event(
            0, 0, "SPool", "pool", "s_pool", "spool", "softmax_pool", power, cycles, clk,
            {"channels": channels, "num_points": num_points, "k_neighbors": k_neighbors, "N": n_total},
        )
    ]


def gen_affine_norm_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    """GAM 内仿射归一化（源码中为 mean/std + (x-mean)/(std+eps) + affine_alpha*x+affine_beta），
    并非简单 LayerNorm。暂按默认 1 cycle 占位，后续可再优化。"""
    s = int(params.get("s", params.get("num_points", 1)))
    k = int(params.get("k", params.get("k_neighbors", 1)))
    d = int(params.get("d", params.get("num_features", 1)))
    use_xyz = bool(params.get("use_xyz", False))
    coord_dims = int(params.get("coord_dims", 3))
    d_eff = d + (coord_dims if use_xyz else 0)
    l_work = max(1, s * k * d_eff)
    p_fixed = 32
    cycles = 2 * (safe_div_ceil(l_work, p_fixed) + 1) + 46

    clk = _clock(hw_config)
    power = _logic_power(hw_config, "layernorm", 56.0)
    return [
        _event(
            0, 0, "AffineNorm", "normalize", "affine_norm", "affine_norm", "groupnorm_affine",
            power, cycles, clk,
            {"S": s, "K": k, "D": d_eff, "L": l_work, "fixed_parallelism": p_fixed, "engine_parallelism": 1},
        )
    ]


def gen_adaptive_max_pool_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    """AdaptiveMaxPool1d：对 (B,C,L) 沿 L 做 max 到 1，工作量 B*C*L。"""
    batch_size = int(params.get("batch_size", params.get("batch", 1)))
    input_shape = params.get("input_shape")
    if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 2:
        base_cycles = _prod([int(x) for x in input_shape[:-1]])
    elif "num_elements" in params:
        base_cycles = int(params.get("num_elements", 1))
    else:
        num_features = int(params.get("num_features", params.get("num_points", 1)))
        num_points = int(params.get("num_points", params.get("input_points", 1)))
        # 沿最后一维做 max，工作量 = B × C × L
        base_cycles = max(1, batch_size * num_features * num_points)

    par = _logic_parallelism(hw_config, "adaptive_max_pool", 1)
    cycles = max(1, safe_div_ceil(base_cycles, par))
    power = _logic_power(hw_config, "adaptive_max_pool", 13.2) * par
    clk = _clock(hw_config)
    return [
        _event(
            0, 0, "AdaptiveMaxPool1d", "pool", "adaptive_max_pool1d", "adaptive_max_pool",
            "adaptive_max_pool1d", power, cycles, clk,
            {"base_cycles": base_cycles, "parallelism": par},
        )
    ]


def gen_add_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    num_elements = _resolve_numel(params)
    par = _logic_parallelism(hw_config, "fp32_add", 1)
    per_elem = _logic_cycles_per_element(hw_config, "fp32_add", 1)
    add_cycles = max(1, safe_div_ceil(num_elements, par) * per_elem)
    add_power = _logic_power(hw_config, "fp32_add", 1.16) * par
    clk = _clock(hw_config)

    events: List[Event] = []
    event_id = 0
    cycle = 0

    residual = bool(params.get("residual", False))
    offchip = bool(params.get("offchip", residual))
    if offchip:
        bytes_rd = max(1, math.ceil(num_elements * 2 / 8))
        bw = int(params.get("dram_bw_bytes_per_cycle", 32))
        dram_cycles = safe_div_ceil(bytes_rd, bw)
        dram_power = float(params.get("dram_power_mw", 50.0))
        events.append(_event(event_id, cycle, "ADD", "residual_buffer", "dram_read", "dram", "offchip_residual_read", dram_power, dram_cycles, clk, {"bytes": bytes_rd}))
        event_id += 1
        cycle += dram_cycles

    events.append(_event(event_id, cycle, "ADD", "elementwise", "add", "fp32_add", "elementwise_add", add_power, add_cycles, clk, {"num_elements": num_elements, "residual": residual}))
    event_id += 1
    cycle += add_cycles

    if offchip:
        bytes_wr = max(1, math.ceil(num_elements / 8))
        bw = int(params.get("dram_bw_bytes_per_cycle", 32))
        dram_cycles = safe_div_ceil(bytes_wr, bw)
        dram_power = float(params.get("dram_power_mw", 50.0))
        events.append(_event(event_id, cycle, "ADD", "residual_buffer", "dram_write", "dram", "offchip_residual_write", dram_power, dram_cycles, clk, {"bytes": bytes_wr}))

    return events


def gen_mul_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    num_elements = _resolve_numel(params)
    par = _logic_parallelism(hw_config, "fp32_mul", 1)
    per_elem = _logic_cycles_per_element(hw_config, "fp32_mul", 1)
    cycles = max(1, safe_div_ceil(num_elements, par) * per_elem)
    power = _logic_power(hw_config, "fp32_mul", 2.5) * par
    clk = _clock(hw_config)
    return [_event(0, 0, "MUL", "elementwise", "multiply", "fp32_mul", "elementwise_multiply", power, cycles, clk, {"num_elements": num_elements})]


def gen_div_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    """FP32 逐元素除法，cycle/power/area 由 logic_circuit.fp32_div 统一管理。"""
    num_elements = _resolve_numel(params)
    par = _logic_parallelism(hw_config, "fp32_div", 1)
    per_elem = _logic_cycles_per_element(hw_config, "fp32_div", 10)
    cycles = max(1, safe_div_ceil(num_elements, par) * per_elem)
    power = _logic_power(hw_config, "fp32_div", 2.59) * par
    clk = _clock(hw_config)
    return [_event(0, 0, "DIV", "elementwise", "divide", "fp32_div", "elementwise_divide", power, cycles, clk, {"num_elements": num_elements})]


def gen_kan_rram_lut_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    """
    KAT(GRF) 事件：base_v1/base_v2 用 logic_circuit.grf_base_engine（数字因果多项式）；
    acc 用 rram_lut_macro.rram_lut（RRAM-LUT 查表）。算子名统一为 KAN，breakdown 按 KAN 聚合。
    """
    in_features = int(params.get("in_features", 1))
    out_features = int(params.get("out_features", in_features))
    num_points = int(params.get("num_points", 1))
    variant = str(params.get("variant", "standard"))
    clk = _clock(hw_config)

    # base_v1/base_v2：KAT 由 grf_base_engine 数字实现（因果有理多项式，19 cycle/数据，23.783 mW）
    is_baseline = getattr(hw_config, "is_baseline", False)
    logic = getattr(hw_config, "logic_circuit", None)
    grf = getattr(logic, "grf_base_engine", None) if logic else None
    if is_baseline and grf is not None:
        cycles_per_data = int(getattr(grf, "cycles", 0) or 19)
        par = max(1, int(getattr(grf, "parallelism", 1) or 1))
        num_data = max(1, num_points * in_features)
        total_cycles = safe_div_ceil(num_data, par) * cycles_per_data
        power = float(getattr(grf, "power", 23.783) or 23.783)
        meta = {"variant": variant, "kat_impl": "grf_base_engine"}
        meta["num_points"] = num_points
        meta["in_features"] = in_features
        meta["out_features"] = out_features
        meta["activation_input_shape"] = f"{num_points}×{in_features}"
        meta["activation_output_shape"] = f"{num_points}×{out_features}"
        return [
            _event(
                0, 0, "KAN", "grf_kat", "grf_polynomial", "grf_base_engine", "kan_grf_causal",
                power, total_cycles, clk, meta,
            )
        ]

    # acc：KAT 由 RRAM-LUT 实现（buffer 读 + LUT 求值 + buffer 写）
    lut_factor = 4 if variant == "standard" else 2
    lut_parallel = int(params.get("lut_parallel", max(1, getattr(hw_config, "cim_parallelism", 128))))
    work = max(1, num_points * in_features * lut_factor)
    lut_cycles = safe_div_ceil(work, lut_parallel)
    buffer_cycles = safe_div_ceil(max(1, num_points * (in_features + out_features)), lut_parallel)

    lut_power = 5.0
    if hasattr(hw_config, "rram_lut_macro") and getattr(hw_config.rram_lut_macro, "rram_lut", None) is not None:
        lut_power = float(getattr(hw_config.rram_lut_macro.rram_lut, "power", lut_power) or lut_power)

    buffer_power = 0.5
    if hasattr(hw_config, "sram_cim_macro") and getattr(hw_config.sram_cim_macro, "output_buffer", None) is not None:
        ob = hw_config.sram_cim_macro.output_buffer
        if hasattr(ob, "get_read_power_mw"):
            buffer_power = float(ob.get_read_power_mw(getattr(hw_config, "clock_period", 1.0)))
        else:
            buffer_power = float(getattr(ob, "power", buffer_power) or buffer_power)

    kat_mode = str(params.get("kat_mode", "lut")).lower()
    lut_cfg = getattr(getattr(hw_config, "rram_lut_macro", None), "rram_lut", None)
    post_mac_power = float(getattr(lut_cfg, "post_mac_power", 0.0) or 0.0)
    post_mac_cycles = int(getattr(lut_cfg, "post_mac_cycles", 0) or 0)
    meta = {"variant": variant, "kat_impl": "rram_lut", "kat_mode": kat_mode}
    meta["num_points"] = num_points
    meta["in_features"] = in_features
    meta["out_features"] = out_features
    meta["activation_input_shape"] = f"{num_points}×{in_features}"
    meta["activation_output_shape"] = f"{num_points}×{out_features}"
    events = [
        _event(0, 0, "KAN", "buffer_io", "sram_read", "buffer", "kan_input_read", buffer_power, buffer_cycles, clk, meta),
        _event(1, buffer_cycles, "KAN", "rram_lut", "lut_eval", "rram_lut", "kan_basis_eval", lut_power, lut_cycles, clk, meta),
    ]
    cursor = buffer_cycles + lut_cycles
    if kat_mode == "lut_mac":
        mac_cycles = max(1, post_mac_cycles or safe_div_ceil(max(1, num_points * out_features), lut_parallel))
        mac_power = post_mac_power if post_mac_power > 0 else lut_power
        events.append(_event(2, cursor, "KAN", "rram_lut", "lut_post_mac", "rram_lut", "kan_post_lut_mac", mac_power, mac_cycles, clk, meta))
        cursor += mac_cycles
    events.append(_event(3, cursor, "KAN", "buffer_io", "sram_write", "buffer", "kan_output_write", buffer_power, buffer_cycles, clk, meta))
    return events


def _try_scalesim_conv1d(params: Dict[str, Any], hw_config: "HardwareConfig"):
    """若启用 use_scalesim_for_systolic 且 SCALE-Sim 可用，返回 (total_cycles, metadata)；否则返回 None。"""
    if not getattr(hw_config, "use_scalesim_for_systolic", False):
        return None
    try:
        from ..integration.scalesim_adapter import run_scalesim_conv1d_layer
    except ImportError:
        return None
    root = getattr(hw_config, "scalesim_root", "") or None
    cfg = getattr(hw_config, "scalesim_config_path", "") or None
    res = run_scalesim_conv1d_layer(
        in_channels=int(params.get("in_channels", 1)),
        out_channels=int(params.get("out_channels", 1)),
        kernel_size=int(params.get("kernel_size", 1)),
        num_points=int(params.get("num_points", 1)),
        layer_name=str(params.get("layer_name", "conv1d")),
        scalesim_root=root,
        config_path=cfg,
        hw_config=hw_config,
    )
    if not res.success:
        return None
    return (max(1, res.total_cycles), res.to_metadata())


def _try_scalesim_linear(params: Dict[str, Any], hw_config: "HardwareConfig"):
    """若启用 use_scalesim_for_systolic 且 SCALE-Sim 可用，返回 (total_cycles, metadata)；否则返回 None。"""
    if not getattr(hw_config, "use_scalesim_for_systolic", False):
        return None
    try:
        from ..integration.scalesim_adapter import run_scalesim_linear_layer
    except ImportError:
        return None
    root = getattr(hw_config, "scalesim_root", "") or None
    cfg = getattr(hw_config, "scalesim_config_path", "") or None
    res = run_scalesim_linear_layer(
        in_features=int(params.get("in_features", 1)),
        out_features=int(params.get("out_features", 1)),
        tokens=int(params.get("tokens", 1)),
        layer_name=str(params.get("layer_name", "linear")),
        scalesim_root=root,
        config_path=cfg,
        hw_config=hw_config,
    )
    if not res.success:
        return None
    return (max(1, res.total_cycles), res.to_metadata())


def gen_conv1d_systolic_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    """
    Conv1d 的 baseline 路径（base_v1/base_v2）：仅使用 SCALE-Sim 脉动模型（im2col + 脉动阵列），FXP16 计算，不考虑量化/反量化。
    若未启用 use_scalesim_for_systolic 或 SCALE-Sim 不可用/失败，则 raise；不回退到其他路径。
    """
    scalesim_out = _try_scalesim_conv1d(params, hw_config)
    if scalesim_out is None:
        raise RuntimeError(
            "Conv1d in baseline (base_v1/base_v2) requires SCALE-Sim: set use_scalesim_for_systolic=True, "
            "ensure SCALE-Sim is installed and scalesim_root/config are valid, and that the layer run succeeds."
        )

    in_ch = int(params.get("in_channels", 1))
    out_ch = int(params.get("out_channels", 1))
    kernel = int(params.get("kernel_size", 1))
    num_points = int(params.get("num_points", 1))
    layer_name = str(params.get("layer_name", ""))

    im2col_cfg = getattr(hw_config, "im2col", None)
    clk = _clock(hw_config)
    im2col_power = float(getattr(im2col_cfg, "power", 15.0) or 15.0) if im2col_cfg else 15.0
    im2col_cpo = max(1, int(getattr(im2col_cfg, "cycles_per_output", 1) or 1)) if im2col_cfg else 1
    L = max(1, num_points)
    im2col_cycles = max(1, L * im2col_cpo)
    M, K, N = out_ch, max(1, in_ch * kernel), L

    systolic_cycles, scalesim_meta = scalesim_out
    # 激活 shape 供输出/核对 SCALE-Sim 周期：Conv1d 映射为 ifmap (1,L) channels=in_ch, filter (1,kernel) num_filter=out_ch
    systolic_meta_base = {
        "M": M, "K": K, "N": N, "layer_name": layer_name, "source": "scalesim",
        "num_points": L, "in_channels": in_ch, "out_channels": out_ch, "kernel_size": kernel,
        "activation_input_shape": str(params.get("activation_input_shape") or f"1×{L} ch={in_ch}"),
        "activation_output_shape": str(params.get("activation_output_shape") or f"1×{L} ch={out_ch}"),
    }
    systolic_meta_base.update(scalesim_meta)
    mac_energy_nj = scalesim_meta.get("scalesim_mac_energy_nj")
    if mac_energy_nj is not None and systolic_cycles > 0 and clk > 0:
        sa_power = float(mac_energy_nj) * 1000.0 / (systolic_cycles * clk)
    else:
        macs = M * K * N
        sa_cfg = getattr(hw_config, "systolic_array", None)
        mac_power_mw = float(getattr(sa_cfg, "mac_power_mw", 7.144) or 7.144) if sa_cfg else 7.144
        sa_power = macs * mac_power_mw / max(1, systolic_cycles)

    events: List[Event] = []
    cycle = 0
    ev_id = 0
    events.append(_event(
        ev_id, cycle, "Conv1d", "im2col", "data_reorder", "im2col", "conv_im2col",
        im2col_power, im2col_cycles, clk,
        {"M": M, "K": K, "N": N, "layer_name": layer_name},
    ))
    ev_id += 1
    cycle += im2col_cycles
    events.append(_event(
        ev_id, cycle, "Conv1d", "systolic", "matrix_multiply", "systolic_array", "conv_systolic",
        sa_power, systolic_cycles, clk,
        systolic_meta_base,
    ))
    events.extend(_scalesim_memory_events(cycle, ev_id + 1, "Conv1d", clk, hw_config, systolic_meta_base))
    return events


def gen_linear_systolic_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    """
    Linear 的 baseline 路径（base_v1/base_v2）：仅使用 SCALE-Sim 脉动模型（N×N 脉动阵列，无 im2col），FXP16 计算，不考虑量化/反量化。
    若未启用 use_scalesim_for_systolic 或 SCALE-Sim 不可用/失败，则 raise；不回退到其他路径。
    """
    scalesim_out = _try_scalesim_linear(params, hw_config)
    if scalesim_out is None:
        raise RuntimeError(
            "Linear in baseline (base_v1/base_v2) requires SCALE-Sim: set use_scalesim_for_systolic=True, "
            "ensure SCALE-Sim is installed and scalesim_root/config are valid, and that the layer run succeeds."
        )

    in_f = int(params.get("in_features", 1))
    out_f = int(params.get("out_features", 1))
    tokens = int(params.get("tokens", 1))
    layer_name = str(params.get("layer_name", ""))

    clk = _clock(hw_config)
    M, K, N = out_f, in_f, max(1, tokens)

    systolic_cycles, scalesim_meta = scalesim_out
    # 激活 shape 供输出/核对 SCALE-Sim 周期：Linear 映射为 ifmap (tokens,1) ch=in_f, filter (1,1) num_filter=out_f
    meta = {
        "M": M, "K": K, "N": N, "layer_name": layer_name, "source": "scalesim",
        "tokens": N, "in_features": in_f, "out_features": out_f,
        "activation_input_shape": str(params.get("activation_input_shape") or f"{N}×1 ch={in_f}"),
        "activation_output_shape": str(params.get("activation_output_shape") or f"{N}×1 ch={out_f}"),
    }
    meta.update(scalesim_meta)
    mac_energy_nj = meta.get("scalesim_mac_energy_nj")
    if mac_energy_nj is not None and systolic_cycles > 0 and clk > 0:
        sa_power = float(mac_energy_nj) * 1000.0 / (systolic_cycles * clk)
    else:
        macs = M * K * N
        sa_cfg = getattr(hw_config, "systolic_array", None)
        mac_power_mw = float(getattr(sa_cfg, "mac_power_mw", 7.144) or 7.144) if sa_cfg else 7.144
        sa_power = macs * mac_power_mw / max(1, systolic_cycles)

    ev = _event(
        0, 0, "Linear", "systolic", "matrix_multiply", "systolic_array", "linear_systolic",
        sa_power, systolic_cycles, clk,
        meta,
    )
    return [ev] + _scalesim_memory_events(0, 1, "Linear", clk, hw_config, meta)


def gen_conv1d_rram_cim_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    """Conv1d 的 PE 级 RRAM-CIM 事件（无 MNSIM/NeuroSim 全 NetStruct 时使用）；可选前量化+后反量化。算子名用 Conv1d（PascalCase）。"""
    in_ch = int(params.get("in_channels", 1))
    out_ch = int(params.get("out_channels", 1))
    kernel = int(params.get("kernel_size", 1))
    num_points = int(params.get("num_points", 1))
    layer_name = str(params.get("layer_name", ""))

    fallback_power = 25.0
    if hasattr(hw_config, "rram_cim_macro"):
        fallback_power = float(getattr(hw_config.rram_cim_macro, "power", fallback_power) or fallback_power)

    cim_backend = getattr(hw_config, "cim_backend", "mnsim") or "mnsim"
    if cim_backend == "neurosim":
        est = estimate_conv1d_neurosim(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel,
            num_points=num_points,
            clock_period_ns=_clock(hw_config),
            neurosim_root=getattr(hw_config, "neurosim_root", "/workspace/DNN_NeuroSim_V1.4/Inference_pytorch"),
            fallback_power_mw=fallback_power,
        )
        assert isinstance(est, NeuroSimEstimate)
    else:
        est = estimate_rram_cim_op(
            in_features=max(1, in_ch * kernel),
            out_features=max(1, out_ch),
            tokens=max(1, num_points),
            clock_period_ns=_clock(hw_config),
            cim_parallelism=max(1, getattr(hw_config, "cim_parallelism", 128)),
            fallback_power_mw=fallback_power,
        )
    compute_ev = _event(
        0, 0, "Conv1d", "rram_cim", "matrix_vector", "rram_cim", "conv1d_rram_cim",
        est.avg_power_mw, est.cycles, _clock(hw_config),
        {"source": est.source, "latency_ns": est.latency_ns, "energy_nj": est.energy_nj},
    )

    if not getattr(hw_config, "enable_w8a8_quant_dequant", False):
        return [compute_ev]

    # W8A8：前量化(激活 FP32→INT8)、后反量化(输出 INT8→FP32)，MNSIM 为 INT 计算
    in_numel = max(1, in_ch * num_points)
    out_numel = max(1, out_ch * num_points)
    q_evs = gen_quant_events(in_numel, hw_config, layer_name=layer_name, op_name="Conv1d")
    d_evs = gen_dequant_events(out_numel, hw_config, layer_name=layer_name, op_name="Conv1d")
    q_ev, d_ev = q_evs[0], d_evs[0]
    c0 = q_ev.cycle + q_ev.duration
    compute_ev.cycle = c0
    d_ev.cycle = c0 + compute_ev.duration
    return [q_ev, compute_ev, d_ev]


def gen_linear_rram_cim_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    """Linear 的 PE 级 RRAM-CIM 事件（无 MNSIM/NeuroSim 全 NetStruct 时使用）；可选前量化+后反量化。算子名用 Linear（PascalCase）。"""
    in_f = int(params.get("in_features", 1))
    out_f = int(params.get("out_features", 1))
    tokens = int(params.get("tokens", 1))
    layer_name = str(params.get("layer_name", ""))

    fallback_power = 20.0
    if hasattr(hw_config, "rram_cim_macro"):
        fallback_power = float(getattr(hw_config.rram_cim_macro, "power", fallback_power) or fallback_power)

    cim_backend = getattr(hw_config, "cim_backend", "mnsim") or "mnsim"
    if cim_backend == "neurosim":
        est = estimate_linear_neurosim(
            in_features=in_f,
            out_features=out_f,
            tokens=max(1, tokens),
            clock_period_ns=_clock(hw_config),
            neurosim_root=getattr(hw_config, "neurosim_root", "/workspace/DNN_NeuroSim_V1.4/Inference_pytorch"),
            fallback_power_mw=fallback_power,
        )
        assert isinstance(est, NeuroSimEstimate)
    else:
        est = estimate_rram_cim_op(
            in_features=in_f,
            out_features=out_f,
            tokens=max(1, tokens),
            clock_period_ns=_clock(hw_config),
            cim_parallelism=max(1, getattr(hw_config, "cim_parallelism", 128)),
            fallback_power_mw=fallback_power,
        )
    compute_ev = _event(
        0, 0, "Linear", "rram_cim", "matrix_vector", "rram_cim", "linear_rram_cim",
        est.avg_power_mw, est.cycles, _clock(hw_config),
        {"source": est.source, "latency_ns": est.latency_ns, "energy_nj": est.energy_nj},
    )

    if not getattr(hw_config, "enable_w8a8_quant_dequant", False):
        return [compute_ev]

    # W8A8：前量化(激活 FP32→INT8)、后反量化(输出 INT8→FP32)，MNSIM 为 INT 计算
    in_numel = max(1, in_f * max(1, tokens))
    out_numel = max(1, out_f * max(1, tokens))
    q_evs = gen_quant_events(in_numel, hw_config, layer_name=layer_name, op_name="Linear")
    d_evs = gen_dequant_events(out_numel, hw_config, layer_name=layer_name, op_name="Linear")
    q_ev, d_ev = q_evs[0], d_evs[0]
    c0 = q_ev.cycle + q_ev.duration
    compute_ev.cycle = c0
    d_ev.cycle = c0 + compute_ev.duration
    return [q_ev, compute_ev, d_ev]
