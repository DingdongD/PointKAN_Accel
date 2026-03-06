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


def _resolve_numel(params: Dict[str, Any], default_features: int = 1, default_points: int = 1) -> int:
    """从 params 解析元素总数（num_elements 或 num_features * num_points）。"""
    num_elements = int(params.get("num_elements", 0))
    if num_elements > 0:
        return num_elements
    num_features = int(params.get("num_features", params.get("normalized_shape", default_features)))
    num_points = int(params.get("num_points", default_points))
    return max(1, num_features * num_points)


def _quant_dequant_power_cycles(hw_config: "HardwareConfig", numel: int) -> tuple:
    """从 hw_config.logic_circuit.quant_dequant 取 power/cycles；cycles 为 0 时按 numel/parallelism 估算。"""
    qd = getattr(
        getattr(hw_config, "logic_circuit", None),
        "quant_dequant",
        None,
    )
    power = float(getattr(qd, "power", 0.0) or 0.0)
    cycles_cfg = getattr(qd, "cycles", 0) or 0
    vec = max(1, getattr(hw_config, "cim_parallelism", 128))
    cycles = max(1, cycles_cfg) if cycles_cfg else max(1, safe_div_ceil(numel, vec))
    return power, cycles


def gen_quant_events(
    numel: int,
    hw_config: "HardwareConfig",
    layer_name: str = "",
    op_name: str = "Conv1d",
) -> List[Event]:
    """W8A8：激活 FP32→INT8 量化事件，在 MNSIM INT 计算前插入；op_name 用于 operator_sequence 显示（PascalCase）。"""
    clk = _clock(hw_config)
    power, cycles = _quant_dequant_power_cycles(hw_config, numel)
    return [
        _event(
            0, 0,
            op_name, "quant", "quantize",
            "quant_dequant", "activation_fp32_to_int8",
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
    power, cycles = _quant_dequant_power_cycles(hw_config, numel)
    return [
        _event(
            0, 0,
            op_name, "dequant", "dequantize",
            "quant_dequant", "output_int8_to_fp32",
            power, cycles, clk,
            {"layer_name": layer_name, "numel": numel, "w8a8": True},
        )
    ]


def gen_bn_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    """BatchNorm1d：按向量宽度与 logic_circuit.batchnorm1d 的 cycles/power 生成归一化事件；BN 融合时可由 dispatcher 跳过。"""
    numel = _resolve_numel(params)
    vec = int(params.get("vector_width", getattr(hw_config, "cim_parallelism", 128)))
    cycles = safe_div_ceil(numel, vec) * _logic_cycles(hw_config, "batchnorm1d", 1)
    power = _logic_power(hw_config, "batchnorm1d", 2.0)
    clk = _clock(hw_config)
    return [_event(0, 0, "BN", "normalize", "batchnorm1d", "batchnorm1d", "batchnorm1d_affine", power, cycles, clk, {"numel": numel})]


def gen_layernorm_events(params: Dict[str, Any], hw_config: "HardwareConfig", op_name: str = "LayerNorm") -> List[Event]:
    numel = _resolve_numel(params, default_features=int(params.get("normalized_shape", 1)))
    vec = int(params.get("vector_width", getattr(hw_config, "cim_parallelism", 128)))
    cycles = safe_div_ceil(numel, vec) * _logic_cycles(hw_config, "layernorm", 2)
    power = _logic_power(hw_config, "layernorm", 3.0)
    clk = _clock(hw_config)
    return [_event(0, 0, op_name, "normalize", "layernorm", "layernorm", "layernorm_affine", power, cycles, clk, {"numel": numel})]


def gen_relu_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    numel = _resolve_numel(params)
    vec = int(params.get("vector_width", getattr(hw_config, "cim_parallelism", 128)))
    cycles = safe_div_ceil(numel, vec) * _logic_cycles(hw_config, "relu", 1)
    power = _logic_power(hw_config, "relu", 1.0)
    clk = _clock(hw_config)
    return [_event(0, 0, "ReLU", "activation", "relu", "relu", "relu", power, cycles, clk, {"numel": numel})]


def gen_spool_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    channels = int(params.get("in_channels", 1))
    num_points = int(params.get("num_points", 1))
    k_neighbors = int(params.get("k_neighbors", 1))
    work = max(1, channels * num_points * k_neighbors)
    vec = int(params.get("vector_width", getattr(hw_config, "cim_parallelism", 128)))
    cycles = safe_div_ceil(work, vec) * _logic_cycles(hw_config, "spool", 2)
    power = _logic_power(hw_config, "spool", 2.5)
    clk = _clock(hw_config)
    return [
        _event(
            0, 0, "SPool", "pool", "s_pool", "spool", "softmax_pool", power, cycles, clk,
            {"channels": channels, "num_points": num_points, "k_neighbors": k_neighbors},
        )
    ]


def gen_affine_norm_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    p = dict(params)
    p.setdefault("normalized_shape", int(p.get("num_features", 1)))
    events = gen_layernorm_events(p, hw_config, op_name="AffineNorm")
    events[0].sub_stage = "affine_norm"
    events[0].operation = "groupnorm_affine"
    return events


def gen_adaptive_max_pool_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    num_features = int(params.get("num_features", 1))
    num_points = int(params.get("num_points", params.get("input_points", 1)))
    output_size = int(params.get("output_size", 1))
    work = max(1, num_features * max(1, num_points // max(1, output_size)))
    vec = int(params.get("vector_width", getattr(hw_config, "cim_parallelism", 128)))
    cycles = safe_div_ceil(work, vec)
    power = _logic_power(hw_config, "spool", 1.2)
    clk = _clock(hw_config)
    return [
        _event(
            0, 0, "AdaptiveMaxPool1d", "pool", "adaptive_max_pool1d", "spool",
            "adaptive_max_pool1d", power, cycles, clk,
            {"num_features": num_features, "num_points": num_points, "output_size": output_size},
        )
    ]


def gen_add_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    num_elements = _resolve_numel(params)
    vec = int(params.get("vector_width", getattr(hw_config, "cim_parallelism", 128)))
    add_cycles = safe_div_ceil(num_elements, vec)
    add_power = float(params.get("power_mw", 0.8))
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

    events.append(_event(event_id, cycle, "ADD", "elementwise", "add", "adder", "elementwise_add", add_power, add_cycles, clk, {"num_elements": num_elements, "residual": residual}))
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
    vec = int(params.get("vector_width", getattr(hw_config, "cim_parallelism", 128)))
    cycles = safe_div_ceil(num_elements, vec)
    power = float(params.get("power_mw", 1.1))
    clk = _clock(hw_config)
    return [_event(0, 0, "MUL", "elementwise", "multiply", "multiplier", "elementwise_multiply", power, cycles, clk, {"num_elements": num_elements})]


def gen_kan_rram_lut_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    in_features = int(params.get("in_features", 1))
    out_features = int(params.get("out_features", in_features))
    num_points = int(params.get("num_points", 1))
    variant = str(params.get("variant", "standard"))

    lut_factor = 4 if variant == "standard" else 2
    lut_parallel = int(params.get("lut_parallel", max(1, getattr(hw_config, "cim_parallelism", 128))))
    work = max(1, num_points * in_features * lut_factor)
    lut_cycles = safe_div_ceil(work, lut_parallel)
    buffer_cycles = safe_div_ceil(max(1, num_points * (in_features + out_features)), lut_parallel)

    lut_power = 5.0
    if hasattr(hw_config, "rram_lut_macro") and getattr(hw_config.rram_lut_macro, "rram_lut", None) is not None:
        lut_power = float(getattr(hw_config.rram_lut_macro.rram_lut, "power", lut_power) or lut_power)

    buffer_power = 0.5
    if hasattr(hw_config, "sram_cim_macro") and getattr(hw_config.sram_cim_macro, "buffer", None) is not None:
        buffer_power = float(getattr(hw_config.sram_cim_macro.buffer, "power", buffer_power) or buffer_power)

    clk = _clock(hw_config)
    return [
        _event(0, 0, "KAN", "buffer_io", "sram_read", "buffer", "kan_input_read", buffer_power, buffer_cycles, clk, {"variant": variant}),
        _event(1, buffer_cycles, "KAN", "rram_lut", "lut_eval", "rram_lut", "kan_basis_eval", lut_power, lut_cycles, clk, {"variant": variant}),
        _event(2, buffer_cycles + lut_cycles, "KAN", "buffer_io", "sram_write", "buffer", "kan_output_write", buffer_power, buffer_cycles, clk, {"variant": variant}),
    ]


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
