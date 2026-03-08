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


def _resolve_numel(params: Dict[str, Any], default_features: int = 1, default_points: int = 1) -> int:
    """从 params 解析元素总数（num_elements 或 num_features * num_points）。"""
    num_elements = int(params.get("num_elements", 0))
    if num_elements > 0:
        return num_elements
    num_features = int(params.get("num_features", params.get("normalized_shape", default_features)))
    num_points = int(params.get("num_points", default_points))
    return max(1, num_features * num_points)


def _quant_dequant_power_cycles(hw_config: "HardwareConfig", numel: int) -> tuple:
    """量化/反量化：默认 1 cycle，功耗按并行度线性放大。"""
    qd = getattr(
        getattr(hw_config, "logic_circuit", None),
        "quant_dequant",
        None,
    )
    power_base = float(getattr(qd, "power", 2.59) or 2.59)
    cycles_cfg = int(getattr(qd, "cycles", 1) or 1)
    par = max(1, int(getattr(qd, "parallelism", 1) or 1))
    power = power_base * par
    cycles = max(1, safe_div_ceil(cycles_cfg, par))
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
    """BatchNorm1d：按向量宽度与 logic_circuit.batchnorm1d 的 cycles/power 生成归一化事件；BN 融合时可由 dispatcher 跳过。
    vec（vector_width）：未传时默认用 hw_config.cim_parallelism，表示本逻辑电路一次处理的元素数（与 CIM 阵列并行度同名，此处仅作 BN 的“每向量”粒度）。
    总周期 = ceil(numel/vec) * per_vector_cycles，其中 per_vector_cycles 来自 logic_circuit.batchnorm1d.cycles，默认 1。"""
    numel = _resolve_numel(params)
    vec = int(params.get("vector_width", getattr(hw_config, "cim_parallelism", 128)))
    per_vec = _logic_cycles(hw_config, "batchnorm1d", 1)
    cycles = safe_div_ceil(numel, vec) * per_vec
    power = _logic_power(hw_config, "batchnorm1d", 2.0)
    clk = _clock(hw_config)
    return [_event(0, 0, "BN", "normalize", "batchnorm1d", "batchnorm1d", "batchnorm1d_affine", power, cycles, clk, {"numel": numel})]


def gen_layernorm_events(params: Dict[str, Any], hw_config: "HardwareConfig", op_name: str = "LayerNorm") -> List[Event]:
    batch_size = int(params.get("batch_size", params.get("batch", 1)))
    num_points = int(params.get("num_points", params.get("tokens", 1)))
    features = int(params.get("num_features", params.get("normalized_shape", 1)))
    n_vec = max(1, batch_size * num_points)
    base_cycles = (6 * features + 32) + n_vec * 10 * features
    if params.get("num_elements", 0):
        numel = int(params["num_elements"])
    else:
        numel = max(1, batch_size * num_points * features)

    par = _logic_parallelism(hw_config, "layernorm", 1)
    cycles = max(1, safe_div_ceil(base_cycles, par))
    power = _logic_power(hw_config, "layernorm", 180.0) * par
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

    per_point_cycles = (max(1, k_neighbors) / 4.0) * 23.0 + 14.0
    base_cycles = max(1, int(math.ceil(batch_size * channels * num_points * per_point_cycles)))
    par = _logic_parallelism(hw_config, "spool", 1)
    cycles = max(1, safe_div_ceil(base_cycles, par))
    power = _logic_power(hw_config, "spool", 17.0) * par
    clk = _clock(hw_config)
    return [
        _event(
            0, 0, "SPool", "pool", "s_pool", "spool", "softmax_pool", power, cycles, clk,
            {"channels": channels, "num_points": num_points, "k_neighbors": k_neighbors, "parallelism": par},
        )
    ]


def gen_affine_norm_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    """GAM 内仿射归一化（源码中为 mean/std + (x-mean)/(std+eps) + affine_alpha*x+affine_beta），
    并非简单 LayerNorm。暂按默认 1 cycle 占位，后续可再优化。"""
    clk = _clock(hw_config)
    power = _logic_power(hw_config, "layernorm", 1.0)
    return [
        _event(
            0, 0, "AffineNorm", "normalize", "affine_norm", "affine_norm", "groupnorm_affine",
            power, 1, clk,
            {"num_features": int(params.get("num_features", 1)), "placeholder_cycles": 1},
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

    par = _logic_parallelism(hw_config, "spool", 1)
    cycles = max(1, safe_div_ceil(base_cycles, par))
    power = _logic_power(hw_config, "spool", 13.2) * par
    clk = _clock(hw_config)
    return [
        _event(
            0, 0, "AdaptiveMaxPool1d", "pool", "adaptive_max_pool1d", "spool",
            "adaptive_max_pool1d", power, cycles, clk,
            {"base_cycles": base_cycles, "parallelism": par},
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
