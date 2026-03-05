"""
Operator Dispatcher
====================

Pure bridge: maps op_type strings to Operator classes, instantiates them,
and calls generate_events().

Dispatcher does NOT contain any cycle/power/event logic itself.
All computation details live in each Operator class.

Responsibilities:
    1. op_type string → handler lookup
    2. Extract params from OperatorRef → build XxxParams
    3. Instantiate XxxOperator(params, hw_config)
    4. Return operator.generate_events() → List[Event]
    5. dispatch_layer: chain operators sequentially with cycle offsets

Usage:
    from operators.dispatcher import dispatch_operator, dispatch_layer

    events = dispatch_operator(op_ref, hw_config)
    layer_events = dispatch_layer(layer.operators, hw_config)
"""

import math
from typing import Dict, Any, List, Callable, Optional, TYPE_CHECKING

from ..simulator.event import Event
from .common_ops import (
    gen_conv1d_rram_cim_events,
    gen_linear_rram_cim_events,
    gen_bn_events,
    gen_layernorm_events,
    gen_relu_events,
    gen_spool_events,
    gen_kan_rram_lut_events,
    gen_affine_norm_events,
    gen_adaptive_max_pool_events,
    gen_add_events,
    gen_mul_events,
)

if TYPE_CHECKING:
    from ..configs.hw_config import HardwareConfig
    from ..network.pointkan_net import OperatorRef


# =============================================================================
# Per-op_type handlers
# =============================================================================
# Each handler signature:
#   (op_ref: OperatorRef, hw_config: HardwareConfig) -> List[Event]


def _gen_conv1d(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """Conv1d: MNSIM full NetStruct result if context provided, else PE-level adapter."""
    if context and "mnsim_latency_ns" in context and "mnsim_energy_nj" in context:
        idx = context.get("mnsim_index", 0)
        lat_ns = context["mnsim_latency_ns"]
        en_nj = context["mnsim_energy_nj"]
        if idx < len(lat_ns) and idx < len(en_nj):
            clock_ns = getattr(hw_config, "clock_period", 1.0)
            cycles = max(1, int(math.ceil(lat_ns[idx] / clock_ns)))
            power_mw = (en_nj[idx] * 1e3 / (cycles * clock_ns)) if (cycles * clock_ns) > 0 else 0.0
            context["mnsim_index"] = idx + 1
            return [
                Event(
                    event_id=0,
                    cycle=0,
                    operator="conv1d",
                    circuit_module="mnsim_tile_noc",
                    operation="conv1d_mnsim_full",
                    power=power_mw,
                    duration=cycles,
                    clock_period_ns=clock_ns,
                    metadata={"source": "mnsim_full"},
                )
            ]
    return gen_conv1d_rram_cim_events(op_ref.params, hw_config)


def _gen_linear(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """Linear: MNSIM full NetStruct result if context provided, else PE-level adapter."""
    if context and "mnsim_latency_ns" in context and "mnsim_energy_nj" in context:
        idx = context.get("mnsim_index", 0)
        lat_ns = context["mnsim_latency_ns"]
        en_nj = context["mnsim_energy_nj"]
        if idx < len(lat_ns) and idx < len(en_nj):
            clock_ns = getattr(hw_config, "clock_period", 1.0)
            cycles = max(1, int(math.ceil(lat_ns[idx] / clock_ns)))
            power_mw = (en_nj[idx] * 1e3 / (cycles * clock_ns)) if (cycles * clock_ns) > 0 else 0.0
            context["mnsim_index"] = idx + 1
            return [
                Event(
                    event_id=0,
                    cycle=0,
                    operator="linear",
                    circuit_module="mnsim_tile_noc",
                    operation="linear_mnsim_full",
                    power=power_mw,
                    duration=cycles,
                    clock_period_ns=clock_ns,
                    metadata={"source": "mnsim_full"},
                )
            ]
    return gen_linear_rram_cim_events(op_ref.params, hw_config)


def _gen_bn(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """BatchNorm1d generic logic-op model."""
    return gen_bn_events(op_ref.params, hw_config)


def _gen_layernorm(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """LayerNorm generic logic-op model."""
    return gen_layernorm_events(op_ref.params, hw_config)


def _gen_relu(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """ReLU generic logic-op model."""
    return gen_relu_events(op_ref.params, hw_config)


def _gen_spool(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """S-Pool generic logic-op model."""
    return gen_spool_events(op_ref.params, hw_config)


def _gen_kan(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """KAN LUT model (B-spline / GRF basis on RRAM-LUT)."""
    return gen_kan_rram_lut_events(op_ref.params, hw_config)


def _gen_fps(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """FPS → FPSOperator.generate_events()."""
    from .fps import FPSOperator, FPSParams

    params = op_ref.params
    fps_params = FPSParams(
        input_points=params.get('input_points', 1024),
        sampled_points_num=params.get('output_points', 512),
    )
    return FPSOperator(params=fps_params, hw_config=hw_config).generate_events()


def _gen_knn(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """KNN → KNNOperator.generate_events()."""
    from .knn import KNNOperator, KNNParams

    params = op_ref.params
    knn_params = KNNParams(
        input_points=params.get('input_points', 1024),
        query_points=params.get('query_points', 512),
        k=params.get('k', 16),
    )
    return KNNOperator(params=knn_params, hw_config=hw_config).generate_events()


def _gen_stub(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """Stub handler for ops not yet mapped to CIM (pipeline completeness only)."""
    return []


def _gen_affine_norm(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """AffineNorm generic logic-op model."""
    return gen_affine_norm_events(op_ref.params, hw_config)


def _gen_adaptive_max_pool(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """AdaptiveMaxPool1d generic logic-op model."""
    return gen_adaptive_max_pool_events(op_ref.params, hw_config)


def _gen_add(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """Element-wise add, supports residual off-chip buffering."""
    return gen_add_events(op_ref.params, hw_config)


def _gen_mul(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """Element-wise multiplication generic model."""
    return gen_mul_events(op_ref.params, hw_config)


# =============================================================================
# Dispatcher Registry & Public API
# =============================================================================

_OP_TYPE_HANDLERS: Dict[str, Callable] = {
    'conv1d':      _gen_conv1d,
    'linear':      _gen_linear,
    'bn':          _gen_bn,
    'batchnorm1d': _gen_bn,       # alias
    'layernorm':   _gen_layernorm,
    'relu':        _gen_relu,
    's-pool':      _gen_spool,
    'fps':         _gen_fps,
    'knn':         _gen_knn,
    'kan':         _gen_kan,
    'affine_norm':           _gen_affine_norm,
    'adaptive_max_pool1d':   _gen_adaptive_max_pool,
    'add':                   _gen_add,
    'mul':                   _gen_mul,
    'multiplication':        _gen_mul,
    'element_multiply':      _gen_mul,
}


def dispatch_operator(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """
    Dispatch an OperatorRef to its Operator and return events.

    Args:
        op_ref: Operator reference from network layer definition.
        hw_config: Hardware configuration.
        context: Optional dict, e.g. mnsim_latency_ns, mnsim_energy_nj, mnsim_index for conv1d/linear.

    Returns:
        List of Event objects with relative cycles starting from 0.

    Raises:
        ValueError: If op_type is not registered.
    """
    handler = _OP_TYPE_HANDLERS.get(op_ref.op_type)
    if handler is None:
        raise ValueError(
            f"Unknown op_type '{op_ref.op_type}'. "
            f"Supported: {list_supported_op_types()}"
        )
    return handler(op_ref, hw_config, context)


def dispatch_layer(
    layer_operators: List['OperatorRef'],
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """
    Dispatch all operators in a layer sequentially.

    Each operator's events are offset so it starts after the previous one ends.
    If context contains mnsim_latency_ns/mnsim_energy_nj/mnsim_index, conv1d/linear
    use MNSIM full NetStruct results (one Event per op).

    Args:
        layer_operators: Ordered list of OperatorRef from a LayerDef.
        hw_config: Hardware configuration.
        context: Optional MNSIM context (mutated: mnsim_index incremented).

    Returns:
        Combined list of Events with correct cycle offsets.
    """
    all_events: List[Event] = []
    current_cycle = 0

    for op_ref in layer_operators:
        events = dispatch_operator(op_ref, hw_config, context)
        for e in events:
            e.cycle += current_cycle
        all_events.extend(events)

        if events:
            current_cycle = max(e.cycle + e.duration for e in events)

    return all_events


def list_supported_op_types() -> List[str]:
    """Return list of all supported op_type strings."""
    return sorted(_OP_TYPE_HANDLERS.keys())


def register_op_type(op_type: str, handler: Callable) -> None:
    """
    Register a custom op_type handler.

    Args:
        op_type: The operator type string.
        handler: Function(op_ref, hw_config, context?) -> List[Event].
    """
    _OP_TYPE_HANDLERS[op_type] = handler
