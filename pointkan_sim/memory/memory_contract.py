"""
PointKAN memory contract analysis.

目标：在“先明确 DRAM/SRAM 交互量”阶段给出可审计的逐层/逐算子字节清单，
用于后续 pipeline / non-pipeline 调度建模。

分级口径（简化）：
- DRAM
- SRAM (不同用途用 name 区分：xyz/feature/weight/dist/state/... buffers)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .feature_residency import build_feature_residency_plan


DRAM = "DRAM"
SRAM = "SRAM"
REGFILE = "REGFILE"
COMPUTE = "COMPUTE"


@dataclass
class MemoryContractPolicy:
    """Memory analysis policy knobs."""
    bytes_per_element: int = 4
    coord_dims: int = 3
    coord_bytes_per_dim: int = 4
    distance_bytes: int = 4
    index_bytes: int = 4

    # SRAM capacities (bytes)
    inter_layer_sram_bytes: int = 64 * 1024
    gam_workspace_sram_bytes: int = 1 * 1024 * 1024
    dist_sram_bytes: int = 4 * 1024
    fps_min_distance_table_sram_bytes: int = 4 * 1024
    sort_cache_sram_bytes: int = 256
    lfp_workspace_sram_bytes: int = 512 * 1024
    gfp_workspace_sram_bytes: int = 512 * 1024

    # LFP/GFP tensor strategy: stream_through or materialize
    lfp_tensor_strategy: str = "stream_through"
    gfp_tensor_strategy: str = "stream_through"

    # Materialization knobs
    materialize_fps_distance_table: bool = True
    materialize_knn_distance_table: bool = True
    materialize_knn_neighbor_features: bool = True

    # CIM staging model
    model_cim_staging: bool = True
    cim_batch_points: int = 128

    @property
    def point_xyz_bytes(self) -> int:
        return self.coord_dims * self.coord_bytes_per_dim


@dataclass
class MemoryContractRecord:
    layer_name: str
    op_name: str
    tensor_name: str
    src_tier: str
    src_name: str
    dst_tier: str
    dst_name: str
    bytes: int
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "op_name": self.op_name,
            "tensor_name": self.tensor_name,
            "src_tier": self.src_tier,
            "src_name": self.src_name,
            "dst_tier": self.dst_tier,
            "dst_name": self.dst_name,
            "src": self.src_tier,
            "dst": self.dst_tier,
            "bytes": int(self.bytes),
            "note": self.note,
        }


@dataclass
class BufferRequirement:
    layer_name: str
    op_name: str
    buffer_tier: str
    buffer_name: str
    peak_bytes: int
    purpose: str
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "op_name": self.op_name,
            "buffer_tier": self.buffer_tier,
            "buffer_name": self.buffer_name,
            "peak_bytes": int(self.peak_bytes),
            "peak_mib": float(self.peak_bytes) / float(1024 * 1024),
            "purpose": self.purpose,
            "note": self.note,
        }


@dataclass
class BufferRelationship:
    buffer_name: str
    shared_group: str
    role: str
    physical_sharing: str
    mutual_exclusion: str
    reset_scope: str
    scheduler_resource_id: str
    pipeline_behavior: str
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "buffer_name": self.buffer_name,
            "shared_group": self.shared_group,
            "role": self.role,
            "physical_sharing": self.physical_sharing,
            "mutual_exclusion": self.mutual_exclusion,
            "reset_scope": self.reset_scope,
            "scheduler_resource_id": self.scheduler_resource_id,
            "pipeline_behavior": self.pipeline_behavior,
            "note": self.note,
        }


@dataclass
class MemoryContractReport:
    records: List[MemoryContractRecord]
    route_bytes_by_tier: Dict[Tuple[str, str], int]
    route_bytes_detailed: Dict[Tuple[str, str], int]
    per_layer_bytes: Dict[str, int]
    suggested_sram_bytes: Dict[str, int]
    buffer_peak_bytes: Dict[str, int]
    buffer_requirements: List[BufferRequirement]
    buffer_relationships: List[BufferRelationship]
    policy: MemoryContractPolicy
    hw_config: Any
    feature_residency_plan: List[Dict[str, Any]]



def _safe_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _op_display(op: Any) -> str:
    t = str(getattr(op, "op_type", "") or "").strip()
    return t if t else "unknown"


def _append_record(
    out: List[MemoryContractRecord],
    layer_name: str,
    op_name: str,
    tensor_name: str,
    src_tier: str,
    src_name: str,
    dst_tier: str,
    dst_name: str,
    nbytes: int,
    note: str = "",
) -> None:
    if nbytes <= 0:
        return
    out.append(
        MemoryContractRecord(
            layer_name=layer_name,
            op_name=op_name,
            tensor_name=tensor_name,
            src_tier=src_tier,
            src_name=src_name,
            dst_tier=dst_tier,
            dst_name=dst_name,
            bytes=int(nbytes),
            note=note,
        )
    )


def _append_buffer_requirement(
    out: List[BufferRequirement],
    layer_name: str,
    op_name: str,
    buffer_tier: str,
    buffer_name: str,
    peak_bytes: int,
    purpose: str,
    note: str = "",
) -> None:
    if peak_bytes <= 0:
        return
    out.append(
        BufferRequirement(
            layer_name=layer_name,
            op_name=op_name,
            buffer_tier=buffer_tier,
            buffer_name=buffer_name,
            peak_bytes=int(peak_bytes),
            purpose=purpose,
            note=note,
        )
    )


def _update_peak(peak_map: Dict[str, int], key: str, value: int) -> None:
    if value <= 0:
        return
    peak_map[key] = max(int(value), int(peak_map.get(key, 0)))


def _memory_access_energy_nj(tier: str, is_read: bool, hw_config: Any) -> Tuple[float, str, int]:
    if tier == DRAM:
        dram = getattr(hw_config, "dram", None)
        block = max(1, int(getattr(dram, "block_size_bytes", 64) or 64)) if dram is not None else 64
        if dram is None:
            return 0.0, "dram_unconfigured", block
        energy = getattr(dram, "read_energy_nj_per_access", None) if is_read else getattr(dram, "write_energy_nj_per_access", None)
        if energy is not None:
            return float(energy), "dramsim3_per_access", block
        bw = max(1, int(getattr(dram, "bandwidth_bytes_per_cycle", 32) or 32))
        clk = float(getattr(hw_config, "clock_period", 1.0) or 1.0)
        power_mw = float(getattr(dram, "read_power_mw", 50.0) if is_read else getattr(dram, "write_power_mw", 50.0) or 50.0)
        cycles = max(1, math.ceil(block / bw))
        return power_mw * cycles * clk / 1000.0, "roofline_power_proxy", block
    if tier == SRAM:
        buf = getattr(hw_config, "inter_layer_buffer", None)
        block = max(1, int(getattr(buf, "block_size_bytes", 64) or 64)) if buf is not None else 64
        if buf is None:
            return 0.0, "sram_unconfigured", block
        energy = getattr(buf, "read_energy_nj_per_access", None) if is_read else getattr(buf, "write_energy_nj_per_access", None)
        if energy is not None:
            return float(energy), "cacti_inter_layer_proxy", block
        bw = max(1, int(getattr(buf, "bandwidth_bytes_per_cycle", 256) or 256))
        clk = float(getattr(hw_config, "clock_period", 1.0) or 1.0)
        power_mw = float(getattr(buf, "read_power_mw", 5.0) if is_read else getattr(buf, "write_power_mw", 5.0) or 5.0)
        cycles = max(1, math.ceil(block / bw))
        return power_mw * cycles * clk / 1000.0, "roofline_power_proxy", block
    return 0.0, "non_memory_tier", 1


def _estimate_record_energy_nj(record: MemoryContractRecord, hw_config: Any) -> Tuple[float, str, str]:
    total = 0.0
    models: List[str] = []
    if record.src_tier in (DRAM, SRAM):
        e, model, block = _memory_access_energy_nj(record.src_tier, True, hw_config)
        if e > 0:
            total += math.ceil(record.bytes / max(1, block)) * e
        models.append(f"{record.src_tier.lower()}_read:{model}")
    if record.dst_tier in (DRAM, SRAM):
        e, model, block = _memory_access_energy_nj(record.dst_tier, False, hw_config)
        if e > 0:
            total += math.ceil(record.bytes / max(1, block)) * e
        models.append(f"{record.dst_tier.lower()}_write:{model}")
    scope = "static_proxy_per_transaction_not_runtime_total"
    return float(total), ";".join(models) if models else "no_memory_energy", scope


def _build_buffer_relationships(hw_config: Any) -> List[BufferRelationship]:
    pipeline_mode = "without_pipeline"
    double_buffering = False
    slots = 1
    buf = getattr(hw_config, "inter_layer_buffer", None)
    if buf is not None:
        pipeline_mode = str(getattr(buf, "pipeline_mode", "without_pipeline") or "without_pipeline")
        double_buffering = bool(getattr(buf, "double_buffering", False))
        slots = int(getattr(buf, "buffer_slots", 0) or 0)
        if slots <= 0:
            slots = 2 if (pipeline_mode == "pipeline" and double_buffering) else 1
    return [
        BufferRelationship("xyz_sram_buffer", "point_cloud_frontend", "xyz/query/key cache", "shared across FPS/KNN distance generation", "serialized by distance generator front-end when same bank/port is reused", "stage boundary", "xyz_sram_shared", "can prefetch next query tile while previous distance batch computes when bank budget allows", "Shared source for baseline dist2engine and acc SRAM-CIM array"),
        BufferRelationship("feature_sram_buffer", "layer_intermediate", "persistent layer input/output activations", "shared across layers", f"exclusive per slot; slots={slots}", "layer boundary", "feature_sram_slot_<n>", f"{pipeline_mode}; double_buffering={double_buffering}", "Without pipeline a single slot is reused; with double buffering adjacent layers alternate slots"),
        BufferRelationship("weight_sram_buffer", "weight_staging", "Conv/Linear weight tile buffer", "shared across Conv/Linear ops over time", "exclusive within a layer's active weight staging window", "operator boundary", "weight_sram_shared", "prefetchable ahead of compute; currently not overlapped globally", "Weights still sourced from DRAM in current runtime path"),
        BufferRelationship("dist_sram_buffer", "distance_path", "shared distance vector buffer for FPS/KNN", "shared physical SRAM reused by FPS and KNN", "FPS and KNN are mutually exclusive in one GAM stage; sorter read vs writer must serialize if same bank", "query/iteration boundary", "dist_sram_shared", "base_v1 full-vector; base_v2/acc stream 64-distance batches", "Physical SRAM shared, but FPS uses extra min-distance table and KNN consumes current distance vector directly"),
        BufferRelationship("fps_min_distance_table_sram", "distance_path", "FPS historical minimum distance table", "FPS-only logical region", "exclusive with FPS update and FPS sorter read on same region", "FPS iteration boundary", "fps_min_table_shared", "updated every sampled-center iteration; can ping-pong with dist_sram in future", "Needed only by FPS"),
        BufferRelationship("sort_cache_sram", "distance_path", "streaming sorter cache for base_v2/acc", "shared by FPS/KNN streaming sorters over time", "exclusive between FPS and KNN sorter cache use", "64-distance batch boundary", "sort_cache_shared", "inactive in base_v1; active in base_v2/acc streaming sort", "Stores previous batch top candidates before merge with next batch"),
        BufferRelationship("nonlinear_workspace_sram", "nonlinear_workspace", "AffineNorm/SPool/Pool/KAN temporary tensors", "shared across nonlinear ops when lifetimes do not overlap", "exclusive across ops within the same layer execution window", "operator boundary", "nonlinear_workspace_shared", "can be partitioned later into GAM/LFP/GFP workspaces", "Current contract still exports finer names such as spool_workspace_sram and kan_workspace_sram"),
        BufferRelationship("state_sram_buffer", "operator_local_state", "topk/index/psum/norm-reduce small state SRAM", "shared over time by operator-local stateful micro-steps", "exclusive per operator instance or batch window", "operator or batch boundary", "operator_local_state_sram", "not double-buffered by default; can be folded into dedicated local SRAM later", "Distance accumulation internal state is not modeled as a standalone buffer; other small state is grouped under operator-local SRAM"),
    ]


def _emit_layer_io_records(
    records: List[MemoryContractRecord],
    layer_name: str,
    input_bytes: int,
    output_bytes: int,
    is_first_layer: bool,
    is_last_layer: bool,
    p: MemoryContractPolicy,
) -> int:
    peak = 0

    if is_first_layer and input_bytes > 0:
        _append_record(
            records,
            layer_name=layer_name,
            op_name="layer_input",
            tensor_name="point_cloud_input",
            src_tier=DRAM,
            src_name="offchip_dram",
            dst_tier=SRAM,
            dst_name="xyz_sram_buffer",
            nbytes=input_bytes,
            note="Phase1 initial point-cloud load",
        )

    if (not is_first_layer) and input_bytes > 0:
        onchip = min(input_bytes, p.inter_layer_sram_bytes)
        spill = max(0, input_bytes - onchip)
        peak = max(peak, onchip)
        _append_record(
            records,
            layer_name=layer_name,
            op_name="layer_input",
            tensor_name="inter_layer_activation",
            src_tier=SRAM,
            src_name="inter_layer_sram",
            dst_tier=COMPUTE,
            dst_name="compute_fabric",
            nbytes=onchip,
            note="Layer input read from inter-layer SRAM",
        )
        _append_record(
            records,
            layer_name=layer_name,
            op_name="layer_input",
            tensor_name="inter_layer_spill",
            src_tier=DRAM,
            src_name="offchip_dram",
            dst_tier=COMPUTE,
            dst_name="compute_fabric",
            nbytes=spill,
            note="Spill read when layer input exceeds inter-layer SRAM capacity",
        )

    if (not is_last_layer) and output_bytes > 0:
        onchip = min(output_bytes, p.inter_layer_sram_bytes)
        spill = max(0, output_bytes - onchip)
        peak = max(peak, onchip)
        _append_record(
            records,
            layer_name=layer_name,
            op_name="layer_output",
            tensor_name="inter_layer_activation",
            src_tier=COMPUTE,
            src_name="compute_fabric",
            dst_tier=SRAM,
            dst_name="inter_layer_sram",
            nbytes=onchip,
            note="Layer output write to inter-layer SRAM",
        )
        _append_record(
            records,
            layer_name=layer_name,
            op_name="layer_output",
            tensor_name="inter_layer_spill",
            src_tier=COMPUTE,
            src_name="compute_fabric",
            dst_tier=DRAM,
            dst_name="offchip_dram",
            nbytes=spill,
            note="Spill write when layer output exceeds inter-layer SRAM capacity",
        )

    return peak


def _tensor_storage_by_strategy(
    strategy: str,
    tensor_bytes: int,
    sram_capacity_bytes: int,
    sram_name: str,
    reg_name: str,
) -> Tuple[str, str]:
    if strategy == "materialize":
        if tensor_bytes <= sram_capacity_bytes:
            return SRAM, sram_name
        return DRAM, "offchip_dram"
    return REGFILE, reg_name


def _infer_points_channels(shape: Any) -> Tuple[int, int]:
    in_shape = getattr(shape, "input_shape", None)
    if isinstance(in_shape, tuple) and len(in_shape) >= 2:
        return _safe_int(in_shape[0], 1), _safe_int(in_shape[-1], 1)
    if isinstance(in_shape, tuple) and len(in_shape) == 1:
        return 1, _safe_int(in_shape[0], 1)
    return 1, 1


def _infer_points_channels_from_shape(shape: Any) -> Tuple[int, int]:
    input_shape = getattr(shape, "input_shape", None)
    output_shape = getattr(shape, "output_shape", None)
    shape_ref = output_shape if isinstance(output_shape, tuple) and output_shape else input_shape
    if isinstance(shape_ref, tuple) and len(shape_ref) >= 2:
        return _safe_int(shape_ref[0], 1), _safe_int(shape_ref[-1], 1)
    if isinstance(shape_ref, tuple) and len(shape_ref) == 1:
        return 1, _safe_int(shape_ref[0], 1)
    return 1, 1


def _infer_points_channels_for_op(shape: Any, params: Dict[str, Any], feature_key: str = "num_features") -> Tuple[int, int]:
    shape_points, shape_channels = _infer_points_channels_from_shape(shape)
    points = _safe_int(
        params.get("num_points", params.get("query_points", params.get("output_points", shape_points))),
        shape_points,
    )
    channels = _safe_int(
        params.get(feature_key, params.get("out_channels", params.get("out_features", params.get("in_channels", params.get("in_features", shape_channels))))),
        shape_channels,
    )
    return max(1, points), max(1, channels)


def _shared_feature_buffer_bytes(feature_residency_plan: List[Dict[str, Any]]) -> int:
    peak = 0
    for row in feature_residency_plan:
        peak = max(peak, _safe_int(row.get("input_sram_bytes", 0), 0), _safe_int(row.get("output_sram_bytes", 0), 0))
    return peak


def _emit_conv_linear_contract(
    records: List[MemoryContractRecord],
    buffer_requirements: List[BufferRequirement],
    buffer_peaks: Dict[str, int],
    layer_name: str,
    op_name: str,
    params: Dict[str, Any],
    shape: Any,
    p: MemoryContractPolicy,
) -> None:
    points, _ = _infer_points_channels_from_shape(shape)
    if op_name == "conv1d":
        in_dim = _safe_int(params.get("in_channels", 1), 1)
        out_dim = _safe_int(params.get("out_channels", 1), 1)
        kernel = _safe_int(params.get("kernel_size", 1), 1)
        vectors = max(1, _safe_int(params.get("num_points", points), points))
        weight_bytes = in_dim * out_dim * kernel * p.bytes_per_element
        purpose = "Conv1d weight staging for systolic/CIM engine"
        psum_name = "conv_psum_reg"
        engine = "conv_engine"
    else:
        in_dim = _safe_int(params.get("in_features", 1), 1)
        out_dim = _safe_int(params.get("out_features", 1), 1)
        vectors = max(1, _safe_int(params.get("tokens", params.get("num_points", points)), points))
        weight_bytes = in_dim * out_dim * p.bytes_per_element
        purpose = "Linear weight staging for systolic/CIM engine"
        psum_name = "linear_psum_reg"
        engine = "linear_engine"
    input_bytes = max(1, vectors) * max(1, in_dim) * p.bytes_per_element
    output_bytes = max(1, vectors) * max(1, out_dim) * p.bytes_per_element
    psum_bytes = output_bytes
    feature_stage_bytes = min(input_bytes, p.inter_layer_sram_bytes)

    _append_record(records, layer_name, op_name, "input_activation", SRAM, "feature_sram_buffer", COMPUTE, engine, input_bytes, "Feature activations consumed from persistent feature SRAM or prior refill")
    _append_record(records, layer_name, op_name, "weights", DRAM, "offchip_dram", SRAM, "weight_sram_buffer", weight_bytes, purpose)
    _append_record(records, layer_name, op_name, "weights_staged", SRAM, "weight_sram_buffer", COMPUTE, engine, weight_bytes, "Weights streamed from local SRAM to compute engine")
    _append_record(records, layer_name, op_name, "partial_sums", COMPUTE, engine, REGFILE, psum_name, psum_bytes, "Partial sums accumulated before output commit")
    _append_record(records, layer_name, op_name, "output_activation", REGFILE, psum_name, SRAM, "feature_sram_buffer", output_bytes, "Operator output committed back to persistent feature SRAM")

    _append_buffer_requirement(buffer_requirements, layer_name, op_name, SRAM, "feature_sram_buffer", feature_stage_bytes, "Shared persistent feature SRAM window for Conv/Linear input/output staging")
    _append_buffer_requirement(buffer_requirements, layer_name, op_name, SRAM, "weight_sram_buffer", weight_bytes, purpose)
    _append_buffer_requirement(buffer_requirements, layer_name, op_name, REGFILE, psum_name, psum_bytes, "Output-channel partial sums before write-back")
    _update_peak(buffer_peaks, "feature_sram_buffer", feature_stage_bytes)
    _update_peak(buffer_peaks, "weight_sram_buffer", weight_bytes)
    _update_peak(buffer_peaks, psum_name, psum_bytes)


def _emit_norm_contract(
    records: List[MemoryContractRecord],
    buffer_requirements: List[BufferRequirement],
    buffer_peaks: Dict[str, int],
    layer_name: str,
    op_name: str,
    params: Dict[str, Any],
    shape: Any,
    p: MemoryContractPolicy,
) -> None:
    points, channels = _infer_points_channels_for_op(shape, params, feature_key="num_features")
    tensor_bytes = max(1, points) * max(1, channels) * p.bytes_per_element
    if op_name == "batchnorm1d":
        param_bytes = 4 * max(1, channels) * p.bytes_per_element
        reduce_bytes = 2 * max(1, channels) * p.bytes_per_element
        buffer_name = "bn_param_sram"
        reduce_name = "bn_reduce_reg"
        engine = "bn_engine"
        purpose = "BN affine/stat parameter staging"
    elif op_name == "bn":
        param_bytes = 4 * max(1, channels) * p.bytes_per_element
        reduce_bytes = 2 * max(1, channels) * p.bytes_per_element
        buffer_name = "bn_param_sram"
        reduce_name = "bn_reduce_reg"
        engine = "bn_engine"
        purpose = "BN affine/stat parameter staging"
    elif op_name == "layernorm":
        param_bytes = 2 * max(1, channels) * p.bytes_per_element
        reduce_bytes = 2 * max(1, channels) * p.bytes_per_element
        buffer_name = "layernorm_param_sram"
        reduce_name = "layernorm_reduce_reg"
        engine = "layernorm_engine"
        purpose = "LayerNorm gamma/beta staging"
    else:
        param_bytes = 2 * max(1, channels) * p.bytes_per_element
        reduce_bytes = 2 * max(1, channels) * p.bytes_per_element
        buffer_name = "affine_norm_param_sram"
        reduce_name = "affine_norm_reduce_reg"
        engine = "affine_norm_engine"
        purpose = "AffineNorm alpha/beta staging"

    _append_record(records, layer_name, op_name, "input_activation", SRAM, "feature_sram_buffer", COMPUTE, engine, tensor_bytes, "Normalization input activations")
    _append_record(records, layer_name, op_name, "norm_parameters", DRAM, "offchip_dram", SRAM, buffer_name, param_bytes, purpose)
    _append_record(records, layer_name, op_name, "norm_parameters_staged", SRAM, buffer_name, COMPUTE, engine, param_bytes, "Normalization parameters streamed from local SRAM")
    _append_record(records, layer_name, op_name, "reduction_state", COMPUTE, engine, REGFILE, reduce_name, reduce_bytes, "Mean/variance or affine reduction state")
    _append_record(records, layer_name, op_name, "output_activation", COMPUTE, engine, SRAM, "feature_sram_buffer", tensor_bytes, "Normalized activations written back to persistent feature SRAM")

    _append_buffer_requirement(buffer_requirements, layer_name, op_name, SRAM, buffer_name, param_bytes, purpose)
    _append_buffer_requirement(buffer_requirements, layer_name, op_name, REGFILE, reduce_name, reduce_bytes, "Normalization reduction state")
    _update_peak(buffer_peaks, buffer_name, param_bytes)
    _update_peak(buffer_peaks, reduce_name, reduce_bytes)


def _emit_relu_contract(
    records: List[MemoryContractRecord],
    buffer_requirements: List[BufferRequirement],
    buffer_peaks: Dict[str, int],
    layer_name: str,
    params: Dict[str, Any],
    shape: Any,
    p: MemoryContractPolicy,
) -> None:
    points, channels = _infer_points_channels_for_op(shape, params, feature_key="num_features")
    tensor_bytes = max(1, points) * max(1, channels) * p.bytes_per_element
    lane_bytes = max(1, channels) * p.bytes_per_element
    _append_record(records, layer_name, "relu", "input_activation", SRAM, "feature_sram_buffer", COMPUTE, "relu_engine", tensor_bytes, "Streaming activation read for ReLU")
    _append_record(records, layer_name, "relu", "activation_lane", COMPUTE, "relu_engine", REGFILE, "relu_lane_reg", lane_bytes, "Per-lane compare/max state")
    _append_record(records, layer_name, "relu", "output_activation", COMPUTE, "relu_engine", SRAM, "feature_sram_buffer", tensor_bytes, "ReLU output written back to feature SRAM")
    _append_buffer_requirement(buffer_requirements, layer_name, "relu", REGFILE, "relu_lane_reg", lane_bytes, "Streaming activation lane register")
    _update_peak(buffer_peaks, "relu_lane_reg", lane_bytes)


def _emit_pool_contract(
    records: List[MemoryContractRecord],
    buffer_requirements: List[BufferRequirement],
    buffer_peaks: Dict[str, int],
    layer_name: str,
    op_name: str,
    params: Dict[str, Any],
    shape: Any,
    p: MemoryContractPolicy,
) -> None:
    points, channels = _infer_points_channels_for_op(shape, params, feature_key="num_features")
    if op_name == "s-pool":
        k = _safe_int(params.get("k_neighbors", 1), 1)
        in_channels = _safe_int(params.get("in_channels", channels), channels)
        workspace_bytes = max(1, points) * max(1, k) * max(1, in_channels) * p.bytes_per_element
        output_bytes = max(1, points) * max(1, in_channels) * p.bytes_per_element
        workspace_name = "spool_workspace_sram"
        reduce_name = "spool_reduce_reg"
        engine = "spool_engine"
        purpose = "Neighbor window workspace for softmax pooling"
    else:
        num_points = _safe_int(params.get("num_points", points), points)
        num_features = _safe_int(params.get("num_features", channels), channels)
        workspace_bytes = max(1, num_points) * max(1, num_features) * p.bytes_per_element
        output_bytes = max(1, num_features) * p.bytes_per_element
        workspace_name = "adaptive_pool_workspace_sram"
        reduce_name = "adaptive_pool_reduce_reg"
        engine = "adaptive_pool_engine"
        purpose = "Reduction window workspace for adaptive max pool"

    _append_record(records, layer_name, op_name, "pool_input", SRAM, "feature_sram_buffer", SRAM, workspace_name, workspace_bytes, "Pool input window materialized before reduction")
    _append_record(records, layer_name, op_name, "pool_workspace", SRAM, workspace_name, COMPUTE, engine, workspace_bytes, "Pool workspace streamed to reduction engine")
    _append_record(records, layer_name, op_name, "pool_reduce_state", COMPUTE, engine, REGFILE, reduce_name, output_bytes, "Per-channel max/accumulator state")
    _append_record(records, layer_name, op_name, "pool_output", REGFILE, reduce_name, SRAM, workspace_name, output_bytes, "Pool result kept on-chip for downstream operator or layer output")

    _append_buffer_requirement(buffer_requirements, layer_name, op_name, SRAM, workspace_name, workspace_bytes, purpose)
    _append_buffer_requirement(buffer_requirements, layer_name, op_name, REGFILE, reduce_name, output_bytes, "Reduction state for pooling")
    _update_peak(buffer_peaks, workspace_name, workspace_bytes)
    _update_peak(buffer_peaks, reduce_name, output_bytes)


def _emit_kan_contract(
    records: List[MemoryContractRecord],
    buffer_requirements: List[BufferRequirement],
    buffer_peaks: Dict[str, int],
    layer_name: str,
    params: Dict[str, Any],
    shape: Any,
    p: MemoryContractPolicy,
) -> None:
    points = _safe_int(params.get("num_points", _infer_points_channels_from_shape(shape)[0]), 1)
    out_features = _safe_int(params.get("out_features", params.get("in_features", _infer_points_channels_from_shape(shape)[1])), 1)
    tensor_bytes = max(1, points) * max(1, out_features) * p.bytes_per_element
    coeff_bytes = max(1, out_features) * 10 * p.bytes_per_element
    _append_record(records, layer_name, "kan", "input_activation", SRAM, "feature_sram_buffer", COMPUTE, "kan_engine", tensor_bytes, "KAN/KAT input activations")
    _append_record(records, layer_name, "kan", "activation_coefficients", DRAM, "offchip_dram", SRAM, "kan_param_sram", coeff_bytes, "Rational activation coefficients staged on-chip")
    _append_record(records, layer_name, "kan", "activation_coefficients_staged", SRAM, "kan_param_sram", COMPUTE, "kan_engine", coeff_bytes, "KAN parameters streamed from local SRAM")
    _append_record(records, layer_name, "kan", "kan_workspace", COMPUTE, "kan_engine", SRAM, "kan_workspace_sram", tensor_bytes, "KAN nonlinear output workspace")
    _append_buffer_requirement(buffer_requirements, layer_name, "kan", SRAM, "kan_param_sram", coeff_bytes, "KAN rational/LUT coefficient buffer")
    _append_buffer_requirement(buffer_requirements, layer_name, "kan", SRAM, "kan_workspace_sram", tensor_bytes, "KAN activation workspace/output hold")
    _update_peak(buffer_peaks, "kan_param_sram", coeff_bytes)
    _update_peak(buffer_peaks, "kan_workspace_sram", tensor_bytes)

def _emit_lfp_gfp_intermediate_records(
    records: List[MemoryContractRecord],
    layer_name: str,
    layer_type: str,
    points: int,
    channels: int,
    p: MemoryContractPolicy,
) -> int:
    t = layer_type.lower()
    if t not in ("lfp", "stage"):
        return 0

    strategy = p.lfp_tensor_strategy if t == "lfp" else p.gfp_tensor_strategy
    cap = p.lfp_workspace_sram_bytes if t == "lfp" else p.gfp_workspace_sram_bytes
    ws_name = "lfp_workspace_sram" if t == "lfp" else "gfp_workspace_sram"

    expanded_bytes = max(1, points) * max(1, channels) * p.bytes_per_element
    pool_bytes = max(1, channels) * p.bytes_per_element

    tier_a, name_a = _tensor_storage_by_strategy(strategy, expanded_bytes, cap, ws_name, f"{t}_local_path_reg")
    tier_b, name_b = _tensor_storage_by_strategy(strategy, expanded_bytes, cap, ws_name, f"{t}_center_path_reg")
    tier_p, name_p = _tensor_storage_by_strategy(strategy, pool_bytes, cap, ws_name, f"{t}_pool_path_reg")

    _append_record(
        records, layer_name, f"{t}_path", "local_path_tensor",
        COMPUTE, "compute_fabric", tier_a, name_a, expanded_bytes,
        f"{t} local-path intermediate ({strategy})",
    )
    _append_record(
        records, layer_name, f"{t}_path", "center_path_tensor",
        COMPUTE, "compute_fabric", tier_b, name_b, expanded_bytes,
        f"{t} center-path intermediate ({strategy})",
    )
    _append_record(
        records, layer_name, f"{t}_path", "pooled_local_tensor",
        COMPUTE, "compute_fabric", tier_p, name_p, pool_bytes,
        f"{t} pooled tensor ({strategy})",
    )
    return max(expanded_bytes, pool_bytes)


def _emit_add_records(
    records: List[MemoryContractRecord],
    layer_name: str,
    layer_type: str,
    params: Dict[str, Any],
    p: MemoryContractPolicy,
) -> None:
    nfeat = _safe_int(params.get("num_features", 1), 1)
    npts = _safe_int(params.get("num_points", 1), 1)
    tensor_bytes = max(1, nfeat) * max(1, npts) * p.bytes_per_element
    residual = bool(params.get("residual", False))

    lt = layer_type.lower()
    if lt == "lfp":
        strategy = p.lfp_tensor_strategy
        cap = p.lfp_workspace_sram_bytes
        sram_name = "lfp_workspace_sram"
    elif lt == "stage":
        strategy = p.gfp_tensor_strategy
        cap = p.gfp_workspace_sram_bytes
        sram_name = "gfp_workspace_sram"
    else:
        strategy = "stream_through"
        cap = 0
        sram_name = "generic_workspace_sram"

    src_tier, src_name = _tensor_storage_by_strategy(strategy, tensor_bytes, cap, sram_name, f"{lt}_add_input_reg")
    out_tier, out_name = _tensor_storage_by_strategy(strategy, tensor_bytes, cap, sram_name, f"{lt}_add_output_reg")

    _append_record(
        records, layer_name, "add", "main_tensor_read",
        src_tier, src_name, COMPUTE, "add_unit", tensor_bytes,
        f"add main input read ({strategy})",
    )
    _append_record(
        records, layer_name, "add", "residual_tensor_read",
        src_tier, src_name, COMPUTE, "add_unit", tensor_bytes,
        f"add residual input read (residual={residual}, {strategy})",
    )
    _append_record(
        records, layer_name, "add", "add_output_write",
        COMPUTE, "add_unit", out_tier, out_name, tensor_bytes,
        f"add output write ({strategy})",
    )


def analyze_memory_contract(
    layers: List[Any],
    activation_shapes: List[Any],
    hw_config: Any,
    policy: Optional[MemoryContractPolicy] = None,
) -> MemoryContractReport:
    """
    Build a conservative memory contract for full PointKAN inference.

    增强点：
    - add 显式 memory term: main_tensor_read + residual_tensor_read + add_output_write
    - LFP/GFP 中间张量: stream-through vs materialize 策略
    - FPS/KNN 距离路径: 共享 dist_sram_buffer + FPS min_distance_table + 可选流式 sort_cache_sram
    """
    p = policy or MemoryContractPolicy()

    inter_buf = getattr(hw_config, "inter_layer_buffer", None)
    if inter_buf is not None:
        p.inter_layer_sram_bytes = _safe_int(
            getattr(inter_buf, "size_bytes", p.inter_layer_sram_bytes),
            p.inter_layer_sram_bytes,
        )

    records: List[MemoryContractRecord] = []
    buffer_requirements: List[BufferRequirement] = []
    buffer_peak_bytes: Dict[str, int] = {}
    peak_inter_layer = 0
    peak_dist_sram_shared = 0
    peak_fps_min_table = 0
    peak_sort_cache = 0
    peak_gam_workspace = 0
    peak_lfp_workspace = 0
    peak_gfp_workspace = 0

    for i, layer in enumerate(layers):
        layer_name = str(getattr(layer, "name", f"layer_{i}") or f"layer_{i}")
        layer_type = str(getattr(layer, "layer_type", "") or "")
        shape = activation_shapes[i] if i < len(activation_shapes) else None
        input_bytes = _safe_int(getattr(shape, "input_bytes", 0), 0) if shape is not None else 0
        output_bytes = _safe_int(getattr(shape, "output_bytes", 0), 0) if shape is not None else 0

        peak_inter_layer = max(
            peak_inter_layer,
            _emit_layer_io_records(
                records=records,
                layer_name=layer_name,
                input_bytes=input_bytes,
                output_bytes=output_bytes,
                is_first_layer=(i == 0),
                is_last_layer=(i == len(layers) - 1),
                p=p,
            ),
        )

        # LFP/GFP intermediate-expanded tensors
        points, channels = _infer_points_channels(shape) if shape is not None else (1, 1)
        ws_peak = _emit_lfp_gfp_intermediate_records(records, layer_name, layer_type, points, channels, p)
        if layer_type.lower() == "lfp":
            peak_lfp_workspace = max(peak_lfp_workspace, ws_peak)
        if layer_type.lower() == "stage":
            peak_gfp_workspace = max(peak_gfp_workspace, ws_peak)

        for op in list(getattr(layer, "operators", []) or []):
            op_name = _op_display(op)
            params = getattr(op, "params", {}) or {}

            if op_name in ("conv1d", "linear"):
                _emit_conv_linear_contract(records, buffer_requirements, buffer_peak_bytes, layer_name, op_name, params, shape, p)

            elif op_name in ("bn", "batchnorm1d", "layernorm", "affine_norm"):
                _emit_norm_contract(records, buffer_requirements, buffer_peak_bytes, layer_name, op_name, params, shape, p)
                if op_name == "affine_norm":
                    s = _safe_int(params.get("num_points", 1), 1)
                    k = _safe_int(params.get("k_neighbors", 1), 1)
                    d = _safe_int(params.get("num_features", 1), 1)
                    use_xyz = bool(params.get("use_xyz", False))
                    coord_dims = _safe_int(params.get("coord_dims", p.coord_dims), p.coord_dims)
                    d_eff = d + (coord_dims if use_xyz else 0)
                    affine_workspace_bytes = max(1, s) * max(1, k) * max(1, d_eff) * p.bytes_per_element
                    _append_buffer_requirement(buffer_requirements, layer_name, op_name, SRAM, "affine_norm_workspace_sram", affine_workspace_bytes, "Neighborhood tensor workspace for GAM affine normalization")
                    _update_peak(buffer_peak_bytes, "affine_norm_workspace_sram", affine_workspace_bytes)
                    peak_gam_workspace = max(peak_gam_workspace, affine_workspace_bytes)

            elif op_name == "relu":
                _emit_relu_contract(records, buffer_requirements, buffer_peak_bytes, layer_name, params, shape, p)

            elif op_name in ("s-pool", "adaptive_max_pool1d"):
                _emit_pool_contract(records, buffer_requirements, buffer_peak_bytes, layer_name, op_name, params, shape, p)
                if op_name == "s-pool":
                    points_sp, channels_sp = _infer_points_channels_for_op(shape, params, feature_key="in_channels")
                    k_sp = _safe_int(params.get("k_neighbors", 1), 1)
                    peak_gam_workspace = max(peak_gam_workspace, points_sp * channels_sp * k_sp * p.bytes_per_element)
                elif layer_type.lower() == "lfp":
                    pool_points, pool_channels = _infer_points_channels_for_op(shape, params)
                    peak_lfp_workspace = max(peak_lfp_workspace, pool_points * pool_channels * p.bytes_per_element)

            elif op_name == "kan":
                _emit_kan_contract(records, buffer_requirements, buffer_peak_bytes, layer_name, params, shape, p)
                kan_points = _safe_int(params.get("num_points", points), points)
                kan_feats = _safe_int(params.get("out_features", params.get("in_features", channels)), channels)
                if layer_type.lower() == "lfp":
                    peak_lfp_workspace = max(peak_lfp_workspace, kan_points * kan_feats * p.bytes_per_element)
                elif layer_type.lower() == "stage":
                    peak_gfp_workspace = max(peak_gfp_workspace, kan_points * kan_feats * p.bytes_per_element)

            elif op_name == "fps":
                n_ref = _safe_int(params.get("input_points", 1024), 1024)
                n_out = _safe_int(params.get("output_points", 512), 512)
                query_bytes = n_out * p.point_xyz_bytes
                scan_bytes = n_out * n_ref * p.point_xyz_bytes
                dist_vec_bytes = n_ref * p.distance_bytes
                sort_rw_bytes = n_out * dist_vec_bytes
                fps_idx_bytes = n_out * p.index_bytes
                sort_cache_bytes = max(1, p.distance_bytes + p.index_bytes)
                sort_mode = "base_v1_full_vector" if not (getattr(hw_config, "is_baseline_v2", False) or getattr(hw_config, "is_acc", False)) else ("base_v2_streaming" if getattr(hw_config, "is_baseline_v2", False) else "acc_streaming")

                peak_dist_sram_shared = max(peak_dist_sram_shared, dist_vec_bytes)
                peak_fps_min_table = max(peak_fps_min_table, dist_vec_bytes)
                if sort_mode != "base_v1_full_vector":
                    peak_sort_cache = max(peak_sort_cache, sort_cache_bytes)

                _append_record(
                    records, layer_name, "fps", "query_points",
                    DRAM, "offchip_dram", COMPUTE, "fps_engine",
                    query_bytes, "Phase2 query-point reads"
                )

                if p.model_cim_staging:
                    _append_record(
                        records, layer_name, "fps", "cim_array_preload",
                        SRAM, "xyz_sram_buffer", SRAM, "sram_cim_array",
                        n_ref * p.point_xyz_bytes,
                        "Load key point cloud into distance generator array",
                    )
                    _append_record(
                        records, layer_name, "fps", "distance_vector_write",
                        COMPUTE, "distance_generator", SRAM, "dist_sram_buffer",
                        sort_rw_bytes,
                        "Distance results written to shared dist SRAM; shift-add local state already accounted inside engine",
                    )
                else:
                    _append_record(
                        records, layer_name, "fps", "key_scan_xyz",
                        SRAM, "xyz_sram_buffer", COMPUTE, "fps_engine",
                        scan_bytes, "Distance scan against key set"
                    )
                    if p.materialize_fps_distance_table:
                        _append_record(
                            records, layer_name, "fps", "distance_vector_write",
                            COMPUTE, "fps_engine", SRAM, "dist_sram_buffer",
                            sort_rw_bytes, "Per-iteration distance vector writes into shared dist SRAM"
                        )

                _append_record(
                    records, layer_name, "fps", "min_distance_table_read",
                    SRAM, "fps_min_distance_table_sram", COMPUTE, "fps_min_update",
                    sort_rw_bytes, "Read historical min-distance table before update"
                )
                _append_record(
                    records, layer_name, "fps", "min_distance_table_write",
                    COMPUTE, "fps_min_update", SRAM, "fps_min_distance_table_sram",
                    sort_rw_bytes, "Update FPS historical min-distance table"
                )
                _append_record(
                    records, layer_name, "fps", "sorter_input_read",
                    SRAM, "fps_min_distance_table_sram", COMPUTE, "fps_sorter",
                    sort_rw_bytes, f"Sorter consumes FPS min-distance table ({sort_mode})"
                )
                if sort_mode != "base_v1_full_vector":
                    _append_record(
                        records, layer_name, "fps", "sort_cache_write",
                        COMPUTE, "fps_sorter", SRAM, "sort_cache_sram",
                        n_out * sort_cache_bytes, "Streaming FPS local-max cache across 64-distance batches"
                    )
                    _append_buffer_requirement(buffer_requirements, layer_name, "fps", SRAM, "sort_cache_sram", sort_cache_bytes, f"FPS streaming sorter cache ({sort_mode})")

                _append_buffer_requirement(buffer_requirements, layer_name, "fps", SRAM, "dist_sram_buffer", dist_vec_bytes, "Shared distance SRAM reused by FPS/KNN")
                _append_buffer_requirement(buffer_requirements, layer_name, "fps", SRAM, "fps_min_distance_table_sram", dist_vec_bytes, "Historical minimum distance table for FPS")
                _update_peak(buffer_peak_bytes, "dist_sram_buffer", dist_vec_bytes)
                _update_peak(buffer_peak_bytes, "fps_min_distance_table_sram", dist_vec_bytes)
                if sort_mode != "base_v1_full_vector":
                    _update_peak(buffer_peak_bytes, "sort_cache_sram", sort_cache_bytes)

                _append_record(
                    records, layer_name, "fps", "fps_index",
                    COMPUTE, "fps_sorter", REGFILE, "fps_index_regfile",
                    fps_idx_bytes, "Sampled center indices"
                )

            elif op_name == "knn":
                n_ref = _safe_int(params.get("input_points", 1024), 1024)
                n_query = _safe_int(params.get("query_points", 512), 512)
                k = _safe_int(params.get("k", 24), 24)
                query_bytes = n_query * p.point_xyz_bytes
                scan_bytes = n_query * n_ref * p.point_xyz_bytes
                dist_vec_bytes = n_ref * p.distance_bytes
                sort_rw_bytes = n_query * dist_vec_bytes
                topk_idx_bytes = n_query * k * p.index_bytes
                sort_cache_bytes = max(32, k) * (p.distance_bytes + p.index_bytes)
                sort_mode = "base_v1_full_vector" if not (getattr(hw_config, "is_baseline_v2", False) or getattr(hw_config, "is_acc", False)) else ("base_v2_streaming" if getattr(hw_config, "is_baseline_v2", False) else "acc_streaming")

                peak_dist_sram_shared = max(peak_dist_sram_shared, dist_vec_bytes)
                if sort_mode != "base_v1_full_vector":
                    peak_sort_cache = max(peak_sort_cache, sort_cache_bytes)

                _append_record(
                    records, layer_name, "knn", "query_points",
                    DRAM, "offchip_dram", COMPUTE, "knn_engine",
                    query_bytes, "Phase2 query-point reads"
                )

                if p.model_cim_staging:
                    _append_record(
                        records, layer_name, "knn", "cim_array_preload",
                        SRAM, "xyz_sram_buffer", SRAM, "sram_cim_array",
                        n_ref * p.point_xyz_bytes,
                        "Load key point cloud into distance generator array",
                    )
                    _append_record(
                        records, layer_name, "knn", "distance_vector_write",
                        COMPUTE, "distance_generator", SRAM, "dist_sram_buffer",
                        sort_rw_bytes,
                        "Distance results written to shared dist SRAM; shift-add local state already accounted inside engine",
                    )
                else:
                    _append_record(
                        records, layer_name, "knn", "key_scan_xyz",
                        SRAM, "xyz_sram_buffer", COMPUTE, "knn_engine",
                        scan_bytes, "Distance scan against key set"
                    )
                    if p.materialize_knn_distance_table:
                        _append_record(
                            records, layer_name, "knn", "distance_vector_write",
                            COMPUTE, "knn_engine", SRAM, "dist_sram_buffer",
                            sort_rw_bytes, "Per-query distance vector writes into shared dist SRAM"
                        )

                _append_record(
                    records, layer_name, "knn", "sorter_input_read",
                    SRAM, "dist_sram_buffer", COMPUTE, "knn_sorter",
                    sort_rw_bytes, f"Sorter consumes shared distance SRAM ({sort_mode})"
                )
                if sort_mode != "base_v1_full_vector":
                    _append_record(
                        records, layer_name, "knn", "sort_cache_write",
                        COMPUTE, "knn_sorter", SRAM, "sort_cache_sram",
                        n_query * sort_cache_bytes, "Streaming KNN top-k cache across 64-distance batches"
                    )
                    _append_buffer_requirement(buffer_requirements, layer_name, "knn", SRAM, "sort_cache_sram", sort_cache_bytes, f"KNN streaming sorter cache ({sort_mode})")

                _append_buffer_requirement(buffer_requirements, layer_name, "knn", SRAM, "dist_sram_buffer", dist_vec_bytes, "Shared distance SRAM reused by FPS/KNN")
                _update_peak(buffer_peak_bytes, "dist_sram_buffer", dist_vec_bytes)
                if sort_mode != "base_v1_full_vector":
                    _update_peak(buffer_peak_bytes, "sort_cache_sram", sort_cache_bytes)

                _append_record(
                    records, layer_name, "knn", "knn_topk_index",
                    COMPUTE, "knn_sorter", REGFILE, "knn_topk_regfile",
                    topk_idx_bytes, "Top-k neighbor indices"
                )

                gather_c = _safe_int(params.get("feature_channels", params.get("in_channels", 0)), 0)
                if gather_c <= 0:
                    if shape is not None:
                        _, c = _infer_points_channels(shape)
                        gather_c = c
                    else:
                        gather_c = 1

                neigh_bytes = n_query * k * gather_c * p.bytes_per_element
                peak_gam_workspace = max(peak_gam_workspace, neigh_bytes)
                if p.materialize_knn_neighbor_features:
                    if neigh_bytes <= p.gam_workspace_sram_bytes:
                        _append_record(
                            records, layer_name, "knn", "neighbor_features",
                            SRAM, "inter_layer_sram", SRAM, "knn_workspace_sram",
                            neigh_bytes, "Gathered neighbor features kept on-chip"
                        )
                    else:
                        _append_record(
                            records, layer_name, "knn", "neighbor_features_spill",
                            SRAM, "inter_layer_sram", DRAM, "offchip_dram",
                            neigh_bytes, "Gather workspace exceeds SRAM, spill to DRAM"
                        )

            elif op_name == "add":
                _emit_add_records(records, layer_name, layer_type, params, p)
                add_points, add_channels = _infer_points_channels_for_op(shape, params)
                add_tensor_bytes = max(1, add_points) * max(1, add_channels) * p.bytes_per_element
                _append_buffer_requirement(buffer_requirements, layer_name, op_name, REGFILE, "add_lane_reg", max(1, add_channels) * p.bytes_per_element, "Streaming element-wise add lane register")
                _update_peak(buffer_peak_bytes, "add_lane_reg", max(1, add_channels) * p.bytes_per_element)
                if layer_type.lower() == "lfp":
                    peak_lfp_workspace = max(peak_lfp_workspace, add_tensor_bytes)
                elif layer_type.lower() == "stage":
                    peak_gfp_workspace = max(peak_gfp_workspace, add_tensor_bytes)

    route_bytes_by_tier: Dict[Tuple[str, str], int] = {}
    route_bytes_detailed: Dict[Tuple[str, str], int] = {}
    per_layer_bytes: Dict[str, int] = {}
    for r in records:
        key_tier = (r.src_tier, r.dst_tier)
        key_detail = (f"{r.src_tier}:{r.src_name}", f"{r.dst_tier}:{r.dst_name}")
        route_bytes_by_tier[key_tier] = route_bytes_by_tier.get(key_tier, 0) + r.bytes
        route_bytes_detailed[key_detail] = route_bytes_detailed.get(key_detail, 0) + r.bytes
        per_layer_bytes[r.layer_name] = per_layer_bytes.get(r.layer_name, 0) + r.bytes

    feature_residency_plan = [entry.to_dict() for entry in build_feature_residency_plan(layers, activation_shapes, hw_config)]
    buffer_relationships = _build_buffer_relationships(hw_config)
    _update_peak(buffer_peak_bytes, "feature_sram_buffer", _shared_feature_buffer_bytes(feature_residency_plan))
    _update_peak(buffer_peak_bytes, "dist_sram_buffer", peak_dist_sram_shared)
    _update_peak(buffer_peak_bytes, "fps_min_distance_table_sram", peak_fps_min_table)
    _update_peak(buffer_peak_bytes, "sort_cache_sram", peak_sort_cache)
    _update_peak(buffer_peak_bytes, "gam_workspace_sram", peak_gam_workspace)
    _update_peak(buffer_peak_bytes, "lfp_workspace_sram", peak_lfp_workspace)
    _update_peak(buffer_peak_bytes, "gfp_workspace_sram", peak_gfp_workspace)

    suggested_sram_bytes = {
        "feature_sram_buffer_min": buffer_peak_bytes.get("feature_sram_buffer", peak_inter_layer),
        "dist_sram_buffer_min": peak_dist_sram_shared,
        "fps_min_distance_table_sram_min": peak_fps_min_table,
        "sort_cache_sram_min": peak_sort_cache,
        "gam_workspace_sram_min": peak_gam_workspace,
        "lfp_workspace_sram_min": peak_lfp_workspace,
        "gfp_workspace_sram_min": peak_gfp_workspace,
        "weight_sram_buffer_min": buffer_peak_bytes.get("weight_sram_buffer", 0),
        "norm_param_sram_min": max(
            buffer_peak_bytes.get("bn_param_sram", 0),
            buffer_peak_bytes.get("layernorm_param_sram", 0),
            buffer_peak_bytes.get("affine_norm_param_sram", 0),
        ),
        "pool_workspace_sram_min": max(
            buffer_peak_bytes.get("spool_workspace_sram", 0),
            buffer_peak_bytes.get("adaptive_pool_workspace_sram", 0),
        ),
        "kan_workspace_sram_min": buffer_peak_bytes.get("kan_workspace_sram", 0),
    }

    return MemoryContractReport(
        records=records,
        route_bytes_by_tier=route_bytes_by_tier,
        route_bytes_detailed=route_bytes_detailed,
        per_layer_bytes=per_layer_bytes,
        suggested_sram_bytes=suggested_sram_bytes,
        buffer_peak_bytes=buffer_peak_bytes,
        buffer_requirements=buffer_requirements,
        buffer_relationships=buffer_relationships,
        policy=p,
        hw_config=hw_config,
        feature_residency_plan=feature_residency_plan,
    )


def format_memory_contract_report(report: MemoryContractReport) -> str:
    lines: List[str] = []
    lines.append("# PointKAN Memory Contract Report")
    lines.append("")
    lines.append("## Suggested SRAM Minimum")
    for k, v in sorted(report.suggested_sram_bytes.items()):
        lines.append(f"- {k}: {v} B ({v / (1024 * 1024):.4f} MiB)")

    lines.append("")
    lines.append("")
    lines.append("## Energy Estimate Note")
    lines.append("- memory_contract 中的 estimated_energy_nj 是静态代理值：按每条事务的字节数乘以当前 SRAM/DRAM 单次访问参数估算。")
    lines.append("- 它用于比较不同事务/不同 buffer 的相对强弱，不是 runtime 总能耗，不能与 breakdown 中的 SRAM/DRAM 总能耗直接对比或求和。")
    lines.append("- 尤其对 spool_workspace_sram、kan_workspace_sram 等逻辑 buffer，目前会复用 inter_layer SRAM 的 CACTI 参数做代理，因此更应视为 rough proxy。")

    lines.append("")
    lines.append("## Feature Residency Plan")
    lines.append("layer_name	layer_type	input_bytes	input_sram_bytes	input_dram_bytes	output_bytes	output_sram_bytes	output_dram_bytes	usable_buffer_bytes	total_buffer_bytes	buffer_slots	pipeline_mode	double_buffering	note")
    for row in report.feature_residency_plan:
        lines.append(
            "{layer_name}\t{layer_type}\t{input_bytes}\t{input_sram_bytes}\t{input_dram_bytes}\t{output_bytes}\t{output_sram_bytes}\t{output_dram_bytes}\t{usable_buffer_bytes}\t{total_buffer_bytes}\t{buffer_slots}\t{pipeline_mode}\t{double_buffering}\t{note}".format(
                layer_name=row.get("layer_name", ""),
                layer_type=row.get("layer_type", ""),
                input_bytes=row.get("input_bytes", 0),
                input_sram_bytes=row.get("input_sram_bytes", 0),
                input_dram_bytes=row.get("input_dram_bytes", 0),
                output_bytes=row.get("output_bytes", 0),
                output_sram_bytes=row.get("output_sram_bytes", 0),
                output_dram_bytes=row.get("output_dram_bytes", 0),
                usable_buffer_bytes=row.get("usable_buffer_bytes", 0),
                total_buffer_bytes=row.get("total_buffer_bytes", 0),
                buffer_slots=row.get("buffer_slots", 0),
                pipeline_mode=row.get("pipeline_mode", ""),
                double_buffering=row.get("double_buffering", False),
                note=row.get("note", ""),
            )
        )

    lines.append("")
    lines.append("## Buffer Peak Summary")
    lines.append("buffer_name	peak_bytes	MiB")
    for name, b in sorted(report.buffer_peak_bytes.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"{name}	{b}	{b / (1024 * 1024):.6f}")

    lines.append("")
    lines.append("## Operator Buffer Requirements")
    lines.append("layer_name	op_name	buffer_tier	buffer_name	peak_bytes	MiB	purpose	note")
    for row in sorted(report.buffer_requirements, key=lambda r: (-r.peak_bytes, r.layer_name, r.op_name, r.buffer_name)):
        lines.append(
            f"{row.layer_name}	{row.op_name}	{row.buffer_tier}	{row.buffer_name}	{row.peak_bytes}	{row.peak_bytes / (1024 * 1024):.6f}	{row.purpose}	{row.note}"
        )

    lines.append("")
    lines.append("## Buffer Relationship Table")
    lines.append("buffer_name	shared_group	role	physical_sharing	mutual_exclusion	reset_scope	scheduler_resource_id	pipeline_behavior	note")
    for row in report.buffer_relationships:
        lines.append(
            f"{row.buffer_name}	{row.shared_group}	{row.role}	{row.physical_sharing}	{row.mutual_exclusion}	{row.reset_scope}	{row.scheduler_resource_id}	{row.pipeline_behavior}	{row.note}"
        )

    lines.append("")
    lines.append("## Route Summary (Tier)")
    lines.append("src_tier\tdst_tier\tbytes\tMiB")
    for (src, dst), b in sorted(report.route_bytes_by_tier.items(), key=lambda x: (-x[1], x[0][0], x[0][1])):
        lines.append(f"{src}\t{dst}\t{b}\t{b / (1024 * 1024):.6f}")

    lines.append("")
    lines.append("## Route Summary (Detailed)")
    lines.append("src\tdst\tbytes\tMiB")
    for (src, dst), b in sorted(report.route_bytes_detailed.items(), key=lambda x: (-x[1], x[0][0], x[0][1])):
        lines.append(f"{src}\t{dst}\t{b}\t{b / (1024 * 1024):.6f}")

    lines.append("")
    lines.append("## Per-Layer Total")
    lines.append("layer_name\tbytes\tMiB")
    for layer, b in sorted(report.per_layer_bytes.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"{layer}\t{b}\t{b / (1024 * 1024):.6f}")

    lines.append("")
    lines.append("## Detailed Records")
    lines.append("layer_name\top_name\ttensor_name\tsrc_tier\tsrc_name\tdst_tier\tdst_name\tbytes\tMiB\tnote")
    for r in report.records:
        lines.append(
            f"{r.layer_name}\t{r.op_name}\t{r.tensor_name}\t{r.src_tier}\t{r.src_name}\t"
            f"{r.dst_tier}\t{r.dst_name}\t{r.bytes}\t{r.bytes / (1024 * 1024):.6f}\t{r.note}"
        )

    lines.append("")
    return "\n".join(lines)


def iter_memory_contract_rows(report: MemoryContractReport) -> Iterable[Dict[str, Any]]:
    for r in report.records:
        row = r.to_dict()
        row["src_tier"] = r.src_tier
        row["src_name"] = r.src_name
        row["dst_tier"] = r.dst_tier
        row["dst_name"] = r.dst_name
        row["mib"] = float(r.bytes) / float(1024 * 1024)
        est_nj, est_model, est_scope = _estimate_record_energy_nj(r, getattr(report, "hw_config", None))
        row["estimated_energy_nj"] = est_nj
        row["estimated_energy_model"] = est_model
        row["estimated_energy_scope"] = est_scope
        yield row


def iter_buffer_requirement_rows(report: MemoryContractReport) -> Iterable[Dict[str, Any]]:
    for r in report.buffer_requirements:
        yield r.to_dict()


def iter_buffer_relationship_rows(report: MemoryContractReport) -> Iterable[Dict[str, Any]]:
    for r in report.buffer_relationships:
        yield r.to_dict()
