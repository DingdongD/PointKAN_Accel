from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class FeatureResidencyEntry:
    layer_name: str
    layer_type: str
    input_bytes: int
    input_sram_bytes: int
    input_dram_bytes: int
    output_bytes: int
    output_sram_bytes: int
    output_dram_bytes: int
    usable_buffer_bytes: int
    total_buffer_bytes: int
    buffer_slots: int
    pipeline_mode: str
    double_buffering: bool
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "layer_type": self.layer_type,
            "input_bytes": int(self.input_bytes),
            "input_sram_bytes": int(self.input_sram_bytes),
            "input_dram_bytes": int(self.input_dram_bytes),
            "output_bytes": int(self.output_bytes),
            "output_sram_bytes": int(self.output_sram_bytes),
            "output_dram_bytes": int(self.output_dram_bytes),
            "usable_buffer_bytes": int(self.usable_buffer_bytes),
            "total_buffer_bytes": int(self.total_buffer_bytes),
            "buffer_slots": int(self.buffer_slots),
            "pipeline_mode": self.pipeline_mode,
            "double_buffering": bool(self.double_buffering),
            "note": self.note,
        }


def get_feature_buffer_slots(hw_config: Any) -> int:
    buf = getattr(hw_config, "inter_layer_buffer", None)
    if buf is None:
        return 1
    slots = int(getattr(buf, "buffer_slots", 0) or 0)
    if slots > 0:
        return max(1, slots)
    pipeline_mode = str(getattr(buf, "pipeline_mode", "without_pipeline") or "without_pipeline").strip().lower()
    double_buffering = bool(getattr(buf, "double_buffering", False))
    if pipeline_mode == "pipeline" and double_buffering:
        return 2
    return 1


def get_effective_feature_buffer_bytes(hw_config: Any) -> int:
    buf = getattr(hw_config, "inter_layer_buffer", None)
    if buf is None:
        return 0
    total_bytes = max(1, int(getattr(buf, "size_bytes", 0) or 0))
    slots = get_feature_buffer_slots(hw_config)
    return max(1, total_bytes // max(1, slots))


def build_feature_residency_plan(layers: List[Any], activation_shapes: List[Any], hw_config: Any) -> List[FeatureResidencyEntry]:
    buf = getattr(hw_config, "inter_layer_buffer", None)
    if buf is None:
        return []

    total_bytes = max(1, int(getattr(buf, "size_bytes", 0) or 0))
    usable_bytes = get_effective_feature_buffer_bytes(hw_config)
    slots = get_feature_buffer_slots(hw_config)
    pipeline_mode = str(getattr(buf, "pipeline_mode", "without_pipeline") or "without_pipeline").strip().lower()
    double_buffering = bool(getattr(buf, "double_buffering", False))

    plan: List[FeatureResidencyEntry] = []
    prev_output_sram_bytes = 0

    for idx, layer in enumerate(layers):
        shape = activation_shapes[idx] if idx < len(activation_shapes) else None
        input_bytes = int(getattr(shape, "input_bytes", 0) or 0) if shape is not None else 0
        output_bytes = int(getattr(shape, "output_bytes", 0) or 0) if shape is not None else 0
        layer_name = str(getattr(layer, "name", f"layer_{idx}") or f"layer_{idx}")
        layer_type = str(getattr(layer, "layer_type", "") or "")

        if idx == 0:
            input_sram_bytes = min(input_bytes, usable_bytes)
            input_dram_bytes = 0
            note = "first-layer input assumed prefetched before compute"
        else:
            input_sram_bytes = min(input_bytes, prev_output_sram_bytes, usable_bytes)
            input_dram_bytes = max(0, input_bytes - input_sram_bytes)
            note = "input prefers persistent feature SRAM; spill bytes refill from DRAM"

        output_sram_bytes = min(output_bytes, usable_bytes)
        output_dram_bytes = max(0, output_bytes - output_sram_bytes)
        if pipeline_mode == "pipeline" and double_buffering:
            note += "; double-buffered feature SRAM reserves one slot per in-flight tensor"
        elif pipeline_mode == "pipeline":
            note += "; pipeline requested without double buffering, so read/write overlap is capacity-safe only if externally scheduled"

        plan.append(
            FeatureResidencyEntry(
                layer_name=layer_name,
                layer_type=layer_type,
                input_bytes=input_bytes,
                input_sram_bytes=input_sram_bytes,
                input_dram_bytes=input_dram_bytes,
                output_bytes=output_bytes,
                output_sram_bytes=output_sram_bytes,
                output_dram_bytes=output_dram_bytes,
                usable_buffer_bytes=usable_bytes,
                total_buffer_bytes=total_bytes,
                buffer_slots=slots,
                pipeline_mode=pipeline_mode,
                double_buffering=double_buffering,
                note=note,
            )
        )
        prev_output_sram_bytes = output_sram_bytes

    return plan


def plan_by_layer_name(plan: List[FeatureResidencyEntry]) -> Dict[str, FeatureResidencyEntry]:
    return {entry.layer_name: entry for entry in plan}
