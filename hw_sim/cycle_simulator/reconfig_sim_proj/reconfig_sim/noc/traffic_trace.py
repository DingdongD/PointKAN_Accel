from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from reconfig_sim.dispatch.dispatcher import DispatchResult, HardwareConfig, TileDesc, TileStats


@dataclass(frozen=True)
class TrafficEvent:
    """A single NoC traffic event produced by dispatcher scheduling."""

    cycle: int
    src: int
    dst: int
    bytes: int
    flow: str
    op_name: str
    tile_index: int
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrafficTrace:
    """Collection of traffic events with optional metadata."""

    events: List[TrafficEvent]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": dict(self.metadata),
            "events": [
                {
                    "cycle": e.cycle,
                    "src": e.src,
                    "dst": e.dst,
                    "bytes": e.bytes,
                    "flow": e.flow,
                    "op_name": e.op_name,
                    "tile_index": e.tile_index,
                    "meta": dict(e.meta),
                }
                for e in self.events
            ],
        }


def _tile_payload_bytes(tile: TileDesc, hw: HardwareConfig) -> Dict[str, int]:
    word_bytes = hw.word_bits // 8
    acc_bytes = hw.acc_bits // 8

    if tile.mode == "OS":
        return {
            "a": tile.mt * tile.kt * word_bytes,
            "b": tile.kt * tile.nt * word_bytes,
            "c": tile.mt * tile.nt * acc_bytes,
        }
    return {
        "a": tile.kt * word_bytes,
        "b": tile.kt * tile.nt * word_bytes,
        "psum_in": tile.nt * acc_bytes,
        "psum_out": tile.nt * acc_bytes,
    }


def build_traffic_trace(
    schedule: Sequence[TileDesc],
    hw: HardwareConfig,
    *,
    tile_stats: Optional[Sequence[TileStats]] = None,
    total_cycles: Optional[int] = None,
    dram_node: int = 0,
    array_node: int = 1,
    start_cycle: int = 0,
    inter_tile_gap: int = 0,
) -> TrafficTrace:
    """Build NoC traffic trace from dispatcher tile schedule.

    If `tile_stats` is provided, event cycle cursor follows per-tile cycles; otherwise
    events are emitted with `inter_tile_gap` spacing only.
    """

    events: List[TrafficEvent] = []
    cycle_cursor = start_cycle

    for tile_idx, tile in enumerate(schedule):
        payloads = _tile_payload_bytes(tile, hw)

        for flow, payload_bytes in payloads.items():
            if payload_bytes <= 0:
                continue
            src = dram_node
            dst = array_node
            if flow in {"c", "psum_out"}:
                src, dst = array_node, dram_node
            events.append(
                TrafficEvent(
                    cycle=cycle_cursor,
                    src=src,
                    dst=dst,
                    bytes=payload_bytes,
                    flow=flow,
                    op_name=tile.op_name,
                    tile_index=tile_idx,
                    meta={"mode": tile.mode, "m0": tile.m0, "n0": tile.n0, "k0": tile.k0},
                )
            )

        tile_cycles = tile_stats[tile_idx].cycles if tile_stats and tile_idx < len(tile_stats) else 0
        cycle_cursor += tile_cycles + inter_tile_gap

    resolved_total_cycles = total_cycles if total_cycles is not None else max(cycle_cursor - start_cycle, 0)
    return TrafficTrace(
        events=events,
        metadata={
            "total_events": len(events),
            "total_cycles": resolved_total_cycles,
            "dram_node": dram_node,
            "array_node": array_node,
        },
    )


def dispatch_to_traffic_trace(
    dispatch_result: DispatchResult,
    hw: HardwareConfig,
    *,
    dram_node: int = 0,
    array_node: int = 1,
    start_cycle: int = 0,
    inter_tile_gap: int = 0,
) -> TrafficTrace:
    """Backward-compatible adapter for DispatchResult -> trace conversion."""
    return build_traffic_trace(
        dispatch_result.schedule,
        hw,
        tile_stats=dispatch_result.tile_stats,
        total_cycles=dispatch_result.total_cycles,
        dram_node=dram_node,
        array_node=array_node,
        start_cycle=start_cycle,
        inter_tile_gap=inter_tile_gap,
    )


def merge_traces(traces: Iterable[TrafficTrace]) -> TrafficTrace:
    events: List[TrafficEvent] = []
    for trace in traces:
        events.extend(trace.events)
    events.sort(key=lambda e: e.cycle)
    return TrafficTrace(events=events, metadata={"total_events": len(events)})
