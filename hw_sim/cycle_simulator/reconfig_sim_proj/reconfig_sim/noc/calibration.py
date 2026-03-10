from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from reconfig_sim.dispatch.dispatcher import HardwareConfig, ModeAwareDispatcher
from reconfig_sim.dispatch.policies import FixedModePolicy, MappingPolicy
from reconfig_sim.noc.booksim_adapter import BookSimAdapter
from reconfig_sim.noc.traffic_trace import build_traffic_trace
from reconfig_sim.workloads.op_desc import OpDesc


@dataclass
class CalibrationResult:
    mode: str
    dispatch_cycles: int
    dispatch_energy: float
    noc_bytes: int
    noc_packets: int
    events: int


def evaluate_policy_traffic(
    ops: List[OpDesc],
    hw: HardwareConfig,
    policy: MappingPolicy,
    *,
    packet_bytes: int = 16,
) -> CalibrationResult:
    dispatcher = ModeAwareDispatcher(hw, policy)
    dispatch_result = dispatcher.dispatch(ops)
    trace = build_traffic_trace(
        dispatch_result.schedule,
        hw,
        tile_stats=dispatch_result.tile_stats,
        total_cycles=dispatch_result.total_cycles,
    )
    summary = BookSimAdapter(packet_bytes=packet_bytes).summarize(trace)
    # 对 FixedModePolicy 使用实际模式名 "OS"/"WS"，其它策略用类名
    mode_name = getattr(policy, "mode", policy.__class__.__name__)
    return CalibrationResult(
        mode=mode_name,
        dispatch_cycles=dispatch_result.total_cycles,
        dispatch_energy=dispatch_result.total_energy,
        noc_bytes=summary["bytes"],
        noc_packets=summary["packets"],
        events=summary["events"],
    )


def compare_fixed_modes(ops: List[OpDesc], hw: HardwareConfig, *, packet_bytes: int = 16) -> Dict[str, CalibrationResult]:
    return {
        "OS": evaluate_policy_traffic(ops, hw, FixedModePolicy("OS"), packet_bytes=packet_bytes),
        "WS": evaluate_policy_traffic(ops, hw, FixedModePolicy("WS"), packet_bytes=packet_bytes),
    }
