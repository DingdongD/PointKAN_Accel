from pathlib import Path

from reconfig_sim.dispatch.dispatcher import HardwareConfig, ModeAwareDispatcher
from reconfig_sim.dispatch.policies import RuleBasedPolicy
from reconfig_sim.noc.booksim_adapter import BookSimAdapter
from reconfig_sim.noc.calibration import compare_fixed_modes
from reconfig_sim.noc.traffic_trace import build_traffic_trace, dispatch_to_traffic_trace
from reconfig_sim.workloads.op_desc import OpDesc


def main() -> None:
    hw = HardwareConfig(arr_r=4, arr_c=4)
    ops = [
        OpDesc("fc1", "linear", m=8, n=8, k=8, has_static_weight=True),
        OpDesc("qk", "attention_qk", m=8, n=8, k=16, has_static_weight=False),
    ]

    dispatch_res = ModeAwareDispatcher(hw, RuleBasedPolicy()).dispatch(ops)
    trace_from_dispatch = dispatch_to_traffic_trace(dispatch_res, hw)
    trace_from_schedule = build_traffic_trace(
        dispatch_res.schedule,
        hw,
        tile_stats=dispatch_res.tile_stats,
        total_cycles=dispatch_res.total_cycles,
    )
    assert len(trace_from_dispatch.events) > 0
    assert len(trace_from_dispatch.events) == len(trace_from_schedule.events)

    adapter = BookSimAdapter(packet_bytes=16)
    payload = adapter.export_trace(trace_from_dispatch)
    summary = payload["summary"]
    assert summary["events"] == len(trace_from_dispatch.events)
    assert summary["bytes"] > 0
    assert len(payload["packet_trace_lines"]) == summary["events"]

    compare = compare_fixed_modes(ops, hw)
    assert "OS" in compare and "WS" in compare
    assert compare["OS"].events > 0 and compare["WS"].events > 0

    out_dir = Path("/tmp/reconfig_sim_trace_test")
    packet_path = adapter.write_packet_trace(trace_from_dispatch, str(out_dir / "trace.tr"))
    json_path = adapter.write_trace_json(trace_from_dispatch, str(out_dir / "trace.json"))
    csv_path = adapter.write_trace_csv(trace_from_dispatch, str(out_dir / "trace.csv"))

    assert packet_path.exists()
    assert json_path.exists()
    assert csv_path.exists()
    print("noc trace adapter test passed")


if __name__ == "__main__":
    main()
