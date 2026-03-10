"""最小可运行脚本：policy/dispatcher -> traffic trace -> BookSim trace 导出。"""

from reconfig_sim.dispatch.dispatcher import HardwareConfig, ModeAwareDispatcher
from reconfig_sim.dispatch.policies import RuleBasedPolicy
from reconfig_sim.noc.booksim_adapter import BookSimAdapter
from reconfig_sim.noc.traffic_trace import build_traffic_trace
from reconfig_sim.workloads.op_desc import OpDesc


def build_demo_ops():
    return [
        OpDesc("fc1", "linear", m=64, n=128, k=256, has_static_weight=True),
        OpDesc("attn_qk", "attention_qk", m=128, n=128, k=64, has_static_weight=False),
    ]


def main() -> None:
    hw = HardwareConfig(arr_r=8, arr_c=8)
    ops = build_demo_ops()

    dispatcher = ModeAwareDispatcher(hw, RuleBasedPolicy())
    dispatch_result = dispatcher.dispatch(ops)

    trace = build_traffic_trace(
        dispatch_result.schedule,
        hw,
        tile_stats=dispatch_result.tile_stats,
        total_cycles=dispatch_result.total_cycles,
    )

    adapter = BookSimAdapter(packet_bytes=16)
    payload = adapter.export_trace(trace)

    out_path = adapter.write_packet_trace(trace, "./artifacts/gemm_noc_trace.tr")

    print("mode_hist:", dispatch_result.mode_hist)
    print("dispatch cycles:", dispatch_result.total_cycles)
    print("trace summary:", payload["summary"])
    print("booksim trace:", out_path)


if __name__ == "__main__":
    main()
