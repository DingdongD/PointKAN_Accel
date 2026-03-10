#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# reconfig_sim_proj 仿真与 NoC trace 统一执行脚本
# 用法:
#   ./run_sims.sh [场景] [可选参数]
# 场景:
#   demo        - 周期仿真 demo（GEMM + Conv2D），输出 JSON 报告
#   demo_viz    - 同上，并导出 PNG/GIF 可视化（需 matplotlib）
#   policy      - 策略对比（all_ws / all_os / rule_based），打印周期与能耗
#   trace       - NoC trace 最小示例：dispatch → trace → BookSim .tr 文件
#   trace_full  - trace 并同时输出 .tr / .json / .csv 到 artifacts/
#   calibration - OS/WS 标定对比（compare_fixed_modes），打印 events/bytes/packets
#   test        - 运行全部测试（smoke + policy_dispatch + noc_trace_adapter）
#   all         - 依次执行 demo + policy + trace + calibration + test（不含 viz）
# 可选参数（仅部分场景）:
#   demo_viz 无额外参数；trace_full 无额外参数。
# 示例:
#   ./run_sims.sh
#   ./run_sims.sh demo
#   ./run_sims.sh trace
#   ./run_sims.sh trace_full
#   ./run_sims.sh test
#   ./run_sims.sh all
# -----------------------------------------------------------------------------
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH=.
export PYTHONDONTWRITEBYTECODE=1
ARTIFACTS_DIR="${ARTIFACTS_DIR:-./artifacts}"
export ARTIFACTS_DIR
mkdir -p "$ARTIFACTS_DIR"

run_demo() {
  echo "========== 周期仿真 Demo (run_demo) =========="
  python reconfig_sim/examples/run_demo.py
}

run_demo_viz() {
  echo "========== 周期仿真 Demo + 可视化 (run_demo --viz) =========="
  python reconfig_sim/examples/run_demo.py --viz
}

run_policy() {
  echo "========== 策略对比 (policy_demo) =========="
  python reconfig_sim/examples/policy_demo.py
}

run_trace() {
  echo "========== NoC Trace 最小示例 (dispatcher_trace_minimal) =========="
  python reconfig_sim/examples/dispatcher_trace_minimal.py
}

run_trace_full() {
  echo "========== NoC Trace 全量导出 (.tr + .json + .csv) =========="
  python - << PY
import os
from pathlib import Path
from reconfig_sim.dispatch.dispatcher import HardwareConfig, ModeAwareDispatcher
from reconfig_sim.dispatch.policies import RuleBasedPolicy
from reconfig_sim.noc.booksim_adapter import BookSimAdapter
from reconfig_sim.noc.traffic_trace import build_traffic_trace
from reconfig_sim.workloads.op_desc import OpDesc

artifacts = Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))
artifacts.mkdir(parents=True, exist_ok=True)
hw = HardwareConfig(arr_r=8, arr_c=8)
ops = [
    OpDesc("fc1", "linear", m=64, n=128, k=256, has_static_weight=True),
    OpDesc("attn_qk", "attention_qk", m=128, n=128, k=64, has_static_weight=False),
]
dispatcher = ModeAwareDispatcher(hw, RuleBasedPolicy())
result = dispatcher.dispatch(ops)
trace = build_traffic_trace(
    result.schedule, hw,
    tile_stats=result.tile_stats,
    total_cycles=result.total_cycles,
)
adapter = BookSimAdapter(packet_bytes=16)
adapter.write_packet_trace(trace, str(artifacts / "gemm_noc_trace.tr"))
adapter.write_trace_json(trace, str(artifacts / "gemm_noc_trace.json"))
adapter.write_trace_csv(trace, str(artifacts / "gemm_noc_trace.csv"))
payload = adapter.export_trace(trace)
print("mode_hist:", result.mode_hist)
print("dispatch cycles:", result.total_cycles)
print("trace summary:", payload["summary"])
print("artifacts:", artifacts.resolve())
print("  .tr:", artifacts / "gemm_noc_trace.tr")
print("  .json:", artifacts / "gemm_noc_trace.json")
print("  .csv:", artifacts / "gemm_noc_trace.csv")
PY
}

run_calibration() {
  echo "========== OS/WS 标定对比 (compare_fixed_modes) =========="
  python - << 'PY'
from reconfig_sim.dispatch.dispatcher import HardwareConfig
from reconfig_sim.noc.calibration import compare_fixed_modes
from reconfig_sim.workloads.op_desc import OpDesc

hw = HardwareConfig(arr_r=4, arr_c=4)
ops = [
    OpDesc("fc1", "linear", m=8, n=8, k=8, has_static_weight=True),
    OpDesc("qk", "attention_qk", m=8, n=8, k=16, has_static_weight=False),
]
compare = compare_fixed_modes(ops, hw)
for mode_name, res in compare.items():
    print(f"  {mode_name}: cycles={res.dispatch_cycles} energy={res.dispatch_energy:.2f} events={res.events} bytes={res.noc_bytes} packets={res.noc_packets}")
PY
}

run_test() {
  echo "========== 测试 (smoke + policy_dispatch + noc_trace_adapter) =========="
  python tests/smoke_test.py
  python tests/test_policy_dispatch.py
  python tests/test_noc_trace_adapter.py
  echo "All tests passed."
}

scene="${1:-all}"
case "$scene" in
  demo)       run_demo ;;
  demo_viz)   run_demo_viz ;;
  policy)     run_policy ;;
  trace)      run_trace ;;
  trace_full) run_trace_full ;;
  calibration) run_calibration ;;
  test)       run_test ;;
  all)
    run_demo
    run_policy
    run_trace
    run_calibration
    run_test
    ;;
  help|--help|-h)
    head -n 28 "$0" | tail -n 26
    exit 0
    ;;
  *)
    echo "Unknown scene: $scene"
    echo "Usage: $0 { demo | demo_viz | policy | trace | trace_full | calibration | test | all }"
    exit 1
    ;;
esac
