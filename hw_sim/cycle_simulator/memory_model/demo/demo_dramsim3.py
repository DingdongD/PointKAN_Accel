import os
import sys

# 保证从 memory_model 项目根或 cycle_simulator 运行都能找到 memory_model 包
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import json
from memory_model.dramsim3_runner import DRAMSim3Runner
from memory_model.dram_trace import DRAMTrace
from memory_model.types import DRAMSim3Input


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # DRAMsim3/CACTI 的 bin 在 cycle_simulator 下，与 memory_model 同级
    cycle_sim_root = os.path.dirname(repo_root)

    trace = DRAMTrace()

    # 连续读 4KB，按 64B burst 拆事务
    trace.extend_burst(
        base_addr=0x10000000,
        num_bytes=4096,
        burst_bytes=64,
        is_write=False,
        start_cycle=0,
        issue_gap=1,
    )

    # 延后一段时间写 1KB
    trace.extend_burst(
        base_addr=0x20000000,
        num_bytes=1024,
        burst_bytes=64,
        is_write=True,
        start_cycle=200,
        issue_gap=1,
    )

    trace_path = os.path.join(repo_root, "traces", "generated", "demo.trace")
    trace.dump(trace_path)

    print("Trace stats:")
    print(json.dumps(trace.get_stats(), indent=2))

    runner = DRAMSim3Runner(
        dramsim3_bin=os.path.join(cycle_sim_root, "DRAMsim3", "build", "dramsim3main"),
        work_root=os.path.join(repo_root, "outputs", "dramsim3"),
    )

    result = runner.run(
        DRAMSim3Input(
            config_path=os.path.join(repo_root, "configs", "dramsim3", "DDR4_8Gb_x8_3200.ini"),
            trace_path=trace_path,
            num_cycles=100000,
            output_dir=os.path.join(repo_root, "outputs", "dramsim3", "demo_run"),
        )
    )

    print("STDOUT:")
    print(result.stdout)

    if result.json_stats:
        print("\nTop-level JSON keys:")
        print(list(result.json_stats.keys()))
    else:
        print("\nNo top-level JSON parsed. Check output directory and DRAMsim3 version.")


if __name__ == "__main__":
    main()