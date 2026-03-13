import json
import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from memory_model.cacti_runner import CactiRunner
from memory_model.dramsim3_runner import DRAMSim3Runner
from memory_model.memory_system import MemorySystem
from memory_model.pointcloud_trace_gen import PointCloudTraceGenerator, TensorRegion
from memory_model.sram_buffer import SRAMBuffer
from memory_model.types import CactiInput


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cycle_sim_root = os.path.dirname(repo_root)
    freq_hz = 500_000_000

    cacti_runner = CactiRunner(
        cacti_bin=os.path.join(cycle_sim_root, "cacti", "cacti"),
        template_cfg=os.path.join(cycle_sim_root, "cacti", "cache.cfg"),
        generated_cfg_dir=os.path.join(repo_root, "configs", "cacti", "generated"),
        output_dir=os.path.join(repo_root, "outputs", "cacti"),
    )

    xyz_sram = SRAMBuffer(
        name="xyz_buffer",
        cacti_runner=cacti_runner,
        cacti_cfg=CactiInput(
            size_bytes=16 * 1024,
            block_size_bytes=16,
            output_bus_width_bits=128,
            cache_type="ram",
            technology_u=0.028,
        ),
        freq_hz=freq_hz,
    )

    dist_sram = SRAMBuffer(
        name="dist_buffer",
        cacti_runner=cacti_runner,
        cacti_cfg=CactiInput(
            size_bytes=8 * 1024,
            block_size_bytes=16,
            output_bus_width_bits=128,
            cache_type="ram",
            technology_u=0.028,
        ),
        freq_hz=freq_hz,
    )

    dramsim3_runner = DRAMSim3Runner(
        dramsim3_bin=os.path.join(cycle_sim_root, "DRAMsim3", "build", "dramsim3main"),
        work_root=os.path.join(repo_root, "outputs", "dramsim3"),
    )

    memsys = MemorySystem(
        sram_buffers={
            "xyz_buffer": xyz_sram,
            "dist_buffer": dist_sram,
        },
        dramsim3_runner=dramsim3_runner,
    )

    # -----------------------------
    # SRAM-side runtime simulation
    # -----------------------------
    cur_cycle = 0

    stat1 = memsys.sram_read("xyz_buffer", 1024 * 12, cur_cycle)
    cur_cycle = stat1.end_cycle

    stat2 = memsys.sram_read("dist_buffer", 1024 * 4, cur_cycle)
    cur_cycle = stat2.end_cycle

    # 假设中间计算
    cur_cycle += 500

    stat3 = memsys.sram_write("dist_buffer", 1024 * 4, cur_cycle)
    cur_cycle = stat3.end_cycle

    # -----------------------------
    # DRAM-side trace generation
    # -----------------------------
    trace_gen = PointCloudTraceGenerator(burst_bytes=64, default_issue_gap=1)

    xyz_region = TensorRegion(
        name="xyz",
        base_addr=0x10000000,
        num_elements=1024 * 3,
        bytes_per_element=4,
    )
    min_dist_region = TensorRegion(
        name="min_dist",
        base_addr=0x20000000,
        num_elements=1024,
        bytes_per_element=4,
    )

    end_cycle = trace_gen.gen_fps_pass(
        xyz_region=xyz_region,
        min_dist_region=min_dist_region,
        n_points=1024,
        point_dim=3,
        tile_points=32,
        start_cycle=0,
        issue_gap=1,
        compute_gap=10,
    )

    trace_path = os.path.join(repo_root, "traces", "generated", "fps.trace")
    trace_gen.dump(trace_path)

    print("Trace stats:")
    print(json.dumps(trace_gen.get_trace_stats(), indent=2))

    dram_result = memsys.run_dram_trace(
        config_path=os.path.join(repo_root, "configs", "dramsim3", "DDR4_8Gb_x8_3200.ini"),
        trace_path=trace_path,
        num_cycles=max(100000, end_cycle + 1000),
        output_dir=os.path.join(repo_root, "outputs", "dramsim3", "fps_run"),
    )

    print("=== SRAM Summary ===")
    print(json.dumps(memsys.summary(), indent=2))

    print("\n=== DRAM JSON keys ===")
    print(list(dram_result.json_stats.keys()))


if __name__ == "__main__":
    main()