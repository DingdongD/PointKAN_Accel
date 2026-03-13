# -*- coding: utf-8 -*-
"""
demo_fps_with_refill.py —— 支持 non-overlap / overlap 对比的 FPS 点云流水线仿真

新增：
  - overlap 模式下按 tile 打印 timeline：
      prefetch_issue / prefetch_ready / compute_start / compute_end / buffer_pair
"""

import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from memory_model.cacti_runner import CactiRunner
from memory_model.dramsim3_runner import DRAMSim3Runner
from memory_model.memory_system import MemorySystem
from memory_model.pointcloud_trace_gen import TensorRegion
from memory_model.sram_buffer import SRAMBuffer
from memory_model.types import CactiInput


def cycles_to_ns(cycles: int, freq_hz: int) -> float:
    if freq_hz <= 0:
        return 0.0
    return cycles / freq_hz * 1e9


def build_memsys(
    repo_root: str,
    cycle_sim_root: str,
    freq_hz: int,
    sram_num_banks: int,
    double_buffer: bool,
):
    cacti_runner = CactiRunner(
        cacti_bin=os.path.join(cycle_sim_root, "cacti", "cacti"),
        template_cfg=os.path.join(cycle_sim_root, "cacti", "cache.cfg"),
        generated_cfg_dir=os.path.join(repo_root, "configs", "cacti", "generated"),
        output_dir=os.path.join(repo_root, "outputs", "cacti"),
        reuse_if_exists=True,
    )

    dramsim3_runner = DRAMSim3Runner(
        dramsim3_bin=os.path.join(cycle_sim_root, "DRAMsim3", "build", "dramsim3main"),
        work_root=os.path.join(repo_root, "outputs", "dramsim3"),
    )

    def make_xyz_buffer(name: str):
        return SRAMBuffer(
            name=name,
            cacti_runner=cacti_runner,
            cacti_cfg=CactiInput(
                size_bytes=4 * 1024,
                block_size_bytes=16,
                output_bus_width_bits=128,
                cache_type="ram",
                technology_u=0.028,
                associativity=1,
                read_write_port=1,
                exclusive_read_port=0,
                exclusive_write_port=0,
                single_ended_read_ports=0,
                uca_bank_count=sram_num_banks,
            ),
            freq_hz=freq_hz,
            num_banks=sram_num_banks,
        )

    def make_dist_buffer(name: str):
        return SRAMBuffer(
            name=name,
            cacti_runner=cacti_runner,
            cacti_cfg=CactiInput(
                size_bytes=2 * 1024,
                block_size_bytes=16,
                output_bus_width_bits=128,
                cache_type="ram",
                technology_u=0.028,
                associativity=1,
                read_write_port=1,
                exclusive_read_port=0,
                exclusive_write_port=0,
                single_ended_read_ports=0,
                uca_bank_count=sram_num_banks,
            ),
            freq_hz=freq_hz,
            num_banks=sram_num_banks,
        )

    if double_buffer:
        sram_buffers = {
            "xyz_buffer_0": make_xyz_buffer("xyz_buffer_0"),
            "dist_buffer_0": make_dist_buffer("dist_buffer_0"),
            "xyz_buffer_1": make_xyz_buffer("xyz_buffer_1"),
            "dist_buffer_1": make_dist_buffer("dist_buffer_1"),
        }
    else:
        sram_buffers = {
            "xyz_buffer": make_xyz_buffer("xyz_buffer"),
            "dist_buffer": make_dist_buffer("dist_buffer"),
        }

    memsys = MemorySystem(
        sram_buffers=sram_buffers,
        dramsim3_runner=dramsim3_runner,
    )
    return memsys


def summarize_run(memsys: MemorySystem, end_to_end_cycles: int, compute_cycles: int, tile_timeline=None):
    summary = memsys.summary()
    buffers = summary.get("buffers", {})
    refill_history = summary.get("refill_history", [])

    sram_service_cycles = sum(
        int(b.get("runtime_stats", {}).get("total_busy_cycles", 0))
        for b in buffers.values()
    )
    sram_energy_nj = sum(float(b.get("total_energy_nj", 0) or 0) for b in buffers.values())

    dram_refill_cycles = 0
    dram_energy_nj = 0.0
    for ev in refill_history:
        dram_refill_cycles += int(
            ev.get("dram_latency_cycles", ev.get("dram_done_cycle", 0) - ev.get("request_cycle", 0))
        )
        if ev.get("dram_energy_nj") is not None:
            dram_energy_nj += float(ev["dram_energy_nj"])

    return {
        "end_to_end_cycles": end_to_end_cycles,
        "sram_service_cycles": sram_service_cycles,
        "sram_energy_nj": sram_energy_nj,
        "dram_refill_cycles": dram_refill_cycles,
        "dram_energy_nj": dram_energy_nj,
        "compute_cycles": compute_cycles,
        "refill_history_len": len(refill_history),
        "full_summary": summary,
        "tile_timeline": tile_timeline or [],
    }


def run_fps_pipeline_non_overlap(
    repo_root: str,
    cycle_sim_root: str,
    freq_hz: int,
    sram_num_banks: int = 1,
    use_dram_batch_refill: bool = False,
    n_points: int = 1024,
    point_dim: int = 3,
    tile_points: int = 128,
    compute_gap: int = 100,
):
    memsys = build_memsys(
        repo_root=repo_root,
        cycle_sim_root=cycle_sim_root,
        freq_hz=freq_hz,
        sram_num_banks=sram_num_banks,
        double_buffer=False,
    )

    xyz_region = TensorRegion(
        name="xyz",
        base_addr=0x10000000,
        num_elements=n_points * point_dim,
        bytes_per_element=4,
    )
    min_dist_region = TensorRegion(
        name="min_dist",
        base_addr=0x20000000,
        num_elements=n_points,
        bytes_per_element=4,
    )

    dram_cfg = os.path.join(repo_root, "configs", "dramsim3", "DDR4_8Gb_x8_3200.ini")
    refill_root = os.path.join(repo_root, "outputs", "dramsim3", "fps_refill_runs_non_overlap")

    cur_cycle = 0
    total_compute_cycles = 0

    for tile_start in range(0, n_points, tile_points):
        tile_n = min(tile_points, n_points - tile_start)

        if use_dram_batch_refill:
            batch_dir = os.path.join(refill_root, f"batch_tile_{tile_start}")
            cur_cycle = memsys.refill_batch(
                refill_requests=[
                    {
                        "buffer_name": "xyz_buffer",
                        "tensor_region": xyz_region,
                        "start_elem": tile_start * point_dim,
                        "num_elem": tile_n * point_dim,
                        "request_cycle": cur_cycle,
                    },
                    {
                        "buffer_name": "dist_buffer",
                        "tensor_region": min_dist_region,
                        "start_elem": tile_start,
                        "num_elem": tile_n,
                        "request_cycle": cur_cycle,
                    },
                ],
                dram_config_path=dram_cfg,
                output_dir=batch_dir,
                burst_bytes=64,
                issue_gap=1,
            )
        else:
            cur_cycle = memsys.refill_if_needed(
                buffer_name="xyz_buffer",
                tensor_region=xyz_region,
                start_elem=tile_start * point_dim,
                num_elem=tile_n * point_dim,
                cur_cycle=cur_cycle,
                dram_config_path=dram_cfg,
                output_dir=os.path.join(refill_root, f"xyz_tile_{tile_start}"),
                burst_bytes=64,
                issue_gap=1,
            )
            cur_cycle = memsys.refill_if_needed(
                buffer_name="dist_buffer",
                tensor_region=min_dist_region,
                start_elem=tile_start,
                num_elem=tile_n,
                cur_cycle=cur_cycle,
                dram_config_path=dram_cfg,
                output_dir=os.path.join(refill_root, f"dist_tile_{tile_start}"),
                burst_bytes=64,
                issue_gap=1,
            )

        xyz_buffer = memsys.sram_buffers["xyz_buffer"]
        dist_buffer = memsys.sram_buffers["dist_buffer"]

        xyz_stat = xyz_buffer.read_region(
            region_name="xyz",
            start_elem=tile_start * point_dim,
            num_elem=tile_n * point_dim,
            bytes_per_elem=4,
            cur_cycle=cur_cycle,
        )
        cur_cycle = xyz_stat.end_cycle

        dist_r_stat = dist_buffer.read_region(
            region_name="min_dist",
            start_elem=tile_start,
            num_elem=tile_n,
            bytes_per_elem=4,
            cur_cycle=cur_cycle,
        )
        cur_cycle = dist_r_stat.end_cycle

        cur_cycle += compute_gap
        total_compute_cycles += compute_gap

        dist_w_stat = dist_buffer.write_region(
            region_name="min_dist",
            start_elem=tile_start,
            num_elem=tile_n,
            bytes_per_elem=4,
            cur_cycle=cur_cycle,
        )
        cur_cycle = dist_w_stat.end_cycle

    return memsys, cur_cycle, total_compute_cycles


def prefetch_tile_into_buffers(
    memsys: MemorySystem,
    xyz_buffer_name: str,
    dist_buffer_name: str,
    xyz_region: TensorRegion,
    min_dist_region: TensorRegion,
    tile_start: int,
    tile_n: int,
    point_dim: int,
    request_cycle: int,
    dram_cfg: str,
    output_dir: str,
    use_dram_batch_refill: bool,
):
    if use_dram_batch_refill:
        return memsys.refill_batch(
            refill_requests=[
                {
                    "buffer_name": xyz_buffer_name,
                    "tensor_region": xyz_region,
                    "start_elem": tile_start * point_dim,
                    "num_elem": tile_n * point_dim,
                    "request_cycle": request_cycle,
                },
                {
                    "buffer_name": dist_buffer_name,
                    "tensor_region": min_dist_region,
                    "start_elem": tile_start,
                    "num_elem": tile_n,
                    "request_cycle": request_cycle,
                },
            ],
            dram_config_path=dram_cfg,
            output_dir=output_dir,
            burst_bytes=64,
            issue_gap=1,
        )
    else:
        c1 = memsys.refill_if_needed(
            buffer_name=xyz_buffer_name,
            tensor_region=xyz_region,
            start_elem=tile_start * point_dim,
            num_elem=tile_n * point_dim,
            cur_cycle=request_cycle,
            dram_config_path=dram_cfg,
            output_dir=os.path.join(output_dir, "xyz"),
            burst_bytes=64,
            issue_gap=1,
        )
        c2 = memsys.refill_if_needed(
            buffer_name=dist_buffer_name,
            tensor_region=min_dist_region,
            start_elem=tile_start,
            num_elem=tile_n,
            cur_cycle=request_cycle,
            dram_config_path=dram_cfg,
            output_dir=os.path.join(output_dir, "dist"),
            burst_bytes=64,
            issue_gap=1,
        )
        return max(c1, c2)


def run_fps_pipeline_overlap(
    repo_root: str,
    cycle_sim_root: str,
    freq_hz: int,
    sram_num_banks: int = 1,
    use_dram_batch_refill: bool = True,
    n_points: int = 1024,
    point_dim: int = 3,
    tile_points: int = 128,
    compute_gap: int = 100,
):
    memsys = build_memsys(
        repo_root=repo_root,
        cycle_sim_root=cycle_sim_root,
        freq_hz=freq_hz,
        sram_num_banks=sram_num_banks,
        double_buffer=True,
    )

    xyz_region = TensorRegion(
        name="xyz",
        base_addr=0x10000000,
        num_elements=n_points * point_dim,
        bytes_per_element=4,
    )
    min_dist_region = TensorRegion(
        name="min_dist",
        base_addr=0x20000000,
        num_elements=n_points,
        bytes_per_element=4,
    )

    dram_cfg = os.path.join(repo_root, "configs", "dramsim3", "DDR4_8Gb_x8_3200.ini")
    refill_root = os.path.join(repo_root, "outputs", "dramsim3", "fps_refill_runs_overlap")

    tile_starts = list(range(0, n_points, tile_points))
    n_tiles = len(tile_starts)

    total_compute_cycles = 0
    tile_timeline = []

    def buf_names(tile_idx: int):
        buf_id = tile_idx % 2
        return f"xyz_buffer_{buf_id}", f"dist_buffer_{buf_id}", buf_id

    # 先预取 tile 0
    t0_start = tile_starts[0]
    t0_n = min(tile_points, n_points - t0_start)
    xyz_buf_0, dist_buf_0, pair0 = buf_names(0)

    tile0_prefetch_issue = 0
    tile0_prefetch_ready = prefetch_tile_into_buffers(
        memsys=memsys,
        xyz_buffer_name=xyz_buf_0,
        dist_buffer_name=dist_buf_0,
        xyz_region=xyz_region,
        min_dist_region=min_dist_region,
        tile_start=t0_start,
        tile_n=t0_n,
        point_dim=point_dim,
        request_cycle=tile0_prefetch_issue,
        dram_cfg=dram_cfg,
        output_dir=os.path.join(refill_root, "tile_0_prefetch"),
        use_dram_batch_refill=use_dram_batch_refill,
    )

    cur_cycle = tile0_prefetch_ready
    current_tile_compute_end = 0
    prefetch_ready = {0: tile0_prefetch_ready}
    prefetch_issue = {0: tile0_prefetch_issue}

    for tile_idx, tile_start in enumerate(tile_starts):
        tile_n = min(tile_points, n_points - tile_start)
        xyz_buf, dist_buf, pair_id = buf_names(tile_idx)

        if tile_idx == 0:
            tile_compute_start = cur_cycle
        else:
            tile_compute_start = max(current_tile_compute_end, prefetch_ready[tile_idx])

        # 计算开始时为下一 tile 发起预取
        if tile_idx + 1 < n_tiles:
            next_tile_start = tile_starts[tile_idx + 1]
            next_tile_n = min(tile_points, n_points - next_tile_start)
            next_xyz_buf, next_dist_buf, _ = buf_names(tile_idx + 1)

            next_issue = tile_compute_start
            next_ready = prefetch_tile_into_buffers(
                memsys=memsys,
                xyz_buffer_name=next_xyz_buf,
                dist_buffer_name=next_dist_buf,
                xyz_region=xyz_region,
                min_dist_region=min_dist_region,
                tile_start=next_tile_start,
                tile_n=next_tile_n,
                point_dim=point_dim,
                request_cycle=next_issue,
                dram_cfg=dram_cfg,
                output_dir=os.path.join(refill_root, f"tile_{tile_idx+1}_prefetch"),
                use_dram_batch_refill=use_dram_batch_refill,
            )
            prefetch_issue[tile_idx + 1] = next_issue
            prefetch_ready[tile_idx + 1] = next_ready

        xyz_buffer = memsys.sram_buffers[xyz_buf]
        dist_buffer = memsys.sram_buffers[dist_buf]

        xyz_stat = xyz_buffer.read_region(
            region_name="xyz",
            start_elem=tile_start * point_dim,
            num_elem=tile_n * point_dim,
            bytes_per_elem=4,
            cur_cycle=tile_compute_start,
        )
        c = xyz_stat.end_cycle

        dist_r_stat = dist_buffer.read_region(
            region_name="min_dist",
            start_elem=tile_start,
            num_elem=tile_n,
            bytes_per_elem=4,
            cur_cycle=c,
        )
        c = dist_r_stat.end_cycle

        c += compute_gap
        total_compute_cycles += compute_gap

        dist_w_stat = dist_buffer.write_region(
            region_name="min_dist",
            start_elem=tile_start,
            num_elem=tile_n,
            bytes_per_elem=4,
            cur_cycle=c,
        )
        c = dist_w_stat.end_cycle

        current_tile_compute_end = c

        tile_timeline.append({
            "tile_idx": tile_idx,
            "tile_start_point": tile_start,
            "tile_points": tile_n,
            "buffer_pair": pair_id,
            "prefetch_issue": prefetch_issue.get(tile_idx, None),
            "prefetch_ready": prefetch_ready.get(tile_idx, None),
            "compute_start": tile_compute_start,
            "compute_end": current_tile_compute_end,
        })

    end_to_end_cycles = current_tile_compute_end
    return memsys, end_to_end_cycles, total_compute_cycles, tile_timeline


def print_compare(
    title: str,
    baseline_cycles: int,
    baseline_summary: dict,
    other_cycles: int,
    other_summary: dict,
    freq_hz: int,
    other_name: str,
):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"Baseline cycles: {baseline_cycles}  ({cycles_to_ns(baseline_cycles, freq_hz):.2f} ns)")
    print(f"{other_name} cycles: {other_cycles}  ({cycles_to_ns(other_cycles, freq_hz):.2f} ns)")
    if other_cycles > 0:
        speedup = baseline_cycles / other_cycles
        reduction = (1.0 - other_cycles / baseline_cycles) * 100.0
        print(f"Speedup: {speedup:.4f}x, reduction = {reduction:.2f}%")
    print("")
    print("分项周期：")
    print(f"  SRAM:    baseline {baseline_summary['sram_service_cycles']} -> {other_name} {other_summary['sram_service_cycles']}")
    print(f"  DRAM:    baseline {baseline_summary['dram_refill_cycles']} -> {other_name} {other_summary['dram_refill_cycles']}")
    print(f"  Compute: baseline {baseline_summary['compute_cycles']} -> {other_name} {other_summary['compute_cycles']}")
    print("")
    print("分项能耗 (nJ)：")
    print(f"  SRAM:    baseline {baseline_summary['sram_energy_nj']:.4f} -> {other_name} {other_summary['sram_energy_nj']:.4f}")
    print(f"  DRAM:    baseline {baseline_summary['dram_energy_nj']:.4f} -> {other_name} {other_summary['dram_energy_nj']:.4f}")
    print(f"  Total:   baseline {baseline_summary['sram_energy_nj'] + baseline_summary['dram_energy_nj']:.4f} -> "
          f"{other_name} {other_summary['sram_energy_nj'] + other_summary['dram_energy_nj']:.4f}")


def print_overlap_timeline(tile_timeline, freq_hz: int):
    print("\n" + "=" * 90)
    print("Overlap tile timeline")
    print("=" * 90)
    print(f"{'tile':>4} | {'pair':>4} | {'pf_issue':>9} | {'pf_ready':>9} | {'comp_start':>10} | {'comp_end':>8} | {'pf_wait':>7} | {'hidden_dram':>11}")
    print("-" * 90)

    for ev in tile_timeline:
        tile_idx = ev["tile_idx"]
        pair_id = ev["buffer_pair"]
        pf_issue = ev["prefetch_issue"]
        pf_ready = ev["prefetch_ready"]
        comp_start = ev["compute_start"]
        comp_end = ev["compute_end"]

        pf_wait = 0 if pf_ready is None else max(0, comp_start - pf_ready)
        # 被隐藏的 DRAM latency：prefetch_ready 若早于 compute_start，则这部分被前一 tile compute“吃掉”了
        hidden_dram = 0
        if pf_issue is not None and pf_ready is not None:
            total_pf = pf_ready - pf_issue
            exposed_pf = max(0, pf_ready - comp_start)
            hidden_dram = total_pf - exposed_pf

        print(f"{tile_idx:4d} | {pair_id:4d} | {str(pf_issue):>9} | {str(pf_ready):>9} | {comp_start:10d} | {comp_end:8d} | {pf_wait:7d} | {hidden_dram:11d}")

    print("-" * 90)
    if len(tile_timeline) > 0:
        first_start = tile_timeline[0]["compute_start"]
        last_end = tile_timeline[-1]["compute_end"]
        print(f"Total overlap execution window: {last_end - first_start} cycles ({cycles_to_ns(last_end - first_start, freq_hz):.2f} ns)")
    print("=" * 90)


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cycle_sim_root = os.path.dirname(repo_root)

    freq_hz = 1_000_000_000
    n_points = 1024
    point_dim = 3
    tile_points = 128
    compute_gap = 100

    print("Running baseline: non-overlap, SRAM banks=1, DRAM sequential refill ...")
    memsys_base, base_cycles, base_compute = run_fps_pipeline_non_overlap(
        repo_root=repo_root,
        cycle_sim_root=cycle_sim_root,
        freq_hz=freq_hz,
        sram_num_banks=1,
        use_dram_batch_refill=False,
        n_points=n_points,
        point_dim=point_dim,
        tile_points=tile_points,
        compute_gap=compute_gap,
    )
    base_summary = summarize_run(memsys_base, base_cycles, base_compute)

    print("Running non-overlap improved: SRAM banks=2, DRAM batch refill ...")
    memsys_multi, multi_cycles, multi_compute = run_fps_pipeline_non_overlap(
        repo_root=repo_root,
        cycle_sim_root=cycle_sim_root,
        freq_hz=freq_hz,
        sram_num_banks=2,
        use_dram_batch_refill=True,
        n_points=n_points,
        point_dim=point_dim,
        tile_points=tile_points,
        compute_gap=compute_gap,
    )
    multi_summary = summarize_run(memsys_multi, multi_cycles, multi_compute)

    print("Running overlap: double-buffer, SRAM banks=2, DRAM batch refill ...")
    memsys_ov, ov_cycles, ov_compute, ov_timeline = run_fps_pipeline_overlap(
        repo_root=repo_root,
        cycle_sim_root=cycle_sim_root,
        freq_hz=freq_hz,
        sram_num_banks=2,
        use_dram_batch_refill=True,
        n_points=n_points,
        point_dim=point_dim,
        tile_points=tile_points,
        compute_gap=compute_gap,
    )
    ov_summary = summarize_run(memsys_ov, ov_cycles, ov_compute, tile_timeline=ov_timeline)

    print_compare(
        title="Baseline vs Non-overlap Improved (multi-bank + batch refill)",
        baseline_cycles=base_cycles,
        baseline_summary=base_summary,
        other_cycles=multi_cycles,
        other_summary=multi_summary,
        freq_hz=freq_hz,
        other_name="improved",
    )

    print_compare(
        title="Baseline vs Overlap (double-buffer prefetch)",
        baseline_cycles=base_cycles,
        baseline_summary=base_summary,
        other_cycles=ov_cycles,
        other_summary=ov_summary,
        freq_hz=freq_hz,
        other_name="overlap",
    )

    print_compare(
        title="Non-overlap Improved vs Overlap",
        baseline_cycles=multi_cycles,
        baseline_summary=multi_summary,
        other_cycles=ov_cycles,
        other_summary=ov_summary,
        freq_hz=freq_hz,
        other_name="overlap",
    )

    print_overlap_timeline(ov_timeline, freq_hz)

    print("\n" + "=" * 70)
    print("说明")
    print("=" * 70)
    print("1. baseline:")
    print("   - 单 SRAM bank")
    print("   - xyz/dist refill 串行")
    print("   - tile 间无 overlap")
    print("2. improved:")
    print("   - 多 SRAM bank（block-interleaved）")
    print("   - xyz/dist batch refill 进入同一条 DRAM trace")
    print("   - 但 tile 间仍无 overlap")
    print("3. overlap:")
    print("   - 双缓冲（ping/pong）")
    print("   - 当前 tile compute 时预取下一 tile")
    print("   - 下一 tile 开始时刻 = max(前一 tile compute 结束, 本 tile prefetch ready)")
    print("   - timeline 中 hidden_dram 越大，说明越多 DRAM latency 被前一 tile compute 隐藏")


if __name__ == "__main__":
    main()