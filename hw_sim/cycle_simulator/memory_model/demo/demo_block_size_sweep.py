# -*- coding: utf-8 -*-
"""
block_size 参数实验：扫描不同 block_size，统计 SRAM/DRAM/Compute 的延迟与能耗，
并生成堆叠占比图，输出到 outputs/block_size_experiment/。
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


# CACTI 要求 cache 内块数不能太少，否则报 "no valid data array organizations found"。
# 对 CACTI 使用的 size 做 round-up，保证至少该块数；逻辑容量由 capacity_bytes 单独指定。
MIN_BLOCKS_FOR_CACTI = 32


def run_one_experiment(
    block_size_bytes: int,
    repo_root: str,
    cycle_sim_root: str,
    freq_hz: int,
    n_points: int = 1024,
    point_dim: int = 3,
    tile_points: int = 128,
    compute_gap: int = 100,
) -> dict:
    """跑一次 FPS 流水线，使用给定 block_size。返回各分项周期与能耗。"""
    bus_bits = block_size_bytes * 8
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

    xyz_logical_bytes = 4 * 1024
    dist_logical_bytes = 2 * 1024
    xyz_cacti_bytes = max(xyz_logical_bytes, MIN_BLOCKS_FOR_CACTI * block_size_bytes)
    dist_cacti_bytes = max(dist_logical_bytes, MIN_BLOCKS_FOR_CACTI * block_size_bytes)

    xyz_buffer = SRAMBuffer(
        name="xyz_buffer",
        cacti_runner=cacti_runner,
        cacti_cfg=CactiInput(
            size_bytes=xyz_cacti_bytes,
            block_size_bytes=block_size_bytes,
            output_bus_width_bits=bus_bits,
            cache_type="ram",
            technology_u=0.028,
            associativity=1,
            read_write_port=1,
            exclusive_read_port=0,
            exclusive_write_port=0,
            single_ended_read_ports=0,
            uca_bank_count=1,
        ),
        freq_hz=freq_hz,
        capacity_bytes=xyz_logical_bytes,
    )
    dist_buffer = SRAMBuffer(
        name="dist_buffer",
        cacti_runner=cacti_runner,
        cacti_cfg=CactiInput(
            size_bytes=dist_cacti_bytes,
            block_size_bytes=block_size_bytes,
            output_bus_width_bits=bus_bits,
            cache_type="ram",
            technology_u=0.028,
            associativity=1,
            read_write_port=1,
            exclusive_read_port=0,
            exclusive_write_port=0,
            single_ended_read_ports=0,
            uca_bank_count=1,
        ),
        freq_hz=freq_hz,
        capacity_bytes=dist_logical_bytes,
    )

    memsys = MemorySystem(
        sram_buffers={"xyz_buffer": xyz_buffer, "dist_buffer": dist_buffer},
        dramsim3_runner=dramsim3_runner,
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
    refill_root = os.path.join(
        repo_root, "outputs", "dramsim3", "block_size_sweep", f"block_{block_size_bytes}"
    )

    cur_cycle = 0
    total_compute_cycles = 0

    for tile_start in range(0, n_points, tile_points):
        tile_n = min(tile_points, n_points - tile_start)
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

    summary = memsys.summary()
    buffers = summary.get("buffers", {})
    refill_history = summary.get("refill_history", [])

    sram_cycles = 0
    sram_energy_nj = 0.0
    for _, buf_sum in buffers.items():
        rs = buf_sum.get("runtime_stats", {})
        sram_cycles += int(rs.get("total_busy_cycles", 0))
        sram_energy_nj += float(buf_sum.get("total_energy_nj", 0) or 0)

    dram_cycles = 0
    dram_energy_nj = 0.0
    for ev in refill_history:
        dram_cycles += int(ev.get("dram_latency_cycles", 0))
        en = ev.get("dram_energy_nj")
        if en is not None:
            dram_energy_nj += float(en)

    return {
        "block_size": block_size_bytes,
        "sram_cycles": sram_cycles,
        "dram_cycles": dram_cycles,
        "compute_cycles": total_compute_cycles,
        "total_cycles": cur_cycle,
        "sram_energy_nj": sram_energy_nj,
        "dram_energy_nj": dram_energy_nj,
        "total_energy_nj": sram_energy_nj + dram_energy_nj,
    }


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cycle_sim_root = os.path.dirname(repo_root)
    freq_hz = 500_000_000

    # 可修改为更多/更少取值，如 [8, 16, 32, 64, 128]
    block_sizes = [8, 16, 32, 64, 128]
    results = []
    for bs in block_sizes:
        print(f"Running block_size={bs} ...")
        r = run_one_experiment(
            block_size_bytes=bs,
            repo_root=repo_root,
            cycle_sim_root=cycle_sim_root,
            freq_hz=freq_hz,
        )
        results.append(r)
        print(f"  total_cycles={r['total_cycles']}, sram={r['sram_cycles']}, dram={r['dram_cycles']}, compute={r['compute_cycles']}")
        print(f"  total_energy_nj={r['total_energy_nj']:.2f}, sram={r['sram_energy_nj']:.2f}, dram={r['dram_energy_nj']:.2f}")

    out_dir = os.path.join(repo_root, "outputs", "block_size_experiment")
    os.makedirs(out_dir, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib 未安装，跳过画图。pip install matplotlib 后重试。")
        with open(os.path.join(out_dir, "block_size_sweep_results.json"), "w") as f:
            import json
            json.dump(results, f, indent=2)
        return

    labels = [f"{r['block_size']} B" for r in results]
    x = np.arange(len(labels))
    width = 0.6

    # ---------- 1. 延迟堆叠占比图（周期） ----------
    sram_c = [r["sram_cycles"] for r in results]
    dram_c = [r["dram_cycles"] for r in results]
    comp_c = [r["compute_cycles"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, sram_c, width, label="SRAM", color="#2ecc71")
    ax.bar(x, dram_c, width, bottom=sram_c, label="DRAM", color="#3498db")
    ax.bar(x, comp_c, width, bottom=[s + d for s, d in zip(sram_c, dram_c)], label="Compute", color="#9b59b6")
    ax.set_ylabel("Cycles", fontsize=12)
    ax.set_xlabel("Block size (bytes)", fontsize=12)
    ax.set_title("Latency breakdown by block_size (stacked, cycles)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    total_cycles = [r["total_cycles"] for r in results]
    for i, (s, d, c) in enumerate(zip(sram_c, dram_c, comp_c)):
        t = total_cycles[i]
        if t <= 0:
            continue
        if s > 0 and s / t > 0.05:
            ax.text(i, s / 2, f"{100*s/t:.0f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        if d > 0 and d / t > 0.05:
            ax.text(i, s + d / 2, f"{100*d/t:.0f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        if c > 0 and c / t > 0.05:
            ax.text(i, s + d + c / 2, f"{100*c/t:.0f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "latency_breakdown_stacked.pdf"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "latency_breakdown_stacked.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved latency breakdown -> {out_dir}/latency_breakdown_stacked.pdf|.png")

    # ---------- 2. 能耗堆叠占比图（nJ） ----------
    sram_e = [r["sram_energy_nj"] for r in results]
    dram_e = [r["dram_energy_nj"] for r in results]

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(x, sram_e, width, label="SRAM", color="#2ecc71")
    ax2.bar(x, dram_e, width, bottom=sram_e, label="DRAM", color="#3498db")
    ax2.set_ylabel("Energy (nJ)", fontsize=12)
    ax2.set_xlabel("Block size (bytes)", fontsize=12)
    ax2.set_title("Energy breakdown by block_size (stacked, nJ)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend(loc="upper right")
    ax2.grid(axis="y", alpha=0.3)

    total_energy = [r["total_energy_nj"] for r in results]
    for i, (s, d) in enumerate(zip(sram_e, dram_e)):
        t = total_energy[i]
        if t <= 0:
            continue
        if s > 0 and s / t > 0.05:
            ax2.text(i, s / 2, f"{100*s/t:.0f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        if d > 0 and d / t > 0.05:
            ax2.text(i, s + d / 2, f"{100*d/t:.0f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "energy_breakdown_stacked.pdf"), dpi=150, bbox_inches="tight")
    fig2.savefig(os.path.join(out_dir, "energy_breakdown_stacked.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved energy breakdown -> {out_dir}/energy_breakdown_stacked.pdf|.png")

    # ---------- 3. 可选：总周期 / 总能耗 随 block_size 变化（折线） ----------
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(10, 4))
    ax3a.plot(labels, total_cycles, "o-", color="#e74c3c", linewidth=2, markersize=8)
    ax3a.set_ylabel("Total cycles")
    ax3a.set_xlabel("Block size (bytes)")
    ax3a.set_title("End-to-end cycles vs block_size")
    ax3a.grid(True, alpha=0.3)
    ax3b.plot(labels, total_energy, "s-", color="#e67e22", linewidth=2, markersize=8)
    ax3b.set_ylabel("Total energy (nJ)")
    ax3b.set_xlabel("Block size (bytes)")
    ax3b.set_title("Total energy vs block_size")
    ax3b.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(os.path.join(out_dir, "total_latency_energy_vs_block_size.pdf"), dpi=150, bbox_inches="tight")
    fig3.savefig(os.path.join(out_dir, "total_latency_energy_vs_block_size.png"), dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved total latency/energy vs block_size -> {out_dir}/")

    import json
    with open(os.path.join(out_dir, "block_size_sweep_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results JSON -> {out_dir}/block_size_sweep_results.json")
    print("Done.")


if __name__ == "__main__":
    main()
