# -*- coding: utf-8 -*-
"""
demo_fps_knn_with_refill.py —— 在 FPS 每个 tile 的 center 上加入 KNN 的访存建模

在 demo_fps_with_refill 的 FPS 流水线基础上，每个 tile 在完成 FPS（读 xyz、读/写 min_dist、FPS 计算）
之后，增加 KNN 阶段：
  1. 读 KNN 所需数据（本 tile 内 num_centers 个 center 的 K 个邻居的 xyz，或从 DRAM 拉取的 search window）
  2. KNN 计算周期（knn_compute_gap）

访存模型考虑：
  - 方案 A（本 demo 默认）：in-tile KNN —— 邻居均在当前 tile 的 xyz 内，只增加 SRAM 读量（num_centers*K*point_dim）
    与一次 knn_compute_gap，无额外 refill。
  - 方案 B（可选）：cross-tile KNN —— 邻居可能来自其他 tile，需为 KNN 单独 refill 一块 “search window”
    或整帧 xyz，再按 center 做 KNN；可通过 refill_knn_region=True 和 knn_window_elem 扩展。

扩展 cross-tile KNN 时的实现要点：
  - 增加 TensorRegion knn_search_region（或复用 xyz_region 的更大 start_elem/num_elem）。
  - 每个 tile 在 FPS 之后、KNN 读之前：若邻居超出当前 tile，调用 refill_if_needed/refill_batch
    将 search window（例如以 center 为中心的前后若干点）拉入一块专用 buffer（如 knn_xyz_buffer），
    再对该 buffer 做 read_region（num_centers * K * point_dim）。
  - 或对“整帧 xyz 在 DRAM”的模型：先 refill 整帧（或分层 refill），KNN 读时按 center 索引
    做多次 read_region（每次一个 center 的 K 个邻居），并统计 SRAM/DRAM 周期与能耗。

运行方式（在 memory_model 项目根下）：
  python demo/demo_fps_knn_with_refill.py

----------------------------------------------------------------------
KNN 访存建模合理性分析（当前实现 vs 真实 KNN）
----------------------------------------------------------------------
1) 读量（bytes）
   - 当前：一次 read_region，num_elem = min(num_centers*K*point_dim, tile_xyz_elem)。
     总字节数 = 读入的「邻居坐标」上界，与算法语义一致。
   - 合理点：总数据量上界正确；若邻居严格在 tile 内且无重复，实际可能略少（多 center 共享邻居），
     当前按上界建模偏保守，可接受。

2) 访问模式（连续 vs 分散）
   - 真实 KNN：每个 center 的 K 个邻居在内存中通常是分散的（gather），不是连续一段。
   - 当前：用「一段连续 read_region」等效，即假设数据连续存放。
   - 影响：连续读对 SRAM 更友好（局部性好、bank 冲突少），实际 gather 往往更多 cycle。
     当前模型对 SRAM 延迟偏乐观；若要更贴近硬件，可改为多次 read_region（按 center 或按
     neighbor 块）或引入 gather 惩罚系数。

3) 与 FPS 阶段对 xyz 的重复读
   - FPS 已对当前 tile 做 read_region(tile_xyz_elem)，即整块 tile xyz 读过一遍。
   - KNN 再对同一 buffer 做 read_region(knn_read_elem)，且 knn_read_elem 常取满 tile（见下）。
   - 即：同一块 tile xyz 被建模为「读两次」。若硬件在 FPS 后能复用寄存器/局部缓存，则
     KNN 可能不必再访问 SRAM，当前会高估 SRAM 读；若必须回 SRAM 取数则当前合理。
   - 建议：在文档或配置中注明「假设 KNN 阶段不从 FPS 复用 xyz，需重新读 SRAM」，便于与
     实际架构对齐。

4) 默认参数下的实际读量
   - num_centers_per_tile=32, K=16, point_dim=3 → 理论 32*16*3=1536 elem。
   - tile_points=128 → tile_xyz_elem=384，min(1536,384)=384，即 KNN 阶段读「整 tile xyz」。
   - 因此当前默认即「每个 tile 的 KNN 读 = 整块 tile xyz 再读一次」，与 FPS 读的区间重叠，
     重复读已在上面(3)说明。

5) center 索引与 KNN 输入
   - 真实流程：FPS 产出 center 索引，KNN 根据索引取邻居。当前未显式建模「读 center 索引」，
     仅建模「读邻居坐标」。若索引量很小（如每 tile 几十个 int），可忽略或后续加一次小 read。

6) KNN 计算周期
   - 当前用常数 knn_compute_gap，属一阶近似。更细可写为 f(num_centers, K, search_size)，
     与具体实现（暴力/排序/近似）一致。

结论：当前 KNN 访存建模在「读量上界」和「in-tile 假设」下合理，适合做周期/能耗上界分析；
     连续读与重复读两点会使 SRAM 周期偏乐观或偏多，可根据是否复用 xyz、是否做 gather 建模
     做后续细化。
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
    """与 demo_fps_with_refill 一致的 SRAM/DRAM 内存系统构建。"""
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

    return MemorySystem(
        sram_buffers=sram_buffers,
        dramsim3_runner=dramsim3_runner,
    )


def summarize_run(memsys: MemorySystem, end_to_end_cycles: int, compute_cycles: int, tile_timeline=None):
    """汇总 SRAM/DRAM 周期与能耗，与 demo_fps_with_refill 一致。"""
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


def run_fps_knn_pipeline_non_overlap(
    repo_root: str,
    cycle_sim_root: str,
    freq_hz: int,
    sram_num_banks: int = 1,
    use_dram_batch_refill: bool = False,
    n_points: int = 1024,
    point_dim: int = 3,
    tile_points: int = 128,
    compute_gap: int = 100,
    num_centers_per_tile: int = 32,
    K: int = 16,
    knn_compute_gap: int = 80,
):
    """
    FPS + KNN 流水线（non-overlap）：每个 tile 先做 FPS，再做 in-tile KNN 访存与计算。

    KNN 建模：从当前 tile 的 xyz buffer 中读取 num_centers_per_tile * K * point_dim 个 float（邻居坐标），
    然后消耗 knn_compute_gap 周期。不增加 DRAM refill（邻居假定在本 tile 内）。
    """
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
    refill_root = os.path.join(repo_root, "outputs", "dramsim3", "fps_knn_refill_runs_non_overlap")

    cur_cycle = 0
    total_compute_cycles = 0
    total_knn_compute_cycles = 0

    # KNN 读体积：每个 center 读 K 个邻居，每个邻居 point_dim 个 float
    knn_read_elem_per_tile = num_centers_per_tile * K * point_dim
    knn_read_bytes_per_tile = knn_read_elem_per_tile * 4

    for tile_start in range(0, n_points, tile_points):
        tile_n = min(tile_points, n_points - tile_start)
        tile_xyz_elem = tile_n * point_dim

        # Refill
        if use_dram_batch_refill:
            batch_dir = os.path.join(refill_root, f"batch_tile_{tile_start}")
            cur_cycle = memsys.refill_batch(
                refill_requests=[
                    {
                        "buffer_name": "xyz_buffer",
                        "tensor_region": xyz_region,
                        "start_elem": tile_start * point_dim,
                        "num_elem": tile_xyz_elem,
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
                num_elem=tile_xyz_elem,
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

        # FPS: read xyz
        xyz_stat = xyz_buffer.read_region(
            region_name="xyz",
            start_elem=tile_start * point_dim,
            num_elem=tile_xyz_elem,
            bytes_per_elem=4,
            cur_cycle=cur_cycle,
        )
        cur_cycle = xyz_stat.end_cycle

        # FPS: read min_dist
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

        # FPS: write min_dist
        dist_w_stat = dist_buffer.write_region(
            region_name="min_dist",
            start_elem=tile_start,
            num_elem=tile_n,
            bytes_per_elem=4,
            cur_cycle=cur_cycle,
        )
        cur_cycle = dist_w_stat.end_cycle

        # KNN: 从同一 tile 的 xyz buffer 读邻居坐标（建模为一段连续读，上界；真实 KNN 为 gather 可能更多 cycle）
        # 注意：FPS 已读过本 tile 的 xyz，此处为「不复用、再读一次」的假设
        knn_read_elem = min(knn_read_elem_per_tile, tile_xyz_elem)
        if knn_read_elem > 0:
            knn_stat = xyz_buffer.read_region(
                region_name="xyz",
                start_elem=tile_start * point_dim,
                num_elem=knn_read_elem,
                bytes_per_elem=4,
                cur_cycle=cur_cycle,
            )
            cur_cycle = knn_stat.end_cycle

        cur_cycle += knn_compute_gap
        total_knn_compute_cycles += knn_compute_gap

    return memsys, cur_cycle, total_compute_cycles, total_knn_compute_cycles


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
    """与 demo_fps_with_refill 一致的 prefetch 辅助。"""
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


def run_fps_knn_pipeline_overlap(
    repo_root: str,
    cycle_sim_root: str,
    freq_hz: int,
    sram_num_banks: int = 1,
    use_dram_batch_refill: bool = True,
    n_points: int = 1024,
    point_dim: int = 3,
    tile_points: int = 128,
    compute_gap: int = 100,
    num_centers_per_tile: int = 32,
    K: int = 16,
    knn_compute_gap: int = 80,
):
    """
    FPS + KNN 流水线（overlap）：双缓冲，当前 tile 计算时预取下一 tile；
    每个 tile 内顺序：FPS（读 xyz、读 dist、compute、写 dist）-> KNN 读 -> KNN compute。
    """
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
    refill_root = os.path.join(repo_root, "outputs", "dramsim3", "fps_knn_refill_runs_overlap")

    tile_starts = list(range(0, n_points, tile_points))
    n_tiles = len(tile_starts)

    knn_read_elem_per_tile = num_centers_per_tile * K * point_dim

    total_compute_cycles = 0
    total_knn_compute_cycles = 0
    tile_timeline = []

    def buf_names(tile_idx: int):
        buf_id = tile_idx % 2
        return f"xyz_buffer_{buf_id}", f"dist_buffer_{buf_id}", buf_id

    # Prefetch tile 0
    t0_start = tile_starts[0]
    t0_n = min(tile_points, n_points - t0_start)
    xyz_buf_0, dist_buf_0, _ = buf_names(0)
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
        tile_xyz_elem = tile_n * point_dim
        xyz_buf, dist_buf, pair_id = buf_names(tile_idx)

        if tile_idx == 0:
            tile_compute_start = cur_cycle
        else:
            tile_compute_start = max(current_tile_compute_end, prefetch_ready[tile_idx])

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

        c = tile_compute_start
        xyz_stat = xyz_buffer.read_region(
            region_name="xyz",
            start_elem=tile_start * point_dim,
            num_elem=tile_xyz_elem,
            bytes_per_elem=4,
            cur_cycle=c,
        )
        xyz_read_end = xyz_stat.end_cycle
        c = xyz_read_end

        dist_r_stat = dist_buffer.read_region(
            region_name="min_dist",
            start_elem=tile_start,
            num_elem=tile_n,
            bytes_per_elem=4,
            cur_cycle=c,
        )
        dist_read_end = dist_r_stat.end_cycle
        c = dist_read_end

        fps_compute_start = c
        c += compute_gap
        fps_compute_end = c
        total_compute_cycles += compute_gap

        dist_w_stat = dist_buffer.write_region(
            region_name="min_dist",
            start_elem=tile_start,
            num_elem=tile_n,
            bytes_per_elem=4,
            cur_cycle=c,
        )
        dist_write_end = dist_w_stat.end_cycle
        c = dist_write_end

        knn_read_start = c
        # KNN 读：连续读上界，假设不复用 FPS 已读的 xyz；真实 gather 可能更多 SRAM cycle
        knn_read_elem = min(knn_read_elem_per_tile, tile_xyz_elem)
        if knn_read_elem > 0:
            knn_stat = xyz_buffer.read_region(
                region_name="xyz",
                start_elem=tile_start * point_dim,
                num_elem=knn_read_elem,
                bytes_per_elem=4,
                cur_cycle=c,
            )
            c = knn_stat.end_cycle
        knn_read_end = c

        knn_compute_start = c
        c += knn_compute_gap
        total_knn_compute_cycles += knn_compute_gap
        knn_compute_end = c

        current_tile_compute_end = c

        tile_timeline.append({
            "tile_idx": tile_idx,
            "tile_start_point": tile_start,
            "tile_points": tile_n,
            "buffer_pair": pair_id,
            "prefetch_issue": prefetch_issue.get(tile_idx),
            "prefetch_ready": prefetch_ready.get(tile_idx),
            "compute_start": tile_compute_start,
            "compute_end": current_tile_compute_end,
            "xyz_read_start": tile_compute_start,
            "xyz_read_end": xyz_read_end,
            "dist_read_start": xyz_read_end,
            "dist_read_end": dist_read_end,
            "fps_compute_start": fps_compute_start,
            "fps_compute_end": fps_compute_end,
            "dist_write_start": fps_compute_end,
            "dist_write_end": dist_write_end,
            "knn_read_start": knn_read_start,
            "knn_read_end": knn_read_end,
            "knn_compute_start": knn_compute_start,
            "knn_compute_end": knn_compute_end,
        })

    return memsys, current_tile_compute_end, total_compute_cycles, total_knn_compute_cycles, tile_timeline


def print_overlap_timeline_detail(tile_timeline, freq_hz: int):
    """打印每个 tile 各阶段起止 cycle 的详细表格。"""
    print("\n" + "=" * 120)
    print("Overlap 详细 Timeline（每 tile 各操作起止 cycle）")
    print("=" * 120)
    hdr = (
        f"{'tile':>4} | {'pf_issue':>8} | {'pf_ready':>8} | "
        f"{'SRAM_xyz':>12} | {'SRAM_dist_r':>12} | {'FPS_comp':>10} | "
        f"{'SRAM_dist_w':>12} | {'KNN_read':>10} | {'KNN_comp':>10} | {'tile_end':>8}"
    )
    print(hdr)
    print("-" * 120)
    for ev in tile_timeline:
        ti = ev["tile_idx"]
        pf_i = ev.get("prefetch_issue")
        pf_r = ev.get("prefetch_ready")
        xyz_s, xyz_e = ev.get("xyz_read_start"), ev.get("xyz_read_end")
        dr_s, dr_e = ev.get("dist_read_start"), ev.get("dist_read_end")
        fps_s, fps_e = ev.get("fps_compute_start"), ev.get("fps_compute_end")
        dw_s, dw_e = ev.get("dist_write_start"), ev.get("dist_write_end")
        kr_s, kr_e = ev.get("knn_read_start"), ev.get("knn_read_end")
        kc_s, kc_e = ev.get("knn_compute_start"), ev.get("knn_compute_end")
        end = ev["compute_end"]
        print(
            f"{ti:4d} | {str(pf_i):>8} | {str(pf_r):>8} | "
            f"{xyz_s}-{xyz_e:>6} | {dr_s}-{dr_e:>6} | {fps_s}-{fps_e:>4} | "
            f"{dw_s}-{dw_e:>6} | {kr_s}-{kr_e:>4} | {kc_s}-{kc_e:>4} | {end:8d}"
        )
    print("-" * 120)
    if tile_timeline:
        first = tile_timeline[0]["compute_start"]
        last = tile_timeline[-1]["compute_end"]
        print(f"总执行窗口: {first} ~ {last}  (共 {last - first} cycles, {cycles_to_ns(last - first, freq_hz):.2f} ns)")
    print("=" * 120)


def print_operation_bands(tile_timeline):
    """
    按时间轴打印「哪些 cycle 区间内有哪些操作在进行」。
    便于理解 overlap：同一 cycle 区间内 DRAM 预取与计算可能并行。
    """
    print("\n" + "=" * 100)
    print("各 cycle 区间内正在进行的操作（DRAM=预取, SRAM=读写, FPS/KNN=计算）")
    print("=" * 100)

    events = []
    for ev in tile_timeline:
        ti = ev["tile_idx"]
        if ev.get("prefetch_issue") is not None and ev.get("prefetch_ready") is not None:
            events.append((ev["prefetch_issue"], ev["prefetch_ready"], "DRAM", f"prefetch_tile_{ti}"))
        events.append((ev["xyz_read_start"], ev["xyz_read_end"], "SRAM", f"tile_{ti}_read_xyz"))
        events.append((ev["dist_read_start"], ev["dist_read_end"], "SRAM", f"tile_{ti}_read_dist"))
        events.append((ev["fps_compute_start"], ev["fps_compute_end"], "FPS", f"tile_{ti}_fps_compute"))
        events.append((ev["dist_write_start"], ev["dist_write_end"], "SRAM", f"tile_{ti}_write_dist"))
        events.append((ev["knn_read_start"], ev["knn_read_end"], "SRAM", f"tile_{ti}_knn_read"))
        events.append((ev["knn_compute_start"], ev["knn_compute_end"], "KNN", f"tile_{ti}_knn_compute"))

    events.sort(key=lambda x: (x[0], x[1]))
    cycle_ranges = set()
    for a, b, _, _ in events:
        cycle_ranges.add(a)
        cycle_ranges.add(b)
    points = sorted(cycle_ranges)

    for i in range(len(points) - 1):
        lo, hi = points[i], points[i + 1]
        if lo == hi:
            continue
        active = []
        for (s, e, typ, label) in events:
            if s < hi and e > lo:
                active.append(f"{typ}:{label}")
        active_str = "; ".join(active) if active else "(idle)"
        print(f"  cycles {lo:5d} ~ {hi:5d}  ({hi - lo:4d} cy)  |  {active_str}")
    print("=" * 100)


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cycle_sim_root = os.path.dirname(repo_root)

    freq_hz = 1_000_000_000
    n_points = 1024
    point_dim = 3
    tile_points = 128
    compute_gap = 100
    num_centers_per_tile = 32
    K = 16
    knn_compute_gap = 80

    print("FPS+KNN 参数: num_centers_per_tile=%d, K=%d, knn_compute_gap=%d" % (
        num_centers_per_tile, K, knn_compute_gap))
    print("KNN 访存模型: in-tile（邻居取自当前 tile xyz buffer），无额外 DRAM refill\n")

    print("Running FPS+KNN non-overlap (SRAM banks=1, DRAM sequential) ...")
    memsys_base, base_cycles, base_fps_compute, base_knn_compute = run_fps_knn_pipeline_non_overlap(
        repo_root=repo_root,
        cycle_sim_root=cycle_sim_root,
        freq_hz=freq_hz,
        sram_num_banks=1,
        use_dram_batch_refill=False,
        n_points=n_points,
        point_dim=point_dim,
        tile_points=tile_points,
        compute_gap=compute_gap,
        num_centers_per_tile=num_centers_per_tile,
        K=K,
        knn_compute_gap=knn_compute_gap,
    )
    base_summary = summarize_run(
        memsys_base, base_cycles,
        base_fps_compute + base_knn_compute,
    )

    print("Running FPS+KNN overlap (double-buffer, SRAM banks=2, DRAM batch) ...")
    memsys_ov, ov_cycles, ov_fps_compute, ov_knn_compute, ov_timeline = run_fps_knn_pipeline_overlap(
        repo_root=repo_root,
        cycle_sim_root=cycle_sim_root,
        freq_hz=freq_hz,
        sram_num_banks=2,
        use_dram_batch_refill=True,
        n_points=n_points,
        point_dim=point_dim,
        tile_points=tile_points,
        compute_gap=compute_gap,
        num_centers_per_tile=num_centers_per_tile,
        K=K,
        knn_compute_gap=knn_compute_gap,
    )
    ov_summary = summarize_run(
        memsys_ov, ov_cycles,
        ov_fps_compute + ov_knn_compute,
        tile_timeline=ov_timeline,
    )

    print("\n" + "=" * 70)
    print("FPS+KNN: Non-overlap vs Overlap")
    print("=" * 70)
    print("Total cycles:  non_overlap %d  overlap %d" % (base_cycles, ov_cycles))
    print("FPS compute:   non_overlap %d  overlap %d" % (base_fps_compute, ov_fps_compute))
    print("KNN compute:   non_overlap %d  overlap %d" % (base_knn_compute, ov_knn_compute))
    print("SRAM cycles:  non_overlap %d  overlap %d" % (
        base_summary["sram_service_cycles"], ov_summary["sram_service_cycles"]))
    print("DRAM cycles:  non_overlap %d  overlap %d" % (
        base_summary["dram_refill_cycles"], ov_summary["dram_refill_cycles"]))
    if ov_cycles > 0:
        print("Speedup: %.4fx" % (base_cycles / ov_cycles))
    print("=" * 70)

    print("\n【为何 overlap 下 DRAM cycles 反而更大？】")
    print("  - \"DRAM cycles\" 当前定义为：refill_history 中每条记录的 (dram_done_cycle - request_cycle) 之和。")
    print("  - Non-overlap: 每 tile 两次单独 refill（xyz、dist 各一次），共 16 条记录，每条是「单次」refill 的延迟。")
    print("  - Overlap: 每 tile 一次 refill_batch（xyz+dist 合并一条 trace），但 refill_history 里按请求记 2 条，")
    print("    每条都用了「整批完成时间 - 请求时间」，即同一批的延迟被加了 2 遍；且一批含更多 transaction，")
    print("    整批 wall-clock 延迟也大于单次 refill。所以总和会大于 non-overlap。")
    print("  - 实际 wall-clock 上 overlap 的 DRAM 与 compute 并行，总端到端周期是下降的（Speedup 2.x）。")

    print_overlap_timeline_detail(ov_timeline, freq_hz)
    print_operation_bands(ov_timeline)

    print("\n说明：")
    print("  - 每个 tile：Refill(xyz+dist) -> FPS(读xyz,读dist,计算,写dist) -> KNN(读邻居xyz) -> KNN计算")
    print("  - KNN 读量: num_centers_per_tile * K * point_dim * 4 字节（本 tile 内，无额外 refill）")
    print("  - 若需 cross-tile KNN（邻居来自其他 tile）：可增加 knn_search_window refill 与对应读/计算建模")
    print("  - KNN 访存建模合理性（连续读 vs gather、与 FPS 重复读等）见本文件头部 docstring 分析。")


if __name__ == "__main__":
    main()
