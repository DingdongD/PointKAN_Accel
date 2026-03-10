# -*- coding: utf-8 -*-
"""
每周期 PE 活动可视化：根据 OS/WS 数据流与脉动时序，生成每个 cycle 各 PE 的操作（MAC/clear/preload/drain/idle），
并输出为 ASCII 网格或可序列化的 trace，便于调试与展示。
"""
from __future__ import print_function

from typing import List, Dict, Any, Tuple, Optional

# 操作标签（短名便于网格显示）
OP_CLEAR = "CLR"    # 部分和清零
OP_MAC = "MAC"      # 乘加
OP_PRELOAD = "PRE"  # 权重重载（WS）
OP_DRAIN = "DRN"    # 排空/写回
OP_IDLE = "-"       # 空闲


def _os_phase_cycles(cfg) -> Tuple[int, int, int]:
    """OS 模式各阶段周期数，与 cycle_energy 一致"""
    return (
        cfg.timing.os_clear_cycles,
        cfg.timing.os_drain_extra,
        None,  # compute 由 m,k,n 算
    )


def _ws_phase_cycles(cfg) -> Tuple[int, int, int]:
    """WS 模式各阶段周期数"""
    return (
        cfg.timing.ws_preload_cycles,
        cfg.timing.ws_gap_cycles,
        None,
    )


def generate_tile_cycle_activity(
    tile,
    cfg,
    arr_r: int,
    arr_c: int,
    max_cycles: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    为单个 tile 生成每周期的 PE 活动列表。
    tile: TileDesc（需有 mode, m_tile, k_tile, n_tile）
    cfg: SimConfig（用于 timing）
    arr_r, arr_c: 阵列行/列数（用于生成 arr_r x arr_c 网格，超出 tile 范围的 PE 标为 idle）
    返回: [ {"cycle": c, "phase": "clear"|"compute"|"drain"|"preload"|"gap", "grid": [[op]] }, ... ]
    grid 为 arr_r x arr_c，grid[r][c] 为该周期 PE(r,c) 的操作。
    """
    m, k, n = tile.m_tile, tile.k_tile, tile.n_tile
    trace = []
    cycle = 0

    def make_grid(default_op: str):
        g = [[default_op] * arr_c for _ in range(arr_r)]
        return g

    def set_tile_region(grid, op: str, mr: int, nr: int):
        for r in range(min(mr, arr_r)):
            for c in range(min(nr, arr_c)):
                grid[r][c] = op

    if tile.mode == "OS":
        clear_cycles, drain_cycles, _ = _os_phase_cycles(cfg)
        compute_cycles = k + m + n - 2

        # 1) Clear
        for _ in range(clear_cycles):
            grid = make_grid(OP_IDLE)
            set_tile_region(grid, OP_CLEAR, m, n)
            trace.append({"cycle": cycle, "phase": "clear", "grid": grid})
            cycle += 1
            if max_cycles is not None and cycle >= max_cycles:
                return trace

        # 2) Compute：脉动波前，在局部周期 T 时 PE(i,j) 做 MAC 当且仅当 i+j <= T < i+j+k
        for T in range(compute_cycles):
            grid = make_grid(OP_IDLE)
            for i in range(min(m, arr_r)):
                for j in range(min(n, arr_c)):
                    if i + j <= T < i + j + k:
                        grid[i][j] = OP_MAC
            trace.append({"cycle": cycle, "phase": "compute", "grid": grid})
            cycle += 1
            if max_cycles is not None and cycle >= max_cycles:
                return trace

        # 3) Drain
        for _ in range(drain_cycles):
            grid = make_grid(OP_IDLE)
            set_tile_region(grid, OP_DRAIN, m, n)
            trace.append({"cycle": cycle, "phase": "drain", "grid": grid})
            cycle += 1
            if max_cycles is not None and cycle >= max_cycles:
                return trace

    else:
        # WS
        preload_cycles, gap_cycles, _ = _ws_phase_cycles(cfg)
        compute_cycles = m + n - 1

        # 1) Preload：权重重载，本模型中简化为 tile 区域内 PEs 均为 PRE
        for _ in range(preload_cycles):
            grid = make_grid(OP_IDLE)
            set_tile_region(grid, OP_PRELOAD, m, n)
            trace.append({"cycle": cycle, "phase": "preload", "grid": grid})
            cycle += 1
            if max_cycles is not None and cycle >= max_cycles:
                return trace

        # 2) Compute：WS 波前与 OS 类似，PE(i,j) 在 T 属于 [i+j, i+j+k) 时做 MAC（此处用相同规则便于可视化）
        for T in range(compute_cycles):
            grid = make_grid(OP_IDLE)
            for i in range(min(m, arr_r)):
                for j in range(min(n, arr_c)):
                    if i + j <= T < i + j + k:
                        grid[i][j] = OP_MAC
            trace.append({"cycle": cycle, "phase": "compute", "grid": grid})
            cycle += 1
            if max_cycles is not None and cycle >= max_cycles:
                return trace

        # 3) Gap
        for _ in range(gap_cycles):
            grid = make_grid(OP_IDLE)
            set_tile_region(grid, OP_IDLE, m, n)
            trace.append({"cycle": cycle, "phase": "gap", "grid": grid})
            cycle += 1
            if max_cycles is not None and cycle >= max_cycles:
                return trace

    return trace


def format_cycle_grid(grid: List[List[str]], cell_width: int = 4) -> str:
    """将单周期 grid 格式化为 ASCII 表格字符串"""
    lines = []
    for row in grid:
        line = " ".join(cell.center(cell_width) for cell in row)
        lines.append(line)
    return "\n".join(lines)


def print_tile_schedule(
    tile,
    cfg,
    arr_r: int,
    arr_c: int,
    max_cycles: Optional[int] = 25,
    title: Optional[str] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    生成并打印单个 tile 的每周期 PE 活动（前 max_cycles 个周期）。
    verbose=True 时每周期打印一张 ASCII 网格；否则只生成 trace 不打印。
    返回 trace 列表，便于写 JSON。
    """
    trace = generate_tile_cycle_activity(tile, cfg, arr_r, arr_c, max_cycles=max_cycles)
    if title:
        print(title)
    print("Tile: mode={}, m_tile={}, k_tile={}, n_tile={}".format(
        tile.mode, tile.m_tile, tile.k_tile, tile.n_tile))
    print("Phases: {}".format([t["phase"] for t in trace]))
    print("Total cycles shown: {}".format(len(trace)))
    if not verbose:
        return trace
    for entry in trace:
        print("\n--- Cycle {} [{}] ---".format(entry["cycle"], entry["phase"]))
        print(format_cycle_grid(entry["grid"]))
    return trace


def trace_to_serializable(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将 trace 转为可 JSON 序列化的结构（grid 保持为二维列表）。"""
    return [
        {"cycle": e["cycle"], "phase": e["phase"], "grid": e["grid"]}
        for e in trace
    ]
