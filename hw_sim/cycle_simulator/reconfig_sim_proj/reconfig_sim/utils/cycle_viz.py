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


# ---------- 图片导出：完整过程（所有 tile × 每 cycle） ----------
_OP_COLORS = {
    OP_CLEAR: "#4A90D9",     # 蓝
    OP_MAC: "#50C878",       # 绿
    OP_PRELOAD: "#E6B800",   # 黄
    OP_DRAIN: "#E07850",     # 橙
    OP_IDLE: "#E8E8E8",     # 浅灰
}


def _render_grid_to_image(
    grid: List[List[str]],
    path: str,
    title: str = "",
    cell_size: float = 0.8,
) -> bool:
    """将单周期 grid 画成彩色网格图并保存为 PNG。需 matplotlib。返回是否成功。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return False
    arr_r, arr_c = len(grid), len(grid[0]) if grid else 0
    if arr_r == 0 or arr_c == 0:
        return False
    fig, ax = plt.subplots(1, 1, figsize=(cell_size * arr_c, cell_size * arr_r))
    for i in range(arr_r):
        for j in range(arr_c):
            op = grid[i][j]
            color = _OP_COLORS.get(op, "#E8E8E8")
            rect = mpatches.Rectangle((j, arr_r - 1 - i), 1, 1, facecolor=color, edgecolor="gray", linewidth=0.5)
            ax.add_patch(rect)
            ax.text(j + 0.5, arr_r - 1 - i + 0.5, op, ha="center", va="center", fontsize=8)
    ax.set_xlim(0, arr_c)
    ax.set_ylim(0, arr_r)
    ax.set_aspect("equal")
    ax.set_xticks(range(arr_c))
    ax.set_yticks(range(arr_r))
    ax.set_xticklabels(range(arr_c))
    ax.set_yticklabels(range(arr_r - 1, -1, -1))
    if title:
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    return True


def export_report_to_images(
    report: Dict[str, Any],
    cfg,
    output_dir: str,
    op_type: str = "gemm",
    arr_r: Optional[int] = None,
    arr_c: Optional[int] = None,
) -> Tuple[int, str]:
    """
    将报告中的「所有 tile × 每 cycle」导出为 PNG 图片序列，便于查看完整数据流过程。
    report: run_gemm 或 run_conv2d 的返回（需含 'trace'，每项含 'tile' 字典）。
    output_dir: 图片保存目录，若不存在会创建。
    op_type: 'gemm' 或 'conv2d'，仅用于文件名前缀。
    返回 (总帧数, output_dir)。
    """
    import os
    from ..models.tile import TileDesc

    arr_r = arr_r or getattr(cfg.array, "arr_r", 8)
    arr_c = arr_c or getattr(cfg.array, "arr_c", 8)
    traces = report.get("trace") or []
    if not traces:
        return 0, output_dir

    os.makedirs(output_dir, exist_ok=True)
    frame_count = 0
    for tile_idx, t_stats in enumerate(traces):
        t = t_stats.get("tile") or {}
        tile = TileDesc(
            op_type=t.get("op_type", op_type),
            mode=t.get("mode", "OS"),
            m0=t.get("m0", 0),
            k0=t.get("k0", 0),
            n0=t.get("n0", 0),
            m_tile=t.get("m_tile", arr_r),
            k_tile=t.get("k_tile", arr_r),
            n_tile=t.get("n_tile", arr_c),
            batch_idx=t.get("batch_idx", 0),
            extra=t.get("extra"),
        )
        trace = generate_tile_cycle_activity(tile, cfg, arr_r, arr_c, max_cycles=None)
        for entry in trace:
            title = "{} Tile {} | global frame {} | cycle {} [{}]".format(
                op_type, tile_idx, frame_count, entry["cycle"], entry["phase"]
            )
            path = os.path.join(output_dir, "frame_{:05d}.png".format(frame_count))
            _render_grid_to_image(entry["grid"], path, title=title)
            frame_count += 1

    if frame_count > 0:
        index_path = os.path.join(output_dir, "index.html")
        with open(index_path, "w", encoding="utf-8") as f:
            f.write("<!DOCTYPE html><html><head><meta charset='utf-8'><title>Systolic Frames</title></head><body>\n")
            f.write("<p>Total frames: {} (all tiles, every cycle).</p>\n".format(frame_count))
            f.write("<p>Legend: CLR=clear, MAC=mac, PRE=preload, DRN=drain, -=idle</p>\n")
            for i in range(frame_count):
                f.write('<img src="frame_{:05d}.png" alt="frame {}" style="max-width:360px; margin:4px;" />\n'.format(i, i))
            f.write("</body></html>\n")
        # 将 frame_*.png 拼成一张 GIF 动画
        gif_path = _frames_to_gif(output_dir, gif_filename="animation.gif", duration_ms=200)
        if gif_path:
            index_path = os.path.join(output_dir, "index.html")
            with open(index_path, "a", encoding="utf-8") as f:
                f.write('<p><a href="animation.gif">animation.gif</a> (all frames)</p>\n')
    return frame_count, output_dir


def _frames_to_gif(
    output_dir: str,
    gif_filename: str = "animation.gif",
    duration_ms: int = 200,
) -> Optional[str]:
    """
    将 output_dir 下按数字序的 frame_*.png 拼成一张 GIF，保存为 output_dir/gif_filename。
    duration_ms: 每帧持续时间（毫秒）。返回 GIF 路径，失败返回 None。
    """
    import os
    import glob
    import re
    pattern = os.path.join(output_dir, "frame_*.png")
    files = sorted(glob.glob(pattern), key=lambda p: int(re.search(r"frame_(\d+)\.png", p).group(1)))
    if not files:
        return None
    gif_path = os.path.join(output_dir, gif_filename)
    duration_sec = duration_ms / 1000.0
    # 优先使用 imageio，GIF 质量更好
    try:
        import imageio
        frames = [imageio.v2.imread(p) for p in files]
        imageio.mimsave(gif_path, frames, duration=duration_sec, loop=0)
        return gif_path
    except ImportError:
        pass
    # 回退到 Pillow（首帧须为 P 模式，后续帧会按首帧调色板转换）
    try:
        from PIL import Image
        images_rgb = [Image.open(p).convert("RGB") for p in files]
        first_p = images_rgb[0].quantize(colors=256)
        first_p.save(
            gif_path,
            save_all=True,
            append_images=images_rgb[1:],
            duration=duration_ms,
            loop=0,
        )
        return gif_path
    except Exception:
        return None
