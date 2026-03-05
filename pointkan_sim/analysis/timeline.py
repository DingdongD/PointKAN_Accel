"""
按 Cycle 的模块活动时间线
========================

根据仿真后的 Event 列表，输出「不同 cycle 下有哪些模块在工作」，
便于分析流水线与数据供给关系（前段算完可供给后段）。
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from ..simulator.event import Event


def get_active_at_cycle(events: List[Event], cycle: int) -> List[Event]:
    """返回在给定 cycle 处于活跃的所有事件。"""
    return [e for e in events if e.cycle <= cycle <= e.cycle + getattr(e, "duration", 1) - 1]


def build_module_activity_timeline(
    events: List[Event],
    cycle_stride: int = 1,
    max_cycles: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    按 cycle 步长采样，输出每个采样 cycle 下正在工作的模块与层。

    用于查看「不同 cycle 下有哪些模块在工作」；当前为顺序执行，
    前一层 buffer 写完后下一层才做 buffer 读与计算，时间线上可看到
    数据供给顺序（前段算完供给后段）。

    Args:
        events: 仿真后的全部事件（含 cycle、duration、circuit_module、metadata）
        cycle_stride: 采样步长（每 stride 个 cycle 输出一行，默认 1 可改为 100/1000 以减小输出）
        max_cycles: 最大 cycle 上限，不传则用 events 的 max(cycle+duration)

    Returns:
        [ {"cycle": c, "active_modules": [...], "active_layers": [...], "active_operators": [...] }, ... ]
    """
    if not events:
        return []

    end_cycle = max(e.cycle + getattr(e, "duration", 1) - 1 for e in events)
    if max_cycles is not None:
        end_cycle = min(end_cycle, max_cycles)

    timeline: List[Dict[str, Any]] = []
    for c in range(0, end_cycle + 1, max(1, cycle_stride)):
        active = get_active_at_cycle(events, c)
        modules: Set[str] = set()
        layers: Set[str] = set()
        operators: Set[str] = set()
        for e in active:
            if e.circuit_module:
                modules.add(e.circuit_module)
            if (e.metadata or {}).get("layer_name"):
                layers.add(e.metadata["layer_name"])
            if e.operator:
                operators.add(e.operator)
        timeline.append({
            "cycle": c,
            "active_modules": sorted(modules),
            "active_layers": sorted(layers),
            "active_operators": sorted(operators),
        })
    return timeline


def build_module_activity_segments(
    events: List[Event],
) -> List[Dict[str, Any]]:
    """
    将时间线压缩为「区间段」：每段内活跃的模块集合不变。
    便于看「从 cycle A 到 B 期间，这些模块在工作」。

    Returns:
        [ {"cycle_start": int, "cycle_end": int, "active_modules": [...], "active_layers": [...] }, ... ]
    """
    if not events:
        return []

    boundaries: List[Tuple[int, int, Event]] = []
    for e in events:
        dur = getattr(e, "duration", 1)
        boundaries.append((e.cycle, 1, e))
        boundaries.append((e.cycle + dur, -1, e))
    boundaries.sort(key=lambda x: (x[0], -x[1]))

    segments: List[Dict[str, Any]] = []
    active_now: Set[int] = set()
    seg_start = 0
    event_by_id = {id(e): e for e in events}

    for cycle, delta, event in boundaries:
        eid = id(event)
        if delta == 1:
            if not active_now:
                seg_start = cycle
            active_now.add(eid)
        else:
            # 区间 [seg_start, cycle-1] 内活跃的集合为 current active_now（含本 event）
            mods = {event_by_id[aid].circuit_module for aid in active_now if event_by_id[aid].circuit_module}
            layers = {(event_by_id[aid].metadata or {}).get("layer_name") for aid in active_now}
            layers.discard(None)
            segments.append({
                "cycle_start": seg_start,
                "cycle_end": cycle - 1,
                "active_modules": sorted(mods),
                "active_layers": sorted(layers),
            })
            active_now.discard(eid)
            if active_now:
                seg_start = cycle

    return segments


def timeline_to_csv_rows(timeline: List[Dict[str, Any]]) -> List[str]:
    """将 build_module_activity_timeline 的输出转为 CSV 行（含表头）。
    当前仿真为层间串行（前一层全部算完再算下一层），故每个采样时刻通常仅一个模块在工作；
    若需流水线重叠，需在调度器中支持层间重叠。
    """
    if not timeline:
        return []
    lines = [
        "# 说明: 当前为层间串行执行，每行表示该 cycle 采样点正在工作的模块/层/算子（通常仅一个）",
        "cycle,active_modules,active_layers,active_operators",
    ]
    for row in timeline:
        cycle = row["cycle"]
        mods = ";".join(row.get("active_modules", []))
        layers = ";".join(row.get("active_layers", []))
        ops = ";".join(row.get("active_operators", []))
        lines.append(f"{cycle},\"{mods}\",\"{layers}\",\"{ops}\"")
    return lines
