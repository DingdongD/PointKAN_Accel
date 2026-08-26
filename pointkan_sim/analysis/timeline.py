"""
按 Cycle 的模块活动时间线
========================

根据仿真后的 Event 列表，输出「不同 cycle 下有哪些模块在工作」，
便于分析流水线与数据供给关系（前段算完可供给后段）。
"""

import json
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


def _event_block_name(event: Event, detailed: bool = False) -> str:
    meta = event.metadata or {}
    rid = str(meta.get("resource_id", "") or "").strip()
    layer = str(getattr(event, "layer_name", "") or meta.get("layer_name", "") or "").strip()
    operation = str(getattr(event, "operation", "") or "").strip()
    module = str(getattr(event, "circuit_module", "") or "").strip()
    phase = str(meta.get("phase", "") or "").strip()
    block_idx = meta.get("block_idx")
    output_idx = meta.get("output_idx")
    pair_left = meta.get("pair_left")
    pair_right = meta.get("pair_right")
    interp_weight = meta.get("interp_weight")
    query = meta.get("query_index", meta.get("query"))
    iteration = meta.get("iteration")
    query_alias = meta.get("query")
    if query_alias is None and query is None and iteration is not None:
        query_alias = iteration
    if detailed:
        parts = []
        if layer:
            parts.append(layer)
        if phase:
            parts.append(phase)
        if query is not None:
            try:
                parts.append(f"q{int(query):03d}")
            except Exception:
                parts.append(f"q{query}")
        elif query_alias is not None:
            try:
                parts.append(f"q{int(query_alias):03d}")
            except Exception:
                parts.append(f"q{query_alias}")
        if block_idx is not None:
            try:
                parts.append(f"b{int(block_idx):02d}")
            except Exception:
                parts.append(f"b{block_idx}")
        if output_idx is not None:
            try:
                parts.append(f"o{int(output_idx):02d}")
            except Exception:
                parts.append(f"o{output_idx}")
        if pair_left is not None and pair_right is not None:
            try:
                lr = f"l{int(pair_left):02d}-r{int(pair_right):02d}"
            except Exception:
                lr = f"l{pair_left}-r{pair_right}"
            if interp_weight is not None:
                try:
                    lr += f"-w{float(interp_weight):.3f}"
                except Exception:
                    lr += f"-w{interp_weight}"
            parts.append(lr)
        base = rid or module or str(getattr(event, "operator", "") or "unknown").strip()
        if operation:
            base = f"{base}/{operation}"
        parts.append(base)
        return " | ".join(parts)
    if rid:
        return rid
    if module:
        return module
    op = str(getattr(event, "operator", "") or "").strip()
    if op:
        return op
    return "unknown"


def build_block_intervals(
    events: List[Event],
    detailed: bool = False,
    layer_filter: Optional[str] = None,
    start_cycle: Optional[int] = None,
    end_cycle: Optional[int] = None,
) -> Dict[str, List[Tuple[int, int]]]:
    """
    从 Event 列表构造 block/resource 级 busy 区间。

    返回格式:
      {
        "dram": [(10, 40), (80, 120)],
        "compute_array": [(20, 60)],
        ...
      }
    """
    raw: Dict[str, List[Tuple[int, int]]] = {}
    for event in events:
        meta = getattr(event, "metadata", None) or {}
        layer = str(getattr(event, "layer_name", "") or meta.get("layer_name", "") or "")
        if layer_filter and layer != layer_filter:
            continue
        start = int(getattr(event, "cycle", 0) or 0)
        end = int(getattr(event, "end_cycle", start) or start)
        if end < start:
            continue
        if start_cycle is not None and end < start_cycle:
            continue
        if end_cycle is not None and start > end_cycle:
            continue
        if start_cycle is not None:
            start = max(start, int(start_cycle)) - int(start_cycle)
            end = max(end, int(start_cycle)) - int(start_cycle)
        name = _event_block_name(event, detailed=detailed)
        raw.setdefault(name, []).append((start, end))

    merged: Dict[str, List[Tuple[int, int]]] = {}
    for name, spans in raw.items():
        if not spans:
            continue
        spans = sorted(spans, key=lambda item: (item[0], item[1]))
        compact: List[Tuple[int, int]] = [spans[0]]
        for start, end in spans[1:]:
            last_start, last_end = compact[-1]
            if start <= last_end:
                compact[-1] = (last_start, max(last_end, end))
            else:
                compact.append((start, end))
        merged[name] = compact
    return merged


def block_intervals_to_json_rows(block_intervals: Dict[str, List[Tuple[int, int]]]) -> str:
    serializable = {
        name: [[int(start), int(end)] for start, end in spans]
        for name, spans in block_intervals.items()
    }
    return json.dumps(serializable, indent=2, ensure_ascii=False)




def _timeline_label_sort_key(name: str, block_intervals: Dict[str, List[Tuple[int, int]]]):
    import re
    first_cycle = min((int(span[0]) for span in block_intervals.get(name, []) if span), default=10**18)
    q_match = re.search(r"\|\s*q(\d+)(?:\s*\||$)", name)
    b_match = re.search(r"\|\s*b(\d+)(?:\s*\||$)", name)
    o_match = re.search(r"\|\s*o(\d+)(?:\s*\||$)", name)
    phase_match = re.search(r"\|\s*([^|]+?)\s*\|\s*(?:q\d+\s*\|\s*)?(?:b\d+\s*\|\s*)", name)
    op_order = [
        'query_xyz_buffer_write',
        'query_xyz_buffer_read',
        'cross_term_distance',
        'self_distance_read',
        'distance_merge_accumulate',
        'fps_min_distance_table_read',
        'min_distance_compare',
        'fps_min_distance_table_write',
        'fps_warmup_coarse_top32',
        'fps_coarse_filter_max',
        'temp_max_candidate_append',
        'fps_warmup_fine_max',
        'fps_fine_filter_max',
        'knn_coarse_filter',
        'topk_candidate_write',
        'topk_candidate_aggregate_read',
        'knn_threshold_update',
        'knn_threshold_refine',
        'topk_candidate_refine_write',
        'topk_candidate_fine_read',
        'knn_fine_filter_topk',
        'fps_tempmax_reduce_read',
        'fps_tempmax_reduce_write',
        'fps_tempmax_global_read',
    ]
    op_idx = len(op_order)
    for i, token in enumerate(op_order):
        if token in name:
            op_idx = i
            break
    phase = phase_match.group(1).strip() if phase_match else ''
    phase_order_map = {
        'distance_frontend': 0,
        'distance_frontend_banked': 0,
        'warmup_full_sort': 1,
        'knn_dynamic_threshold': 2,
        'post_predict_coarse_fine': 3,
    }
    phase_idx = phase_order_map.get(phase, 99)
    q = int(q_match.group(1)) if q_match else 10**9
    b = int(b_match.group(1)) if b_match else 10**9
    o = int(o_match.group(1)) if o_match else 10**9
    lr_match = re.search(r"\|\s*l(\d+)-r(\d+)(?:-w([0-9.]+))?\s*\|", name)
    l = int(lr_match.group(1)) if lr_match else 10**9
    r = int(lr_match.group(2)) if lr_match else 10**9
    it = q
    # For ACC stage timelines, show a block's causal chain first, then phase/fallback order.
    return (q, it, b, o, l, r, op_idx, phase_idx, first_cycle, name)


def _compact_timeline_label(name: str, label_width: int) -> str:
    parts = [p.strip() for p in name.split("|")]
    if len(parts) < 2:
        return name[:label_width].ljust(label_width)

    phase_map = {
        "distance_frontend": "dist",
        "warmup_full_sort": "warm",
        "knn_dynamic_threshold": "knn_dyn",
        "post_predict_coarse_fine": "post",
        "predictor_gate": "gate",
        "speculative_overlap": "spec",
    }
    op_map = {
        "query_xyz_buffer_write": "qbuf_wr",
        "query_xyz_buffer_read": "qbuf_rd",
        "xyz_buffer_query_read": "xyz_q_rd",
        "xyz_buffer_key_read": "xyz_k_rd",
        "cross_term_distance": "cross",
        "self_distance_read": "self_rd",
        "distance_merge_accumulate": "merge",
        "min_distance_compare": "min_upd",
        "knn_coarse_filter": "knn_coarse",
        "fps_warmup_coarse_top32": "fps_warm_coarse",
        "topk_candidate_write": "topk_wr",
        "fps_warmup_fine_max": "fps_warm_fine",
        "fps_coarse_filter_max": "fps_coarse",
        "temp_max_candidate_append": "temp_append",
        "fps_tempmax_global_read": "temp_glb_rd",
        "fps_fine_filter_max": "fps_fine",
        "topk_candidate_aggregate_read": "topk_aggr_rd",
        "knn_threshold_update": "th_update",
        "knn_threshold_refine": "th_refine",
        "topk_candidate_fine_read": "topk_fine_rd",
        "knn_fine_filter_topk": "knn_fine",
        "resampler_downsample_lut": "down_lut",
        "resampler_downsample": "downsample",
        "resampler_downsample_step": "down_step",
        "predictor_fc1_linear": "pred_fc1",
        "predictor_kat": "pred_kat",
        "predictor_fc2_linear": "pred_fc2",
        "resampler_upsample_lut": "up_lut",
        "resampler_upsample": "upsample",
        "resampler_upsample_step": "up_step",
        "fps_prefix_sample_write": "prefix_wr",
    }

    phase = phase_map.get(parts[1], parts[1])
    q = next((p for p in parts if p.startswith("q")), "")
    b = next((p for p in parts if p.startswith("b")), "")
    o = next((p for p in parts if p.startswith("o")), "")
    lr = next((p for p in parts if p.startswith("l") and "-r" in p), "")
    base = parts[-1]
    op = base.split("/")[-1].strip()
    op = op_map.get(op, op)
    label_parts = [x for x in (phase, q, b, o, lr, op) if x]
    label = " | ".join(label_parts)
    if len(label) > label_width:
        return label[:label_width]
    return label.ljust(label_width)


def render_ascii_timeline(
    block_intervals: Dict[str, List[Tuple[int, int]]],
    max_cycle: int = 500,
    label_width: int = 56,
    active_char: str = "#",
    idle_char: str = ".",
) -> str:
    """
    将 busy 区间渲染为终端文本版 ASCII timeline。

    - cycle 粒度 = 1
    - 横向表示 cycle
    - 纵向表示 block/module
    - 活动为 active_char，空闲为 idle_char
    """
    if not block_intervals:
        return "(empty timeline)"

    labels = sorted(block_intervals.keys(), key=lambda name: _timeline_label_sort_key(name, block_intervals))
    max_cycle = max(0, int(max_cycle))

    header_100 = " " * (label_width + 3)
    header_10 = " " * (label_width + 3)
    header_1 = " " * (label_width + 3)
    for c in range(max_cycle):
        if c % 100 == 0:
            txt = str(c)
            header_100 += txt[0] if len(txt) >= 1 else " "
        else:
            header_100 += " "
        if c % 10 == 0:
            txt = f"{c:03d}"
            header_10 += txt[-2]
            header_1 += txt[-1]
        else:
            header_10 += " "
            header_1 += " "

    lines = [
        f"ASCII Cycle Timeline (0-{max_cycle-1})",
        header_100,
        header_10,
        header_1,
    ]

    for name in labels:
        chars = [idle_char] * max_cycle
        for start, end in block_intervals.get(name, []):
            s = max(0, int(start))
            e = min(max_cycle - 1, int(end))
            if e < 0 or s >= max_cycle or e < s:
                continue
            for i in range(s, e + 1):
                chars[i] = active_char
        label = _compact_timeline_label(name, label_width)
        lines.append(f"{label} | {''.join(chars)}")
    return "\n".join(lines)


def plot_timeline(
    block_intervals: Dict[str, List[Tuple[int, int]]],
    max_cycle: int = 2000,
    output_path: Optional[str] = None,
    title: str = "Cycle-Level Hardware Execution Timeline",
) -> Optional[str]:
    """
    将 block/resource busy 区间画成 gantt 风格 timeline。

    约束：
    - 横轴为 cycle
    - 纵轴为 block 名称
    - cycle 粒度 = 1
    - 只画前 max_cycle 个 cycle
    """
    if not block_intervals:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clipped: Dict[str, List[Tuple[int, int]]] = {}
    for name, spans in block_intervals.items():
        kept: List[Tuple[int, int]] = []
        for start, end in spans:
            if end < 0 or start > max_cycle:
                continue
            kept.append((max(0, int(start)), min(int(end), int(max_cycle))))
        if kept:
            clipped[name] = kept

    if not clipped:
        return None

    labels = sorted(clipped.keys())
    fig_h = max(6, 0.35 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(20, fig_h))

    y_ticks = []
    y_labels = []
    row_height = 8
    row_gap = 4
    cmap = plt.get_cmap("tab20")

    for idx, name in enumerate(labels):
        y = idx * (row_height + row_gap)
        y_ticks.append(y + row_height / 2)
        y_labels.append(name)
        spans = []
        for start, end in clipped[name]:
            width = max(1, end - start + 1)
            spans.append((start, width))
        ax.broken_barh(
            spans,
            (y, row_height),
            facecolors=cmap(idx % 20),
            edgecolors="black",
            linewidth=0.3,
        )

    ax.set_xlim(0, max_cycle)
    ax.set_ylim(0, len(labels) * (row_height + row_gap))
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Block / Resource")
    ax.set_title(title)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.6)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return output_path
    return None
