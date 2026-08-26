#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from pointkan_sim.analysis.timeline import render_ascii_timeline


BlockIntervals = Dict[str, List[Tuple[int, int]]]


def _load_intervals(path: Path) -> BlockIntervals:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    out: BlockIntervals = {}
    for name, spans in raw.items():
        norm: List[Tuple[int, int]] = []
        for item in spans:
            if not isinstance(item, list) or len(item) != 2:
                continue
            norm.append((int(item[0]), int(item[1])))
        if norm:
            out[str(name)] = norm
    return out


def _clip_intervals(block_intervals: BlockIntervals, start_cycle: int, end_cycle: int) -> BlockIntervals:
    clipped: BlockIntervals = {}
    for name, spans in block_intervals.items():
        kept: List[Tuple[int, int]] = []
        for s, e in spans:
            if e < start_cycle or s > end_cycle:
                continue
            kept.append((max(s, start_cycle) - start_cycle, min(e, end_cycle) - start_cycle))
        if kept:
            clipped[name] = kept
    return clipped


def _is_local_label(name: str) -> bool:
    tokens = (
        "fps_prefix_sample_write",
        "resampler_downsample_lut",
        "resampler_downsample_step",
        "predictor_fc1_linear",
        "predictor_kat",
        "predictor_fc2_linear",
        "resampler_upsample_lut",
        "resampler_upsample_step",
        "fps_coarse_filter_max",
        "temp_max_candidate_append",
        "fps_fine_filter_max",
        "predictor_gate",
        "speculative_overlap",
        "post_predict_coarse_fine",
    )
    return any(token in name for token in tokens)


def _find_anchor(block_intervals: BlockIntervals) -> int:
    predictor_starts = [
        s
        for name, spans in block_intervals.items()
        if any(token in name for token in ("predictor_fc1_linear", "predictor_kat", "predictor_fc2_linear"))
        for s, _ in spans
    ]
    post_starts = [
        s
        for name, spans in block_intervals.items()
        if any(token in name for token in ("fps_coarse_filter_max", "temp_max_candidate_append", "fps_fine_filter_max"))
        for s, _ in spans
    ]
    if predictor_starts:
        return max(0, min(predictor_starts) - 300)
    if post_starts:
        return max(0, min(post_starts) - 500)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a local GAM-stage0 predictor/switch ASCII timeline from block_intervals JSON.")
    parser.add_argument("input_json", type=Path, help="Path to block_intervals_gam_stage0_switch.json")
    parser.add_argument("--output-txt", type=Path, default=None, help="Output text timeline path")
    parser.add_argument("--output-json", type=Path, default=None, help="Output clipped JSON path")
    parser.add_argument("--width", type=int, default=1800, help="Window width in cycles")
    parser.add_argument("--label-width", type=int, default=72, help="ASCII label width")
    args = parser.parse_args()

    block_intervals = _load_intervals(args.input_json)
    filtered = {name: spans for name, spans in block_intervals.items() if _is_local_label(name)}
    anchor = _find_anchor(filtered)
    clipped = _clip_intervals(filtered, anchor, anchor + max(1, int(args.width)))

    output_txt = args.output_txt or args.input_json.with_name("cycle_timeline_gam_stage0_predictor_switch_local.txt")
    output_json = args.output_json or args.input_json.with_name("block_intervals_gam_stage0_predictor_switch_local.json")

    with output_json.open("w", encoding="utf-8") as f:
        json.dump({k: [[s, e] for s, e in v] for k, v in clipped.items()}, f, indent=2, ensure_ascii=False)

    with output_txt.open("w", encoding="utf-8") as f:
        f.write(render_ascii_timeline(clipped, max_cycle=max(1, int(args.width)), label_width=max(24, int(args.label_width))))

    print(f"anchor={anchor}")
    print(f"wrote {output_json}")
    print(f"wrote {output_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
