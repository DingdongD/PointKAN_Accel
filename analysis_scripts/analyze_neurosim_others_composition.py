#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def _detect_prefixes(fieldnames: Iterable[str]) -> List[str]:
    names = set(fieldnames)
    out: List[str] = []
    for cand in ("w8a8_", "w4a4_"):
        if any((cand + key) in names for key in (
            "latency_ns",
            "buffer_latency_ns",
            "ic_latency_ns",
            "core_latency_adc_ns",
            "core_latency_accum_ns",
            "core_latency_other_ns",
        )):
            out.append(cand)
    if out:
        return out
    return [""]


def _base_key(prefix: str, key: str) -> str:
    return f"{prefix}{key}" if prefix else key


def _safe_ratio(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return num / den


def _analyze_row(row: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    latency = _to_float(row.get(_base_key(prefix, "latency_ns")))
    buffer_lat = _to_float(row.get(_base_key(prefix, "buffer_latency_ns")))
    ic_lat = _to_float(row.get(_base_key(prefix, "ic_latency_ns")))
    adc_lat = _to_float(row.get(_base_key(prefix, "core_latency_adc_ns")))
    accum_lat = _to_float(row.get(_base_key(prefix, "core_latency_accum_ns")))
    other_lat = _to_float(row.get(_base_key(prefix, "core_latency_other_ns")))
    explicit_visible = _to_float(row.get(_base_key(prefix, "core_latency_other_visible_ns")))
    explicit_hidden = _to_float(row.get(_base_key(prefix, "core_latency_other_hidden_ns")))
    sub_tile_lat = _to_float(row.get(_base_key(prefix, "other_sub_tile_latency_ns")))
    sub_act_lat = _to_float(row.get(_base_key(prefix, "other_sub_activation_latency_ns")))
    sub_pool_lat = _to_float(row.get(_base_key(prefix, "other_sub_pooling_latency_ns")))
    sub_gb_lat = _to_float(row.get(_base_key(prefix, "other_sub_global_buffer_latency_ns")))
    sub_gic_lat = _to_float(row.get(_base_key(prefix, "other_sub_global_ic_latency_ns")))

    core_sum = adc_lat + accum_lat + other_lat
    visible_buffer_ic = buffer_lat + ic_lat

    # Lower/upper bounds from existing csv (without requiring NeuroSim source modifications)
    hidden_min = max(0.0, other_lat - visible_buffer_ic)
    visible_overlap_max = min(other_lat, visible_buffer_ic)

    # If new NeuroSim explicit fields exist, keep them for comparison.
    hidden_explicit = max(0.0, explicit_hidden)
    visible_explicit = max(0.0, explicit_visible)

    return {
        f"{prefix}core_sum_latency_ns": core_sum,
        f"{prefix}visible_system_buffer_ic_ns": visible_buffer_ic,
        f"{prefix}other_latency_ns": other_lat,
        f"{prefix}other_hidden_min_ns": hidden_min,
        f"{prefix}other_visible_overlap_max_ns": visible_overlap_max,
        f"{prefix}other_visible_explicit_ns": visible_explicit,
        f"{prefix}other_hidden_explicit_ns": hidden_explicit,
        f"{prefix}other_hidden_explicit_share_in_other": _safe_ratio(hidden_explicit, other_lat),
        f"{prefix}other_sub_tile_share_in_other": _safe_ratio(sub_tile_lat, other_lat),
        f"{prefix}other_sub_activation_share_in_other": _safe_ratio(sub_act_lat, other_lat),
        f"{prefix}other_sub_pooling_share_in_other": _safe_ratio(sub_pool_lat, other_lat),
        f"{prefix}other_sub_global_buffer_share_in_other": _safe_ratio(sub_gb_lat, other_lat),
        f"{prefix}other_sub_global_ic_share_in_other": _safe_ratio(sub_gic_lat, other_lat),
        f"{prefix}other_share_in_core_sum": _safe_ratio(other_lat, core_sum),
        f"{prefix}other_hidden_min_share_in_other": _safe_ratio(hidden_min, other_lat),
        f"{prefix}system_buffer_ic_share_in_other": _safe_ratio(visible_buffer_ic, other_lat),
        f"{prefix}core_sum_over_latency": _safe_ratio(core_sum, latency),
    }


def _load_csv(path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _rank_rows(rows: List[Dict[str, Any]], prefixes: List[str], topk: int) -> str:
    lines: List[str] = []
    for prefix in prefixes:
        metric = f"{prefix}other_hidden_min_share_in_other"
        sortable = []
        for row in rows:
            val = _to_float(row.get(metric))
            sortable.append((val, str(row.get("layer_index", "")), str(row.get("layer_name", ""))))
        sortable.sort(reverse=True)
        tag = prefix[:-1] if prefix else "default"
        lines.append(f"[{tag}] top-{topk} by other_hidden_min_share_in_other")
        for val, lidx, lname in sortable[:topk]:
            lines.append(f"  layer={lidx} {lname}, hidden_min_share={val:.4f}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Analyze NeuroSim Others latency composition from per-run/merged layer breakdown CSV."
    )
    ap.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to neurosim layer breakdown CSV (per_run or merged).",
    )
    ap.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV path. Default: <input_stem>_others_composition.csv in same directory.",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=6,
        help="Top-K layers to print by hidden-min share.",
    )
    args = ap.parse_args()

    in_csv = Path(args.input_csv).resolve()
    if not in_csv.exists():
        raise SystemExit(f"Input not found: {in_csv}")

    rows, fieldnames = _load_csv(in_csv)
    if not rows:
        raise SystemExit(f"Input has no rows: {in_csv}")

    prefixes = _detect_prefixes(fieldnames)

    analyzed_rows: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for prefix in prefixes:
            item.update(_analyze_row(row, prefix))
        analyzed_rows.append(item)

    if args.output_csv:
        out_csv = Path(args.output_csv).resolve()
    else:
        out_csv = in_csv.with_name(f"{in_csv.stem}_others_composition.csv")

    new_fields: List[str] = []
    for prefix in prefixes:
        new_fields.extend([
            f"{prefix}core_sum_latency_ns",
            f"{prefix}visible_system_buffer_ic_ns",
            f"{prefix}other_latency_ns",
            f"{prefix}other_hidden_min_ns",
            f"{prefix}other_visible_overlap_max_ns",
            f"{prefix}other_visible_explicit_ns",
            f"{prefix}other_hidden_explicit_ns",
            f"{prefix}other_hidden_explicit_share_in_other",
            f"{prefix}other_sub_tile_share_in_other",
            f"{prefix}other_sub_activation_share_in_other",
            f"{prefix}other_sub_pooling_share_in_other",
            f"{prefix}other_sub_global_buffer_share_in_other",
            f"{prefix}other_sub_global_ic_share_in_other",
            f"{prefix}other_share_in_core_sum",
            f"{prefix}other_hidden_min_share_in_other",
            f"{prefix}system_buffer_ic_share_in_other",
            f"{prefix}core_sum_over_latency",
        ])

    merged_fieldnames = list(fieldnames)
    for nf in new_fields:
        if nf not in merged_fieldnames:
            merged_fieldnames.append(nf)
    _write_csv(out_csv, analyzed_rows, merged_fieldnames)

    print(f"Wrote: {out_csv}")
    print(_rank_rows(analyzed_rows, prefixes, max(1, int(args.topk))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
