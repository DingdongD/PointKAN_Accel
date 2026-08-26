#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_EXP8_RE = re.compile(
    r"^sim_results_(pointnet2-[^_]+)_pointcim_p(\d+)_w(\d+)a(\d+)bit_isoA$"
)


@dataclass(frozen=True)
class RunTag:
    network: str
    num_points: int
    wl_weight: int
    wl_activation: int


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _discover_runs(root: Path) -> Dict[RunTag, Path]:
    runs: Dict[RunTag, Path] = {}
    for p in root.iterdir():
        if not p.is_dir():
            continue
        m = _EXP8_RE.match(p.name)
        if not m:
            continue
        tag = RunTag(
            network=m.group(1),
            num_points=int(m.group(2)),
            wl_weight=int(m.group(3)),
            wl_activation=int(m.group(4)),
        )
        runs[tag] = p
    return runs


def _load_layer_breakdown(results_dir: Path) -> List[Dict[str, Any]]:
    # Prefer exported file; fall back to report.json key.
    p = results_dir / "neurosim_layer_breakdown.json"
    if p.exists():
        rows = _read_json(p)
        return list(rows or [])

    report = results_dir / "report.json"
    if report.exists():
        j = _read_json(report)
        rows = j.get("neurosim_layer_breakdown") or []
        return list(rows or [])

    return []


def _write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _norm_row(row: Dict[str, Any]) -> Dict[str, Any]:
    # Keep a stable subset + pass through extra keys if present.
    keys = [
        "layer_index",
        "layer_name",
        "layer_type",
        "latency_ns",
        "buffer_latency_ns",
        "ic_latency_ns",
        "core_latency_adc_ns",
        "core_latency_accum_ns",
        "core_latency_other_ns",
        "array_latency_ns",
        "adc_latency_precharger_ns",
        "adc_latency_col_delay_ns",
        "adc_latency_senseamp_ns",
        "adc_latency_mlsa_ns",
        "adc_latency_mlencoder_ns",
        "adc_latency_sar_ns",
        "accum_latency_adder_ns",
        "accum_latency_dff_ns",
        "accum_latency_shiftadd_weight_ns",
        "accum_latency_shiftadd_input_ns",
        "accum_latency_adder_tree_ns",
        "accum_latency_tile_ns",
        "core_latency_other_visible_ns",
        "core_latency_other_hidden_ns",
        "core_latency_other_visible_share_pct",
        "core_latency_other_hidden_share_pct",
        "energy_nj",
        "buffer_energy_nj",
        "ic_energy_nj",
        "core_energy_adc_nj",
        "core_energy_accum_nj",
        "core_energy_other_nj",
        "array_energy_nj",
        "adc_energy_precharger_nj",
        "adc_energy_senseamp_nj",
        "adc_energy_mlsa_nj",
        "adc_energy_mlencoder_nj",
        "adc_energy_sar_nj",
        "accum_energy_adder_nj",
        "accum_energy_dff_nj",
        "accum_energy_shiftadd_weight_nj",
        "accum_energy_shiftadd_input_nj",
        "accum_energy_adder_tree_nj",
        "accum_energy_tile_nj",
        "core_energy_other_visible_nj",
        "core_energy_other_hidden_nj",
        "other_sub_tile_latency_ns",
        "other_sub_tile_pe_other_latency_ns",
        "other_sub_tile_buffer_htree_latency_ns",
        "other_sub_tile_pe_wl_decoder_latency_ns",
        "other_sub_tile_pe_wl_driver_latency_ns",
        "other_sub_tile_pe_wl_switch_latency_ns",
        "other_sub_tile_pe_sl_switch_latency_ns",
        "other_sub_tile_pe_mux_latency_ns",
        "other_sub_tile_pe_mux_decoder_latency_ns",
        "other_sub_tile_pe_level_shifter_latency_ns",
        "other_sub_tile_pe_wl_path_latency_ns",
        "other_sub_tile_pe_mux_path_latency_ns",
        "other_sub_tile_pe_selected_path_latency_ns",
        "other_sub_tile_pe_other_multiplier",
        "other_sub_tile_pe_num_col_muxed",
        "other_sub_tile_pe_num_add",
        "other_sub_tile_pe_num_row_activity",
        "other_sub_activation_latency_ns",
        "other_sub_tile_activation_latency_ns",
        "other_sub_chip_activation_latency_ns",
        "other_sub_pooling_latency_ns",
        "other_sub_global_buffer_latency_ns",
        "other_sub_global_ic_latency_ns",
        "other_sub_global_htree_latency_ns",
        "other_sub_global_xybus_latency_ns",
        "other_sub_tile_energy_nj",
        "other_sub_tile_pe_other_energy_nj",
        "other_sub_tile_buffer_htree_energy_nj",
        "other_sub_activation_energy_nj",
        "other_sub_tile_activation_energy_nj",
        "other_sub_chip_activation_energy_nj",
        "other_sub_pooling_energy_nj",
        "other_sub_global_buffer_energy_nj",
        "other_sub_global_ic_energy_nj",
        "other_sub_global_htree_energy_nj",
        "other_sub_global_xybus_energy_nj",
    ]
    out: Dict[str, Any] = {}
    for k in keys:
        out[k] = row.get(k)
    lat_other = _to_float(out.get("core_latency_other_ns"))
    lat_visible = _to_float(out.get("core_latency_other_visible_ns"))
    if lat_visible is None:
        lat_visible = (_to_float(out.get("buffer_latency_ns")) or 0.0) + (_to_float(out.get("ic_latency_ns")) or 0.0)
        out["core_latency_other_visible_ns"] = lat_visible
    if out.get("core_latency_other_hidden_ns") is None and lat_other is not None:
        out["core_latency_other_hidden_ns"] = max(lat_other - lat_visible, 0.0)
    if lat_other and lat_other > 0:
        out["core_latency_other_visible_share_pct"] = lat_visible / lat_other * 100.0
        out["core_latency_other_hidden_share_pct"] = (_to_float(out.get("core_latency_other_hidden_ns")) or 0.0) / lat_other * 100.0
    energy_other = _to_float(out.get("core_energy_other_nj"))
    energy_visible = _to_float(out.get("core_energy_other_visible_nj"))
    if energy_visible is None:
        energy_visible = (_to_float(out.get("buffer_energy_nj")) or 0.0) + (_to_float(out.get("ic_energy_nj")) or 0.0)
        out["core_energy_other_visible_nj"] = energy_visible
    if out.get("core_energy_other_hidden_nj") is None and energy_other is not None:
        out["core_energy_other_hidden_nj"] = max(energy_other - energy_visible, 0.0)
    return out


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _merge_two_runs(
    rows_a: List[Dict[str, Any]],
    rows_b: List[Dict[str, Any]],
    *,
    prefix_a: str,
    prefix_b: str,
) -> List[Dict[str, Any]]:
    def key(r: Dict[str, Any]) -> Tuple[int, str]:
        return int(r.get("layer_index") or 0), str(r.get("layer_name") or "")

    a = {key(r): r for r in rows_a}
    b = {key(r): r for r in rows_b}

    merged: List[Dict[str, Any]] = []
    for k in sorted(set(a) | set(b)):
        ra = a.get(k, {})
        rb = b.get(k, {})
        base = {
            "layer_index": k[0],
            "layer_name": k[1],
            "layer_type": ra.get("layer_type") or rb.get("layer_type"),
        }

        def add(prefix: str, r: Dict[str, Any]) -> None:
            for kk, vv in _norm_row(r).items():
                if kk in {"layer_index", "layer_name", "layer_type"}:
                    continue
                base[f"{prefix}{kk}"] = vv

        add(prefix_a, ra)
        add(prefix_b, rb)

        la = _to_float(ra.get("latency_ns"))
        lb = _to_float(rb.get("latency_ns"))
        base["speedup_latency_a_over_b"] = (la / lb) if (la and lb and lb > 0) else None
        merged.append(base)
    return merged


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default=".",
        help="PointKAN-pytorch 根目录（默认当前目录）",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="analysis_exp8_isoA_neurosim_breakdown",
        help="输出目录（相对 root）",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = (root / args.out_dir).resolve()

    runs = _discover_runs(root)
    if not runs:
        raise SystemExit(f"No exp8 isoA runs found under {root}")

    # 1) Export per-run normalized breakdown CSV
    for tag, rdir in sorted(runs.items(), key=lambda x: (x[0].network, x[0].num_points, x[0].wl_weight, x[0].wl_activation)):
        rows = [_norm_row(r) for r in _load_layer_breakdown(rdir)]
        out = out_dir / "per_run" / f"{rdir.name}_neurosim_layer_breakdown.csv"
        fieldnames = list(rows[0].keys()) if rows else [
            "layer_index",
            "layer_name",
            "layer_type",
            "latency_ns",
            "buffer_latency_ns",
            "ic_latency_ns",
            "core_latency_adc_ns",
            "core_latency_accum_ns",
            "core_latency_other_ns",
            "array_latency_ns",
            "adc_latency_precharger_ns",
            "adc_latency_col_delay_ns",
            "adc_latency_senseamp_ns",
            "adc_latency_mlsa_ns",
            "adc_latency_mlencoder_ns",
            "adc_latency_sar_ns",
            "accum_latency_adder_ns",
            "accum_latency_dff_ns",
            "accum_latency_shiftadd_weight_ns",
            "accum_latency_shiftadd_input_ns",
            "accum_latency_adder_tree_ns",
            "accum_latency_tile_ns",
            "core_latency_other_visible_ns",
            "core_latency_other_hidden_ns",
            "core_latency_other_visible_share_pct",
            "core_latency_other_hidden_share_pct",
            "energy_nj",
            "buffer_energy_nj",
            "ic_energy_nj",
            "core_energy_adc_nj",
            "core_energy_accum_nj",
            "core_energy_other_nj",
            "array_energy_nj",
            "adc_energy_precharger_nj",
            "adc_energy_senseamp_nj",
            "adc_energy_mlsa_nj",
            "adc_energy_mlencoder_nj",
            "adc_energy_sar_nj",
            "accum_energy_adder_nj",
            "accum_energy_dff_nj",
            "accum_energy_shiftadd_weight_nj",
            "accum_energy_shiftadd_input_nj",
            "accum_energy_adder_tree_nj",
            "accum_energy_tile_nj",
            "core_energy_other_visible_nj",
            "core_energy_other_hidden_nj",
            "other_sub_tile_latency_ns",
            "other_sub_tile_pe_other_latency_ns",
            "other_sub_tile_buffer_htree_latency_ns",
            "other_sub_tile_pe_wl_decoder_latency_ns",
            "other_sub_tile_pe_wl_driver_latency_ns",
            "other_sub_tile_pe_wl_switch_latency_ns",
            "other_sub_tile_pe_sl_switch_latency_ns",
            "other_sub_tile_pe_mux_latency_ns",
            "other_sub_tile_pe_mux_decoder_latency_ns",
            "other_sub_tile_pe_level_shifter_latency_ns",
            "other_sub_tile_pe_wl_path_latency_ns",
            "other_sub_tile_pe_mux_path_latency_ns",
            "other_sub_tile_pe_selected_path_latency_ns",
            "other_sub_tile_pe_other_multiplier",
            "other_sub_tile_pe_num_col_muxed",
            "other_sub_tile_pe_num_add",
            "other_sub_tile_pe_num_row_activity",
            "other_sub_activation_latency_ns",
            "other_sub_tile_activation_latency_ns",
            "other_sub_chip_activation_latency_ns",
            "other_sub_pooling_latency_ns",
            "other_sub_global_buffer_latency_ns",
            "other_sub_global_ic_latency_ns",
            "other_sub_global_htree_latency_ns",
            "other_sub_global_xybus_latency_ns",
            "other_sub_tile_energy_nj",
            "other_sub_tile_pe_other_energy_nj",
            "other_sub_tile_buffer_htree_energy_nj",
            "other_sub_activation_energy_nj",
            "other_sub_tile_activation_energy_nj",
            "other_sub_chip_activation_energy_nj",
            "other_sub_pooling_energy_nj",
            "other_sub_global_buffer_energy_nj",
            "other_sub_global_ic_energy_nj",
            "other_sub_global_htree_energy_nj",
            "other_sub_global_xybus_energy_nj",
        ]
        _write_csv(out, fieldnames, rows)

    # 2) Export merged W8A8 vs W4A4 per-network
    by_net: Dict[Tuple[str, int], Dict[Tuple[int, int], Path]] = {}
    for tag, rdir in runs.items():
        by_net.setdefault((tag.network, tag.num_points), {})[(tag.wl_weight, tag.wl_activation)] = rdir

    for (net, pts), mp in sorted(by_net.items()):
        r8 = mp.get((8, 8))
        r4 = mp.get((4, 4))
        if not r8 or not r4:
            continue
        rows8 = [_norm_row(r) for r in _load_layer_breakdown(r8)]
        rows4 = [_norm_row(r) for r in _load_layer_breakdown(r4)]
        merged = _merge_two_runs(rows8, rows4, prefix_a="w8a8_", prefix_b="w4a4_")
        out = out_dir / "merged" / f"{net}_p{pts}_w8a8_vs_w4a4_neurosim_layer_breakdown.csv"
        fieldnames = list(merged[0].keys()) if merged else []
        _write_csv(out, fieldnames, merged)

    print(f"Wrote exp8 isoA neurosim breakdown CSVs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
