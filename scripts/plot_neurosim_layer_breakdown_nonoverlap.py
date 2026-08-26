#!/usr/bin/env python3
"""
Generate non-overlapping per-layer NeuroSim breakdown tables and plots.

This script avoids double counting between buffer / interconnect and the broad
"other" bucket by materializing two plotting schemes:

1. functional:
   Array, ADC Peripheral, Accum, Buffer, Interconnect, Activation/Pooling,
   Other Control

2. hotspot:
   Array, ADC Peripheral, Accum, TileBufferHTree, GlobalBuffer, GlobalIC,
   TilePEOther, Activation/Pooling, Other Hidden
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple


def find_breakdown_input(path: Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    csv_path = path / "neurosim_layer_breakdown.csv"
    json_path = path / "neurosim_layer_breakdown.json"
    if csv_path.is_file():
        return csv_path
    if json_path.is_file():
        return json_path
    raise FileNotFoundError(
        f"No neurosim breakdown file found under {path}. Expected {csv_path.name} or {json_path.name}."
    )


def _coerce_value(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return 0.0
    text = str(value).strip()
    if text == "":
        return 0.0
    try:
        return float(text)
    except ValueError:
        return text


def load_rows(path: Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [{k: _coerce_value(v) for k, v in row.items()} for row in data]
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [{k: _coerce_value(v) for k, v in row.items()} for row in reader]


def _f(row: Dict[str, Any], key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def _layer_label(row: Dict[str, Any], idx: int) -> str:
    name = str(row.get("layer_name", "") or "").strip()
    return name if name else f"layer_{idx}"


Extractor = Callable[[Dict[str, Any]], float]


def functional_latency_components() -> List[Tuple[str, str, Extractor]]:
    return [
        ("Array", "array_latency_ns", lambda r: _f(r, "array_latency_ns")),
        (
            "ADC Peripheral",
            "max(core_latency_adc_ns - array_latency_ns, 0)",
            lambda r: max(0.0, _f(r, "core_latency_adc_ns") - _f(r, "array_latency_ns")),
        ),
        ("Accum", "core_latency_accum_ns", lambda r: _f(r, "core_latency_accum_ns")),
        ("Buffer", "buffer_latency_ns", lambda r: _f(r, "buffer_latency_ns")),
        ("Interconnect", "ic_latency_ns", lambda r: _f(r, "ic_latency_ns")),
        (
            "Activation/Pooling",
            "other_sub_activation_latency_ns + other_sub_pooling_latency_ns",
            lambda r: _f(r, "other_sub_activation_latency_ns") + _f(r, "other_sub_pooling_latency_ns"),
        ),
        ("Other Control", "core_latency_other_hidden_ns", lambda r: _f(r, "core_latency_other_hidden_ns")),
    ]


def functional_energy_components() -> List[Tuple[str, str, Extractor]]:
    return [
        ("Array", "array_energy_nj", lambda r: _f(r, "array_energy_nj")),
        (
            "ADC Peripheral",
            "max(core_energy_adc_nj - array_energy_nj, 0)",
            lambda r: max(0.0, _f(r, "core_energy_adc_nj") - _f(r, "array_energy_nj")),
        ),
        ("Accum", "core_energy_accum_nj", lambda r: _f(r, "core_energy_accum_nj")),
        ("Buffer", "buffer_energy_nj", lambda r: _f(r, "buffer_energy_nj")),
        ("Interconnect", "ic_energy_nj", lambda r: _f(r, "ic_energy_nj")),
        (
            "Activation/Pooling",
            "other_sub_activation_energy_nj + other_sub_pooling_energy_nj",
            lambda r: _f(r, "other_sub_activation_energy_nj") + _f(r, "other_sub_pooling_energy_nj"),
        ),
        ("Other Control", "core_energy_other_hidden_nj", lambda r: _f(r, "core_energy_other_hidden_nj")),
    ]


def hotspot_latency_components() -> List[Tuple[str, str, Extractor]]:
    return [
        ("Array", "array_latency_ns", lambda r: _f(r, "array_latency_ns")),
        (
            "ADC Peripheral",
            "max(core_latency_adc_ns - array_latency_ns, 0)",
            lambda r: max(0.0, _f(r, "core_latency_adc_ns") - _f(r, "array_latency_ns")),
        ),
        ("Accum", "core_latency_accum_ns", lambda r: _f(r, "core_latency_accum_ns")),
        ("TileBufferHTree", "other_sub_tile_buffer_htree_latency_ns", lambda r: _f(r, "other_sub_tile_buffer_htree_latency_ns")),
        ("GlobalBuffer", "other_sub_global_buffer_latency_ns", lambda r: _f(r, "other_sub_global_buffer_latency_ns")),
        ("GlobalIC", "other_sub_global_ic_latency_ns", lambda r: _f(r, "other_sub_global_ic_latency_ns")),
        ("TilePEOther", "other_sub_tile_pe_other_latency_ns", lambda r: _f(r, "other_sub_tile_pe_other_latency_ns")),
        (
            "Activation/Pooling",
            "other_sub_activation_latency_ns + other_sub_pooling_latency_ns",
            lambda r: _f(r, "other_sub_activation_latency_ns") + _f(r, "other_sub_pooling_latency_ns"),
        ),
        ("Other Hidden", "core_latency_other_hidden_ns", lambda r: _f(r, "core_latency_other_hidden_ns")),
    ]


def hotspot_energy_components() -> List[Tuple[str, str, Extractor]]:
    return [
        ("Array", "array_energy_nj", lambda r: _f(r, "array_energy_nj")),
        (
            "ADC Peripheral",
            "max(core_energy_adc_nj - array_energy_nj, 0)",
            lambda r: max(0.0, _f(r, "core_energy_adc_nj") - _f(r, "array_energy_nj")),
        ),
        ("Accum", "core_energy_accum_nj", lambda r: _f(r, "core_energy_accum_nj")),
        ("TileBufferHTree", "other_sub_tile_buffer_htree_energy_nj", lambda r: _f(r, "other_sub_tile_buffer_htree_energy_nj")),
        ("GlobalBuffer", "other_sub_global_buffer_energy_nj", lambda r: _f(r, "other_sub_global_buffer_energy_nj")),
        ("GlobalIC", "other_sub_global_ic_energy_nj", lambda r: _f(r, "other_sub_global_ic_energy_nj")),
        ("TilePEOther", "other_sub_tile_pe_other_energy_nj", lambda r: _f(r, "other_sub_tile_pe_other_energy_nj")),
        (
            "Activation/Pooling",
            "other_sub_activation_energy_nj + other_sub_pooling_energy_nj",
            lambda r: _f(r, "other_sub_activation_energy_nj") + _f(r, "other_sub_pooling_energy_nj"),
        ),
        ("Other Hidden", "core_energy_other_hidden_nj", lambda r: _f(r, "core_energy_other_hidden_nj")),
    ]


def _build_series(rows: List[Dict[str, Any]], components: List[Tuple[str, str, Extractor]]) -> Dict[str, Any]:
    return {
        "labels": [_layer_label(row, idx) for idx, row in enumerate(rows)],
        "components": {name: [extractor(row) for row in rows] for name, _, extractor in components},
    }


def _write_long_csv(
    rows: List[Dict[str, Any]],
    components: List[Tuple[str, str, Extractor]],
    out_csv: Path,
    value_key: str,
) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["layer_name", "component", value_key])
        writer.writeheader()
        for idx, row in enumerate(rows):
            layer_name = _layer_label(row, idx)
            for name, _, extractor in components:
                writer.writerow({"layer_name": layer_name, "component": name, value_key: extractor(row)})
    return out_csv


def _write_mapping_md(out_md: Path) -> Path:
    lines = [
        "# NeuroSim Non-overlap Breakdown Mapping",
        "",
        "## Functional scheme",
        "",
        "| Component | Latency field mapping | Energy field mapping |",
        "|---|---|---|",
    ]
    latency = {name: expr for name, expr, _ in functional_latency_components()}
    energy = {name: expr for name, expr, _ in functional_energy_components()}
    for name in latency:
        lines.append(f"| {name} | `{latency[name]}` | `{energy[name]}` |")

    lines += [
        "",
        "## Hotspot scheme",
        "",
        "| Component | Latency field mapping | Energy field mapping |",
        "|---|---|---|",
    ]
    latency_h = {name: expr for name, expr, _ in hotspot_latency_components()}
    energy_h = {name: expr for name, expr, _ in hotspot_energy_components()}
    for name in latency_h:
        lines.append(f"| {name} | `{latency_h[name]}` | `{energy_h[name]}` |")

    lines += [
        "",
        "## Notes",
        "",
        "- `core_latency_other_ns` and `core_energy_other_nj` are not used directly here because they overlap with buffer/interconnect sub-categories.",
        "- `ADC Peripheral` subtracts `Array` from the broad ADC bucket to avoid double counting.",
        "- Functional scheme is better for a clean paper figure; Hotspot scheme is better for debugging why `other` is large.",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_md


def _plot_stacked_bar(series: Dict[str, Any], title: str, ylabel: str, out_png: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping PNG generation.")
        return None

    labels = list(series["labels"])
    components = dict(series["components"])
    if not labels:
        return None

    fig_width = max(12.0, 0.55 * len(labels) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    x = list(range(len(labels)))
    bottoms = [0.0] * len(labels)
    palette = ["#2c7fb8", "#f28e2b", "#59a14f", "#af7aa1", "#e15759", "#76b7b2", "#edc948", "#b07aa1", "#ff9da7"]

    for color_idx, (component, values) in enumerate(components.items()):
        vals = [float(v) for v in values]
        ax.bar(x, vals, bottom=bottoms, label=component, color=palette[color_idx % len(palette)], width=0.78)
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=min(4, max(1, len(components))), fontsize=8.5)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_png


def run(input_path: Path, out_dir: Path) -> Dict[str, List[Path]]:
    resolved_input = find_breakdown_input(input_path)
    rows = load_rows(resolved_input)
    if not rows:
        raise ValueError(f"No rows found in {resolved_input}")

    out_dir.mkdir(parents=True, exist_ok=True)
    written_tables: List[Path] = []
    written_plots: List[Path] = []

    written_tables.append(_write_mapping_md(out_dir / "neurosim_nonoverlap_mapping.md"))
    written_tables.append(
        _write_long_csv(rows, functional_latency_components(), out_dir / "neurosim_nonoverlap_functional_latency_long.csv", "value_ns")
    )
    written_tables.append(
        _write_long_csv(rows, functional_energy_components(), out_dir / "neurosim_nonoverlap_functional_energy_long.csv", "value_nj")
    )
    written_tables.append(
        _write_long_csv(rows, hotspot_latency_components(), out_dir / "neurosim_nonoverlap_hotspot_latency_long.csv", "value_ns")
    )
    written_tables.append(
        _write_long_csv(rows, hotspot_energy_components(), out_dir / "neurosim_nonoverlap_hotspot_energy_long.csv", "value_nj")
    )

    plot_specs = [
        (
            _build_series(rows, functional_latency_components()),
            "NeuroSim Layer Breakdown (Non-overlap, Functional): Latency",
            "Latency (ns)",
            "neurosim_nonoverlap_functional_latency.png",
        ),
        (
            _build_series(rows, functional_energy_components()),
            "NeuroSim Layer Breakdown (Non-overlap, Functional): Energy",
            "Energy (nJ)",
            "neurosim_nonoverlap_functional_energy.png",
        ),
        (
            _build_series(rows, hotspot_latency_components()),
            "NeuroSim Layer Breakdown (Non-overlap, Hotspot): Latency",
            "Latency (ns)",
            "neurosim_nonoverlap_hotspot_latency.png",
        ),
        (
            _build_series(rows, hotspot_energy_components()),
            "NeuroSim Layer Breakdown (Non-overlap, Hotspot): Energy",
            "Energy (nJ)",
            "neurosim_nonoverlap_hotspot_energy.png",
        ),
    ]

    for series, title, ylabel, filename in plot_specs:
        out = _plot_stacked_bar(series, title, ylabel, out_dir / filename)
        if out is not None:
            written_plots.append(out)

    return {"tables": written_tables, "plots": written_plots}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate non-overlapping NeuroSim per-layer breakdown plots/tables.")
    parser.add_argument(
        "input",
        nargs="?",
        default="analysis_subarray_pointnet2_pointcim_auto/runs/pointnet2-partseg-ssg_pointcim_p1024_w8a8_suba128x128_auto",
        help="Path to neurosim_layer_breakdown.csv/json or a run directory containing one.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <run>/nonoverlap_breakdown_plots.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    resolved_input = find_breakdown_input(input_path)
    default_out = (input_path if input_path.is_dir() else resolved_input.parent) / "nonoverlap_breakdown_plots"
    out_dir = Path(args.out_dir) if args.out_dir else default_out

    outputs = run(input_path, out_dir)
    print(f"Loaded NeuroSim breakdown from: {resolved_input}")
    for table in outputs["tables"]:
        print(f"Wrote table: {table}")
    for plot in outputs["plots"]:
        print(f"Wrote plot: {plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
