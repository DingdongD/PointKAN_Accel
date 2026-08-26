#!/usr/bin/env python3
"""
Plot per-layer NeuroSim breakdown as stacked bar charts.

Supports:
- direct input file: neurosim_layer_breakdown.csv or .json
- sim_results directory auto-discovery

Outputs:
- latency stacked bar PNG
- total energy stacked bar PNG
- ADC sub-breakdown energy stacked bar PNG
- long-format CSV tables for downstream plotting
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


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
        f"No neurosim breakdown file found under {path}. Expected "
        f"{csv_path.name} or {json_path.name}."
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


def load_breakdown_rows(path: Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list in {path}")
        return [{k: _coerce_value(v) for k, v in row.items()} for row in data]

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [{k: _coerce_value(v) for k, v in row.items()} for row in reader]


def _layer_label(row: Dict[str, Any], idx: int) -> str:
    name = str(row.get("layer_name", "") or "").strip()
    if name:
        return name
    return f"layer_{idx}"


def build_latency_series(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = [_layer_label(row, idx) for idx, row in enumerate(rows)]
    return {
        "labels": labels,
        "components": {
            "ADC": [float(row.get("core_latency_adc_ns", 0.0) or 0.0) for row in rows],
            "Accum": [float(row.get("core_latency_accum_ns", 0.0) or 0.0) for row in rows],
            "Other": [float(row.get("core_latency_other_ns", 0.0) or 0.0) for row in rows],
        },
        "unit": "ns",
    }


def build_energy_series(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = [_layer_label(row, idx) for idx, row in enumerate(rows)]
    return {
        "labels": labels,
        "components": {
            "ADC": [float(row.get("core_energy_adc_nj", 0.0) or 0.0) for row in rows],
            "Accum": [float(row.get("core_energy_accum_nj", 0.0) or 0.0) for row in rows],
            "Other": [float(row.get("core_energy_other_nj", 0.0) or 0.0) for row in rows],
        },
        "unit": "nJ",
    }


def build_adc_energy_series(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = [_layer_label(row, idx) for idx, row in enumerate(rows)]
    return {
        "labels": labels,
        "components": {
            "Array": [float(row.get("adc_subarray_energy_nj", 0.0) or 0.0) for row in rows],
            "Precharger": [float(row.get("adc_precharger_energy_nj", 0.0) or 0.0) for row in rows],
            "SenseAmp": [float(row.get("adc_senseamp_energy_nj", 0.0) or 0.0) for row in rows],
            "MLSA": [float(row.get("adc_mlsa_energy_nj", 0.0) or 0.0) for row in rows],
            "ML Encoder": [float(row.get("adc_mlencoder_energy_nj", 0.0) or 0.0) for row in rows],
            "SAR ADC": [float(row.get("adc_sar_energy_nj", 0.0) or 0.0) for row in rows],
        },
        "unit": "nJ",
    }


def _write_long_table(series: Dict[str, Any], out_csv: Path, value_key: str) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    labels = list(series["labels"])
    components = dict(series["components"])
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["layer_name", "component", value_key])
        writer.writeheader()
        for component, values in components.items():
            for label, value in zip(labels, values):
                writer.writerow({"layer_name": label, "component": component, value_key: float(value)})
    return out_csv


def write_plot_tables(rows: List[Dict[str, Any]], out_dir: Path) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    written.append(
        _write_long_table(
            build_latency_series(rows),
            out_dir / "neurosim_layer_breakdown_latency_long.csv",
            "value_ns",
        )
    )
    written.append(
        _write_long_table(
            build_energy_series(rows),
            out_dir / "neurosim_layer_breakdown_energy_long.csv",
            "value_nj",
        )
    )
    written.append(
        _write_long_table(
            build_adc_energy_series(rows),
            out_dir / "neurosim_layer_breakdown_adc_energy_long.csv",
            "value_nj",
        )
    )
    return written


def _plot_stacked_bar(series: Dict[str, Any], title: str, ylabel: str, out_png: Path) -> Optional[Path]:
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

    fig_width = max(10.0, 0.6 * len(labels) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    x = list(range(len(labels)))
    bottoms = [0.0] * len(labels)
    palette = ["#244c9a", "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]

    for color_idx, (component, values) in enumerate(components.items()):
        vals = [float(v) for v in values]
        ax.bar(x, vals, bottom=bottoms, label=component, color=palette[color_idx % len(palette)], width=0.75)
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=min(3, max(1, len(components))), fontsize=9)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_png


def run(input_path: Path, out_dir: Path) -> Dict[str, List[Path]]:
    resolved_input = find_breakdown_input(Path(input_path))
    rows = load_breakdown_rows(resolved_input)
    if not rows:
        raise ValueError(f"No rows found in {resolved_input}")

    out_dir = Path(out_dir)
    written_tables = write_plot_tables(rows, out_dir)
    written_plots: List[Path] = []
    for series, title, ylabel, filename in (
        (build_latency_series(rows), "NeuroSim Layer Breakdown: Latency", "Latency (ns)", "neurosim_layer_breakdown_latency.png"),
        (build_energy_series(rows), "NeuroSim Layer Breakdown: Energy", "Energy (nJ)", "neurosim_layer_breakdown_energy.png"),
        (build_adc_energy_series(rows), "NeuroSim ADC Sub-breakdown: Energy", "Energy (nJ)", "neurosim_layer_breakdown_adc_energy.png"),
    ):
        out = _plot_stacked_bar(series, title, ylabel, out_dir / filename)
        if out is not None:
            written_plots.append(out)

    return {"tables": written_tables, "plots": written_plots}


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot per-layer NeuroSim stacked-bar breakdowns from CSV/JSON.")
    parser.add_argument(
        "input",
        nargs="?",
        default="sim_results_pointcim_knnconv_strict_p1024",
        help="Path to neurosim_layer_breakdown.csv/json or a sim_results directory containing one.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory for generated PNG/CSV outputs. Defaults to the input directory when a directory is given, otherwise the input file's parent.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    resolved_input = find_breakdown_input(input_path)
    default_out_dir = input_path if input_path.is_dir() else resolved_input.parent
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir

    outputs = run(input_path, out_dir)
    print(f"Loaded NeuroSim breakdown from: {resolved_input}")
    for table in outputs["tables"]:
        print(f"Wrote table: {table}")
    for plot in outputs["plots"]:
        print(f"Wrote plot: {plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
