#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np


ROOT = Path("/workspace/PointKAN-pytorch")
PLOT_DIR = ROOT / "hw_sim" / "plot_figs"


def _find_input(path: Path) -> Path:
    if path.is_file():
        return path
    csv_path = path / "neurosim_layer_breakdown.csv"
    json_path = path / "neurosim_layer_breakdown.json"
    if csv_path.is_file():
        return csv_path
    if json_path.is_file():
        return json_path
    raise FileNotFoundError(f"No neurosim breakdown file found under {path}")


def _coerce(v: Any) -> Any:
    if isinstance(v, (float, int)):
        return float(v)
    if v is None:
        return 0.0
    t = str(v).strip()
    if t == "":
        return 0.0
    try:
        return float(t)
    except ValueError:
        return t


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    path = _find_input(path)
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [{k: _coerce(v) for k, v in row.items()} for row in data]
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [{k: _coerce(v) for k, v in row.items()} for row in reader]


def _f(row: Dict[str, Any], key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def _simplify_layer_name(name: str) -> str:
    out = str(name)
    out = out.replace(".mlp_convs.", ".")
    out = out.replace(".mlp.", ".")
    return out


Extractor = Callable[[Dict[str, Any]], float]


def functional_latency_components() -> List[Tuple[str, Extractor]]:
    return [
        ("Array", lambda r: _f(r, "array_latency_ns")),
        ("ADC", lambda r: max(0.0, _f(r, "core_latency_adc_ns") - _f(r, "array_latency_ns"))),
        ("Accum", lambda r: _f(r, "core_latency_accum_ns")),
        ("Buffer", lambda r: _f(r, "buffer_latency_ns")),
        ("Interconnect", lambda r: _f(r, "ic_latency_ns")),
        ("Activation/Pooling", lambda r: _f(r, "other_sub_activation_latency_ns") + _f(r, "other_sub_pooling_latency_ns")),
        ("Others", lambda r: _f(r, "core_latency_other_hidden_ns")),
    ]


def functional_energy_components() -> List[Tuple[str, Extractor]]:
    return [
        ("Array", lambda r: _f(r, "array_energy_nj")),
        ("ADC", lambda r: max(0.0, _f(r, "core_energy_adc_nj") - _f(r, "array_energy_nj"))),
        ("Accum", lambda r: _f(r, "core_energy_accum_nj")),
        ("Buffer", lambda r: _f(r, "buffer_energy_nj")),
        ("Interconnect", lambda r: _f(r, "ic_energy_nj")),
        ("Activation/Pooling", lambda r: _f(r, "other_sub_activation_energy_nj") + _f(r, "other_sub_pooling_energy_nj")),
        ("Others", lambda r: _f(r, "core_energy_other_hidden_nj")),
    ]


def hotspot_latency_components() -> List[Tuple[str, Extractor]]:
    return [
        ("Array", lambda r: _f(r, "array_latency_ns")),
        ("ADC", lambda r: max(0.0, _f(r, "core_latency_adc_ns") - _f(r, "array_latency_ns"))),
        ("Accum", lambda r: _f(r, "core_latency_accum_ns")),
        ("TileBufferHTree", lambda r: _f(r, "other_sub_tile_buffer_htree_latency_ns")),
        ("GlobalBuffer", lambda r: _f(r, "other_sub_global_buffer_latency_ns")),
        ("GlobalIC", lambda r: _f(r, "other_sub_global_ic_latency_ns")),
        ("TilePEOther", lambda r: _f(r, "other_sub_tile_pe_other_latency_ns")),
        ("Activation/Pooling", lambda r: _f(r, "other_sub_activation_latency_ns") + _f(r, "other_sub_pooling_latency_ns")),
        ("Others", lambda r: _f(r, "core_latency_other_hidden_ns")),
    ]


def hotspot_energy_components() -> List[Tuple[str, Extractor]]:
    return [
        ("Array", lambda r: _f(r, "array_energy_nj")),
        ("ADC", lambda r: max(0.0, _f(r, "core_energy_adc_nj") - _f(r, "array_energy_nj"))),
        ("Accum", lambda r: _f(r, "core_energy_accum_nj")),
        ("TileBufferHTree", lambda r: _f(r, "other_sub_tile_buffer_htree_energy_nj")),
        ("GlobalBuffer", lambda r: _f(r, "other_sub_global_buffer_energy_nj")),
        ("GlobalIC", lambda r: _f(r, "other_sub_global_ic_energy_nj")),
        ("TilePEOther", lambda r: _f(r, "other_sub_tile_pe_other_energy_nj")),
        ("Activation/Pooling", lambda r: _f(r, "other_sub_activation_energy_nj") + _f(r, "other_sub_pooling_energy_nj")),
        ("Others", lambda r: _f(r, "core_energy_other_hidden_nj")),
    ]


def _select_components(scheme: str, metric: str) -> List[Tuple[str, Extractor]]:
    if scheme == "functional" and metric == "latency":
        return functional_latency_components()
    if scheme == "functional" and metric == "energy":
        return functional_energy_components()
    if scheme == "hotspot" and metric == "latency":
        return hotspot_latency_components()
    if scheme == "hotspot" and metric == "energy":
        return hotspot_energy_components()
    raise ValueError(f"unsupported scheme={scheme} metric={metric}")


def _load_font():
    from matplotlib import font_manager

    candidate_fonts = [
        PLOT_DIR / "TIMES.TTF",
        PLOT_DIR / "TIMESBD.TTF",
        PLOT_DIR / "TIMESBI.TTF",
        PLOT_DIR / "TIMESI.TTF",
    ]
    font_name = None
    for fp in candidate_fonts:
        if fp.exists():
            font_manager.fontManager.addfont(str(fp))
            font_name = font_manager.FontProperties(fname=str(fp)).get_name()
            break
    return font_name or "DejaVu Sans"


def _prepare_plot_style():
    import matplotlib.pyplot as plt

    font_name = _load_font()
    font_size = 24
    legend_size = 22
    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": font_size - 2,
            "ytick.labelsize": font_size - 4,
            "legend.fontsize": legend_size,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
            "hatch.linewidth": 1.5,
        }
    )


def _build_model_plot_rows(rows: List[Dict[str, Any]], components: List[Tuple[str, Extractor]]) -> List[Dict[str, Any]]:
    out = []
    for idx, row in enumerate(rows):
        layer_name = str(row.get("layer_name", "") or "")
        label = f"L{idx}: {_simplify_layer_name(layer_name)}"
        values = {name: extractor(row) for name, extractor in components}
        total = float(sum(values.values()))
        total = total if total > 0 else float("nan")
        pct = {name: (val / total * 100.0 if total == total else 0.0) for name, val in values.items()}
        out.append(
            {
                "layer_index": idx,
                "layer_name": layer_name,
                "layer_label": label,
                "values": values,
                "pct": pct,
                "total": 0.0 if total != total else total,
            }
        )
    return out


def _write_merged_csv(model_tables: Dict[str, List[Dict[str, Any]]], component_names: List[str], metric: str, out_csv: Path) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model_key", "model_title", "layer_index", "layer_name", "layer_label", f"total_{metric}"] + [
        f"{name}_{metric}" for name in component_names
    ] + [f"{name}_pct" for name in component_names]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_key, rows in model_tables.items():
            model_title = {
                "cls": "PointNet++ Cls-SSG",
                "partseg": "PointNet++ PartSeg-SSG",
                "semseg": "PointNet++ Sem-SSG",
            }[model_key]
            for row in rows:
                rec = {
                    "model_key": model_key,
                    "model_title": model_title,
                    "layer_index": row["layer_index"],
                    "layer_name": row["layer_name"],
                    "layer_label": row["layer_label"],
                    f"total_{metric}": row["total"],
                }
                for name in component_names:
                    rec[f"{name}_{metric}"] = row["values"][name]
                    rec[f"{name}_pct"] = row["pct"][name]
                writer.writerow(rec)
    return out_csv


def _draw_subplot(ax, rows: List[Dict[str, Any]], component_names: List[str], colors: Dict[str, str], hatches: Dict[str, str], subtitle: str, xlabel: str):
    import matplotlib.ticker as mtick

    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    for name in component_names:
        vals = np.array([r["pct"][name] for r in rows], dtype=float)
        ax.barh(
            y,
            vals,
            left=left,
            height=0.78,
            color=colors[name],
            hatch=hatches[name],
            edgecolor="white",
            linewidth=0.8,
            zorder=2,
        )
        ax.barh(y, vals, left=left, height=0.78, color="none", edgecolor="white", linewidth=0.8, zorder=3)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([r["layer_label"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_title(subtitle, pad=12)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.tick_params(axis="both", width=1.1)


def _plot_three_models(model_tables: Dict[str, List[Dict[str, Any]]], component_names: List[str], metric: str, scheme: str, out_base: Path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    _prepare_plot_style()
    colors = {
        "Buffer": "#4F81BD",
        "Interconnect": "#C0504D",
        "ADC": "#9BBB59",
        "Accum": "#F79646",
        "Others": "#8064A2",
        "Array": "#2C7FB8",
        "Activation/Pooling": "#76B7B2",
        "TileBufferHTree": "#4F81BD",
        "GlobalBuffer": "#59A14F",
        "GlobalIC": "#C0504D",
        "TilePEOther": "#F28E2B",
    }
    hatches = {
        "Buffer": "//",
        "Interconnect": "\\\\",
        "ADC": "xx",
        "Accum": "--",
        "Others": "++",
        "Array": "..",
        "Activation/Pooling": "oo",
        "TileBufferHTree": "//",
        "GlobalBuffer": "..",
        "GlobalIC": "\\\\",
        "TilePEOther": "--",
    }

    max_layers = max(len(v) for v in model_tables.values())
    fig_h = max(9.0, 0.45 * max_layers + 3.0)
    fig, axes = plt.subplots(1, 3, figsize=(34, fig_h), gridspec_kw={"wspace": 0.18})
    order = [
        ("cls", "(a) Classification"),
        ("partseg", "(b) Part Segmentation"),
        ("semseg", "(c) Semantic Segmentation"),
    ]
    xlabel = "Breakdown Ratio"
    for ax, (key, title) in zip(axes, order):
        _draw_subplot(ax, model_tables[key], component_names, colors, hatches, title, xlabel)

    legend_handles = [
        Patch(facecolor=colors[name], edgecolor="white", hatch=hatches[name], linewidth=0.8, label=name) for name in component_names
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(len(component_names), 7),
        frameon=False,
        columnspacing=1.0,
        handlelength=1.8,
        handletextpad=0.5,
        borderaxespad=0.2,
        labelspacing=0.4,
    )
    plt.tight_layout()
    fig.subplots_adjust(top=0.88, left=0.10, right=0.98, bottom=0.08)
    for ext, kwargs in {
        ".pdf": {"transparent": True, "bbox_inches": "tight", "pad_inches": 0.03},
        ".png": {"dpi": 600, "transparent": True, "bbox_inches": "tight", "pad_inches": 0.03},
        ".svg": {"transparent": True, "bbox_inches": "tight", "pad_inches": 0.03},
    }.items():
        plt.savefig(str(out_base) + ext, **kwargs)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot three PointNet++ models using non-overlapping NeuroSim breakdown.")
    parser.add_argument("--scheme", choices=["functional", "hotspot"], default="functional")
    parser.add_argument(
        "--cls-run",
        default=str(ROOT / "analysis_subarray_pointnet2_pointcim_auto" / "runs" / "pointnet2-cls-ssg_pointcim_p1024_w8a8_suba128x128_auto"),
    )
    parser.add_argument(
        "--partseg-run",
        default=str(ROOT / "analysis_subarray_pointnet2_pointcim_auto" / "runs" / "pointnet2-partseg-ssg_pointcim_p1024_w8a8_suba128x128_auto"),
    )
    parser.add_argument(
        "--semseg-run",
        default=str(ROOT / "analysis_subarray_pointnet2_pointcim_auto" / "runs" / "pointnet2-sem-ssg_pointcim_p4096_w8a8_suba128x128_auto"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(PLOT_DIR),
    )
    parser.add_argument(
        "--out-prefix",
        default="pointnet2_three_models",
    )
    args = parser.parse_args()

    run_dirs = {
        "cls": Path(args.cls_run),
        "partseg": Path(args.partseg_run),
        "semseg": Path(args.semseg_run),
    }
    for key, path in run_dirs.items():
        _find_input(path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in ("latency", "energy"):
        components = _select_components(args.scheme, metric)
        component_names = [name for name, _ in components]
        model_tables = {key: _build_model_plot_rows(_load_rows(path), components) for key, path in run_dirs.items()}
        merged_csv = out_dir / f"{args.out_prefix}_{args.scheme}_{metric}_merged.csv"
        _write_merged_csv(model_tables, component_names, metric, merged_csv)
        metric_title = "layer_latency_breakdown_ratio" if metric == "latency" else "layer_energy_breakdown_ratio"
        out_base = out_dir / f"{args.out_prefix}_{args.scheme}_{metric_title}"
        _plot_three_models(model_tables, component_names, metric, args.scheme, out_base)
        print(f"Wrote merged CSV: {merged_csv}")
        print(f"Wrote plots: {out_base}.pdf/.png/.svg")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
