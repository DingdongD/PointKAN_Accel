#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


ROOT = Path("/workspace/PointKAN-pytorch")
PLOT_DIR = ROOT / "hw_sim" / "plot_figs"


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


def _prepare_plot_style() -> None:
    import matplotlib.pyplot as plt

    font_name = _load_font()
    font_size = 26
    legend_size = 24
    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": font_size - 2,
            "ytick.labelsize": font_size - 5,
            "legend.fontsize": legend_size,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
            "hatch.linewidth": 1.5,
        }
    )


def _simplify_layer_name(name: str) -> str:
    out = str(name)
    out = out.replace(".mlp_convs.", ".")
    out = out.replace(".mlp.", ".")
    return out


def _read_merged_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _metric_columns(rows: List[Dict[str, str]]) -> Tuple[List[str], str]:
    first = rows[0]
    total_cols = [k for k in first if k.startswith("total_")]
    if len(total_cols) != 1:
        raise ValueError(f"Expected exactly one total_* column, got {total_cols}")
    total_col = total_cols[0]
    component_cols = [
        k
        for k in first
        if (k.endswith("_latency") or k.endswith("_energy")) and not k.startswith("total_")
    ]
    return component_cols, total_col


def _build_compare_rows(rows_w8: List[Dict[str, str]], rows_w4: List[Dict[str, str]], model_key: str) -> Tuple[List[Dict[str, object]], List[str], str]:
    rows8 = [r for r in rows_w8 if r["model_key"] == model_key]
    rows4 = [r for r in rows_w4 if r["model_key"] == model_key]
    comp_cols8, total_col8 = _metric_columns(rows8)
    comp_cols4, total_col4 = _metric_columns(rows4)
    if comp_cols8 != comp_cols4 or total_col8 != total_col4:
        raise ValueError("W8A8 and W4A4 merged CSV component sets do not match")

    by_layer4 = {r["layer_name"]: r for r in rows4}
    compare_rows: List[Dict[str, object]] = []
    for row8 in rows8:
        layer_name = row8["layer_name"]
        if layer_name not in by_layer4:
            raise KeyError(f"Missing W4A4 row for layer {layer_name}")
        row4 = by_layer4[layer_name]
        compare_rows.append(
            {
                "layer_index": int(float(row8["layer_index"])),
                "layer_name": layer_name,
                "layer_label": f"L{int(float(row8['layer_index']))}",
                "w8_total": float(row8[total_col8]),
                "w4_total": float(row4[total_col4]),
                "w8_values": {c: float(row8[c]) for c in comp_cols8},
                "w4_values": {c: float(row4[c]) for c in comp_cols8},
            }
        )
    compare_rows.sort(key=lambda r: int(r["layer_index"]))  # type: ignore[arg-type]
    return compare_rows, comp_cols8, total_col8


def _component_display_name(col_name: str) -> str:
    if col_name.endswith("_latency"):
        return col_name[: -len("_latency")]
    if col_name.endswith("_energy"):
        return col_name[: -len("_energy")]
    return col_name


def _component_style(component_names: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    colors = {
        "Array": "#2C7FB8",
        "ADC": "#9BBB59",
        "Accum": "#F79646",
        "Buffer": "#4F81BD",
        "Interconnect": "#C0504D",
        "Activation/Pooling": "#76B7B2",
        "Others": "#8064A2",
        "TileBufferHTree": "#4F81BD",
        "GlobalBuffer": "#59A14F",
        "GlobalIC": "#C0504D",
        "TilePEOther": "#F28E2B",
    }
    hatches = {
        "Array": "..",
        "ADC": "xx",
        "Accum": "--",
        "Buffer": "//",
        "Interconnect": "\\\\",
        "Activation/Pooling": "oo",
        "Others": "++",
        "TileBufferHTree": "//",
        "GlobalBuffer": "..",
        "GlobalIC": "\\\\",
        "TilePEOther": "--",
    }
    return (
        {name: colors.get(name, "#777777") for name in component_names},
        {name: hatches.get(name, "") for name in component_names},
    )


def _format_total_text(value: float, total_col: str) -> str:
    if total_col.endswith("latency"):
        return f"{value / 1e6:.2f} ms"
    return f"{value:.1f} nJ"


def _draw_one_subplot(ax, compare_rows: List[Dict[str, object]], component_cols: List[str], component_names: List[str], colors: Dict[str, str], hatches: Dict[str, str], bw_key: str, title: str, total_col: str, text_size: float) -> None:
    y = np.arange(len(compare_rows))
    left = np.zeros(len(compare_rows))
    value_key = "w8_values" if bw_key == "w8" else "w4_values"
    total_key = "w8_total" if bw_key == "w8" else "w4_total"

    for col, comp_name in zip(component_cols, component_names):
        vals = np.array([float(r[value_key][col]) for r in compare_rows], dtype=float)  # type: ignore[index]
        ax.barh(
            y,
            vals,
            left=left,
            height=0.78,
            color=colors[comp_name],
            hatch=hatches[comp_name],
            edgecolor="white",
            linewidth=0.8,
            zorder=2,
        )
        ax.barh(
            y,
            vals,
            left=left,
            height=0.78,
            color="none",
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        left += vals

    labels = [str(r["layer_label"]) for r in compare_rows]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    unit = "Latency (ns)" if total_col.endswith("latency") else "Energy (nJ)"
    ax.set_xlabel(unit)
    ax.set_title(title, pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.35, zorder=0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.tick_params(axis="both", width=1.1)

    totals = [float(r[total_key]) for r in compare_rows]  # type: ignore[arg-type]
    max_total = max(totals) if totals else 1.0
    ax.set_xlim(0, max_total * 1.18)
    for yi, total in enumerate(totals):
        ax.text(
            total + max_total * 0.01,
            yi,
            _format_total_text(total, total_col),
            va="center",
            ha="left",
            fontsize=text_size,
        )


def _plot_model(
    compare_rows: List[Dict[str, object]],
    component_cols: List[str],
    total_col: str,
    model_title: str,
    out_base: Path,
    formats: Iterable[str],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    _prepare_plot_style()
    component_names = [_component_display_name(c) for c in component_cols]
    colors, hatches = _component_style(component_names)

    n = len(compare_rows)
    fig_h = max(8.0, 0.48 * n + 2.5)
    fig, axes = plt.subplots(1, 2, figsize=(28, fig_h), gridspec_kw={"wspace": 0.18})

    _draw_one_subplot(
        axes[0],
        compare_rows,
        component_cols,
        component_names,
        colors,
        hatches,
        "w8",
        "(a) BioAP-Base",
        total_col,
        14,
    )
    _draw_one_subplot(
        axes[1],
        compare_rows,
        component_cols,
        component_names,
        colors,
        hatches,
        "w4",
        "(b) BioAP-SGBA",
        total_col,
        14,
    )
    fig.suptitle(f"{model_title}: Layer-wise Absolute {'Latency' if total_col.endswith('latency') else 'Energy'} Breakdown", y=0.995)

    component_handles = [
        Patch(facecolor=colors[name], edgecolor="white", hatch=hatches[name], linewidth=0.8, label=name)
        for name in component_names
    ]
    fig.legend(
        handles=component_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.045),
        ncol=min(len(component_handles), 7),
        frameon=False,
        columnspacing=0.95,
        handlelength=1.8,
        handletextpad=0.5,
        borderaxespad=0.2,
        labelspacing=0.4,
    )
    plt.tight_layout()
    fig.subplots_adjust(top=0.89, left=0.12, right=0.98, bottom=0.06)
    format_kwargs = {
        ".pdf": {"transparent": True, "bbox_inches": "tight", "pad_inches": 0.03},
        ".png": {"dpi": 300, "transparent": True, "bbox_inches": "tight", "pad_inches": 0.03},
        ".svg": {"transparent": True, "bbox_inches": "tight", "pad_inches": 0.03},
    }
    for ext in formats:
        kwargs = format_kwargs[ext]
        plt.savefig(str(out_base) + ext, **kwargs)
    plt.close(fig)


def run(
    latency_w8: Path,
    latency_w4: Path,
    energy_w8: Path,
    energy_w4: Path,
    out_dir: Path,
    model_keys: Iterable[str] | None = None,
    metrics: Iterable[str] = ("latency", "energy"),
    formats: Iterable[str] = (".pdf", ".png", ".svg"),
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []
    rows_latency_w8 = _read_merged_csv(latency_w8)
    rows_latency_w4 = _read_merged_csv(latency_w4)
    rows_energy_w8 = _read_merged_csv(energy_w8)
    rows_energy_w4 = _read_merged_csv(energy_w4)

    model_titles = {
        "cls": "PointNet++ Cls-SSG",
        "partseg": "PointNet++ PartSeg-SSG",
        "semseg": "PointNet++ Sem-SSG",
    }
    selected_model_keys = list(model_keys) if model_keys is not None else list(model_titles.keys())
    selected_metrics = set(metrics)
    selected_formats = list(formats)
    for model_key in selected_model_keys:
        model_title = model_titles[model_key]
        if "latency" in selected_metrics:
            compare_rows, component_cols, total_col = _build_compare_rows(rows_latency_w8, rows_latency_w4, model_key)
            out_base = out_dir / f"pointnet2_{model_key}_w8a8_vs_w4a4_absolute_latency_breakdown"
            _plot_model(compare_rows, component_cols, total_col, model_title, out_base, selected_formats)
            outputs.extend([out_base.with_suffix(ext) for ext in selected_formats])

        if "energy" in selected_metrics:
            compare_rows, component_cols, total_col = _build_compare_rows(rows_energy_w8, rows_energy_w4, model_key)
            out_base = out_dir / f"pointnet2_{model_key}_w8a8_vs_w4a4_absolute_energy_breakdown"
            _plot_model(compare_rows, component_cols, total_col, model_title, out_base, selected_formats)
            outputs.extend([out_base.with_suffix(ext) for ext in selected_formats])
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot absolute per-layer W8A8 vs W4A4 breakdowns for three PointNet++ models.")
    parser.add_argument("--latency-w8", default=str(PLOT_DIR / "pointnet2_three_models_w8a8_functional_latency_merged.csv"))
    parser.add_argument("--latency-w4", default=str(PLOT_DIR / "pointnet2_three_models_w4a4_functional_latency_merged.csv"))
    parser.add_argument("--energy-w8", default=str(PLOT_DIR / "pointnet2_three_models_w8a8_functional_energy_merged.csv"))
    parser.add_argument("--energy-w4", default=str(PLOT_DIR / "pointnet2_three_models_w4a4_functional_energy_merged.csv"))
    parser.add_argument("--out-dir", default=str(PLOT_DIR))
    parser.add_argument(
        "--models",
        default="cls,partseg,semseg",
        help="Comma-separated model keys: cls,partseg,semseg",
    )
    parser.add_argument(
        "--formats",
        default="pdf,png,svg",
        help="Comma-separated output formats without dots, e.g. pdf,png",
    )
    parser.add_argument(
        "--metrics",
        default="latency,energy",
        help="Comma-separated metrics: latency,energy",
    )
    args = parser.parse_args()

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    formats = [f".{fmt.strip().lstrip('.')}" for fmt in args.formats.split(",") if fmt.strip()]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    outputs = run(
        Path(args.latency_w8),
        Path(args.latency_w4),
        Path(args.energy_w8),
        Path(args.energy_w4),
        Path(args.out_dir),
        model_keys=model_keys,
        metrics=metrics,
        formats=formats,
    )
    for out in outputs:
        print(f"Wrote plot: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
