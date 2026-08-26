import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.ticker as ticker
from matplotlib.patches import Patch


# =========================
# 1) Font
# =========================
candidate_fonts = [
    "/root/.fonts/Arial.ttf",
    "/root/.fonts/arial.ttf",
    "/root/.fonts/ARIALN.TTF",
    "/root/.fonts/arialbd.ttf",
    "/home/oai/share/fonts/Arial.ttf",
]

font_name = None
for fp in candidate_fonts:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        font_name = font_manager.FontProperties(fname=fp).get_name()
        break

if font_name is None:
    font_name = "DejaVu Sans"

print("Loaded font:", font_name)


# =========================
# 2) Style
# =========================
FONT_SIZE = 22

plt.rcParams.update({
    "font.family": font_name,
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 3,
    "ytick.labelsize": FONT_SIZE - 3,
    "legend.fontsize": FONT_SIZE - 3,
    "figure.titlesize": FONT_SIZE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
    "hatch.linewidth": 2.2,
})


# =========================
# 3) Colors / hatch
# =========================
COLOR_BASELINE = "#d9f2d0"
COLOR_OBAS = "#ffc000"
COLOR_OBAS_PIM = "#00b0f0"

PATTERN_BASELINE = "///"
PATTERN_OBAS = "\\\\"
PATTERN_OBAS_PIM = "xx"

MODE_LABELS = ["GRF Baseline", "GRF(OBAS)", "GRF(OBAS+PIM)"]
MODE_COLORS = [COLOR_BASELINE, COLOR_OBAS, COLOR_OBAS_PIM]
MODE_HATCHES = [PATTERN_BASELINE, PATTERN_OBAS, PATTERN_OBAS_PIM]


# =========================
# 4) Data: center-only GRF/KAT, 4 center KATs per model
# =========================
datasets = ["ModelNet40", "ShapeNet", "ScanObjectNN"]
x = np.arange(len(datasets))

# cycles at 1 ns/cycle. Convert to ms for plotting.
latency_cycles = np.array([
    [2_490_641, 131_349, 131_236],
    [1_167_633, 61_717, 61_604],
    [2_490_641, 131_349, 131_236],
], dtype=float)
latency_ms = latency_cycles / 1e6

# nJ
energy_nj = np.array([
    [59_292.157514, 137.111738, 75.681878],
    [27_826.661348, 97.734842, 36.304982],
    [59_292.157514, 137.111738, 75.681878],
], dtype=float)

latency_speedup = np.concatenate(
    [np.ones((len(datasets), 1)), latency_cycles[:, [0]] / latency_cycles[:, 1:]],
    axis=1,
)
energy_reduction = np.concatenate(
    [np.ones((len(datasets), 1)), energy_nj[:, [0]] / energy_nj[:, 1:]],
    axis=1,
)


# =========================
# 5) Plot helpers
# =========================
def style_axes(ax, logy=False):
    if logy:
        ax.set_yscale("log", base=10)
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10))
        ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))

    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(False)

    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
        spine.set_zorder(10)

    ax.tick_params(width=1.4)


def draw_white_hatch_bars(ax, xs, ys, width, color, hatch, label=None, zbase=2):
    ax.bar(
        xs, ys, width=width,
        color=color,
        hatch=hatch,
        edgecolor="white",
        linewidth=1.0,
        label=label,
        zorder=zbase,
    )
    ax.bar(
        xs, ys, width=width,
        color="none",
        edgecolor="white",
        linewidth=1.0,
        zorder=zbase + 1,
    )


def grouped_bars(ax, values, ylabel, title, logy=False):
    width = 0.23
    offsets = np.array([-width, 0.0, width])
    for idx, (label, color, hatch) in enumerate(zip(MODE_LABELS, MODE_COLORS, MODE_HATCHES)):
        draw_white_hatch_bars(ax, x + offsets[idx], values[:, idx], width, color, hatch, label=label)

    ax.set_title(title, pad=10)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=18, ha="right")
    ax.margins(x=0.05)
    style_axes(ax, logy=logy)


def gain_bars(ax, values, ylabel, title):
    width = 0.23
    labels = MODE_LABELS
    colors = MODE_COLORS
    hatches = MODE_HATCHES
    offsets = np.array([-width, 0.0, width])

    for idx, (label, color, hatch) in enumerate(zip(labels, colors, hatches)):
        ys = values[:, idx]
        draw_white_hatch_bars(ax, x + offsets[idx], ys, width, color, hatch, label=label)
        for bx, by in zip(x + offsets[idx], ys):
            ax.text(
                bx, by * 1.015, f"{by:.1f}",
                ha="center", va="bottom",
                fontsize=FONT_SIZE - 7,
                rotation=90,
                clip_on=False,
            )

    ax.set_title(title, pad=10)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=18, ha="right")
    ax.margins(x=0.08)
    ymax = np.max(values) * 1.18
    ax.set_ylim(0, ymax)
    style_axes(ax, logy=False)


def plot_grf_center_obas(fname="grf_center_obas_analysis", figsize=(11, 4.8)):
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"wspace": 0.32})

    gain_bars(axes[0], latency_speedup, "Speedup", "Latency Gain")
    gain_bars(axes[1], energy_reduction, "Reduction", "Energy Gain")

    legend_handles = [
        Patch(facecolor=c, edgecolor="white", hatch=h, label=l)
        for c, h, l in zip(MODE_COLORS, MODE_HATCHES, MODE_LABELS)
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
        columnspacing=1.6,
        handlelength=2.0,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_dir = os.path.dirname(os.path.abspath(__file__))
    for ext in ("pdf", "png", "svg"):
        out_path = os.path.join(out_dir, f"{fname}.{ext}")
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
        print("Saved:", out_path)
    plt.close(fig)


if __name__ == "__main__":
    plot_grf_center_obas()
