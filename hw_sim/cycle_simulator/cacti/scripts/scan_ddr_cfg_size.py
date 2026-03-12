import csv
import math
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


CACTI_DIR = Path(__file__).resolve().parent.parent
CACTI_BIN = CACTI_DIR / "cacti"
BASE_CFG = CACTI_DIR / "ddr3.cfg"

OUT_CFG_DIR = CACTI_DIR / "outputs" / "generated_ddr3_cfgs"
OUT_LOG_DIR = CACTI_DIR / "outputs" / "ddr3_logs"
OUT_CSV = CACTI_DIR / "outputs" / "ddr3_size_scan_results.csv"


def replace_or_append_param(cfg_text: str, param_name: str, new_value: str) -> str:
    pattern = re.compile(
        rf"^(?!\s*//)\s*{re.escape(param_name)}\s+.*$",
        flags=re.MULTILINE,
    )
    replacement = f"{param_name} {new_value}"
    if pattern.search(cfg_text):
        cfg_text = pattern.sub(replacement, cfg_text, count=1)
    else:
        cfg_text += f"\n{replacement}\n"
    return cfg_text


def generate_cfg(base_cfg_path: Path, out_cfg_path: Path, size_bytes: int):
    text = base_cfg_path.read_text()

    text = replace_or_append_param(text, "-size (bytes)", str(size_bytes))

    out_cfg_path.write_text(text)


def run_cacti(cfg_path: Path) -> str:
    cmd = [str(CACTI_BIN), "-infile", str(cfg_path)]
    result = subprocess.run(
        cmd,
        cwd=str(CACTI_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout


def grab_float(patterns, text):
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


def parse_output(output: str) -> dict:
    metrics = {}

    # ===== DDR3 IO 部分 =====
    metrics["io_dynamic_power_mw"] = grab_float(
        [r"IO Dynamic Power \(mW\)\s*=\s*([0-9eE\.\+\-]+)"], output
    )
    metrics["phy_power_mw"] = grab_float(
        [r"PHY Power \(mW\)\s*=\s*([0-9eE\.\+\-]+)"], output
    )
    metrics["io_term_power_mw"] = grab_float(
        [r"IO Termination and Bias Power \(mW\)\s*=\s*([0-9eE\.\+\-]+)"], output
    )
    metrics["phy_wakeup_us"] = grab_float(
        [r"PHY Wakeup Time \(us\)\s*=\s*([0-9eE\.\+\-]+)"], output
    )

    # ===== cache / SRAM array 部分 =====
    metrics["access_time_ns"] = grab_float(
        [r"Access time \(ns\)\s*:\s*([0-9eE\.\+\-]+)"], output
    )
    metrics["cycle_time_ns"] = grab_float(
        [r"Cycle time \(ns\)\s*:\s*([0-9eE\.\+\-]+)"], output
    )
    metrics["read_energy_nj"] = grab_float(
        [r"Total dynamic read energy per access \(nJ\)\s*:\s*([0-9eE\.\+\-]+)"], output
    )
    metrics["write_energy_nj"] = grab_float(
        [r"Total dynamic write energy per access \(nJ\)\s*:\s*([0-9eE\.\+\-]+)"], output
    )
    metrics["leakage_mw"] = grab_float(
        [r"Total leakage power of a bank \(mW\)\s*:\s*([0-9eE\.\+\-]+)"], output
    )

    m_hw = re.search(
        r"Cache height x width \(mm\)\s*:\s*([0-9eE\.\+\-]+)\s*x\s*([0-9eE\.\+\-]+)",
        output,
        flags=re.IGNORECASE,
    )
    if m_hw:
        h = float(m_hw.group(1))
        w = float(m_hw.group(2))
        metrics["height_mm"] = h
        metrics["width_mm"] = w
        metrics["area_mm2"] = h * w
    else:
        metrics["height_mm"] = None
        metrics["width_mm"] = None
        metrics["area_mm2"] = None

    return metrics


def size_to_label(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes // (1024 * 1024)}MB"
    return f"{size_bytes // 1024}KB"


def maybe_plot(x_kb, y, ylabel, title, out_png, logx=True):
    xs = []
    ys = []
    for a, b in zip(x_kb, y):
        if b is not None:
            xs.append(a)
            ys.append(b)

    if not xs:
        print(f"[WARN] Skip plot {title}, no valid data.")
        return

    plt.figure(figsize=(7, 5))
    plt.plot(xs, ys, marker="o")
    if logx:
        plt.xscale("log", base=2)
    plt.xlabel("Cache size (KB)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    if not CACTI_BIN.exists():
        raise FileNotFoundError(f"Cannot find CACTI binary: {CACTI_BIN}")
    if not BASE_CFG.exists():
        raise FileNotFoundError(f"Cannot find base cfg: {BASE_CFG}")

    OUT_CFG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    (CACTI_DIR / "outputs" / "figures").mkdir(parents=True, exist_ok=True)

    sizes = [
        2 * 1024,
        4 * 1024,
        32 * 1024,
        128 * 1024,
        256 * 1024,
        1 * 1024 * 1024,
        2 * 1024 * 1024,
        4 * 1024 * 1024,
        8 * 1024 * 1024,
        16 * 1024 * 1024,
        32 * 1024 * 1024,
        64 * 1024 * 1024,
        128 * 1024 * 1024,
    ]

    results = []

    for size_bytes in sizes:
        label = size_to_label(size_bytes)
        cfg_path = OUT_CFG_DIR / f"ddr3_{label}.cfg"
        log_path = OUT_LOG_DIR / f"ddr3_{label}.log"

        generate_cfg(BASE_CFG, cfg_path, size_bytes)
        output = run_cacti(cfg_path)
        log_path.write_text(output)

        metrics = parse_output(output)
        row = {
            "size_bytes": size_bytes,
            "size_label": label,
            **metrics,
        }
        results.append(row)
        print(f"[DONE] {label}: {row}")

    fieldnames = [
        "size_bytes",
        "size_label",
        "io_dynamic_power_mw",
        "phy_power_mw",
        "io_term_power_mw",
        "phy_wakeup_us",
        "access_time_ns",
        "cycle_time_ns",
        "read_energy_nj",
        "write_energy_nj",
        "leakage_mw",
        "height_mm",
        "width_mm",
        "area_mm2",
    ]

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    x_kb = [r["size_bytes"] / 1024 for r in results]

    # cache array 动态能耗
    maybe_plot(
        x_kb,
        [r["read_energy_nj"] for r in results],
        ylabel="Read energy per access (nJ)",
        title="ddr3.cfg size scan: Cache read energy/access",
        out_png=CACTI_DIR / "outputs" / "figures" / "ddr3_size_vs_read_energy.png",
    )

    maybe_plot(
        x_kb,
        [r["write_energy_nj"] for r in results],
        ylabel="Write energy per access (nJ)",
        title="ddr3.cfg size scan: Cache write energy/access",
        out_png=CACTI_DIR / "outputs" / "figures" / "ddr3_size_vs_write_energy.png",
    )

    maybe_plot(
        x_kb,
        [r["leakage_mw"] for r in results],
        ylabel="Leakage power (mW)",
        title="ddr3.cfg size scan: Cache leakage",
        out_png=CACTI_DIR / "outputs" / "figures" / "ddr3_size_vs_leakage.png",
    )

    maybe_plot(
        x_kb,
        [r["access_time_ns"] for r in results],
        ylabel="Access time (ns)",
        title="ddr3.cfg size scan: Cache access time",
        out_png=CACTI_DIR / "outputs" / "figures" / "ddr3_size_vs_access_time.png",
    )

    maybe_plot(
        x_kb,
        [r["area_mm2"] for r in results],
        ylabel="Area (mm^2)",
        title="ddr3.cfg size scan: Cache area",
        out_png=CACTI_DIR / "outputs" / "figures" / "ddr3_size_vs_area.png",
    )

    # DDR3 IO power
    maybe_plot(
        x_kb,
        [r["io_dynamic_power_mw"] for r in results],
        ylabel="IO Dynamic Power (mW)",
        title="ddr3.cfg size scan: DDR3 IO Dynamic Power",
        out_png=CACTI_DIR / "outputs" / "figures" / "ddr3_size_vs_io_dynamic_power.png",
    )

    maybe_plot(
        x_kb,
        [r["phy_power_mw"] for r in results],
        ylabel="PHY Power (mW)",
        title="ddr3.cfg size scan: DDR3 PHY Power",
        out_png=CACTI_DIR / "outputs" / "figures" / "ddr3_size_vs_phy_power.png",
    )

    maybe_plot(
        x_kb,
        [r["io_term_power_mw"] for r in results],
        ylabel="IO Termination/Bias Power (mW)",
        title="ddr3.cfg size scan: DDR3 IO Termination/Bias Power",
        out_png=CACTI_DIR / "outputs" / "figures" / "ddr3_size_vs_io_term_power.png",
    )

    print("\nAll done.")
    print(f"CSV saved to: {OUT_CSV}")


if __name__ == "__main__":
    main()
