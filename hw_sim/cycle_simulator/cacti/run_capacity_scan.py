import os
import re
import csv
import math
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


CACTI_DIR = Path(".").resolve()
CACTI_BIN = CACTI_DIR / "cacti"
BASE_CFG = CACTI_DIR / "cache.cfg"

OUT_CFG_DIR = CACTI_DIR / "generated_cfgs"
OUT_LOG_DIR = CACTI_DIR / "logs"
OUT_CSV = CACTI_DIR / "capacity_scan_results.csv"
OUT_PNG1 = CACTI_DIR / "capacity_vs_access_time.png"
OUT_PNG2 = CACTI_DIR / "capacity_vs_area.png"
OUT_PNG3 = CACTI_DIR / "capacity_vs_read_energy.png"


def replace_or_append_param(cfg_text: str, param_name: str, new_value: str) -> str:
    """
    将配置文件中某个有效参数行替换为新值。
    只替换未被 // 注释掉的行。
    若不存在，则追加到文件末尾。
    """
    pattern = re.compile(
        rf'^(?!//)\s*{re.escape(param_name)}\s+.*$',
        flags=re.MULTILINE
    )
    replacement = f"{param_name} {new_value}"
    if pattern.search(cfg_text):
        cfg_text = pattern.sub(replacement, cfg_text, count=1)
    else:
        cfg_text += f"\n{replacement}\n"
    return cfg_text


def generate_cfg(base_cfg_path: Path, out_cfg_path: Path, size_bytes: int,
                 block_size: int = 64,
                 associativity: int = 1,
                 bank_count: int = 1,
                 technology_um: float = 0.032,
                 bus_width: int = 512,
                 temperature_k: int = 300):
    text = base_cfg_path.read_text()

    text = replace_or_append_param(text, "-size (bytes)", str(size_bytes))
    text = replace_or_append_param(text, "-block size (bytes)", str(block_size))
    text = replace_or_append_param(text, "-associativity", str(associativity))
    text = replace_or_append_param(text, "-UCA bank count", str(bank_count))
    text = replace_or_append_param(text, "-technology (u)", f"{technology_um:.3f}")
    text = replace_or_append_param(text, "-output/input bus width", str(bus_width))
    text = replace_or_append_param(text, "-operating temperature (K)", str(temperature_k))

    out_cfg_path.write_text(text)


def run_cacti(cfg_path: Path) -> str:
    cmd = [str(CACTI_BIN), "-infile", str(cfg_path)]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=CACTI_DIR
    )
    return result.stdout


def try_extract(patterns, text):
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


def parse_cacti_output(output: str) -> dict:
    """
    不同版本/配置输出字段可能略有差异，所以这里做“宽松匹配”。
    你第一次跑完后，如果某个字段没抓到，打印 output 看一下字段名，再补 regex。
    """
    metrics = {}

    metrics["access_time_ns"] = try_extract([
        r"Access time \(ns\)\s*:\s*([0-9eE\.\+\-]+)",
        r"access time \(ns\)\s*:\s*([0-9eE\.\+\-]+)",
    ], output)

    metrics["cycle_time_ns"] = try_extract([
        r"Cycle time \(ns\)\s*:\s*([0-9eE\.\+\-]+)",
        r"cycle time \(ns\)\s*:\s*([0-9eE\.\+\-]+)",
    ], output)

    metrics["area_mm2"] = try_extract([
        r"Cache height x width \(mm\)\s*:\s*([0-9eE\.\+\-]+)\s*x\s*([0-9eE\.\+\-]+)",
        r"Cache size \(mm2\)\s*:\s*([0-9eE\.\+\-]+)",
        r"Area \(mm2\)\s*:\s*([0-9eE\.\+\-]+)",
    ], output)

    # 如果是 height x width，则单独算面积
    m_hw = re.search(
        r"Cache height x width \(mm\)\s*:\s*([0-9eE\.\+\-]+)\s*x\s*([0-9eE\.\+\-]+)",
        output, flags=re.IGNORECASE
    )
    if m_hw:
        h = float(m_hw.group(1))
        w = float(m_hw.group(2))
        metrics["area_mm2"] = h * w

    metrics["read_dynamic_nj"] = try_extract([
        r"Total dynamic read energy per access \(nJ\)\s*:\s*([0-9eE\.\+\-]+)",
        r"Dynamic read energy \(nJ\)\s*:\s*([0-9eE\.\+\-]+)",
    ], output)

    metrics["write_dynamic_nj"] = try_extract([
        r"Total dynamic write energy per access \(nJ\)\s*:\s*([0-9eE\.\+\-]+)",
        r"Dynamic write energy \(nJ\)\s*:\s*([0-9eE\.\+\-]+)",
    ], output)

    metrics["leakage_mw"] = try_extract([
        r"Total leakage power of a bank \(mW\)\s*:\s*([0-9eE\.\+\-]+)",
        r"Total leakage power \(mW\)\s*:\s*([0-9eE\.\+\-]+)",
        r"Leakage power \(mW\)\s*:\s*([0-9eE\.\+\-]+)",
    ], output)

    return metrics


def size_to_label(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes // (1024 * 1024)}MB"
    return f"{size_bytes // 1024}KB"


def main():
    if not CACTI_BIN.exists():
        raise FileNotFoundError(f"Cannot find CACTI binary: {CACTI_BIN}")
    if not BASE_CFG.exists():
        raise FileNotFoundError(f"Cannot find base cfg: {BASE_CFG}")

    OUT_CFG_DIR.mkdir(exist_ok=True)
    OUT_LOG_DIR.mkdir(exist_ok=True)

    # 你要扫描的容量点
    capacities = [
        32 * 1024,
        64 * 1024,
        128 * 1024,
        256 * 1024,
        512 * 1024,
        1 * 1024 * 1024,
        2 * 1024 * 1024,
    ]

    results = []

    for size_bytes in capacities:
        tag = size_to_label(size_bytes)
        cfg_path = OUT_CFG_DIR / f"cache_{tag}.cfg"
        log_path = OUT_LOG_DIR / f"cache_{tag}.log"

        generate_cfg(
            base_cfg_path=BASE_CFG,
            out_cfg_path=cfg_path,
            size_bytes=size_bytes,
            block_size=64,
            associativity=1,
            bank_count=1,
            technology_um=0.032,
            bus_width=512,
            temperature_k=300,
        )

        output = run_cacti(cfg_path)
        log_path.write_text(output)

        metrics = parse_cacti_output(output)
        row = {
            "size_bytes": size_bytes,
            "size_label": tag,
            **metrics
        }
        results.append(row)
        print(f"[DONE] {tag}: {row}")

    # 写 CSV
    fieldnames = [
        "size_bytes", "size_label",
        "access_time_ns", "cycle_time_ns",
        "area_mm2", "read_dynamic_nj", "write_dynamic_nj", "leakage_mw"
    ]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # 准备画图数据
    xs = [r["size_bytes"] / 1024 for r in results]  # KB
    access = [r["access_time_ns"] for r in results]
    area = [r["area_mm2"] for r in results]
    read_energy = [r["read_dynamic_nj"] for r in results]

    # 图1：容量 vs access time
    plt.figure(figsize=(7, 5))
    plt.plot(xs, access, marker="o")
    plt.xscale("log", base=2)
    plt.xlabel("Capacity (KB)")
    plt.ylabel("Access time (ns)")
    plt.title("CACTI Capacity Scan: Access Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_PNG1, dpi=200)
    plt.close()

    # 图2：容量 vs area
    plt.figure(figsize=(7, 5))
    plt.plot(xs, area, marker="o")
    plt.xscale("log", base=2)
    plt.xlabel("Capacity (KB)")
    plt.ylabel("Area (mm^2)")
    plt.title("CACTI Capacity Scan: Area")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_PNG2, dpi=200)
    plt.close()

    # 图3：容量 vs read energy
    plt.figure(figsize=(7, 5))
    plt.plot(xs, read_energy, marker="o")
    plt.xscale("log", base=2)
    plt.xlabel("Capacity (KB)")
    plt.ylabel("Read dynamic energy per access (nJ)")
    plt.title("CACTI Capacity Scan: Read Energy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_PNG3, dpi=200)
    plt.close()

    print("\nAll done.")
    print(f"CSV: {OUT_CSV}")
    print(f"PNG: {OUT_PNG1}")
    print(f"PNG: {OUT_PNG2}")
    print(f"PNG: {OUT_PNG3}")


if __name__ == "__main__":
    main()
