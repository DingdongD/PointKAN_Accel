#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


# ============================================================
# 工具函数
# ============================================================

def mw_ns_to_nj(power_mw: float, time_ns: float) -> float:
    # 1 mW * 1 ns = 1e-3 nJ
    return power_mw * time_ns / 1e3


def replace_or_append_param(cfg_text: str, param_name: str, new_value: str) -> str:
    """
    只替换未被 // 注释的有效行。
    如果找不到，则追加到文件末尾。
    """
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


def extract_first(patterns, text, cast=float):
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            try:
                return cast(m.group(1))
            except Exception:
                pass
    return None


# ============================================================
# 数据结构
# ============================================================

@dataclass
class DramIOCalib:
    dram_type: str
    io_type: str
    iostate: str

    bus_freq_mhz: float
    num_dq: int
    burst_bytes: int

    io_dynamic_power_mw: float
    phy_power_mw: float
    io_termination_power_mw: float
    phy_wakeup_time_us: float

    # 作为“协议/控制器固定开销”的可调参数
    read_fixed_latency_ns: float
    write_fixed_latency_ns: float

    # 行未命中惩罚，作为上层模型可调参数
    row_miss_penalty_ns: float = 0.0

    def peak_bandwidth_Bps(self) -> float:
        # DDR 双沿传输
        transfers_per_sec = 2.0 * self.bus_freq_mhz * 1e6
        bytes_per_transfer = self.num_dq / 8.0
        return transfers_per_sec * bytes_per_transfer


@dataclass
class SingleAccessResult:
    op: str
    nbytes: int
    bursts: int
    transfer_time_ns: float
    fixed_latency_ns: float
    wakeup_ns: float
    queue_delay_ns: float
    row_penalty_ns: float
    latency_ns: float
    transfer_energy_nj: float
    background_energy_nj: float
    energy_nj: float


@dataclass
class CombinedAccessResult:
    read: Optional[SingleAccessResult]
    write: Optional[SingleAccessResult]
    latency_ns: float
    energy_nj: float


# ============================================================
# 配置生成
# ============================================================

def generate_io_cfg(
    base_cfg_path: Path,
    out_cfg_path: Path,
    io_state: str,
    bus_freq_mhz: Optional[float] = None,
    num_dq: Optional[int] = None,
    mem_density: Optional[str] = None,
    addr_timing: Optional[float] = None,
):
    text = base_cfg_path.read_text()

    # 你的日志里打印成 iostate : READ
    # cfg 文件里通常是 -iostate "R"/"W"
    text = replace_or_append_param(text, "-iostate", f"\"{io_state}\"")

    if bus_freq_mhz is not None:
        text = replace_or_append_param(text, "-bus_freq", f"{bus_freq_mhz:g} MHz")

    if num_dq is not None:
        text = replace_or_append_param(text, "-num_dq", str(num_dq))

    if mem_density is not None:
        text = replace_or_append_param(text, "-mem_density", mem_density)

    if addr_timing is not None:
        text = replace_or_append_param(text, "-addr_timing", f"{addr_timing:g}")

    out_cfg_path.write_text(text)


# ============================================================
# cfg 元信息解析
# ============================================================

def parse_cfg_meta(cfg_text: str) -> dict:
    dram_type = extract_first(
        [r'^\s*-dram_type\s+"?([A-Za-z0-9_]+)"?'],
        cfg_text,
        cast=str,
    )

    io_type = extract_first(
        [r'^\s*-io_type\s+"?([A-Za-z0-9_]+)"?'],
        cfg_text,
        cast=str,
    )

    # cfg 里一般是 R/W；输出里可能是 READ/WRITE
    iostate = extract_first(
        [r'^\s*-iostate\s+"?([A-Za-z]+)"?'],
        cfg_text,
        cast=str,
    )

    bus_freq_mhz = extract_first(
        [
            r"^\s*-bus_freq\s+([0-9eE\.\+\-]+)\s*MHz",
            r"^\s*-bus_freq\s+([0-9eE\.\+\-]+)",
        ],
        cfg_text,
        cast=float,
    )

    num_dq = extract_first(
        [r"^\s*-num_dq\s+([0-9]+)"],
        cfg_text,
        cast=int,
    )

    return {
        "dram_type": dram_type,
        "io_type": io_type,
        "iostate": iostate,
        "bus_freq_mhz": bus_freq_mhz,
        "num_dq": num_dq,
    }


# ============================================================
# 仅解析 DRAM IO / PHY 字段
# ============================================================

def parse_cacti_dram_io_metrics(output: str) -> dict:
    return {
        "io_dynamic_power_mw": extract_first(
            [r"IO Dynamic Power \(mW\)\s*=\s*([0-9eE\.\+\-]+)"],
            output,
            cast=float,
        ),
        "phy_power_mw": extract_first(
            [r"PHY Power \(mW\)\s*=\s*([0-9eE\.\+\-]+)"],
            output,
            cast=float,
        ),
        "phy_wakeup_time_us": extract_first(
            [r"PHY Wakeup Time \(us\)\s*=\s*([0-9eE\.\+\-]+)"],
            output,
            cast=float,
        ),
        "io_termination_power_mw": extract_first(
            [r"IO Termination and Bias Power \(mW\)\s*=\s*([0-9eE\.\+\-]+)"],
            output,
            cast=float,
        ),
        "io_timing_margin_ps": extract_first(
            [r"IO Timing Margin \(ps\)\s*=\s*([0-9eE\.\+\-]+)"],
            output,
            cast=float,
        ),
        # 你的日志里就是 Votlage，按原样兼容
        "io_voltage_margin_v": extract_first(
            [r"IO Votlage Margin \(V\)\s*=\s*([0-9eE\.\+\-]+)"],
            output,
            cast=float,
        ),
        "printed_iostate": extract_first(
            [r"^\s*iostate\s*:\s*([A-Za-z]+)"],
            output,
            cast=str,
        ),
        "printed_io_type": extract_first(
            [r"^\s*io_type\s*:\s*([A-Za-z0-9_]+)"],
            output,
            cast=str,
        ),
    }


# ============================================================
# 调 CACTI
# ============================================================

def run_cacti(cacti_bin: Path, cfg_path: Path, workdir: Path) -> str:
    cmd = [str(cacti_bin), "-infile", str(cfg_path)]
    result = subprocess.run(
        cmd,
        cwd=str(workdir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout


# ============================================================
# 标定器
# ============================================================

class CactiDramIOCalibrator:
    def __init__(
        self,
        cacti_dir: Path,
        cacti_bin: str = "cacti",
        base_cfg: str = "ddr3.cfg",
        work_subdir: str = "outputs/dram_io_calib_runs",
    ):
        self.cacti_dir = Path(cacti_dir).resolve()
        self.cacti_bin = self.cacti_dir / cacti_bin
        self.base_cfg = self.cacti_dir / base_cfg
        self.workdir = self.cacti_dir / work_subdir

    def calibrate(
        self,
        burst_bytes: int = 64,
        bus_freq_mhz: Optional[float] = None,
        num_dq: Optional[int] = None,
        mem_density: Optional[str] = None,
        addr_timing: Optional[float] = None,
        # 这两个是你仿真器的固定协议/控制器开销，不取 SRAM access time
        read_fixed_latency_ns: float = 15.0,
        write_fixed_latency_ns: float = 15.0,
        row_miss_penalty_ns: float = 0.0,
        # fallback
        fallback_io_dynamic_power_mw: float = 1750.0,
        fallback_phy_power_mw: float = 230.0,
        fallback_io_termination_power_mw: float = 3800.0,
        fallback_phy_wakeup_time_us: float = 27.5,
        verbose: bool = True,
    ) -> DramIOCalib:

        if not self.cacti_bin.exists():
            raise FileNotFoundError(f"Cannot find CACTI binary: {self.cacti_bin}")
        if not self.base_cfg.exists():
            raise FileNotFoundError(f"Cannot find base cfg: {self.base_cfg}")

        self.workdir.mkdir(exist_ok=True)

        read_cfg = self.workdir / "ddr3_read.cfg"
        write_cfg = self.workdir / "ddr3_write.cfg"
        read_log = self.workdir / "ddr3_read.log"
        write_log = self.workdir / "ddr3_write.log"

        generate_io_cfg(
            self.base_cfg,
            read_cfg,
            io_state="R",
            bus_freq_mhz=bus_freq_mhz,
            num_dq=num_dq,
            mem_density=mem_density,
            addr_timing=addr_timing,
        )
        generate_io_cfg(
            self.base_cfg,
            write_cfg,
            io_state="W",
            bus_freq_mhz=bus_freq_mhz,
            num_dq=num_dq,
            mem_density=mem_density,
            addr_timing=addr_timing,
        )

        read_out = run_cacti(self.cacti_bin, read_cfg, self.cacti_dir)
        write_out = run_cacti(self.cacti_bin, write_cfg, self.cacti_dir)

        read_log.write_text(read_out)
        write_log.write_text(write_out)

        read_cfg_meta = parse_cfg_meta(read_cfg.read_text())
        write_cfg_meta = parse_cfg_meta(write_cfg.read_text())

        read_io = parse_cacti_dram_io_metrics(read_out)
        write_io = parse_cacti_dram_io_metrics(write_out)

        dram_type = (
            read_cfg_meta["dram_type"]
            or write_cfg_meta["dram_type"]
            or "DRAM"
        )
        io_type = (
            read_io["printed_io_type"]
            or read_cfg_meta["io_type"]
            or "DDR3"
        )

        # 频率 / dq 优先从 cfg 取
        final_bus_freq_mhz = float(
            read_cfg_meta["bus_freq_mhz"]
            or write_cfg_meta["bus_freq_mhz"]
            or bus_freq_mhz
            or 800.0
        )
        final_num_dq = int(
            read_cfg_meta["num_dq"]
            or write_cfg_meta["num_dq"]
            or num_dq
            or 64
        )

        # 读写日志里这几项通常一致；这里取 read 优先
        io_dynamic_power_mw = (
            read_io["io_dynamic_power_mw"]
            if read_io["io_dynamic_power_mw"] is not None
            else (
                write_io["io_dynamic_power_mw"]
                if write_io["io_dynamic_power_mw"] is not None
                else fallback_io_dynamic_power_mw
            )
        )
        phy_power_mw = (
            read_io["phy_power_mw"]
            if read_io["phy_power_mw"] is not None
            else (
                write_io["phy_power_mw"]
                if write_io["phy_power_mw"] is not None
                else fallback_phy_power_mw
            )
        )
        io_term_power_mw = (
            read_io["io_termination_power_mw"]
            if read_io["io_termination_power_mw"] is not None
            else (
                write_io["io_termination_power_mw"]
                if write_io["io_termination_power_mw"] is not None
                else fallback_io_termination_power_mw
            )
        )
        phy_wakeup_us = (
            read_io["phy_wakeup_time_us"]
            if read_io["phy_wakeup_time_us"] is not None
            else (
                write_io["phy_wakeup_time_us"]
                if write_io["phy_wakeup_time_us"] is not None
                else fallback_phy_wakeup_time_us
            )
        )

        calib = DramIOCalib(
            dram_type=str(dram_type),
            io_type=str(io_type),
            iostate="RW",
            bus_freq_mhz=final_bus_freq_mhz,
            num_dq=final_num_dq,
            burst_bytes=burst_bytes,
            io_dynamic_power_mw=float(io_dynamic_power_mw),
            phy_power_mw=float(phy_power_mw),
            io_termination_power_mw=float(io_term_power_mw),
            phy_wakeup_time_us=float(phy_wakeup_us),
            read_fixed_latency_ns=float(read_fixed_latency_ns),
            write_fixed_latency_ns=float(write_fixed_latency_ns),
            row_miss_penalty_ns=float(row_miss_penalty_ns),
        )

        calib_json = self.workdir / "dram_io_calib.json"
        calib_json.write_text(json.dumps(asdict(calib), indent=2))

        if verbose:
            print("=== CACTI DRAM IO calibration done ===")
            print(f"Read cfg:   {read_cfg}")
            print(f"Write cfg:  {write_cfg}")
            print(f"Read log:   {read_log}")
            print(f"Write log:  {write_log}")
            print(f"Calib json: {calib_json}")
            print()
            print(json.dumps(asdict(calib), indent=2))
            print()
            print("Read IO metrics parsed:")
            print(json.dumps(read_io, indent=2))
            print("Write IO metrics parsed:")
            print(json.dumps(write_io, indent=2))

        return calib


# ============================================================
# 运行时 DRAM external memory 模型
# ============================================================

class DramModel:
    def __init__(self, calib: DramIOCalib):
        self.c = calib

    def peak_bandwidth_Bps(self) -> float:
        return self.c.peak_bandwidth_Bps()

    def access(
        self,
        op: str,
        nbytes: int,
        row_hit_rate: float = 1.0,
        queue_delay_ns: float = 0.0,
        from_sleep: bool = False,
    ) -> SingleAccessResult:
        assert op in ("read", "write")
        assert nbytes >= 0

        if nbytes == 0:
            return SingleAccessResult(
                op=op,
                nbytes=0,
                bursts=0,
                transfer_time_ns=0.0,
                fixed_latency_ns=0.0,
                wakeup_ns=0.0,
                queue_delay_ns=0.0,
                row_penalty_ns=0.0,
                latency_ns=0.0,
                transfer_energy_nj=0.0,
                background_energy_nj=0.0,
                energy_nj=0.0,
            )

        bursts = math.ceil(nbytes / self.c.burst_bytes)
        bw = self.peak_bandwidth_Bps()
        transfer_time_ns = nbytes / bw * 1e9

        fixed_latency_ns = (
            self.c.read_fixed_latency_ns if op == "read" else self.c.write_fixed_latency_ns
        )
        wakeup_ns = self.c.phy_wakeup_time_us * 1000.0 if from_sleep else 0.0

        row_miss_rate = max(0.0, 1.0 - row_hit_rate)
        row_penalty_ns = bursts * row_miss_rate * self.c.row_miss_penalty_ns

        latency_ns = wakeup_ns + fixed_latency_ns + transfer_time_ns + queue_delay_ns + row_penalty_ns

        # 传输动态能耗
        transfer_energy_nj = mw_ns_to_nj(self.c.io_dynamic_power_mw, transfer_time_ns)

        # PHY + termination/bias 背景能耗，按整个请求服务时长算
        background_energy_nj = mw_ns_to_nj(
            self.c.phy_power_mw + self.c.io_termination_power_mw,
            latency_ns
        )

        energy_nj = transfer_energy_nj + background_energy_nj

        return SingleAccessResult(
            op=op,
            nbytes=nbytes,
            bursts=bursts,
            transfer_time_ns=transfer_time_ns,
            fixed_latency_ns=fixed_latency_ns,
            wakeup_ns=wakeup_ns,
            queue_delay_ns=queue_delay_ns,
            row_penalty_ns=row_penalty_ns,
            latency_ns=latency_ns,
            transfer_energy_nj=transfer_energy_nj,
            background_energy_nj=background_energy_nj,
            energy_nj=energy_nj,
        )

    def dram_access(
        self,
        read_bytes: int = 0,
        write_bytes: int = 0,
        row_hit_rate: float = 1.0,
        read_queue_delay_ns: float = 0.0,
        write_queue_delay_ns: float = 0.0,
        read_from_sleep: bool = False,
        write_from_sleep: bool = False,
        overlap_rw: bool = False,
    ) -> CombinedAccessResult:

        read_res = None
        write_res = None

        if read_bytes > 0:
            read_res = self.access(
                "read",
                read_bytes,
                row_hit_rate=row_hit_rate,
                queue_delay_ns=read_queue_delay_ns,
                from_sleep=read_from_sleep,
            )

        if write_bytes > 0:
            write_res = self.access(
                "write",
                write_bytes,
                row_hit_rate=row_hit_rate,
                queue_delay_ns=write_queue_delay_ns,
                from_sleep=write_from_sleep,
            )

        total_energy = 0.0
        if read_res is not None:
            total_energy += read_res.energy_nj
        if write_res is not None:
            total_energy += write_res.energy_nj

        if read_res is None and write_res is None:
            total_latency = 0.0
        elif read_res is None:
            total_latency = write_res.latency_ns
        elif write_res is None:
            total_latency = read_res.latency_ns
        else:
            total_latency = max(read_res.latency_ns, write_res.latency_ns) if overlap_rw else (
                read_res.latency_ns + write_res.latency_ns
            )

        return CombinedAccessResult(
            read=read_res,
            write=write_res,
            latency_ns=total_latency,
            energy_nj=total_energy,
        )


# ============================================================
# 对外构建接口
# ============================================================

def build_dram_model_from_cacti_io(
    cacti_dir: str,
    cacti_bin: str = "cacti",
    base_cfg: str = "ddr3.cfg",
    burst_bytes: int = 64,
    bus_freq_mhz: Optional[float] = None,
    num_dq: Optional[int] = None,
    mem_density: Optional[str] = None,
    addr_timing: Optional[float] = None,
    read_fixed_latency_ns: float = 15.0,
    write_fixed_latency_ns: float = 15.0,
    row_miss_penalty_ns: float = 0.0,
    fallback_io_dynamic_power_mw: float = 1750.0,
    fallback_phy_power_mw: float = 230.0,
    fallback_io_termination_power_mw: float = 3800.0,
    fallback_phy_wakeup_time_us: float = 27.5,
    verbose: bool = True,
) -> DramModel:
    calib = CactiDramIOCalibrator(
        cacti_dir=Path(cacti_dir),
        cacti_bin=cacti_bin,
        base_cfg=base_cfg,
    ).calibrate(
        burst_bytes=burst_bytes,
        bus_freq_mhz=bus_freq_mhz,
        num_dq=num_dq,
        mem_density=mem_density,
        addr_timing=addr_timing,
        read_fixed_latency_ns=read_fixed_latency_ns,  # read_fixed_latency_ns 和 write_fixed_latency_ns 表示的是与传输字节数无关的固定事务开销。 
        write_fixed_latency_ns=write_fixed_latency_ns, # 
        row_miss_penalty_ns=row_miss_penalty_ns,
        fallback_io_dynamic_power_mw=fallback_io_dynamic_power_mw,
        fallback_phy_power_mw=fallback_phy_power_mw,
        fallback_io_termination_power_mw=fallback_io_termination_power_mw,
        fallback_phy_wakeup_time_us=fallback_phy_wakeup_time_us,
        verbose=verbose,
    )
    return DramModel(calib)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="CACTI DDR IO based external DRAM latency/energy model"
    )

    parser.add_argument("--cacti-dir", type=str, default=".")
    parser.add_argument("--cacti-bin", type=str, default="cacti")
    parser.add_argument("--base-cfg", type=str, default="ddr3.cfg")

    parser.add_argument("--burst-bytes", type=int, default=256)
    parser.add_argument("--bus-freq-mhz", type=float, default=None)
    parser.add_argument("--num-dq", type=int, default=None)
    parser.add_argument("--mem-density", type=str, default=None)
    parser.add_argument("--addr-timing", type=float, default=None)

    parser.add_argument("--read-fixed-latency-ns", type=float, default=.0)
    parser.add_argument("--write-fixed-latency-ns", type=float, default=.0)
    parser.add_argument("--row-miss-penalty-ns", type=float, default=0.0)

    parser.add_argument("--read-bytes", type=int, default=0)
    parser.add_argument("--write-bytes", type=int, default=0)
    parser.add_argument("--row-hit-rate", type=float, default=1.0)
    parser.add_argument("--read-queue-delay-ns", type=float, default=0.0)
    parser.add_argument("--write-queue-delay-ns", type=float, default=0.0)
    parser.add_argument("--read-from-sleep", action="store_true")
    parser.add_argument("--write-from-sleep", action="store_true")
    parser.add_argument("--overlap-rw", action="store_true")

    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    dram = build_dram_model_from_cacti_io(
        cacti_dir=args.cacti_dir,
        cacti_bin=args.cacti_bin,
        base_cfg=args.base_cfg,
        burst_bytes=args.burst_bytes,
        bus_freq_mhz=args.bus_freq_mhz,
        num_dq=args.num_dq,
        mem_density=args.mem_density,
        addr_timing=args.addr_timing,
        read_fixed_latency_ns=args.read_fixed_latency_ns,
        write_fixed_latency_ns=args.write_fixed_latency_ns,
        row_miss_penalty_ns=args.row_miss_penalty_ns,
        verbose=(not args.quiet),
    )

    result = dram.dram_access(
        read_bytes=args.read_bytes,
        write_bytes=args.write_bytes,
        row_hit_rate=args.row_hit_rate,
        read_queue_delay_ns=args.read_queue_delay_ns,
        write_queue_delay_ns=args.write_queue_delay_ns,
        read_from_sleep=args.read_from_sleep,
        write_from_sleep=args.write_from_sleep,
        overlap_rw=args.overlap_rw,
    )

    print("\n=== Runtime access result ===")
    print(json.dumps(asdict(result), indent=2, default=lambda o: asdict(o)))


if __name__ == "__main__":
    main()