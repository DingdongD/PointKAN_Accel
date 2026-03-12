#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
profile.py

统一的 CPU / GPU 推理性能分析器（命令级）
--------------------------------------------------
功能：
1. 执行用户给定的推理命令
2. 统计 wall time / latency
3. GPU 侧优先使用 NVML 采样功耗、利用率、显存占用
4. 若 NVML 不可用，则自动回退到 nvidia-smi dmon
5. CPU 侧使用 psutil 采样 CPU 利用率、RSS 内存
6. 可选启用 perf stat，并自动解析 cycles / instructions / cache misses / RAPL energy
7. 可选从 /sys/class/powercap 读取 RAPL，作为 CPU energy 的回退源
8. 对 GPU 功率-时间曲线做梯形积分，得到 energy (J)
9. 输出每次运行的 JSON / CSV 汇总结果

说明：
- 这是“命令级” profiler，适合：
    python profile.py --cmd "python infer.py --batch 1" --mode gpu
- 如果你后面要直接适配你自己的 network + dataset，建议另外再加一个 profile_torch_infer.py
"""

import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

# psutil：用于采样 CPU 利用率、进程内存
try:
    import psutil
except Exception:
    psutil = None

# pynvml：用于读取 NVIDIA GPU 功耗 / 利用率 / 显存
try:
    import pynvml
except Exception:
    pynvml = None


def now_ns() -> int:
    """返回高精度单调时钟，单位 ns"""
    return time.perf_counter_ns()


def command_exists(cmd: str) -> bool:
    """检查系统命令是否存在"""
    return shutil.which(cmd) is not None


def shell_quote_single(s: str) -> str:
    """
    将字符串安全包成单引号 shell 字符串
    例如 abc'd -> 'abc'"'"'d'
    """
    return "'" + s.replace("'", "'\"'\"'") + "'"


def trapezoid_energy_j(samples: List[Dict[str, Any]], power_key: str = "power_w") -> float:
    """
    使用梯形积分法计算能量：
        E = ∫ P(t) dt
    离散形式：
        E += 0.5 * (P_i + P_{i+1}) * Δt

    参数：
        samples: 采样点列表，每个元素至少包含：
                 - t_rel_s: 相对时间（秒）
                 - power_w: 功率（瓦）
        power_key: 功率字段名，默认 power_w

    返回：
        energy，单位焦耳 J
    """
    if len(samples) < 2:
        return 0.0

    energy_j = 0.0
    for a, b in zip(samples[:-1], samples[1:]):
        ta = a.get("t_rel_s")
        tb = b.get("t_rel_s")
        pa = a.get(power_key)
        pb = b.get(power_key)
        if ta is None or tb is None or pa is None or pb is None:
            continue
        dt = max(0.0, float(tb) - float(ta))
        energy_j += 0.5 * (float(pa) + float(pb)) * dt

    return energy_j


def write_csv(path: Path, rows: List[Dict[str, Any]]):
    """
    将字典列表写入 CSV
    自动收集所有列名
    """
    if not rows:
        return

    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_perf_value(value_str: str):
    """
    解析 perf 输出的计数字段
    例如：
      "1,234,567" -> 1234567
      "12.34" -> 12.34
      "<not counted>" -> None
    """
    value_str = value_str.strip()
    if value_str in ("", "<not counted>", "<not supported>", "<not enabled>", "not counted"):
        return None

    # 去掉千位分隔符和空格
    value_str = value_str.replace(" ", "").replace(",", "")

    try:
        if "." in value_str:
            return float(value_str)
        return int(value_str)
    except Exception:
        return None


def parse_perf_stat_csv(perf_path: Path) -> Dict[str, Any]:
    """
    解析 perf stat -x, 输出

    典型格式（不同版本略有差异）：
      value,unit,event,run_time,percent,metric,metric_unit

    返回：
      {
        "cycles": ...,
        "instructions": ...,
        "cache_misses": ...,
        "power_energy_pkg_": ...,
      }
    """
    result: Dict[str, Any] = {}
    if not perf_path.exists():
        return result

    with perf_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 3:
                continue

            value_str = parts[0]
            event_name = parts[2]

            value = parse_perf_value(value_str)
            if value is None:
                continue

            key = event_name.replace("/", "_").replace("-", "_").replace(":", "_")
            result[key] = value

    return result


def discover_rapl_energy_nodes() -> Dict[str, Path]:
    """
    扫描系统中的 RAPL energy_uj 节点
    返回:
        {名字: energy_uj路径}
    """
    base = Path("/sys/class/powercap")
    nodes: Dict[str, Path] = {}
    if not base.exists():
        return nodes

    for p in base.rglob("energy_uj"):
        name_file = p.parent / "name"
        if name_file.exists():
            try:
                name = name_file.read_text().strip()
            except Exception:
                name = p.parent.name
        else:
            name = p.parent.name
        # 若重名，给一个去重 key
        key = name
        idx = 1
        while key in nodes:
            idx += 1
            key = f"{name}_{idx}"
        nodes[key] = p

    return nodes


def read_rapl_energy_uj(nodes: Dict[str, Path]) -> Dict[str, int]:
    """
    读取所有 RAPL 节点当前 energy_uj 值
    """
    vals: Dict[str, int] = {}
    for name, path in nodes.items():
        try:
            vals[name] = int(path.read_text().strip())
        except Exception:
            pass
    return vals


def diff_rapl_j(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, float]:
    """
    根据运行前后 energy_uj 读数，计算能耗增量（J）
    注意：
    - 这里对计数器回卷做了简单保护，若 after < before 则忽略
    """
    out: Dict[str, float] = {}
    for k, v0 in before.items():
        if k not in after:
            continue
        v1 = after[k]
        dv_uj = v1 - v0
        if dv_uj >= 0:
            out[k] = dv_uj / 1e6
    return out


def start_dmon_logger(out_path: Path, gpu_index: int, interval_s: int = 1):
    """
    启动 nvidia-smi dmon 采样子进程
    输出写入 out_path
    """
    if not command_exists("nvidia-smi"):
        return None, None

    f = out_path.open("w")
    proc = subprocess.Popen(
        [
            "nvidia-smi",
            "dmon",
            "-i", str(gpu_index),
            "-s", "pucvmet",
            "-d", str(interval_s),
        ],
        stdout=f,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, f


def stop_dmon_logger(proc, file_obj):
    """停止 dmon 子进程"""
    if proc is not None:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    if file_obj is not None:
        try:
            file_obj.close()
        except Exception:
            pass


def parse_dmon_file(dmon_path: Path, interval_s: float = 1.0) -> List[Dict[str, Any]]:
    """
    解析 nvidia-smi dmon 输出文本，转换成统一采样格式

    由于 dmon 各版本列名可能略有差异，这里做宽松解析。
    常见数据行大致形式：
      gpu pwr gtemp mtemp sm mem enc dec mclk pclk ...
    或其他近似列顺序

    返回字段尽量统一为：
    - t_rel_s
    - power_w
    - gpu_util_percent
    - mem_util_percent
    - temp_c
    """
    samples: List[Dict[str, Any]] = []
    if not dmon_path.exists():
        return samples

    data_idx = 0
    with dmon_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = re.split(r"\s+", line)
            if len(parts) < 3:
                continue

            numeric = []
            for p in parts:
                try:
                    numeric.append(float(p))
                except ValueError:
                    pass

            # 至少要有 gpu_id + power + 其他一个量
            if len(numeric) < 3:
                continue

            row: Dict[str, Any] = {
                "t_rel_s": data_idx * interval_s,
            }

            # 常见经验映射：
            # [0]=gpu_id, [1]=power, [2]=gtemp, [3]=mtemp, [4]=sm, [5]=mem, ...
            if len(numeric) >= 2:
                row["power_w"] = numeric[1]
            if len(numeric) >= 3:
                row["temp_c"] = numeric[2]
            if len(numeric) >= 5:
                row["gpu_util_percent"] = numeric[4]
            if len(numeric) >= 6:
                row["mem_util_percent"] = numeric[5]

            samples.append(row)
            data_idx += 1

    return samples


@dataclass
class RunSummary:
    """单次运行的摘要结果"""
    mode: str
    command: str
    returncode: int
    wall_time_s: float
    latency_s: float

    gpu_backend: Optional[str]
    gpu_energy_j: Optional[float]
    gpu_avg_power_w: Optional[float]
    gpu_peak_power_w: Optional[float]

    cpu_avg_percent: Optional[float]
    rss_peak_mb: Optional[float]

    cpu_energy_pkg_j: Optional[float]
    cpu_energy_ram_j: Optional[float]
    cpu_energy_total_j: Optional[float]
    rapl_source: Optional[str]

    perf_stats: Optional[Dict[str, Any]]

    stdout_path: str
    stderr_path: str
    gpu_csv_path: Optional[str]
    cpu_csv_path: Optional[str]
    dmon_path: Optional[str]
    perf_path: Optional[str]
    inference_only_s: Optional[float] = None  # 仅 model 前向时间总和（子进程写 inference_only.json 时填入）
    inference_only_per_forward_s: Optional[float] = None  # 单次前向时间 = inference_only_s / num_batches
    # 仅推理阶段 CPU 能耗（子进程内 pyRAPL Measurement 包住推理循环写入 inference_only.json）
    cpu_energy_inference_pkg_j: Optional[float] = None
    cpu_energy_inference_ram_j: Optional[float] = None
    cpu_energy_inference_total_j: Optional[float] = None
    gpu_idle_power_w: Optional[float] = None  # 跑前采样的 GPU 空闲功率，用于能耗扣减


class GPUSampler(threading.Thread):
    """
    GPU 采样线程：
    周期性读取 NVML 指标
    - 功耗 power_w
    - GPU 利用率 gpu_util_percent
    - 显存控制器利用率 mem_util_percent
    - 显存占用 mem_used_mb
    """

    def __init__(self, interval_s: float, gpu_index: int):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.gpu_index = gpu_index
        self.stop_evt = threading.Event()
        self.samples: List[Dict[str, Any]] = []
        self.handle = None
        self.backend = None

    def setup(self) -> bool:
        """
        初始化 NVML，并拿到指定 GPU 设备句柄
        返回：
            True  -> 成功
            False -> 失败
        """
        if pynvml is None:
            return False

        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self.backend = "pynvml"
            return True
        except Exception:
            return False

    def run(self):
        """线程主循环：定期采样 GPU 指标"""
        t0 = time.perf_counter()

        while not self.stop_evt.is_set():
            ts = time.time()
            rel = time.perf_counter() - t0

            row = {
                "ts": ts,
                "t_rel_s": rel,
            }

            try:
                p_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                row["power_w"] = p_mw / 1000.0
            except Exception:
                row["power_w"] = None

            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                row["gpu_util_percent"] = getattr(util, "gpu", None)
                row["mem_util_percent"] = getattr(util, "memory", None)
            except Exception:
                row["gpu_util_percent"] = None
                row["mem_util_percent"] = None

            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                row["mem_used_mb"] = mem.used / (1024 * 1024)
            except Exception:
                row["mem_used_mb"] = None

            self.samples.append(row)
            time.sleep(self.interval_s)

    def stop(self):
        """停止线程并清理 NVML"""
        self.stop_evt.set()
        self.join(timeout=2)
        if self.backend == "pynvml":
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


class CPUSampler(threading.Thread):
    """
    CPU 采样线程：
    周期性采样目标进程的
    - cpu_percent
    - rss_mb
    """

    def __init__(self, interval_s: float, pid: int):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.pid = pid
        self.stop_evt = threading.Event()
        self.samples: List[Dict[str, Any]] = []

    def run(self):
        if psutil is None:
            return

        t0 = time.perf_counter()

        try:
            proc = psutil.Process(self.pid)
            proc.cpu_percent(interval=None)
        except Exception:
            return

        while not self.stop_evt.is_set():
            ts = time.time()
            rel = time.perf_counter() - t0

            row = {
                "ts": ts,
                "t_rel_s": rel,
            }

            try:
                row["cpu_percent"] = proc.cpu_percent(interval=None)
            except Exception:
                row["cpu_percent"] = None

            try:
                mi = proc.memory_info()
                row["rss_mb"] = mi.rss / (1024 * 1024)
            except Exception:
                row["rss_mb"] = None

            self.samples.append(row)
            time.sleep(self.interval_s)

    def stop(self):
        """停止线程"""
        self.stop_evt.set()
        self.join(timeout=2)


def aggregate(summaries: List[RunSummary]) -> Dict[str, Any]:
    """对多次 repeat 的结果做聚合统计"""

    def avg(vals):
        vals = [v for v in vals if v is not None]
        return (sum(vals) / len(vals)) if vals else None

    # 聚合 perf event
    perf_agg: Dict[str, List[float]] = {}
    for s in summaries:
        if not s.perf_stats:
            continue
        for k, v in s.perf_stats.items():
            if v is None:
                continue
            perf_agg.setdefault(k, []).append(float(v))

    perf_mean = {k: sum(vs) / len(vs) for k, vs in perf_agg.items() if vs}

    return {
        "num_runs": len(summaries),
        "latency_s_mean": avg([s.latency_s for s in summaries]),
        "latency_s_min": min([s.latency_s for s in summaries]) if summaries else None,
        "latency_s_max": max([s.latency_s for s in summaries]) if summaries else None,
        "inference_only_s_mean": avg([s.inference_only_s for s in summaries if getattr(s, "inference_only_s", None) is not None]),
        "inference_only_per_forward_s_mean": avg([s.inference_only_per_forward_s for s in summaries if getattr(s, "inference_only_per_forward_s", None) is not None]),

        "gpu_energy_j_mean": avg([s.gpu_energy_j for s in summaries]),
        "gpu_idle_power_w_mean": avg([s.gpu_idle_power_w for s in summaries if getattr(s, "gpu_idle_power_w", None) is not None]),
        "gpu_avg_power_w_mean": avg([s.gpu_avg_power_w for s in summaries]),
        "gpu_peak_power_w_max": max(
            [s.gpu_peak_power_w for s in summaries if s.gpu_peak_power_w is not None],
            default=None,
        ),

        "cpu_avg_percent_mean": avg([s.cpu_avg_percent for s in summaries]),
        "rss_peak_mb_max": max(
            [s.rss_peak_mb for s in summaries if s.rss_peak_mb is not None],
            default=None,
        ),

        "cpu_energy_pkg_j_mean": avg([s.cpu_energy_pkg_j for s in summaries]),
        "cpu_energy_ram_j_mean": avg([s.cpu_energy_ram_j for s in summaries]),
        "cpu_energy_total_j_mean": avg([s.cpu_energy_total_j for s in summaries]),
        "cpu_energy_inference_pkg_j_mean": avg([s.cpu_energy_inference_pkg_j for s in summaries if getattr(s, "cpu_energy_inference_pkg_j", None) is not None]),
        "cpu_energy_inference_ram_j_mean": avg([s.cpu_energy_inference_ram_j for s in summaries if getattr(s, "cpu_energy_inference_ram_j", None) is not None]),
        "cpu_energy_inference_total_j_mean": avg([s.cpu_energy_inference_total_j for s in summaries if getattr(s, "cpu_energy_inference_total_j", None) is not None]),

        "perf_stats_mean": perf_mean if perf_mean else None,
    }


def sample_gpu_idle_power(gpu_index: int, duration_s: float = 2.0, interval_s: float = 0.1) -> Optional[float]:
    """
    在启动子进程前采样 GPU 空闲功率（瓦），用于后续从总能耗中扣减空闲分量。
    采样 duration_s 秒，每 interval_s 秒取一次 NVML 功率，返回平均值。
    """
    if pynvml is None:
        return None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        t0 = time.perf_counter()
        powers: List[float] = []
        while time.perf_counter() - t0 < duration_s:
            p_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            powers.append(p_mw / 1000.0)
            time.sleep(interval_s)
        pynvml.nvmlShutdown()
        return sum(powers) / len(powers) if powers else None
    except Exception:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        return None


def run_command(
    cmd: str,
    out_dir: Path,
    tag: str,
    sample_gpu: bool,
    sample_cpu: bool,
    gpu_interval_ms: int,
    cpu_interval_ms: int,
    gpu_index: int,
    enable_perf: bool = False,
) -> RunSummary:
    """
    执行一次目标命令，并在运行期间进行 CPU/GPU 采样。
    GPU 能耗会扣减「空闲功率 × 运行时间」得到增量能耗。
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    # 先测 GPU 空闲功率（用于能耗扣减）
    gpu_idle_power_w: Optional[float] = None
    if sample_gpu and pynvml is not None:
        gpu_idle_power_w = sample_gpu_idle_power(
            gpu_index, duration_s=2.0, interval_s=max(0.05, gpu_interval_ms / 1000.0)
        )

    stdout_path = out_dir / f"{tag}.stdout.txt"
    stderr_path = out_dir / f"{tag}.stderr.txt"
    gpu_csv_path = out_dir / f"{tag}_gpu_samples.csv"
    cpu_csv_path = out_dir / f"{tag}_cpu_samples.csv"
    perf_path = out_dir / f"{tag}_perf.csv"
    dmon_path = out_dir / f"{tag}_dmon.txt"

    gpu_sampler = None
    cpu_sampler = None
    dmon_proc = None
    dmon_file = None

    gpu_backend = None
    gpu_energy = None
    gpu_avg_power = None
    gpu_peak_power = None

    cpu_avg = None
    rss_peak = None

    perf_stats: Dict[str, Any] = {}
    cpu_energy_pkg_j = None
    cpu_energy_ram_j = None
    cpu_energy_total_j = None
    rapl_source = None

    rapl_nodes = discover_rapl_energy_nodes()
    rapl_before = read_rapl_energy_uj(rapl_nodes) if rapl_nodes else {}

    # 如果启用 perf，则将用户命令包在 perf stat 外层
    if enable_perf and command_exists("perf"):
        perf_events = ",".join([
            "cycles",
            "instructions",
            "branches",
            "branch-misses",
            "cache-references",
            "cache-misses",
            "task-clock",
            "power/energy-pkg/",
            "power/energy-ram/",
        ])
        full_cmd = (
            f"perf stat -x, -ddd "
            f"-e {shell_quote_single(perf_events)} "
            f"-o {shell_quote_single(str(perf_path))} "
            f"bash -lc {shell_quote_single(cmd)}"
        )
    else:
        full_cmd = cmd

    inference_only_s = None
    cpu_energy_inference_pkg_j = None
    cpu_energy_inference_ram_j = None
    cpu_energy_inference_total_j = None
    child_env = dict(os.environ, PROFILE_RUN_DIR=str(out_dir), PROFILE_RUN_TAG=tag)

    with stdout_path.open("w") as fo, stderr_path.open("w") as fe:
        start = now_ns()

        proc = subprocess.Popen(
            full_cmd,
            shell=True,
            stdout=fo,
            stderr=fe,
            preexec_fn=os.setsid,
            env=child_env,
        )

        # GPU 采样：优先 NVML，失败则回退到 dmon
        if sample_gpu:
            gpu_sampler = GPUSampler(gpu_interval_ms / 1000.0, gpu_index)
            if gpu_sampler.setup():
                gpu_sampler.start()
                gpu_backend = "pynvml"
            else:
                gpu_sampler = None
                dmon_interval_s = max(1, int(round(gpu_interval_ms / 1000.0)))
                dmon_proc, dmon_file = start_dmon_logger(
                    dmon_path, gpu_index, interval_s=dmon_interval_s
                )
                if dmon_proc is not None:
                    gpu_backend = "dmon"

        # CPU 采样
        if sample_cpu:
            cpu_sampler = CPUSampler(cpu_interval_ms / 1000.0, proc.pid)
            cpu_sampler.start()

        ret = proc.wait()
        end = now_ns()

        inference_only_per_forward_s = None
        infer_path = out_dir / f"{tag}_inference_only.json"
        if infer_path.exists():
            try:
                with infer_path.open("r", encoding="utf-8") as f:
                    infer_data = json.load(f)
                inference_only_s = infer_data.get("inference_only_s")
                num_b = infer_data.get("num_batches") or 0
                if num_b and inference_only_s is not None and inference_only_s >= 0:
                    inference_only_per_forward_s = inference_only_s / num_b
                # 推理阶段 CPU 能耗（pyRAPL 在进程内仅包住推理循环）
                v = infer_data.get("cpu_energy_inference_pkg_j")
                if v is not None:
                    cpu_energy_inference_pkg_j = float(v)
                v = infer_data.get("cpu_energy_inference_ram_j")
                if v is not None:
                    cpu_energy_inference_ram_j = float(v)
                v = infer_data.get("cpu_energy_inference_total_j")
                if v is not None:
                    cpu_energy_inference_total_j = float(v)
            except Exception:
                pass

        if gpu_sampler:
            gpu_sampler.stop()
        if dmon_proc:
            stop_dmon_logger(dmon_proc, dmon_file)
        if cpu_sampler:
            cpu_sampler.stop()

    rapl_after = read_rapl_energy_uj(rapl_nodes) if rapl_nodes else {}
    cpu_energy_rapl = diff_rapl_j(rapl_before, rapl_after) if rapl_before and rapl_after else {}

    wall_s = (end - start) / 1e9

    # 汇总 GPU：优先 NVML，否则 dmon
    if gpu_sampler and gpu_sampler.samples:
        write_csv(gpu_csv_path, gpu_sampler.samples)

        powers = [
            float(r["power_w"])
            for r in gpu_sampler.samples
            if r.get("power_w") is not None
        ]
        gpu_energy = trapezoid_energy_j(gpu_sampler.samples)

        if powers:
            gpu_avg_power = sum(powers) / len(powers)
            gpu_peak_power = max(powers)

    elif dmon_path.exists():
        dmon_interval_s = max(1, int(round(gpu_interval_ms / 1000.0)))
        dmon_samples = parse_dmon_file(dmon_path, interval_s=dmon_interval_s)
        if dmon_samples:
            write_csv(gpu_csv_path, dmon_samples)

            powers = [
                float(r["power_w"])
                for r in dmon_samples
                if r.get("power_w") is not None
            ]
            gpu_energy = trapezoid_energy_j(dmon_samples)

            if powers:
                gpu_avg_power = sum(powers) / len(powers)
                gpu_peak_power = max(powers)
        else:
            gpu_csv_path = None
    else:
        gpu_csv_path = None

    # GPU 能耗扣减空闲分量：用「推理时长」扣减避免 wall 内大量非推理时间导致扣成 0
    subtract_s = inference_only_s if inference_only_s is not None and inference_only_s >= 0 else wall_s
    if gpu_energy is not None and gpu_idle_power_w is not None and subtract_s is not None:
        gpu_energy = max(0.0, float(gpu_energy) - gpu_idle_power_w * subtract_s)

    # 汇总 CPU
    if cpu_sampler and cpu_sampler.samples:
        write_csv(cpu_csv_path, cpu_sampler.samples)

        cpus = [
            float(r["cpu_percent"])
            for r in cpu_sampler.samples
            if r.get("cpu_percent") is not None
        ]
        rss = [
            float(r["rss_mb"])
            for r in cpu_sampler.samples
            if r.get("rss_mb") is not None
        ]

        if cpus:
            cpu_avg = sum(cpus) / len(cpus)
        if rss:
            rss_peak = max(rss)
    else:
        cpu_csv_path = None

    # 解析 perf
    if perf_path.exists():
        perf_stats = parse_perf_stat_csv(perf_path)

    # CPU energy：优先 perf 里的 RAPL
    if perf_stats:
        if "power_energy_pkg_" in perf_stats:
            try:
                cpu_energy_pkg_j = float(perf_stats["power_energy_pkg_"])
                rapl_source = "perf"
            except Exception:
                pass
        if "power_energy_ram_" in perf_stats:
            try:
                cpu_energy_ram_j = float(perf_stats["power_energy_ram_"])
                rapl_source = "perf"
            except Exception:
                pass

    # 若 perf 没拿到，则尝试 sysfs RAPL
    if cpu_energy_pkg_j is None and cpu_energy_ram_j is None and cpu_energy_rapl:
        rapl_source = "sysfs"
        for k, v in cpu_energy_rapl.items():
            lk = k.lower()
            if "package" in lk or "pkg" in lk:
                cpu_energy_pkg_j = v
            elif "dram" in lk or "ram" in lk:
                cpu_energy_ram_j = v

        # 某些平台没有 package/dram 清晰命名，则退化为全部求和
        if cpu_energy_pkg_j is None and cpu_energy_ram_j is None:
            total = sum(cpu_energy_rapl.values()) if cpu_energy_rapl else None
            if total is not None:
                cpu_energy_total_j = total

    vals = [x for x in [cpu_energy_pkg_j, cpu_energy_ram_j] if x is not None]
    if vals:
        cpu_energy_total_j = sum(vals)

    return RunSummary(
        mode="gpu" if sample_gpu else "cpu",
        command=cmd,
        returncode=ret,
        wall_time_s=wall_s,
        latency_s=wall_s,
        inference_only_s=inference_only_s,
        inference_only_per_forward_s=inference_only_per_forward_s,

        gpu_backend=gpu_backend,
        gpu_energy_j=gpu_energy,
        gpu_avg_power_w=gpu_avg_power,
        gpu_peak_power_w=gpu_peak_power,
        gpu_idle_power_w=gpu_idle_power_w,

        cpu_avg_percent=cpu_avg,
        rss_peak_mb=rss_peak,

        cpu_energy_pkg_j=cpu_energy_pkg_j,
        cpu_energy_ram_j=cpu_energy_ram_j,
        cpu_energy_total_j=cpu_energy_total_j,
        rapl_source=rapl_source,
        cpu_energy_inference_pkg_j=cpu_energy_inference_pkg_j,
        cpu_energy_inference_ram_j=cpu_energy_inference_ram_j,
        cpu_energy_inference_total_j=cpu_energy_inference_total_j,

        perf_stats=perf_stats if perf_stats else None,

        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        gpu_csv_path=str(gpu_csv_path) if gpu_csv_path and Path(gpu_csv_path).exists() else None,
        cpu_csv_path=str(cpu_csv_path) if cpu_csv_path and Path(cpu_csv_path).exists() else None,
        dmon_path=str(dmon_path) if dmon_path.exists() else None,
        perf_path=str(perf_path) if perf_path.exists() else None,
    )


def main():
    """命令行入口"""
    ap = argparse.ArgumentParser(description="统一 CPU/GPU 推理分析器")
    ap.add_argument("--cmd", required=True, help="要执行的 shell 命令")
    ap.add_argument("--out-dir", default="outputs", help="输出目录")
    ap.add_argument("--mode", choices=["cpu", "gpu"], default="gpu", help="分析模式")
    ap.add_argument("--warmup", type=int, default=1, help="warmup 次数")
    ap.add_argument("--repeat", type=int, default=3, help="正式统计次数")
    ap.add_argument("--gpu-index", type=int, default=0, help="GPU 编号")
    ap.add_argument("--gpu-sample-ms", type=int, default=50, help="GPU 采样周期 ms")
    ap.add_argument("--cpu-sample-ms", type=int, default=50, help="CPU 采样周期 ms")
    ap.add_argument("--enable-perf", action="store_true", help="是否启用 perf stat")
    ap.add_argument("--print-inference-only", action="store_true", help="仅打印 model 推理相关指标，不打印进程级 latency_s 等")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    run_dir = out_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = vars(args).copy()
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    # warmup：跑若干次，但不纳入最终统计
    for i in range(args.warmup):
        run_command(
            args.cmd,
            run_dir,
            f"warmup_{i}",
            sample_gpu=(args.mode == "gpu"),
            sample_cpu=True,
            gpu_interval_ms=args.gpu_sample_ms,
            cpu_interval_ms=args.cpu_sample_ms,
            gpu_index=args.gpu_index,
            enable_perf=False,  # warmup 默认不跑 perf，避免影响统计
        )

    # repeat：正式统计
    summaries = []
    for i in range(args.repeat):
        s = run_command(
            args.cmd,
            run_dir,
            f"run_{i}",
            sample_gpu=(args.mode == "gpu"),
            sample_cpu=True,
            gpu_interval_ms=args.gpu_sample_ms,
            cpu_interval_ms=args.cpu_sample_ms,
            gpu_index=args.gpu_index,
            enable_perf=args.enable_perf,
        )
        summaries.append(s)
        (run_dir / f"run_{i}.json").write_text(
            json.dumps(asdict(s), indent=2, ensure_ascii=False)
        )

    summary = {
        "aggregate": aggregate(summaries),
        "runs": [asdict(s) for s in summaries],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    if args.print_inference_only:
        agg = summary["aggregate"]
        out = {
            "aggregate": {
                "num_runs": agg.get("num_runs"),
                "inference_only_per_forward_s_mean": agg.get("inference_only_per_forward_s_mean"),
                "inference_only_s_mean": agg.get("inference_only_s_mean"),
                "gpu_energy_j_mean": agg.get("gpu_energy_j_mean"),
                "gpu_avg_power_w_mean": agg.get("gpu_avg_power_w_mean"),
                "gpu_peak_power_w_max": agg.get("gpu_peak_power_w_max"),
                "cpu_energy_inference_pkg_j_mean": agg.get("cpu_energy_inference_pkg_j_mean"),
                "cpu_energy_inference_ram_j_mean": agg.get("cpu_energy_inference_ram_j_mean"),
                "cpu_energy_inference_total_j_mean": agg.get("cpu_energy_inference_total_j_mean"),
            },
            "returncodes": [s.returncode for s in summaries],
        }
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()