import os
import time
import glob
import statistics
from dataclasses import dataclass, asdict
from typing import Callable, Any, Dict, List, Optional

import numpy as np


# =========================
# RAPL 读取
# =========================

@dataclass
class RaplZone:
    name: str
    path: str
    energy_file: str
    max_energy_file: Optional[str] = None


def discover_rapl_zones(base: str = "/sys/class/powercap") -> List[RaplZone]:
    """
    自动发现可读的 RAPL zone
    典型路径:
      /sys/class/powercap/intel-rapl:0/
      /sys/class/powercap/intel-rapl:0/intel-rapl:0:0/
    """
    zones = []
    if not os.path.isdir(base):
        return zones

    # 递归找所有 name 文件
    for name_file in glob.glob(os.path.join(base, "**", "name"), recursive=True):
        zone_dir = os.path.dirname(name_file)
        energy_file = os.path.join(zone_dir, "energy_uj")
        max_energy_file = os.path.join(zone_dir, "max_energy_range_uj")

        if os.path.isfile(energy_file):
            try:
                with open(name_file, "r") as f:
                    name = f.read().strip()
            except Exception:
                name = os.path.basename(zone_dir)

            zones.append(
                RaplZone(
                    name=name,
                    path=zone_dir,
                    energy_file=energy_file,
                    max_energy_file=max_energy_file if os.path.isfile(max_energy_file) else None,
                )
            )
    return zones


def read_text(path: str) -> str:
    with open(path, "r") as f:
        return f.read().strip()


def read_energy_uj(zone: RaplZone) -> int:
    return int(read_text(zone.energy_file))


def read_max_energy_uj(zone: RaplZone) -> Optional[int]:
    if zone.max_energy_file and os.path.isfile(zone.max_energy_file):
        return int(read_text(zone.max_energy_file))
    return None


def energy_delta_uj(e0: int, e1: int, max_range_uj: Optional[int]) -> int:
    """
    处理 energy counter 回绕
    """
    if e1 >= e0:
        return e1 - e0
    if max_range_uj is None:
        # 无法确定回绕范围时，返回保守值 0
        return 0
    return (max_range_uj - e0) + e1


def read_all_zone_energies(zones: List[RaplZone]) -> Dict[str, int]:
    return {z.path: read_energy_uj(z) for z in zones}


# =========================
# Profile 结果
# =========================

@dataclass
class ProfileResult:
    repeats: int
    warmup: int

    latency_ms_list: List[float]
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_min: float
    latency_ms_max: float

    total_energy_j: Dict[str, float]
    avg_energy_j_per_run: Dict[str, float]
    avg_power_w: Dict[str, float]

    metadata: Dict[str, Any]


# =========================
# 核心 profile 函数
# =========================

def profile_cpu_inference(
    infer_fn: Callable[[], Any],
    warmup: int = 10,
    repeats: int = 50,
    sleep_s: float = 0.0,
) -> ProfileResult:
    """
    infer_fn: 不带参数的推理函数，例如 lambda: model(x)
    warmup : 预热次数
    repeats: 正式测量次数
    """

    # 1) 预热
    for _ in range(warmup):
        _ = infer_fn()

    zones = discover_rapl_zones()
    energy_supported = len(zones) > 0

    before = {}
    after = {}

    if energy_supported:
        before = read_all_zone_energies(zones)

    latency_ms_list = []

    t_batch_start = time.perf_counter_ns()

    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        _ = infer_fn()
        t1 = time.perf_counter_ns()
        latency_ms_list.append((t1 - t0) / 1e6)

        if sleep_s > 0:
            time.sleep(sleep_s)

    t_batch_end = time.perf_counter_ns()
    total_time_s = (t_batch_end - t_batch_start) / 1e9

    if energy_supported:
        after = read_all_zone_energies(zones)

    total_energy_j = {}
    avg_energy_j_per_run = {}
    avg_power_w = {}

    if energy_supported:
        zone_map = {z.path: z for z in zones}
        for path in before:
            z = zone_map[path]
            e0 = before[path]
            e1 = after[path]
            max_range = read_max_energy_uj(z)
            delta_j = energy_delta_uj(e0, e1, max_range) / 1e6  # uJ -> J

            key = f"{z.name} [{path}]"
            total_energy_j[key] = delta_j
            avg_energy_j_per_run[key] = delta_j / repeats
            avg_power_w[key] = delta_j / total_time_s if total_time_s > 0 else float("nan")

    return ProfileResult(
        repeats=repeats,
        warmup=warmup,
        latency_ms_list=latency_ms_list,
        latency_ms_mean=float(statistics.mean(latency_ms_list)),
        latency_ms_p50=float(np.percentile(latency_ms_list, 50)),
        latency_ms_p95=float(np.percentile(latency_ms_list, 95)),
        latency_ms_min=float(min(latency_ms_list)),
        latency_ms_max=float(max(latency_ms_list)),
        total_energy_j=total_energy_j,
        avg_energy_j_per_run=avg_energy_j_per_run,
        avg_power_w=avg_power_w,
        metadata={
            "rapl_supported": energy_supported,
            "num_rapl_zones": len(zones),
            "zones": [{"name": z.name, "path": z.path} for z in zones],
            "total_measured_window_s": total_time_s,
        },
    )


# =========================
# 示例：你自己的模型
# =========================

def dummy_infer():
    # 用矩阵乘法模拟一点 CPU 推理负载
    a = np.random.randn(512, 512).astype(np.float32)
    b = np.random.randn(512, 512).astype(np.float32)
    c = a @ b
    return c


def pretty_print_result(res: ProfileResult):
    print("===== CPU Inference Profile =====")
    print(f"warmup={res.warmup}, repeats={res.repeats}")
    print(f"latency mean: {res.latency_ms_mean:.4f} ms")
    print(f"latency p50 : {res.latency_ms_p50:.4f} ms")
    print(f"latency p95 : {res.latency_ms_p95:.4f} ms")
    print(f"latency min : {res.latency_ms_min:.4f} ms")
    print(f"latency max : {res.latency_ms_max:.4f} ms")
    print()

    print("RAPL supported:", res.metadata["rapl_supported"])
    print("RAPL zones:")
    for z in res.metadata["zones"]:
        print(f"  - {z['name']}: {z['path']}")
    print()

    if res.total_energy_j:
        print("Energy:")
        for k, v in res.total_energy_j.items():
            print(f"  total {k}: {v:.6f} J")
        print()

        print("Average energy per inference:")
        for k, v in res.avg_energy_j_per_run.items():
            print(f"  {k}: {v:.6f} J/run")
        print()

        print("Average power:")
        for k, v in res.avg_power_w.items():
            print(f"  {k}: {v:.6f} W")
    else:
        print("No RAPL energy data available on this machine.")


if __name__ == "__main__":
    res = profile_cpu_inference(
        infer_fn=dummy_infer,
        warmup=5,
        repeats=20,
    )
    pretty_print_result(res)
