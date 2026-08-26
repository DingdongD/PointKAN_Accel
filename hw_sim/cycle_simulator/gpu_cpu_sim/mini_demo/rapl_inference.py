#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 pyRAPL 在 Python 进程内测量「仅 model(x) 前向」的 CPU 能耗（Measurement 只包住 forward 循环）。
并提供统一的 model(x) 计时：GPU 用 torch.cuda.Event，CPU 用 time.time() 前后打点。

与 profiler 通过 perf/sysfs 读到的整段 RAPL 不同：此处能量只对应 forward 段。可选 baseline（空循环）扣除。

返回能耗单位：焦耳 (J)。若 pyRAPL 不可用则返回 None。
"""

import time
from typing import Any, Callable, Dict, Optional, Tuple


def time_forward_seconds(device: Any, forward_fn: Callable[[], None]) -> float:
    """
    测量一次前向的耗时（秒）。GPU 用 torch.cuda.Event 记录 kernel 时间，CPU 用 time.time() 前后打点。
    forward_fn 应只执行一次 model(x)，无参数、无返回值。
    """
    if getattr(device, "type", None) == "cuda":
        import torch
        torch.cuda.synchronize()
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        forward_fn()
        end_ev.record()
        torch.cuda.synchronize()
        return start_ev.elapsed_time(end_ev) / 1000.0  # ms -> s
    t0 = time.time()
    forward_fn()
    return time.time() - t0

_rapl_available = False
_rapl_setup_done = False


def _init_rapl() -> bool:
    global _rapl_available, _rapl_setup_done
    if _rapl_setup_done:
        return _rapl_available
    _rapl_setup_done = True
    try:
        import pyRAPL
        pyRAPL.setup()
        _rapl_available = True
    except Exception:
        _rapl_available = False
    return _rapl_available


def _sample_cpu_idle_power_j_per_s(duration_s: float = 1.0) -> Optional[float]:
    """
    测量 CPU 空闲功率：在 duration_s 秒内仅 sleep，用 pyRAPL 测能耗，返回 J/s（即瓦）。
    用于从推理阶段能耗中扣减「空闲功率 × 推理时间」。
    """
    if not _init_rapl():
        return None
    try:
        import pyRAPL
        m = pyRAPL.Measurement("idle")
        m.begin()
        time.sleep(max(0.1, duration_s))
        m.end()
        pkg_uj = m.result.pkg if m.result and getattr(m.result, "pkg", None) else []
        dram_uj = m.result.dram if m.result and getattr(m.result, "dram", None) else []
        pkg_j = sum(pkg_uj) / 1e6 if pkg_uj else 0.0
        dram_j = sum(dram_uj) / 1e6 if dram_uj else 0.0
        total_j = pkg_j + dram_j
        return total_j / duration_s
    except Exception:
        return None


def measure_cpu_energy_inference(fn: Callable[[], Any]) -> Tuple[Any, Optional[Dict[str, float]]]:
    """
    在 pyRAPL Measurement 内执行 fn()，返回 (fn 的返回值, 能耗字典或 None)。

    能耗字典键：cpu_energy_pkg_j, cpu_energy_ram_j, cpu_energy_total_j（单位焦耳）。
    会先测 CPU 空闲功率，再从推理阶段总能耗中扣减「空闲功率 × 推理时间」得到增量能耗。
    若 pyRAPL 不可用或测量失败，第二项为 None。
    """
    if not _init_rapl():
        return fn(), None
    result = None
    try:
        import pyRAPL
        idle_j_per_s = _sample_cpu_idle_power_j_per_s(duration_s=1.0)

        m = pyRAPL.Measurement("inference")
        m.begin()
        result = fn()
        m.end()
        # pkg/dram 为微焦，多 socket 为列表
        pkg_uj = m.result.pkg if m.result and getattr(m.result, "pkg", None) else []
        dram_uj = m.result.dram if m.result and getattr(m.result, "dram", None) else []
        pkg_j = sum(pkg_uj) / 1e6 if pkg_uj else 0.0
        dram_j = sum(dram_uj) / 1e6 if dram_uj else 0.0
        total_j = pkg_j + dram_j

        # 扣减空闲分量：E_corrected = E_total - P_idle * T_inference
        infer_total_s = None
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            infer_total_s = result[1]
        if idle_j_per_s is not None and infer_total_s is not None and total_j is not None:
            corrected_total_j = max(0.0, total_j - idle_j_per_s * infer_total_s)
            if total_j > 0:
                factor = corrected_total_j / total_j
                pkg_j = pkg_j * factor
                dram_j = dram_j * factor
                total_j = corrected_total_j
            else:
                total_j = corrected_total_j

        return result, {
            "cpu_energy_pkg_j": pkg_j,
            "cpu_energy_ram_j": dram_j,
            "cpu_energy_total_j": total_j,
        }
    except Exception:
        if result is not None:
            return result, None
        raise
