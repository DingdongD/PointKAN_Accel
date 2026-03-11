#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 pyRAPL 在进程内测量「仅模型推理」阶段的 CPU 能耗（包住推理循环，不关心子进程/整进程）。
并提供统一的 model(x) 推理计时：GPU 用 torch.cuda.Event，CPU 用 time.time() 前后打点。

与 profiler 的 perf/sysfs 整进程 RAPL 不同：这里在 Python 里 Measurement.begin/end 只包住
model forward 循环，得到纯推理对应的 CPU 能耗。可选做 baseline（空循环）扣除。

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


def measure_cpu_energy_inference(fn: Callable[[], Any]) -> Tuple[Any, Optional[Dict[str, float]]]:
    """
    在 pyRAPL Measurement 内执行 fn()，返回 (fn 的返回值, 能耗字典或 None)。

    能耗字典键：cpu_energy_pkg_j, cpu_energy_ram_j, cpu_energy_total_j（单位焦耳）。
    若 pyRAPL 不可用或测量失败，第二项为 None。
    """
    if not _init_rapl():
        return fn(), None
    result = None
    try:
        import pyRAPL
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
        return result, {
            "cpu_energy_pkg_j": pkg_j,
            "cpu_energy_ram_j": dram_j,
            "cpu_energy_total_j": total_j,
        }
    except Exception:
        if result is not None:
            return result, None
        raise
