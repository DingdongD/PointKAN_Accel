"""
Utilities Module
================

公共工具：日志、数学工具、单位换算（units 预留）。
"""

from .logger import setup_logger, get_logger
from .math_utils import safe_div_ceil
from .units import (
    energy_nJ_from_ns,
    energy_to_nj,
    power_from_energy_nJ,
    power_to_mw,
    cycles_to_ns,
    cycles_to_ms,
    time_to_ns,
    freq_mhz_from_period_ns,
    area_to_mm2,
)

__all__ = [
    'setup_logger',
    'get_logger',
    'safe_div_ceil',
    'energy_nJ_from_ns',
    'energy_to_nj',
    'power_from_energy_nJ',
    'power_to_mw',
    'cycles_to_ns',
    'cycles_to_ms',
    'time_to_ns',
    'freq_mhz_from_period_ns',
    'area_to_mm2',
]
