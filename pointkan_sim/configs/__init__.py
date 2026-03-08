"""
Configuration Module
====================

Hardware accelerator configurations.
"""

from .hw_config import (
    get_accelerator_config,
    list_accelerator_configs,
    CircuitModule,
    set_digital_clock_freq_hz,
    set_logic_parallelism,
)

__all__ = [
    'get_accelerator_config',
    'list_accelerator_configs',
    'CircuitModule',
    'set_digital_clock_freq_hz',
    'set_logic_parallelism',
]
