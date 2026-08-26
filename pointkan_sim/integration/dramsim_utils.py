"""
Small shared DRAMSim3 config helpers.
"""

from __future__ import annotations

import configparser
from typing import Optional


def read_dramsim3_ini(config_path: str) -> Optional[configparser.ConfigParser]:
    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    parser.optionxform = str
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            content = fh.read()
    except OSError:
        return None

    ini_text = content if content.lstrip().startswith("[") else "[system]\n" + content
    try:
        parser.read_string(ini_text)
    except configparser.Error:
        return None
    return parser


def parse_dramsim3_burst_bytes(config_path: str, *, default: int = 64) -> int:
    parser = read_dramsim3_ini(config_path)
    if parser is None:
        return default
    try:
        bus_width = int(parser.get("system", "bus_width", fallback="64"))
    except Exception:
        bus_width = 64
    try:
        burst_length = int(parser.get("dram_structure", "BL", fallback="8"))
    except Exception:
        burst_length = 8
    return max(1, int((max(1, bus_width) * max(1, burst_length)) // 8))
