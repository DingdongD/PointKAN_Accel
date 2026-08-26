"""
Shared lightweight helpers used across pointkan_sim modules.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

import numpy as np

from ..simulator.event import Event


def sanitize_log_tag(raw: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9._-]+", "_", str(raw or "").strip())
    return tag.strip("._-") or "sim"


def split_bytes(total_bytes: int, chunk_bytes: int) -> List[int]:
    total = max(0, int(total_bytes))
    chunk = max(1, int(chunk_bytes))
    out: List[int] = []
    while total > 0:
        take = min(total, chunk)
        out.append(take)
        total -= take
    return out


def retag_events(
    events: List[Event],
    *,
    operator: Optional[str] = None,
    circuit_module: Optional[str] = None,
    operation_replace: Optional[Dict[str, str]] = None,
    metadata_updates: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    if not events:
        return events
    repl = operation_replace or {}
    meta_updates = metadata_updates or {}
    for ev in events:
        if operator:
            ev.operator = operator
        if circuit_module:
            ev.circuit_module = circuit_module
        if repl and getattr(ev, "operation", None):
            op_name = str(ev.operation)
            for old, new in repl.items():
                op_name = op_name.replace(old, new)
            ev.operation = op_name
        if meta_updates:
            ev.metadata = dict(ev.metadata or {})
            ev.metadata.update(meta_updates)
    return events


def stretch_input_conv1d(
    in_c: int,
    in_w: int,
    k_w: int,
    stride: int = 1,
) -> np.ndarray:
    num_patches = max(1, (in_w - k_w) // stride + 1)
    k_dim = in_c * k_w
    return np.random.randn(1, num_patches, k_dim).astype(np.float32)


def index_bytes_for_points(num_points: int, default_bytes: Optional[int] = None) -> int:
    n = max(2, int(num_points))
    bits = int(math.ceil(math.log2(n)))
    bytes_needed = max(1, int(math.ceil(bits / 8.0)))
    if default_bytes is None or int(default_bytes) <= 0:
        return bytes_needed
    return max(1, min(int(default_bytes), bytes_needed))


def resolve_neurosim_sar_column_power_w(
    power_w: Optional[float] = None,
    power_uw: Optional[float] = None,
) -> Optional[float]:
    if power_w is not None:
        value = float(power_w)
        return value if value > 0 else None
    if power_uw is not None:
        value = float(power_uw)
        return (value * 1e-6) if value > 0 else None
    return None
