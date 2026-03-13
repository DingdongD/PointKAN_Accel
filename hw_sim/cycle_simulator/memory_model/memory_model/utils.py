import json
import math
import os
import hashlib
from typing import Any, Dict


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def ns_to_cycles(time_ns: float, freq_hz: int) -> int:
    clock_period_ns = 1e9 / freq_hz
    return max(1, math.ceil(time_ns / clock_period_ns))


def stable_hash_dict(d: Dict[str, Any]) -> str:
    s = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]


def load_json_if_exists(path: str):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)