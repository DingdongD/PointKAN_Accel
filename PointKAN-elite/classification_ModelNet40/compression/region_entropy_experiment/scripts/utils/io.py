"""读写 modelnet40_test.npz。"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np


def load_test_npz(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    z = np.load(path, allow_pickle=False)
    points = z["points"]
    labels = z["labels"]
    sample_ids = z["sample_ids"]
    meta_path = path.replace(".npz", "_id_map.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        id_strings = m.get("id_strings", [])
    else:
        id_strings = [str(int(i)) for i in sample_ids]
    return points, labels, sample_ids, id_strings


def save_id_map(npz_path: str, id_strings: List[str]) -> None:
    meta_path = npz_path.replace(".npz", "_id_map.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"id_strings": id_strings}, f, ensure_ascii=False, indent=0)
