# -*- coding: utf-8 -*-
"""
报告工具：将仿真结果等对象以 JSON 形式写入文件。
"""
import json
from pathlib import Path


def write_json(obj, path):
    """将 obj 序列化为带缩进的 JSON 并写入 path（Path 或 str）。"""
    path = Path(path)
    path.write_text(json.dumps(obj, indent=2))
