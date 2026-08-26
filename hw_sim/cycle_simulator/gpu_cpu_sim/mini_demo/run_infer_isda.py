#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""兼容入口：已合并至 run_infer_random.py。将 --model X 转为 --task isda_X 后转发。"""
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(SCRIPT_DIR, "run_infer_random.py")


def main():
    out: list[str] = []
    i = 0
    argv = sys.argv[1:]
    while i < len(argv):
        if argv[i] == "--model" and i + 1 < len(argv):
            out.extend(["--task", f"isda_{argv[i + 1]}"])
            i += 2
            continue
        out.append(argv[i])
        i += 1
    rc = subprocess.call([sys.executable, TARGET] + out)
    sys.exit(rc)


if __name__ == "__main__":
    main()
