#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KITTI 入口：默认 --dataset kitti，k=256/512/1024，颜色 A/B/C 对应最大/中/小 K。
等价于: python fps_multi_k_visualize_core.py --dataset kitti ...
"""
import sys

from fps_multi_k_visualize_core import build_parser, run


def main():
    p = build_parser()
    if "--dataset" not in sys.argv:
        p.set_defaults(dataset="kitti")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
