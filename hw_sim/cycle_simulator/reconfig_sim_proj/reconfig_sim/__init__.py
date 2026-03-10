# -*- coding: utf-8 -*-
"""
可重构脉动阵列周期仿真器：支持 OS/WS 数据流的 GEMM 与 Conv2D(im2col) 周期与能耗估计。
"""
from .configs.defaults import SimConfig
from .simulator import ReconfigSimulator
