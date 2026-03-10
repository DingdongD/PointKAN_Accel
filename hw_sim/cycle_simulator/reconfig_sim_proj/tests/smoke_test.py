# -*- coding: utf-8 -*-
"""
冒烟测试：用 4x4x4 GEMM、AUTO 模式跑一次仿真，检查周期、能耗、分块覆盖及与 golden 一致。
"""
import numpy as np
from reconfig_sim.configs.defaults import SimConfig
from reconfig_sim.simulator import ReconfigSimulator
from reconfig_sim.workloads.gemm import GemmWorkload

cfg = SimConfig()
sim = ReconfigSimulator(cfg)
wl = GemmWorkload.random(m=4, k=4, n=4, word_size=cfg.array.word_size, seed=1)
report = sim.run_gemm(wl, mode='AUTO')
assert report['total_cycles'] > 0
assert report['total_energy'] > 0
assert report['correct'] is True, "tiling coverage check failed: {}".format(report.get('tiling_check'))
assert report['golden_checksum'] == int(np.sum(wl.golden().astype(np.int64))), "golden checksum mismatch"
print('smoke_test passed')
