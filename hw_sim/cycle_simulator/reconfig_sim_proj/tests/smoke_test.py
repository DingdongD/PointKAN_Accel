# -*- coding: utf-8 -*-
"""
冒烟测试：用 4x4x4 GEMM、AUTO 模式跑一次仿真，检查周期与能耗是否为正。
"""
from reconfig_sim.configs.defaults import SimConfig
from reconfig_sim.simulator import ReconfigSimulator
from reconfig_sim.workloads.gemm import GemmWorkload

cfg = SimConfig()
sim = ReconfigSimulator(cfg)
wl = GemmWorkload.random(m=4, k=4, n=4, word_size=cfg.array.word_size, seed=1)
report = sim.run_gemm(wl, mode='AUTO')
assert report['total_cycles'] > 0
assert report['total_energy'] > 0
print('smoke_test passed')
