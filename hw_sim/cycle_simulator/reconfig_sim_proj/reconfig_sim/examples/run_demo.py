# -*- coding: utf-8 -*-
"""
示例：使用默认配置跑一次 GEMM 与一次 Conv2D 仿真，并输出周期/能耗与 JSON 报告。
同时输出「每个 cycle 工作的 PE 与操作」的可视化（首块前若干周期 ASCII + 完整 trace 写 JSON）。
运行：在工程根目录执行 PYTHONPATH=. python reconfig_sim/examples/run_demo.py
"""
from reconfig_sim.configs.defaults import SimConfig
from reconfig_sim.simulator import ReconfigSimulator
from reconfig_sim.workloads.gemm import GemmWorkload
from reconfig_sim.workloads.conv2d import Conv2DWorkload
from reconfig_sim.utils.report import write_json
from reconfig_sim.utils.cycle_viz import (
    print_tile_schedule,
    trace_to_serializable,
    generate_tile_cycle_activity,
)
from reconfig_sim.models.tile import TileDesc

cfg = SimConfig()
sim = ReconfigSimulator(cfg)

# 8x8x8 GEMM 与 1x3x8x8 -> 8 通道 3x3 卷积，随机数据
gemm = GemmWorkload.random(m=16, k=8, n=8, word_size=cfg.array.word_size, seed=42)
conv = Conv2DWorkload.random(n=1, c_in=3, h=8, w=8, c_out=8, k_h=3, k_w=3,
                             stride=1, padding=1, word_size=cfg.array.word_size, seed=7)

gemm_report = sim.run_gemm(gemm, mode='AUTO')
conv_report = sim.run_conv2d(conv, mode='AUTO')
report = {
    'gemm': gemm_report,
    'conv2d': conv_report,
    'noc_note': sim.noc_model.why_not_booksim(),
    'config': cfg.to_dict(),
}
write_json(report, 'reconfig_sim_demo_report.json')
print('GEMM cycles:', gemm_report['total_cycles'], 'energy:', round(gemm_report['total_energy'], 2))
print('Conv cycles:', conv_report['total_cycles'], 'energy:', round(conv_report['total_energy'], 2))
print('Wrote reconfig_sim_demo_report.json')

# ----- 每周期 PE 工作与操作可视化 -----
# 说明：仿真器报告的「总 cycle」= 阵列阶段周期(clear+compute+drain) + NoC 传输周期 + buffer stall。
# 可视化仅展示「阵列阶段」内各 cycle 的 PE 活动，不包含 NoC/buffer 的周期。
arr_r, arr_c = cfg.array.arr_r, cfg.array.arr_c
max_cycles_show = 25  # 控制台只展示前 25 个周期，避免刷屏
if gemm_report.get('trace'):
    t0_stats = gemm_report['trace'][0]
    phase_cycles = t0_stats.get('phase_cycles', {})
    noc_cycles = t0_stats.get('noc', {}).get('cycles', 0)
    buf_stalls = t0_stats.get('buffer', {}).get('stalls', 0)
    array_cycles = sum(phase_cycles.values())
    total_tile_cycles = t0_stats.get('cycles', 0)
    print('\n========== GEMM 首块周期构成（总 {} cycle）=========='.format(total_tile_cycles))
    print('  阵列阶段(clear+compute+drain) = {} | NoC = {} | buffer stall = {}'.format(
        array_cycles, noc_cycles, buf_stalls))
    print('  下方可视化仅展示「阵列阶段」内各 cycle 的 PE 活动（前 {} 周期）\n'.format(max_cycles_show))
    t0 = gemm_report['trace'][0]['tile']
    tile0 = TileDesc(
        op_type=t0['op_type'],
        mode=t0['mode'],
        m0=t0['m0'], k0=t0['k0'], n0=t0['n0'],
        m_tile=t0['m_tile'], k_tile=t0['k_tile'], n_tile=t0['n_tile'],
        batch_idx=t0.get('batch_idx', 0),
        extra=t0.get('extra'),
    )
    trace_viz = print_tile_schedule(
        tile0, cfg, arr_r, arr_c,
        max_cycles=max_cycles_show,
        title='Tile#0 (mode={})'.format(tile0.mode),
        verbose=True,
    )
    # 将首块完整 trace（不截断）写入 JSON，便于外部绘图或分析
    full_trace = generate_tile_cycle_activity(tile0, cfg, arr_r, arr_c, max_cycles=None)
    cycle_trace_report = {
        'tile': tile0.to_dict(),
        'array_shape': [arr_r, arr_c],
        'op_legend': {'CLR': 'clear', 'MAC': 'mac', 'PRE': 'preload', 'DRN': 'drain', '-': 'idle'},
        'cycle_trace': trace_to_serializable(full_trace),
    }
    write_json(cycle_trace_report, 'reconfig_sim_cycle_trace_tile0.json')
    print('\nWrote reconfig_sim_cycle_trace_tile0.json (full per-cycle PE ops for tile0)')
else:
    print('No trace in gemm_report, skip cycle viz.')
