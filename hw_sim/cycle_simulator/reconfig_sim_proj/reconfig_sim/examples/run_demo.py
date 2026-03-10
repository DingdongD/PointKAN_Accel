# -*- coding: utf-8 -*-
"""
示例：使用默认配置跑一次 GEMM 与一次 Conv2D 仿真，输出周期/能耗与 JSON 报告，
并将「所有 tile × 每 cycle」的 PE 活动导出为 PNG 图片序列（见 systolic_viz_gemm/、systolic_viz_conv2d/）。
运行：在工程根目录执行 PYTHONPATH=. python reconfig_sim/examples/run_demo.py
依赖：matplotlib（用于导出 PNG）。
"""
from reconfig_sim.configs.defaults import SimConfig
from reconfig_sim.simulator import ReconfigSimulator
from reconfig_sim.workloads.gemm import GemmWorkload
from reconfig_sim.workloads.conv2d import Conv2DWorkload
from reconfig_sim.utils.report import write_json
from reconfig_sim.utils.cycle_viz import export_report_to_images

cfg = SimConfig()
sim = ReconfigSimulator(cfg)

# 8x8x8 GEMM 与 1x3x8x8 -> 8 通道 3x3 卷积，随机数据
gemm = GemmWorkload.random(m=16, k=8, n=8, word_size=cfg.array.word_size, seed=42)
conv = Conv2DWorkload.random(n=1, c_in=3, h=8, w=8, c_out=8, k_h=3, k_w=3,
                             stride=1, padding=1, word_size=cfg.array.word_size, seed=7)

gemm_report = sim.run_gemm(gemm, mode='OS')
conv_report = sim.run_conv2d(conv, mode='OS')
report = {
    'gemm': gemm_report,
    'conv2d': conv_report,
    'noc_note': sim.noc_model.why_not_booksim(),
    'config': cfg.to_dict(),
}
write_json(report, 'reconfig_sim_demo_report.json')
print('GEMM cycles:', gemm_report['total_cycles'], 'energy:', round(gemm_report['total_energy'], 2))
print('GEMM correct:', gemm_report['correct'], '| result_check:', gemm_report.get('result_check', '-'))
print('Conv cycles:', conv_report['total_cycles'], 'energy:', round(conv_report['total_energy'], 2))
print('Conv correct:', conv_report['correct'], '| result_check:', conv_report.get('result_check', '-'))
print('Wrote reconfig_sim_demo_report.json')

# ----- 完整过程导出为图片（所有 tile × 每 cycle），便于分析数据流 -----
arr_r, arr_c = cfg.array.arr_r, cfg.array.arr_c
n_gemm, dir_gemm = export_report_to_images(gemm_report, cfg, "systolic_viz_gemm", op_type="gemm", arr_r=arr_r, arr_c=arr_c)
n_conv, dir_conv = export_report_to_images(conv_report, cfg, "systolic_viz_conv2d", op_type="conv2d", arr_r=arr_r, arr_c=arr_c)
if n_gemm > 0:
    print("GEMM: saved {} frames + animation.gif to {} (open index.html or animation.gif)".format(n_gemm, dir_gemm))
if n_conv > 0:
    print("Conv2D: saved {} frames + animation.gif to {} (open index.html or animation.gif)".format(n_conv, dir_conv))
if n_gemm == 0 and n_conv == 0:
    print("Viz: no frames (install matplotlib for PNG export)")
