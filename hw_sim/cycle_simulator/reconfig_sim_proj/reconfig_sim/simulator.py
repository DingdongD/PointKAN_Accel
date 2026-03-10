# -*- coding: utf-8 -*-
"""
可重构脉动阵列周期仿真器：支持 OS/WS 数据流、GEMM 与 Conv2D(im2col->GEMM)，输出周期/能耗与模式分解。
"""
from dataclasses import asdict
from typing import Dict, Any, List, Optional
import numpy as np

from .configs.defaults import SimConfig
from .models.buffer_model import BufferModel
from .models.noc_model import NoCModel
from .models.cycle_energy import CycleEnergyModel
from .schedulers.tiler import GemmTiler, Conv2DTiler
from .schedulers.dispatcher import Dispatcher
from .utils.im2col import conv2d_to_gemm
from .utils.tiling_check import (
    verify_gemm_tiling_coverage,
    verify_conv2d_tiling_coverage,
    verify_gemm_result,
    verify_conv2d_result,
)
from .workloads.gemm import GemmWorkload
from .workloads.conv2d import Conv2DWorkload


class ReconfigSimulator:
    """
    可重构脉动阵列仿真器：对给定 GEMM/Conv2D 工作负载进行分块、分配 OS/WS 模式，
    并汇总周期、能耗及每块 trace。依赖配置中的阵列、缓冲、NoC、时序与能耗参数。
    """

    def __init__(self, cfg: Optional[SimConfig] = None):
        self.cfg = cfg or SimConfig()
        self.buffer_model = BufferModel(self.cfg.buffer)
        self.noc_model = NoCModel(self.cfg.noc)
        self.cem = CycleEnergyModel(self.cfg, self.buffer_model, self.noc_model)

    def _simulate_tiles(self, tiles, mode_request: str):
        """
        对已分块列表逐块做周期/能耗估计，累加总周期、总能耗、总 MAC，并按 OS/WS 统计。
        每块会加上模式切换惩罚（若与上一块模式不同）。
        """
        dispatcher = Dispatcher(self.cfg)
        traces = []
        total_cycles = 0
        total_energy = 0.0
        total_macs = 0
        mode_breakdown = {'OS': {'tiles': 0, 'cycles': 0, 'energy': 0.0},
                          'WS': {'tiles': 0, 'cycles': 0, 'energy': 0.0}}
        for tile in tiles:
            switch_pen = dispatcher.mode_switch_penalty(tile.mode)
            stats = self.cem.estimate_tile(tile, word_bytes=self.cfg.array.word_size // 8)
            stats.cycles += switch_pen
            stats.energy += switch_pen * (self.cfg.energy.control_energy_per_cycle + self.cfg.energy.dispatcher_energy_per_cycle)
            traces.append(stats.to_dict())
            total_cycles += stats.cycles
            total_energy += stats.energy
            total_macs += stats.macs
            mode_breakdown[tile.mode]['tiles'] += 1
            mode_breakdown[tile.mode]['cycles'] += stats.cycles
            mode_breakdown[tile.mode]['energy'] += stats.energy
        return {
            'total_cycles': total_cycles,
            'total_energy': total_energy,
            'total_macs': total_macs,
            'mode_request': mode_request,
            'mode_breakdown': mode_breakdown,
            'trace': traces,
        }

    def run_gemm(self, wl: GemmWorkload, mode: str = 'AUTO') -> Dict[str, Any]:
        """
        运行 GEMM 仿真。mode 为 'OS'/'WS' 时按该模式统一分块；为 'AUTO' 时先用 OS 脚印分块再逐块分配模式。
        返回包含 total_cycles、total_energy、mode_breakdown、trace、shape、golden_checksum 等字段的统计字典。
        """
        base_tiler = GemmTiler(self.cfg.array.arr_r, self.cfg.array.arr_c)
        if mode in ('OS', 'WS'):
            tiles = base_tiler.tile(wl.m, wl.k, wl.n, mode)
        else:
            os_tiles = base_tiler.tile(wl.m, wl.k, wl.n, 'OS')
            ws_tiles = base_tiler.tile(wl.m, wl.k, wl.n, 'WS')
            tiles = os_tiles
            tiles = Dispatcher(self.cfg).assign_modes(tiles, 'AUTO')
        stats = self._simulate_tiles(tiles, mode)
        golden = wl.golden()
        tiling_ok, tiling_msg = verify_gemm_tiling_coverage(tiles, wl.m, wl.n)
        result_ok, result_msg = verify_gemm_result(
            wl.a, wl.b, golden, tiles, wl.m, wl.k, wl.n
        )
        stats['op_type'] = 'gemm'
        stats['shape'] = {'m': wl.m, 'k': wl.k, 'n': wl.n}
        stats['correct'] = tiling_ok and result_ok
        stats['tiling_check'] = tiling_msg
        stats['result_check'] = result_msg
        stats['golden_checksum'] = int(np.sum(golden.astype(np.int64)))
        return stats

    def run_conv2d(self, wl: Conv2DWorkload, mode: str = 'AUTO') -> Dict[str, Any]:
        """
        将 Conv2D 转为 im2col GEMM 后分块仿真。mode 含义同 run_gemm。
        返回统计中额外包含 shape（卷积参数）、im2col（rows/k_dim/cout）与 golden_checksum。
        """
        cols, w_mat, y = conv2d_to_gemm(wl.x, wl.wgt, wl.bias, stride=wl.stride, padding=wl.padding)
        rows = cols.shape[0]
        k = cols.shape[1]
        cout = w_mat.shape[1]
        tiler = Conv2DTiler(self.cfg.array.arr_r, self.cfg.array.arr_c)
        if mode in ('OS', 'WS'):
            tiles = tiler.tile_im2col_gemm(rows, k, cout, mode)
        else:
            tiles = tiler.tile_im2col_gemm(rows, k, cout, 'OS')
            tiles = Dispatcher(self.cfg).assign_modes(tiles, 'AUTO')
        stats = self._simulate_tiles(tiles, mode)
        stats['op_type'] = 'conv2d_im2col'
        stats['shape'] = {
            'n': wl.n, 'c_in': wl.c_in, 'h': wl.h, 'w': wl.w,
            'c_out': wl.c_out, 'k_h': wl.k_h, 'k_w': wl.k_w,
            'stride': wl.stride, 'padding': wl.padding,
        }
        stats['im2col'] = {
            'rows': rows,
            'k_dim': k,
            'cout': cout,
        }
        tiling_ok, tiling_msg = verify_conv2d_tiling_coverage(tiles, rows, cout)
        n, oh, ow = y.shape[0], y.shape[2], y.shape[3]
        result_ok, result_msg = verify_conv2d_result(
            cols, w_mat, y, tiles, rows, k, cout, n, oh, ow, bias=wl.bias
        )
        stats['correct'] = tiling_ok and result_ok
        stats['tiling_check'] = tiling_msg
        stats['result_check'] = result_msg
        stats['golden_checksum'] = int(np.sum(y.astype(np.int64)))
        return stats
