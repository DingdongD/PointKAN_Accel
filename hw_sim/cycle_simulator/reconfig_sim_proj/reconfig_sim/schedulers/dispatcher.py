# -*- coding: utf-8 -*-
"""
调度器（Dispatcher）：为各 tile 分配 OS/WS 模式（AUTO 时按启发式），并计算模式切换惩罚周期。
"""
from typing import List
from ..models.tile import TileDesc


class Dispatcher:
    """按请求模式（OS/WS/AUTO）为分块列表分配数据流模式，并记录上一次模式以计算切换惩罚"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.prev_mode = None  # 上一块使用的模式，用于 mode_switch_penalty

    def choose_mode_for_shape(self, m_tile: int, k_tile: int, n_tile: int) -> str:
        """
        根据块形状的简单启发式选择 OS 或 WS。
        倾向 OS：输出元素多且能较好填满阵列、K 维不太小；
        倾向 WS：K 维分块占主导、按行累加输出更有利。
        """
        arr_r = self.cfg.array.arr_r
        arr_c = self.cfg.array.arr_c
        os_score = (m_tile * n_tile) / max(1, arr_r * arr_c) + 0.25 * min(k_tile, arr_r)
        ws_score = (n_tile / max(1, arr_c)) + 0.35 * min(k_tile, arr_r) - 0.15 * max(0, m_tile - 1)
        return 'OS' if os_score >= ws_score else 'WS'

    def assign_modes(self, tiles: List[TileDesc], requested_mode: str) -> List[TileDesc]:
        """
        为每个 tile 设置 mode。requested_mode 为 'OS'/'WS' 时全部统一；为 'AUTO' 时逐块启发式选择。
        """
        out = []
        if requested_mode in ('OS', 'WS'):
            for t in tiles:
                t.mode = requested_mode
                out.append(t)
            return out
        if requested_mode != 'AUTO':
            raise ValueError(f'Unknown mode {requested_mode}')
        for t in tiles:
            t.mode = self.choose_mode_for_shape(t.m_tile, t.k_tile, t.n_tile)
            out.append(t)
        return out

    def mode_switch_penalty(self, curr_mode: str) -> int:
        """若当前块模式与上一块不同，返回模式切换惩罚周期数；否则返回 0。会更新 prev_mode。"""
        if self.prev_mode is None or self.prev_mode == curr_mode:
            self.prev_mode = curr_mode
            return 0
        self.prev_mode = curr_mode
        return self.cfg.timing.mode_switch_cycles
