# -*- coding: utf-8 -*-
"""
仿真输出与算子序列的键名常量
============================

统一 operator_sequence、memory_transactions 等字典/TSV 的字段名，
便于读写一致、避免拼写错误，并与 operator_sequence.txt 表头一一对应。

算子名称（operator 列）统一采用 PascalCase，与网络层 op_type（小写 conv1d/linear）
区分：op_type 用于派发逻辑，operator 用于报告与 operator_sequence.txt 显示。
"""

# ----- 硬件算子序列中「算子名称」统一形式（PascalCase，与 operator_sequence.txt 一致） -----
# 卷积 / 全连接：输出到 operator_sequence 时一律用下列名称，避免出现 conv1d 与 Conv1d 混用
OP_NAME_CONV1D = "Conv1d"
OP_NAME_LINEAR = "Linear"
# 其他常用算子名（FPS, KNN, BN, ReLU, LayerBuffer 等）在各自生成 Event 时已用 PascalCase，此处仅作说明

# ----- 硬件算子序列单条记录的键名（与 operator_sequence.txt 列一致） -----
# 序号（从 0 起）
KEY_IDX = "idx"
# 起始周期
KEY_CYCLE = "cycle"
# 结束周期（含）
KEY_END_CYCLE = "end_cycle"
# 持续周期数
KEY_DURATION = "duration"
# 算子名称（PascalCase：Conv1d, Linear, LayerBuffer, FPS, KNN, BN, ReLU 等）
KEY_OPERATOR = "operator"
# 阶段（如 buffer_io, quant, distance）
KEY_STAGE = "stage"
# 具体操作描述（如 phase1_point_cloud_to_sram_ii, buffer_write）
KEY_OPERATION = "operation"
# 所属层名（如 embedding, GAM-stage0）
KEY_LAYER_NAME = "layer_name"
# 能耗（nJ）
KEY_ENERGY_NJ = "energy_nj"
# 电路模块（如 DRAM, SRAM-II；内存事务导出时使用）
KEY_CIRCUIT_MODULE = "circuit_module"

# operator_sequence.txt 表头列顺序（不含 circuit_module）
OPERATOR_SEQUENCE_HEADER = (
    f"# {KEY_IDX}\t{KEY_CYCLE}\t{KEY_END_CYCLE}\t{KEY_DURATION}\t"
    f"{KEY_OPERATOR}\t{KEY_STAGE}\t{KEY_OPERATION}\t{KEY_LAYER_NAME}\t{KEY_ENERGY_NJ}\n"
)
# memory_transactions.txt 表头列顺序（含 circuit_module）
MEMORY_TRANSACTIONS_HEADER = (
    f"# {KEY_IDX}\t{KEY_CYCLE}\t{KEY_END_CYCLE}\t{KEY_DURATION}\t"
    f"{KEY_OPERATOR}\t{KEY_STAGE}\t{KEY_OPERATION}\t{KEY_CIRCUIT_MODULE}\t{KEY_LAYER_NAME}\t{KEY_ENERGY_NJ}\n"
)
