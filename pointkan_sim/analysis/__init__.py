"""
分析模块（Analysis）
====================

仿真后分析：按层/按算子/按电路模块聚合周期与能耗（analyze_simulation_events_breakdown），
供 main 与 PPA 报告使用。算子级 breakdown 可视化由 integration/ppa_report 的饼图承担。
"""

from .breakdown import (
    ModuleBreakdown,
    StageBreakdown,
    OperatorBreakdown,
    analyze_operator_breakdown,
    format_breakdown_table,
    analyze_simulation_events_breakdown,
)

__all__ = [
    'ModuleBreakdown',
    'StageBreakdown',
    'OperatorBreakdown',
    'analyze_operator_breakdown',
    'format_breakdown_table',
    'analyze_simulation_events_breakdown',
]
