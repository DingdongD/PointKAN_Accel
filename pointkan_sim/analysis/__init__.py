"""
Analysis Module
===============

Provides post-simulation analysis tools including:
- Energy breakdown by circuit module
- Cycle breakdown by stage/sub-stage
- Network-level analysis
"""

from .breakdown import (
    ModuleBreakdown,
    StageBreakdown,
    OperatorBreakdown,
    analyze_operator_breakdown,
    analyze_network_breakdown,
    format_breakdown_table,
    format_network_breakdown,
    print_breakdown,
    print_network_breakdown
)

__all__ = [
    'ModuleBreakdown',
    'StageBreakdown',
    'OperatorBreakdown',
    'analyze_operator_breakdown',
    'analyze_network_breakdown',
    'format_breakdown_table',
    'format_network_breakdown',
    'print_breakdown',
    'print_network_breakdown'
]
