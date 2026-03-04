"""
Point Cloud CIM PPA Simulator (New)
====================================

Hardware PPA (Performance, Power, Area) evaluator for point cloud neural
network deployment on CIM accelerators.

Modules:
--------
- configs: Hardware accelerator configuration (baseline / acc)
- operators: Point cloud & NN operators
- network: Neural network model definitions (PointKAN, PointKAN-Elite)
- simulator: Cycle-level event-driven simulation
- analysis: Post-simulation breakdown analysis
- report: Report generation
- utils: Utility functions and logging
"""

__version__ = "0.2.0"
__author__ = "PointCloud CIM Team"
