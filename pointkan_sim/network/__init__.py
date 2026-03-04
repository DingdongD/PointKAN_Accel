"""
Network Model Module
====================

Neural network definitions for point cloud processing.
Supported networks:
- PointKAN (standard B-spline KAN)
- PointKAN-Elite (Efficient_KAN with rational activation)
"""

from .pointkan_net import (
    OperatorRef,
    LayerDef,
    PointKANConfig,
    create_pointkan_classification,
    POINTKAN_LAYERS,
)
from .pointkan_elite_net import (
    PointKANEliteConfig,
    create_pointkan_elite_classification,
    POINTKAN_ELITE_LAYERS,
)

__all__ = [
    'OperatorRef',
    'LayerDef',
    'PointKANConfig',
    'create_pointkan_classification',
    'POINTKAN_LAYERS',
    'PointKANEliteConfig',
    'create_pointkan_elite_classification',
    'POINTKAN_ELITE_LAYERS',
]
