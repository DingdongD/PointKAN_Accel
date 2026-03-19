"""
Network Model Module
====================

Neural network definitions for point cloud processing.
Supported networks:
- PointKAN (standard B-spline KAN)
- PointKAN-Elite (Efficient_KAN with rational activation)
- PointNet2ClsSSG (ModelNet40 classification)
- PointNet2PartSegSSG (ShapeNetPart part segmentation)
- PointNet2SemSSG (S3DIS semantic segmentation)
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
from .pointnet2_net import (
    create_pointnet2_cls_ssg,
    create_pointnet2_partseg_ssg,
    create_pointnet2_sem_ssg,
    POINTNET2_CLS_SSG_LAYERS,
    POINTNET2_PARTSEG_SSG_LAYERS,
    POINTNET2_SEM_SSG_LAYERS,
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
    'create_pointnet2_cls_ssg',
    'create_pointnet2_partseg_ssg',
    'create_pointnet2_sem_ssg',
    'POINTNET2_CLS_SSG_LAYERS',
    'POINTNET2_PARTSEG_SSG_LAYERS',
    'POINTNET2_SEM_SSG_LAYERS',
]
