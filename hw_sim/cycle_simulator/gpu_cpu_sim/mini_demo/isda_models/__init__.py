"""
从 ISEDA26 ``model_back.py`` 精简移植的点云网络，用于 mini_demo 跨平台推理性能评估。

**DGCNN**
  - ``DGCNN_cls``：分类
  - ``DGCNN_partseg``、``DGCNN_semseg_s3dis``、``Transform_Net``：见 ``dgcnn_seg.py``
  - 语义分割默认用 ``get_graph_feature(..., dim9=True)``（非原版 bit-kNN），便于 CPU/GPU 一致。

**PointNet++**（FPS + ball query 为 PyTorch，不依赖 ``pointnet2_ops``）
  - 分类：``PointNet2ClsSSG``、``PointNet2ClsMSG``
  - 部件：``PointNet2PartSegSSG``、``PointNet2PartSegMSG``
  - 语义：``PointNet2SemSSG``、``PointNet2SemMSG``

同目录 ``pointnet2_ops_lib`` 仅供 PointKAN-elite 等使用；本包不 import。
"""

from .dgcnn import DGCNN_cls, get_graph_feature, knn
from .dgcnn_seg import DGCNN_partseg, DGCNN_semseg_s3dis, Transform_Net
from .pointnet2 import (
    PointNet2ClsMSG,
    PointNet2ClsSSG,
    PointNet2PartSegMSG,
    PointNet2PartSegSSG,
    PointNet2SemMSG,
    PointNet2SemSSG,
)

# run_infer_random / 脚本共用的 ISEDA 子任务名（前缀 isda_）
ISDA_MODEL_NAMES = (
    "dgcnn",
    "pn2_ssg",
    "pn2_msg",
    "dgcnn_partseg",
    "dgcnn_semseg",
    "pn2_part_ssg",
    "pn2_part_msg",
    "pn2_sem_ssg",
    "pn2_sem_msg",
)

__all__ = [
    "ISDA_MODEL_NAMES",
    "DGCNN_cls",
    "DGCNN_partseg",
    "DGCNN_semseg_s3dis",
    "Transform_Net",
    "knn",
    "get_graph_feature",
    "PointNet2ClsMSG",
    "PointNet2ClsSSG",
    "PointNet2PartSegMSG",
    "PointNet2PartSegSSG",
    "PointNet2SemMSG",
    "PointNet2SemSSG",
]
