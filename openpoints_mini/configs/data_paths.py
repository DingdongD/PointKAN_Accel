"""
本机数据集根路径约定（与 PointKAN-elite 推理/训练脚本一致）

参考：
- ``PointKAN-elite/classification_ModelNet40/data.py`` 中 ``load_data``：
  ``DATA_DIR = '/workspace/VoxelNet/pc_data/'``，HDF5 在 ``modelnet40_ply_hdf5_2048/``。
- ``PointKAN-elite/part_segmentation/util/data_util.py`` 中 ``PartNormalDataset``：
  ``root = '/workspace/VoxelNet/pc_data/shapenetcore_partanno_segmentation_benchmark_v0_normal'``。
- S3DIS：原始对齐版在 ``Stanford3dDataset_v1.2_Aligned_Version/``（多为 Area 子目录）；
  **OpenPoints 的 ``S3DIS`` 数据集类**要求 ``data_root/raw/*.npy``（每文件 ``(N,7)=xyz+rgb+label``）。
  你当前机器上这类 npy 集中在 ``s3dis_trainval/``，需通过符号链接挂成 ``.../raw``（见 ``scripts/prepare_s3dis_openpoints.sh``）。

可通过环境变量 ``PC_DATA_ROOT`` 覆盖根目录（默认 ``/workspace/VoxelNet/pc_data``）。

**两种 S3DIS 训练数据不要混用：**

1. **Indoor3D HDF5**（``indoor3d_sem_seg_hdf5_data*``）：``/workspace/ISEDA26/pointnet_seg.py`` 与
   ``openpoints_mini/train_s3dis_pointnext_iseda26_data.py`` 使用此种。
2. **整房 npy + raw/**（``s3dis_trainval`` 经 ``prepare_s3dis_openpoints.sh`` 链到 ``s3dis_for_openpoints/raw``）：
   供 FractalCloud/OpenPoints 自带 ``dataset/s3dis/s3dis.py`` 使用。
"""
from __future__ import annotations

import os

PC_DATA_ROOT = os.environ.get("PC_DATA_ROOT", "/workspace/VoxelNet/pc_data")

# ModelNet40（与 PointKAN-elite classification data.py 一致）
MODELNET40_PLY_HDF5_2048 = os.path.join(PC_DATA_ROOT, "modelnet40_ply_hdf5_2048")

# ShapeNet 部件分割（与 PointKAN-elite part_segmentation data_util 一致）
SHAPENET_PART_NORMAL = os.path.join(
    PC_DATA_ROOT, "shapenetcore_partanno_segmentation_benchmark_v0_normal"
)

# 原始 Stanford S3DIS 对齐数据（Area_* 子目录）
STANFORD_S3DIS_ALIGNED = os.path.join(PC_DATA_ROOT, "Stanford3dDataset_v1.2_Aligned_Version")

# 已合并为单文件 npy 的房间（OpenPoints 训练常用格式）
S3DIS_TRAINVAL_NPY = os.path.join(PC_DATA_ROOT, "s3dis_trainval")

# 供 OpenPoints ``S3DIS(..., data_root=...)`` 使用：其内部读取 ``os.path.join(data_root, 'raw')``
S3DIS_OPENPOINTS_DEFAULT = os.path.join(PC_DATA_ROOT, "s3dis_for_openpoints")

# ---------- 与 ISEDA26 ``pointnet_seg.py`` / ``data_config.py`` 一致（Indoor3D HDF5）----------
# ``load_data_semseg`` 读取的是块数据，不是上面的 npy 整房文件。
S3DIS_INDOOR3D_HDF5_TRAIN = os.path.join(PC_DATA_ROOT, "indoor3d_sem_seg_hdf5_data")
S3DIS_INDOOR3D_HDF5_TEST = os.path.join(PC_DATA_ROOT, "indoor3d_sem_seg_hdf5_data_test")


def get_s3dis_openpoints_data_root(create_symlink: bool = False) -> str:
    """
    返回适用于 OpenPoints ``dataset/s3dis/s3dis.py`` 的 ``data_root``。

    若目录 ``<PC_DATA_ROOT>/s3dis_for_openpoints/raw`` 不存在或为空，且
    ``create_symlink=True``，则创建 ``raw -> s3dis_trainval`` 的符号链接（与
    ``scripts/prepare_s3dis_openpoints.sh`` 行为一致）。
    """
    root = os.environ.get("S3DIS_OPENPOINTS_ROOT", S3DIS_OPENPOINTS_DEFAULT)
    raw = os.path.join(root, "raw")
    if create_symlink and os.path.isdir(S3DIS_TRAINVAL_NPY):
        if not os.path.islink(raw) and not os.path.isdir(raw):
            os.makedirs(root, exist_ok=True)
        if not os.path.exists(raw) or (os.path.isdir(raw) and not any(os.scandir(raw))):
            if os.path.lexists(raw):
                if os.path.islink(raw):
                    os.unlink(raw)
                else:
                    raise FileExistsError(f"已存在非空路径且非链接: {raw}")
            os.symlink(S3DIS_TRAINVAL_NPY, raw, target_is_directory=True)
    return root


def pointkan_elite_modelnet40_glob(partition: str) -> str:
    """与 ``classification_ModelNet40/data.py`` 中 glob 模式一致，便于检查文件是否存在。"""
    return os.path.join(MODELNET40_PLY_HDF5_2048, f"ply_data_{partition}*.h5")
