#!/usr/bin/env bash
# 为 OpenPoints / FractalCloud 风格 S3DIS Dataset 准备 data_root：raw -> s3dis_trainval
# 依赖：/workspace/VoxelNet/pc_data/s3dis_trainval 下已有 Area_*_*.npy（与 PointKAN 环境一致）
set -euo pipefail
PC_DATA_ROOT="${PC_DATA_ROOT:-/workspace/VoxelNet/pc_data}"
OUT_ROOT="${S3DIS_OPENPOINTS_ROOT:-$PC_DATA_ROOT/s3dis_for_openpoints}"
TRAINVAL="${S3DIS_TRAINVAL_NPY:-$PC_DATA_ROOT/s3dis_trainval}"

if [[ ! -d "$TRAINVAL" ]]; then
  echo "错误: 未找到 S3DIS npy 目录: $TRAINVAL"
  exit 1
fi

mkdir -p "$OUT_ROOT"
if [[ -e "$OUT_ROOT/raw" || -L "$OUT_ROOT/raw" ]]; then
  rm -rf "$OUT_ROOT/raw"
fi
ln -sfn "$TRAINVAL" "$OUT_ROOT/raw"
echo "OpenPoints S3DIS data_root 已就绪: $OUT_ROOT"
echo "  (raw -> $TRAINVAL)"
echo "训练/配置中使用: data_root=$OUT_ROOT"
