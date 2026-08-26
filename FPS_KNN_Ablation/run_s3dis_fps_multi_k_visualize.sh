#!/usr/bin/env bash
# S3DIS：同一起点三次 FPS（1024/2048/4096），3D 三色叠加看重叠
set -euo pipefail
cd "$(dirname "$0")"

DATA_DIR="${DATA_DIR:-/workspace/VoxelNet/pc_data/s3dis_trainval}"
OUT_DIR="${OUT_DIR:-./s3dis_fps_multi_k_outputs}"
# 大场景可先设 MAX_POINTS=40000 加速；全点云设为空或 -1
MAX_POINTS="${MAX_POINTS:--1}"
FILE_INDEX="${FILE_INDEX:-0}"

ARGS=(--dataset s3dis --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" --file_index "$FILE_INDEX" --start_idx 0)
if [[ "$MAX_POINTS" != "-1" && -n "$MAX_POINTS" ]]; then
  ARGS+=(--max_points "$MAX_POINTS")
fi

python3 s3dis_fps_multi_k_visualize.py "${ARGS[@]}"
