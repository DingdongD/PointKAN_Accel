#!/usr/bin/env bash
# KITTI：同一起点三次 FPS（默认 256/512/1024），3D 三色叠加
set -euo pipefail
cd "$(dirname "$0")"

DATA_DIR="${DATA_DIR:-/workspace/VoxelNet/KITTI_Mini/training/velodyne}"
OUT_DIR="${OUT_DIR:-./kitti_fps_multi_k_outputs}"
MAX_POINTS="${MAX_POINTS:--1}"
FILE_INDEX="${FILE_INDEX:-0}"

ARGS=(--dataset kitti --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" --file_index "$FILE_INDEX" --start_idx 0)
if [[ "$MAX_POINTS" != "-1" && -n "$MAX_POINTS" ]]; then
  ARGS+=(--max_points "$MAX_POINTS")
fi

python3 kitti_fps_multi_k_visualize.py "${ARGS[@]}"
