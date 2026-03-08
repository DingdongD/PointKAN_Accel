#!/usr/bin/env bash
# KNN 粗筛+细筛 IoU 分析 — KITTI 点云
# 用法：
#   ./run_knn_coarse_fine_eval_kitti.sh
#   DATA_DIR=/path/to/velodyne MAX_SAMPLES=200 ./run_knn_coarse_fine_eval_kitti.sh
#   NUM_POINTS=2048 FPS_STEPS=512 OUT_JSON=./kitti_knn_iou.json ./run_knn_coarse_fine_eval_kitti.sh

set -e
cd "$(dirname "$0")"

DATA_DIR="${DATA_DIR:-/workspace/VoxelNet/KITTI_Mini/training/velodyne}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
# KITTI 原始尺度：ROI 后约 4 万–6.5 万点，默认 50000；显存不足可改小
NUM_POINTS="${NUM_POINTS:-50000}"
FPS_STEPS="${FPS_STEPS:-2048}"
K_NEIGHBORS="${K_NEIGHBORS:-32}"
MIN_CANDIDATES_RATIO="${MIN_CANDIDATES_RATIO:-1.2}"
OUT_JSON="${OUT_JSON:-./knn_coarse_fine_iou_results_kitti.json}"
DEVICE="${DEVICE:-cuda}"
# 大点数时默认 batch=1 避免 OOM
BATCH_SIZE="${BATCH_SIZE:-1}"
NO_ROI="${NO_ROI:-0}"

EXTRA_ARGS=(
  --data_dir "$DATA_DIR"
  --num_points "$NUM_POINTS"
  --fps_steps "$FPS_STEPS"
  --k_neighbors "$K_NEIGHBORS"
  --min_candidates_ratio "$MIN_CANDIDATES_RATIO"
  --device "$DEVICE"
  --out_json "$OUT_JSON"
  --batch_size "$BATCH_SIZE"
)
[[ "$MAX_SAMPLES" -gt 0 ]] && EXTRA_ARGS+=(--max_samples "$MAX_SAMPLES")
[[ "$NO_ROI" = "1" ]] && EXTRA_ARGS+=(--no_roi)

echo ">>> python run_knn_coarse_fine_eval_kitti.py ${EXTRA_ARGS[*]}"
exec python run_knn_coarse_fine_eval_kitti.py "${EXTRA_ARGS[@]}"
