#!/usr/bin/env bash
# KNN 粗筛+细筛 IoU 分析 — Waymo 点云（KITTI 格式）
# 用法：
#   ./run_knn_coarse_fine_eval_waymo.sh
#   DATA_DIR=/path/to/velodyne MAX_SAMPLES=200 ./run_knn_coarse_fine_eval_waymo.sh
#   NUM_POINTS=40000 FPS_STEPS=512 OUT_JSON=./waymo_knn_iou.json ./run_knn_coarse_fine_eval_waymo.sh
# CPU 测试（耗时用 time.perf_counter 测量）：
#   DEVICE=cpu ./run_knn_coarse_fine_eval_waymo.sh
#   DEVICE=cpu OUT_JSON=./waymo_knn_iou_cpu.json ./run_knn_coarse_fine_eval_waymo.sh

set -e
cd "$(dirname "$0")"

DATA_DIR="${DATA_DIR:-/workspace/VoxelNet/Waymo_Mini/Waymo_mini/kitti_format/training/velodyne}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
NUM_POINTS="${NUM_POINTS:-5000000}"
FPS_STEPS="${FPS_STEPS:-512}"
K_NEIGHBORS="${K_NEIGHBORS:-32}"
MIN_CANDIDATES_RATIO="${MIN_CANDIDATES_RATIO:-1.1}"
DEVICE="${DEVICE:-cuda:3}"
# 未显式设置 OUT_JSON 时按 device 区分保存，避免 CPU/GPU 结果互相覆盖
DEFAULT_JSON_BASE="./knn_coarse_fine_iou_results_waymo"
if [[ "$DEVICE" == "cpu" ]]; then
  OUT_JSON="${OUT_JSON:-${DEFAULT_JSON_BASE}_cpu.json}"
else
  OUT_JSON="${OUT_JSON:-${DEFAULT_JSON_BASE}.json}"
fi
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

echo ">>> python run_knn_coarse_fine_eval_waymo.py ${EXTRA_ARGS[*]}"
exec python run_knn_coarse_fine_eval_waymo.py "${EXTRA_ARGS[@]}"
