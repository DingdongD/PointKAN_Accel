#!/usr/bin/env bash
# KNN 粗筛+细筛 IoU 分析 — NuScenes 点云
set -e
cd "$(dirname "$0")"
DATA_DIR="${DATA_DIR:-/workspace/VoxelNet/Nuscenes_Mini/sweeps/LIDAR_TOP}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
NUM_POINTS="${NUM_POINTS:-50000}"
FPS_STEPS="${FPS_STEPS:-2048}"
K_NEIGHBORS="${K_NEIGHBORS:-32}"
MIN_CANDIDATES_RATIO="${MIN_CANDIDATES_RATIO:-1.2}"
OUT_JSON="${OUT_JSON:-./knn_coarse_fine_iou_results_nuscenes.json}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NO_ROI="${NO_ROI:-0}"
EXTRA_ARGS=(--data_dir "$DATA_DIR" --num_points "$NUM_POINTS" --fps_steps "$FPS_STEPS" --k_neighbors "$K_NEIGHBORS" --min_candidates_ratio "$MIN_CANDIDATES_RATIO" --device "$DEVICE" --out_json "$OUT_JSON" --batch_size "$BATCH_SIZE")
[[ "$MAX_SAMPLES" -gt 0 ]] && EXTRA_ARGS+=(--max_samples "$MAX_SAMPLES")
[[ "$NO_ROI" = "1" ]] && EXTRA_ARGS+=(--no_roi)
echo ">>> python run_knn_coarse_fine_eval_nuscenes.py ${EXTRA_ARGS[*]}"
exec python run_knn_coarse_fine_eval_nuscenes.py "${EXTRA_ARGS[@]}"
