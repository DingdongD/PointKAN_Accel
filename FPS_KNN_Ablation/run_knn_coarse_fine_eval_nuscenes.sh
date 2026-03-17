#!/usr/bin/env bash
# KNN 粗筛+细筛 IoU 分析 — NuScenes 点云
# 用法：./run_knn_coarse_fine_eval_nuscenes.sh
# CPU 测试（耗时用 time.perf_counter 测量）：DEVICE=cpu ./run_knn_coarse_fine_eval_nuscenes.sh
set -e
cd "$(dirname "$0")"
DATA_DIR="${DATA_DIR:-/workspace/VoxelNet/Nuscenes_Mini/sweeps/LIDAR_TOP}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
NUM_POINTS="${NUM_POINTS:-5000000}"
FPS_STEPS="${FPS_STEPS:-512}"
K_NEIGHBORS="${K_NEIGHBORS:-32}"
MIN_CANDIDATES_RATIO="${MIN_CANDIDATES_RATIO:-1.2}"
DEVICE="${DEVICE:-cuda:3}"
# 未显式设置 OUT_JSON 时按 device 区分保存，避免 CPU/GPU 结果互相覆盖
DEFAULT_JSON_BASE="./knn_coarse_fine_iou_results_nuscenes"
if [[ "$DEVICE" == "cpu" ]]; then
  OUT_JSON="${OUT_JSON:-${DEFAULT_JSON_BASE}_cpu.json}"
else
  OUT_JSON="${OUT_JSON:-${DEFAULT_JSON_BASE}.json}"
fi
BATCH_SIZE="${BATCH_SIZE:-16}"
NO_ROI="${NO_ROI:-0}"
EXTRA_ARGS=(--data_dir "$DATA_DIR" --num_points "$NUM_POINTS" --fps_steps "$FPS_STEPS" --k_neighbors "$K_NEIGHBORS" --min_candidates_ratio "$MIN_CANDIDATES_RATIO" --device "$DEVICE" --out_json "$OUT_JSON" --batch_size "$BATCH_SIZE")
[[ "$MAX_SAMPLES" -gt 0 ]] && EXTRA_ARGS+=(--max_samples "$MAX_SAMPLES")
[[ "$NO_ROI" = "1" ]] && EXTRA_ARGS+=(--no_roi)
echo ">>> python run_knn_coarse_fine_eval_nuscenes.py ${EXTRA_ARGS[*]}"
exec python run_knn_coarse_fine_eval_nuscenes.py "${EXTRA_ARGS[@]}"
