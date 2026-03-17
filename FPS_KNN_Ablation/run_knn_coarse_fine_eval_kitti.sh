#!/usr/bin/env bash
# KNN 粗筛+细筛 IoU 分析 — KITTI 点云
# 用法：
#   ./run_knn_coarse_fine_eval_kitti.sh
#   DATA_DIR=/path/to/velodyne MAX_SAMPLES=200 ./run_knn_coarse_fine_eval_kitti.sh
#   NUM_POINTS=2048 FPS_STEPS=512 OUT_JSON=./kitti_knn_iou.json ./run_knn_coarse_fine_eval_kitti.sh
# CPU 测试（耗时用 time.perf_counter 测量）：
#   DEVICE=cpu ./run_knn_coarse_fine_eval_kitti.sh
#   DEVICE=cpu OUT_JSON=./kitti_knn_iou_cpu.json ./run_knn_coarse_fine_eval_kitti.sh

set -e
cd "$(dirname "$0")"

# 只使用一张卡，避免占多卡：仅当前脚本可见的 GPU 为指定卡号，程序内用 cuda:0
# 改卡号示例：CUDA_VISIBLE_DEVICES=3 ./run_knn_coarse_fine_eval_kitti.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
DEVICE="${DEVICE:-cuda:0}"

DATA_DIR="${DATA_DIR:-/workspace/VoxelNet/KITTI_Mini/training/velodyne}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
# KITTI 原始尺度：ROI 后约 4 万–6.5 万点，默认 50000。加速可设 NUM_POINTS=4096 FPS_STEPS=512 MAX_SAMPLES=500
NUM_POINTS="${NUM_POINTS:-50000}"
FPS_STEPS="${FPS_STEPS:-512}"
K_NEIGHBORS="${K_NEIGHBORS:-32}"
MIN_CANDIDATES_RATIO="${MIN_CANDIDATES_RATIO:-1.2}"
# 未显式设置 OUT_JSON 时按 device 区分保存，避免 CPU/GPU 结果互相覆盖
DEFAULT_JSON_BASE="./knn_coarse_fine_iou_results_kitti"
if [[ "$DEVICE" == "cpu" ]]; then
  OUT_JSON="${OUT_JSON:-${DEFAULT_JSON_BASE}_cpu.json}"
else
  OUT_JSON="${OUT_JSON:-${DEFAULT_JSON_BASE}.json}"
fi
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
