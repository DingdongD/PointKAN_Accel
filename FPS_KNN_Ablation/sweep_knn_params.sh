#!/usr/bin/env bash
# 对 KITTI / NuScenes / Waymo 分别跑 KNN 超参数扫描（init_threshold 等），选最佳 IoU。
# 用法：
#   ./sweep_knn_params.sh              # 依次扫描三个数据集（默认 max_samples=30）
#   ./sweep_knn_params.sh waymo      # 只扫描 waymo
#   MAX_SAMPLES=20 ./sweep_knn_params.sh
#   INIT_THRESHOLD="0.05,0.1,0.2,0.3" ./sweep_knn_params.sh kitti

set -e
cd "$(dirname "$0")"

DATASETS="${1:-kitti nuscenes waymo}"
MAX_SAMPLES="${MAX_SAMPLES:-30}"
NUM_POINTS="${NUM_POINTS:-50000}"
FPS_STEPS="${FPS_STEPS:-2048}"
K_NEIGHBORS="${K_NEIGHBORS:-32}"
INIT_THRESHOLD="${INIT_THRESHOLD:-0.01,0.02,0.05,0.08,0.1,0.15,0.2,0.25,0.3,0.4,0.5}"
OUT_DIR="${OUT_DIR:-.}"

for ds in $DATASETS; do
  echo ""
  echo ">>> Sweep dataset: $ds (max_samples=$MAX_SAMPLES)"
  python sweep_knn_params.py \
    --dataset "$ds" \
    --max_samples "$MAX_SAMPLES" \
    --num_points "$NUM_POINTS" \
    --fps_steps "$FPS_STEPS" \
    --k_neighbors "$K_NEIGHBORS" \
    --init_threshold "$INIT_THRESHOLD" \
    --out_json "${OUT_DIR}/sweep_knn_${ds}.json"
done
echo ""
echo "Done. Check ${OUT_DIR}/sweep_knn_*.json for best params."
