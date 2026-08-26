#!/usr/bin/env bash
set -euo pipefail

# 一键跑多数据集 KNN 粗筛+细筛对比
# 默认依次跑: modelnet40 partseg s3dis scannet

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEVICE="${DEVICE:-cuda}"
FPS_STEPS="${FPS_STEPS:-512}"
K_NEIGHBORS="${K_NEIGHBORS:-24}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"   # 0=全量
NUM_POINTS_MODELNET40="${NUM_POINTS_MODELNET40:-1024}"
NUM_POINTS_PARTSEG="${NUM_POINTS_PARTSEG:-2048}"
NUM_POINTS_S3DIS="${NUM_POINTS_S3DIS:-50000}"
NUM_POINTS_SCANNET="${NUM_POINTS_SCANNET:-1024}"

DATASETS=("${@}")
if [ ${#DATASETS[@]} -eq 0 ]; then
  DATASETS=("modelnet40" "partseg" "s3dis" "scannet")
fi

echo ">>> datasets: ${DATASETS[*]}"
echo ">>> device=$DEVICE fps_steps=$FPS_STEPS k_neighbors=$K_NEIGHBORS batch_size=$BATCH_SIZE max_samples=$MAX_SAMPLES"

for ds in "${DATASETS[@]}"; do
  case "$ds" in
    modelnet40) NP="$NUM_POINTS_MODELNET40" ;;
    partseg) NP="$NUM_POINTS_PARTSEG" ;;
    s3dis) NP="$NUM_POINTS_S3DIS" ;;
    scannet) NP="$NUM_POINTS_SCANNET" ;;
    *) echo "Unsupported dataset: $ds"; exit 1 ;;
  esac

  OUT_JSON="./knn_coarse_fine_iou_results_${ds}.json"
  echo "============================================================"
  echo ">>> Running dataset=$ds num_points=$NP out=$OUT_JSON"
  python run_knn_coarse_fine_eval_multidataset.py \
    --dataset "$ds" \
    --num_points "$NP" \
    --fps_steps "$FPS_STEPS" \
    --k_neighbors "$K_NEIGHBORS" \
    --batch_size "$BATCH_SIZE" \
    --max_samples "$MAX_SAMPLES" \
    --device "$DEVICE" \
    --out_json "$OUT_JSON"
done

echo "============================================================"
echo "All done."
