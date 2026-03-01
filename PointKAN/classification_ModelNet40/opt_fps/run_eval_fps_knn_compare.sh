#!/usr/bin/env bash
# FPS-KNN 对比：vanilla vs adaptive（ModelNet40）
# 在 classification_ModelNet40 目录下执行: bash opt_fps/run_eval_fps_knn_compare.sh

set -e
cd "$(dirname "$0")/.."
echo "==> FPS-KNN compare (cwd: $(pwd))"

mkdir -p opt_fps/figures

# 是否在校准集上评估：0=否（默认 test 取前 num_batches），1=是（train 分层采样）
USE_CALIBRATION="${USE_CALIBRATION:-0}"
# 加速：批大小（校准集建议 16 或 32，GPU 批处理）；DataLoader 预取进程数（建议 2~4）
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
[ "$USE_CALIBRATION" = "1" ] && BATCH_SIZE="${BATCH_SIZE:-16}" && NUM_WORKERS="${NUM_WORKERS:-2}"

EXTRA_ARGS=()
if [ "$USE_CALIBRATION" = "1" ]; then
  echo "==> Using calibration set (train, samples_per_class=50), batch_size=$BATCH_SIZE, num_workers=$NUM_WORKERS"
  EXTRA_ARGS=(--partition train --use_calibration --samples_per_class 50)
fi

# 默认：50 样本、K=512, k_neighbors=24，结果写入 txt + JSON，前 3 个样本保存索引
python opt_fps/eval_fps_knn_compare.py \
  --partition test \
  --num_batches 50 \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --K 512 \
  --k_neighbors 24 \
  --fps_seed first \
  --out_txt opt_fps/figures/fps_knn_compare.txt \
  --out_json opt_fps/figures/fps_knn_compare.json \
  --save_indices_n 3 \
  "${EXTRA_ARGS[@]}"

echo "==> Done. Check opt_fps/figures/fps_knn_compare.txt"
