#!/usr/bin/env bash
# 动态阈值 KNN 参数 tol、max_iter、init_threshold 网格搜索：探索对 KNN IoU / iters_used / thresholds 的影响
# 在 classification_ModelNet40 目录下执行: bash opt_fps/run_sweep_tol_max_iter.sh

set -e
cd "$(dirname "$0")/.."
echo "==> Sweep tol & max_iter & init_threshold (cwd: $(pwd))"

mkdir -p opt_fps/figures

# 是否在校准集上 sweep（与 eval_fps_radii 同套分层采样）：0=否（默认 test 取前 num_batches），1=是
USE_CALIBRATION="${USE_CALIBRATION:-0}"
# 加速：校准集上建议 batch_size=16、num_workers=2
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
[ "$USE_CALIBRATION" = "1" ] && BATCH_SIZE="${BATCH_SIZE:-16}" && NUM_WORKERS="${NUM_WORKERS:-2}"

if [ "$USE_CALIBRATION" = "1" ]; then
  echo "==> Using calibration set (train, samples_per_class=50), batch_size=$BATCH_SIZE, num_workers=$NUM_WORKERS"
  python opt_fps/eval_fps_knn_compare.py \
    --partition train \
    --use_calibration \
    --samples_per_class 50 \
    --num_batches 200 \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --K 512 \
    --k_neighbors 24 \
    --sweep_tol "0.01,0.05,0.1" \
    --sweep_max_iter "5,10,15,20" \
    --sweep_init_threshold "0.05,0.1,0.15,0.2" \
    --out_json opt_fps/figures/sweep_tol_max_iter_calibration.json \
    --sweep_out_dir opt_fps/figures
  echo "==> Done. Check opt_fps/figures/sweep_tol_max_iter_calibration.json and *.png"
else
  echo "==> Using partition (test) shuffle, first num_batches, batch_size=$BATCH_SIZE"
  python opt_fps/eval_fps_knn_compare.py \
    --partition test \
    --num_batches 50 \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --K 512 \
    --k_neighbors 24 \
    --sweep_tol "0.01,0.05,0.1" \
    --sweep_max_iter "5,10,15,20" \
    --sweep_init_threshold "0.05,0.1,0.15,0.2" \
    --out_json opt_fps/figures/sweep_tol_max_iter.json \
    --sweep_out_dir opt_fps/figures
  echo "==> Done. Check opt_fps/figures/sweep_tol_max_iter.json and opt_fps/figures/sweep_tol_max_iter_*.png"
fi
