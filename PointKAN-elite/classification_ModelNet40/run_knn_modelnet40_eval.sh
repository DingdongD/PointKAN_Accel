#!/usr/bin/env bash
# ModelNet40：同时跑 Vanilla / Safe / FastNN 三种 KNN，输出对比结果
# 用法：
#   默认（三种模式对比）：  ./run_knn_modelnet40_eval.sh
#   单模式：               MODE=safe ./run_knn_modelnet40_eval.sh  或  MODE=fastnn
#   参数覆盖：             N_REF=256 K=16 MAX_SAMPLES=100 ./run_knn_modelnet40_eval.sh

set -e
cd "$(dirname "$0")"

PARTITION="${PARTITION:-test}"
NUM_POINTS="${NUM_POINTS:-1024}"
N_REF="${N_REF:-512}"
N_QUERIES="${N_QUERIES:-0}"
K="${K:-16}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SEED="${SEED:-42}"
# all = 同时跑 Vanilla / Safe / FastNN 并输出对比表；safe / fastnn = 仅跑该模式

python run_knn_modelnet40_eval.py \
  --partition "$PARTITION" \
  --num_points "$NUM_POINTS" \
  --n_ref "$N_REF" \
  --n_queries "$N_QUERIES" \
  --k "$K" \
  --max_samples "$MAX_SAMPLES" \
  --seed "$SEED" \
  --mode "$MODE" \
  "$@"
