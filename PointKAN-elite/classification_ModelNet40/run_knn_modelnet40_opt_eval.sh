#!/usr/bin/env bash
# ModelNet40 上评估 Float/Int/Strict-Safe/FastNN 四类 KNN，输出三组对比：
#   A. Float Vanilla vs Int Vanilla   -> 量化损失
#   B. Int Vanilla vs Safe Strict      -> strict safe 是否严格无损
#   C. Float Vanilla vs FastNN         -> 近似预比较器损失
#
# 用法：
#   默认（三组对比）：     ./run_knn_modelnet40_opt_eval.sh
#   E2LSH 模式：          MODE=e2lsh ./run_knn_modelnet40_opt_eval.sh
#   覆盖参数：            MAX_SAMPLES=20 N_REF=1024 E2LSH_L=8 E2LSH_W=0.2 ./run_knn_modelnet40_opt_eval.sh

set -e
cd "$(dirname "$0")"

PARTITION="${PARTITION:-test}"
NUM_POINTS="${NUM_POINTS:-1024}"
N_REF="${N_REF:-512}"
N_QUERIES="${N_QUERIES:-0}"
K="${K:-16}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"
SEED="${SEED:-42}"
# dataset_range: 按数据集空间 [COORD_MIN, COORD_MAX] 与 QUANT_BITS 量化（ModelNet40 默认 -1, 1, 16）
QUANT_MODE="${QUANT_MODE:-dataset_range}"
COORD_SCALE="${COORD_SCALE:-1000.0}"
COORD_MIN="${COORD_MIN:--1.0}"
COORD_MAX="${COORD_MAX:-1.0}"
QUANT_BITS="${QUANT_BITS:-16}"

# E2LSH 模式（MODE=e2lsh 时生效）
MODE="${MODE:-default}"
E2LSH_L="${E2LSH_L:-8}"
E2LSH_K="${E2LSH_K:-4}"
E2LSH_W="${E2LSH_W:-0.2}"
E2LSH_TABLE_SIZE="${E2LSH_TABLE_SIZE:-4096}"
E2LSH_C="${E2LSH_C:-4294967291}"
E2LSH_SEED="${E2LSH_SEED:-42}"
E2LSH_CANDIDATE_MODE="${E2LSH_CANDIDATE_MODE:-intersection}"
E2LSH_MIN_HITS="${E2LSH_MIN_HITS:-2}"
E2LSH_SEARCH_MODE="${E2LSH_SEARCH_MODE:-int_vanilla}"
E2LSH_GT_MODE="${E2LSH_GT_MODE:-float_vanilla}"

python run_knn_modelnet40_opt_eval.py \
  --partition "$PARTITION" \
  --num_points "$NUM_POINTS" \
  --n_ref "$N_REF" \
  --n_queries "$N_QUERIES" \
  --k "$K" \
  --max_samples "$MAX_SAMPLES" \
  --seed "$SEED" \
  --quant_mode "$QUANT_MODE" \
  --coord_scale "$COORD_SCALE" \
  --coord_min "$COORD_MIN" \
  --coord_max "$COORD_MAX" \
  --quant_bits "$QUANT_BITS" \
  --mode "$MODE" \
  --e2lsh_L "$E2LSH_L" \
  --e2lsh_K "$E2LSH_K" \
  --e2lsh_w "$E2LSH_W" \
  --e2lsh_table_size "$E2LSH_TABLE_SIZE" \
  --e2lsh_C "$E2LSH_C" \
  --e2lsh_seed "$E2LSH_SEED" \
  --e2lsh_candidate_mode "$E2LSH_CANDIDATE_MODE" \
  --e2lsh_min_hits "$E2LSH_MIN_HITS" \
  --e2lsh_search_mode "$E2LSH_SEARCH_MODE" \
  --e2lsh_gt_mode "$E2LSH_GT_MODE" \
  "$@"
