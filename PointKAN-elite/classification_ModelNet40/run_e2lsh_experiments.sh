#!/usr/bin/env bash
# E2LSH 实验：参数扫 (L, w, candidate_mode) + E2LSH+backend 对比
# 日志与 JSON 写入 runs/e2lsh_experiments/
#
# 用法：
#   全量：     ./run_e2lsh_experiments.sh
#   快速：     MAX_SAMPLES=25 N_REF=512 ./run_e2lsh_experiments.sh
#   只对比：   ./run_e2lsh_experiments.sh --skip_sweep
#   改 E2LSH： E2LSH_K=4 E2LSH_TABLE_SIZE=4096 ./run_e2lsh_experiments.sh

set -e
cd "$(dirname "$0")"

# 数据与评估
PARTITION="${PARTITION:-test}"
NUM_POINTS="${NUM_POINTS:-1024}"
N_REF="${N_REF:-1024}"
N_QUERIES="${N_QUERIES:-0}"
K="${K:-16}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"
SEED="${SEED:-42}"

# 量化 (dataset_range)
QUANT_MODE="${QUANT_MODE:-dataset_range}"
COORD_MIN="${COORD_MIN:--1.0}"
COORD_MAX="${COORD_MAX:-1.0}"
QUANT_BITS="${QUANT_BITS:-16}"

# E2LSH two-level 参数
E2LSH_SEED="${E2LSH_SEED:-42}"
E2LSH_K="${E2LSH_K:-4}"
E2LSH_TABLE_SIZE="${E2LSH_TABLE_SIZE:-4096}"
E2LSH_C="${E2LSH_C:-4294967291}"

# Part2 固定参数（backend 对比时使用的 E2LSH 配置；intersection 在 two-level 下极严，默认 union 以便 Part2 有可读 recall）
PICK_L="${PICK_L:-8}"
PICK_W="${PICK_W:-0.5}"
PICK_CANDIDATE_MODE="${PICK_CANDIDATE_MODE:-union}"

python run_e2lsh_experiments.py \
  --partition "$PARTITION" \
  --num_points "$NUM_POINTS" \
  --n_ref "$N_REF" \
  --n_queries "$N_QUERIES" \
  --k "$K" \
  --max_samples "$MAX_SAMPLES" \
  --seed "$SEED" \
  --quant_mode "$QUANT_MODE" \
  --coord_min "$COORD_MIN" \
  --coord_max "$COORD_MAX" \
  --quant_bits "$QUANT_BITS" \
  --e2lsh_seed "$E2LSH_SEED" \
  --e2lsh_K "$E2LSH_K" \
  --e2lsh_table_size "$E2LSH_TABLE_SIZE" \
  --e2lsh_C "$E2LSH_C" \
  --pick_L "$PICK_L" \
  --pick_w "$PICK_W" \
  --pick_candidate_mode "$PICK_CANDIDATE_MODE" \
  "$@"
