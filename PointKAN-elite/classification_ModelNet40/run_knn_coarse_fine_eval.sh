#!/usr/bin/env bash
# KNN 粗筛+细筛 IoU 分析与 ModelNet40 推理评估
# 用法：
#   仅 IoU/radii 分析（不跑分类推理）：
#     ./run_knn_coarse_fine_eval.sh
#   带分类推理（需存在 checkpoint）：
#     RUN_INFERENCE=1 ./run_knn_coarse_fine_eval.sh
#   指定检查点、样本数、输出 JSON：
#     CHECKPOINT=checkpoints/pointKANElite-xxx RUN_INFERENCE=1 MAX_SAMPLES=500 ./run_knn_coarse_fine_eval.sh

set -e
cd "$(dirname "$0")"

# 默认使用第 3 张 GPU（可用 GPU_ID 覆盖）
GPU_ID="${GPU_ID:-3}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

RUN_INFERENCE="${RUN_INFERENCE:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
FPS_STEPS="${FPS_STEPS:-512}"
K_NEIGHBORS="${K_NEIGHBORS:-24}"
MIN_CANDIDATES_RATIO="${MIN_CANDIDATES_RATIO:-1.1}"
OUT_JSON="${OUT_JSON:-./knn_coarse_fine_iou_results.json}"
CHECKPOINT="${CHECKPOINT:-checkpoints/pointKANElite-pointkan_elite_exp-42}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-64}"

# blockwise coarse-fine（可选）
RUN_BLOCKWISE="${RUN_BLOCKWISE:-0}"          # 1 开启 blockwise 评估
BLOCK_SIZE="${BLOCK_SIZE:-256}"              # blockwise 分块大小
COLLECT_BLOCK_STATS="${COLLECT_BLOCK_STATS:-0}"  # 1 统计每个 block 候选数/最终 masked 数（显存更大）

EXTRA_ARGS=(
  --fps_steps "$FPS_STEPS"
  --k_neighbors "$K_NEIGHBORS"
  --min_candidates_ratio "$MIN_CANDIDATES_RATIO"
  --device "$DEVICE"
  --out_json "$OUT_JSON"
  --batch_size "$BATCH_SIZE"
)
[[ "$MAX_SAMPLES" -gt 0 ]] && EXTRA_ARGS+=(--max_samples "$MAX_SAMPLES")
[[ "$RUN_INFERENCE" = "1" ]] && EXTRA_ARGS+=(--run_inference --checkpoint "$CHECKPOINT")
[[ "$RUN_INFERENCE" != "1" ]] && EXTRA_ARGS+=(--no_inference)
[[ "$RUN_BLOCKWISE" = "1" ]] && EXTRA_ARGS+=(--run_blockwise --block_size "$BLOCK_SIZE")
[[ "$COLLECT_BLOCK_STATS" = "1" ]] && EXTRA_ARGS+=(--collect_block_stats)

echo ">>> python run_knn_coarse_fine_eval.py ${EXTRA_ARGS[*]}"
exec python run_knn_coarse_fine_eval.py "${EXTRA_ARGS[@]}"
