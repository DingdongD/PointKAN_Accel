#!/usr/bin/env bash
# PointKAN-Elite ModelNet40 每层网络 shape 分析
# 参考 inference.sh 的环境变量与默认值，使用 ModelNet40 数据跑前向并输出每层 shape 到日志。
#
# 使用前请在本目录下执行：chmod +x run_shape_analysis.sh
#
# 环境变量（与 inference.sh 对齐）：
#   CHECKPOINT   - 检查点目录或 best_checkpoint.pth 路径
#   NUM_POINTS   - 点云点数 (默认 1024)
#   BATCH_SIZE   - 批大小 (默认 32)
#   WORKERS      - DataLoader workers (默认 4)
#   DEVICE       - cuda | cpu
#   SEED         - 随机种子 (默认 42)
#   PARTITION    - ModelNet40 分区: train | test (默认 test)
#   NUM_BATCHES  - 用于统计的 batch 数 (默认 1)
#   LOG          - 日志文件路径，默认 shape_analysis_<timestamp>.log
#   CSV          - 若非空，则额外输出 CSV 到该路径
#   INCLUDE_WEIGHTS - 非空则输出权重 shape（默认不输出，避免日志过大）
#   MAX_WEIGHT_ITEMS - 每层最多输出多少个权重条目（0 不限制；建议 20/50）
#
# 示例：
#   ./run_shape_analysis.sh
#   CHECKPOINT=checkpoints/pointKANElite-xxx-42 ./run_shape_analysis.sh
#   CSV=logs/shape.csv INCLUDE_WEIGHTS=1 MAX_WEIGHT_ITEMS=50 ./run_shape_analysis.sh

set -e
cd "$(dirname "$0")"

CHECKPOINT="${CHECKPOINT:-checkpoints/pointKANElite-pointkan_elite_exp-42}"
NUM_POINTS="${NUM_POINTS:-1024}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WORKERS="${WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
PARTITION="${PARTITION:-test}"
NUM_BATCHES="${NUM_BATCHES:-1}"
LOG="${LOG:-}"
CSV="${CSV:-}"
INCLUDE_WEIGHTS="${INCLUDE_WEIGHTS:-}"
MAX_WEIGHT_ITEMS="${MAX_WEIGHT_ITEMS:-0}"

EXTRA_ARGS=("${@}")

if [[ " ${EXTRA_ARGS[*]} " == *" --help "* ]]; then
    python run_shape_analysis.py --help
    exit 0
fi

args=(
    --checkpoint "$CHECKPOINT"
    --num_points "$NUM_POINTS"
    --batch_size "$BATCH_SIZE"
    --workers "$WORKERS"
    --device "$DEVICE"
    --seed "$SEED"
    --partition "$PARTITION"
    --num_batches "$NUM_BATCHES"
    --max_weight_items "$MAX_WEIGHT_ITEMS"
)
[[ -n "$LOG" ]] && args+=(--log "$LOG")
[[ -n "$CSV" ]] && args+=(--csv "$CSV")
[[ -n "$INCLUDE_WEIGHTS" ]] && args+=(--include_weights)

echo ">>> 运行: python run_shape_analysis.py ${args[*]} ${EXTRA_ARGS[*]}"
python run_shape_analysis.py "${args[@]}" "${EXTRA_ARGS[@]}"
