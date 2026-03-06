#!/usr/bin/env bash
# 使用真实 checkpoint 与 ModelNet40 数据运行 PPA 仿真（真实前向 + 按样本串行仿真）
#
# 使用（在 PointKAN-pytorch 根目录）:
#   chmod +x run_sim_with_real_eval.sh && ./run_sim_with_real_eval.sh
#
# 环境变量（可选）:
#   CKPT              检查点路径，默认 PointKAN-elite/.../pointKANElite-pointkan_elite_exp-42
#   NUM_POINTS        点云点数，默认 1024
#   BATCH_SIZE        测试 batch 大小，默认 32
#   MAX_BATCHES       限制 batch 数（快速测试），例如 MAX_BATCHES=2
#   USE_REAL_DATA     1=真实数据+MNSIM（默认），0=仅结构仿真
#   MNSIM_FULL        1=启用 MNSIM 全 NetStruct（默认），0=不启用
#   RUN_REAL_INFERENCE 1=先跑 inference_and_kan_analysis 再仿真，0=不跑（默认）
#   DEVICE            cuda（默认）或 cpu
#   SIM_LOG_DIR       日志目录，默认 sim_log
#   OUTPUT_JSON       报告 JSON 文件名，默认 report.json
#   SHOW_SHAPES       1=打印 LayerDef 与真实前向每层 conv/linear 的 shape 与 indata/rdata（默认 0）
#   BN_EN             1=启用 BN 融合（默认 1，BN 不单独计入）；0=关闭融合（保留 BN 事件）
#   QD_EN             1=启用量化/反量化事件（默认 0）
#   EXPORT_MEM_TXN    1=导出内存事务序列文件（默认 1）
#   SIM_RESULTS_DIR   PPA 报告 JSON 与 breakdown 图表输出目录，默认 sim_results
#
# 示例：快速跑 2 个 batch
#   MAX_BATCHES=2 ./run_sim_with_real_eval.sh
# 示例：先跑真实推理再仿真
#   RUN_REAL_INFERENCE=1 ./run_sim_with_real_eval.sh
# 示例：打印 LayerDef 与真实前向每层 shape/indata/rdata（核对 36 vs 20 层）
#   SHOW_SHAPES=1 ./run_sim_with_real_eval.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 检查点：目录（含 best_checkpoint.pth）或 .pth 路径，相对 classification_ModelNet40 或绝对路径
CKPT="${CKPT:-PointKAN-elite/classification_ModelNet40/checkpoints/pointKANElite-pointkan_elite_exp-42}"
NUM_POINTS="${NUM_POINTS:-1024}"
NUM_CLASSES="${NUM_CLASSES:-40}"
BATCH_SIZE="${BATCH_SIZE:-1}"
# 限 batch 数做快速测试（不设则跑全量测试集）
MAX_BATCHES="${MAX_BATCHES:-}"
SIM_LOG_DIR="${SIM_LOG_DIR:-sim_log}"
OUTPUT_JSON="${OUTPUT_JSON:-report.json}"
SIM_RESULTS_DIR="${SIM_RESULTS_DIR:-sim_results}"
DEVICE="${DEVICE:-cuda}"
ACCELERATOR="${ACCELERATOR:-baseline}"

# 默认：真实数据模式 + MNSIM 全 NetStruct（按层真实 indata/rdata 驱动）
USE_REAL_DATA="${USE_REAL_DATA:-1}"
MNSIM_FULL="${MNSIM_FULL:-1}"
RUN_REAL_INFERENCE="${RUN_REAL_INFERENCE:-0}"
SHOW_SHAPES="${SHOW_SHAPES:-0}"
BN_EN="${BN_EN:-1}"
QD_EN="${QD_EN:-0}"
EXPORT_MEM_TXN="${EXPORT_MEM_TXN:-1}"

build_args() {
    local args=(
        --checkpoint "$CKPT"
        --num_points "$NUM_POINTS"
        --num_classes "$NUM_CLASSES"
        --sim_log_dir "$SIM_LOG_DIR"
        --accelerator "$ACCELERATOR"
        --device "$DEVICE"
    )
    [[ "$USE_REAL_DATA" == "1" ]] && args+=(--use_real_data --batch_size "$BATCH_SIZE")
    [[ -n "$MAX_BATCHES" ]] && args+=(--max_batches "$MAX_BATCHES")
    [[ "$MNSIM_FULL" == "1" ]] && args+=(--mnsim_full)
    [[ "$RUN_REAL_INFERENCE" == "1" ]] && args+=(--run_real_inference --no_vis)
    [[ -n "$OUTPUT_JSON" ]] && args+=(--output "$OUTPUT_JSON")
    [[ "$SHOW_SHAPES" == "1" ]] && args+=(--show_shapes)
    args+=(--bn_en "$BN_EN")
    args+=(--qd_en "$QD_EN")
    [[ "$EXPORT_MEM_TXN" == "1" ]] && args+=(--export_memory_transactions)
    args+=(--sim_results_dir "$SIM_RESULTS_DIR")
    printf '%s\n' "${args[@]}"
}

echo "========== run_sim_with_real_eval =========="
echo "  CKPT=$CKPT"
echo "  USE_REAL_DATA=$USE_REAL_DATA MNSIM_FULL=$MNSIM_FULL RUN_REAL_INFERENCE=$RUN_REAL_INFERENCE"
echo "  BATCH_SIZE=$BATCH_SIZE MAX_BATCHES=${MAX_BATCHES:-all} DEVICE=$DEVICE"
echo "  SIM_LOG_DIR=$SIM_LOG_DIR SIM_RESULTS_DIR=$SIM_RESULTS_DIR OUTPUT=$OUTPUT_JSON"
echo "==========================================="

python -m pointkan_sim.run_sim_with_real_eval $(build_args)
