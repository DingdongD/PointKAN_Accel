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
#   USE_REAL_DATA     1=真实数据+MNSIM/NeuroSim（默认），0=仅结构仿真
#   MNSIM_FULL        1=启用 MNSIM 全 NetStruct（默认），0=不启用
#   NEUROSIM_FULL     1=启用 NeuroSim 全网络仿真（DNN_NeuroSim main），0=不启用（默认）；与 MNSIM_FULL 二选一时优先 NeuroSim
#   NEUROSIM_ROOT    NeuroSim 根目录（Inference_pytorch），默认 /workspace/DNN_NeuroSim_V1.4/Inference_pytorch
#   NEUROSIM_CLK_FREQ  NeuroSim 工作频率(Hz)，由脚本传入、可调，不限于 100M（如 100000000=100MHz、200000000=200MHz）；不设则用 NeuroSim 关键路径频率
#   NEUROSIM_TECHNODE  NeuroSim 工艺节点(nm)，如 32；仅支持 130/90/65/45/32/22/14/10/7
#   LOGIC_CLK_FREQ    数字逻辑时钟频率(Hz)，接口可配、默认 1 GHz（如 1000000000）；不设则 hw_config 使用默认 1 GHz
#   RUN_REAL_INFERENCE 1=先跑 inference_and_kan_analysis 再仿真，0=不跑（默认）
#   DEVICE            cuda（默认）或 cpu
#   SIM_LOG_DIR       日志目录，默认 sim_log
#   OUTPUT_JSON       报告 JSON 文件名，默认 report.json
#   SHOW_SHAPES       1=打印 LayerDef 与真实前向每层 conv/linear 的 shape 与 indata/rdata（默认 0）
#   LOGIC_PAR_LAYERNORM / LOGIC_PAR_RELU / LOGIC_PAR_SPOOL / LOGIC_PAR_QUANT_DEQUANT  逻辑电路各模块并行度（≥1，不设则用默认）
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
# 示例：全网络 NeuroSim 仿真（真实数据 + 打印 shape + BN 不融合 + 量化 + 导出内存事务）
#   SHOW_SHAPES=1 BN_EN=0 QD_EN=1 EXPORT_MEM_TXN=1 NEUROSIM_FULL=1 MNSIM_FULL=0 ./run_sim_with_real_eval.sh
# 示例：NeuroSim 100MHz + 32nm 工艺
#   NEUROSIM_FULL=1 NEUROSIM_CLK_FREQ=100000000 NEUROSIM_TECHNODE=32 ./run_sim_with_real_eval.sh

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
NEUROSIM_FULL="${NEUROSIM_FULL:-0}"
NEUROSIM_ROOT="${NEUROSIM_ROOT:-}"
NEUROSIM_CLK_FREQ="${NEUROSIM_CLK_FREQ:-}"
NEUROSIM_TECHNODE="${NEUROSIM_TECHNODE:-32}"
LOGIC_CLK_FREQ="${LOGIC_CLK_FREQ:-}"
RUN_REAL_INFERENCE="${RUN_REAL_INFERENCE:-0}"
SHOW_SHAPES="${SHOW_SHAPES:-0}"
LOGIC_PAR_LAYERNORM="${LOGIC_PAR_LAYERNORM:-}"
LOGIC_PAR_RELU="${LOGIC_PAR_RELU:-}"
LOGIC_PAR_SPOOL="${LOGIC_PAR_SPOOL:-}"
LOGIC_PAR_QUANT_DEQUANT="${LOGIC_PAR_QUANT_DEQUANT:-}"
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
    [[ "$NEUROSIM_FULL" == "1" ]] && args+=(--neurosim_full)
    [[ -n "$NEUROSIM_ROOT" ]] && args+=(--neurosim_root "$NEUROSIM_ROOT")
    [[ -n "$NEUROSIM_CLK_FREQ" ]] && args+=(--neurosim_clk_freq "$NEUROSIM_CLK_FREQ")
    [[ -n "$NEUROSIM_TECHNODE" ]] && args+=(--neurosim_technode "$NEUROSIM_TECHNODE")
    [[ -n "$LOGIC_CLK_FREQ" ]] && args+=(--logic_clk_freq "$LOGIC_CLK_FREQ")
    [[ "$RUN_REAL_INFERENCE" == "1" ]] && args+=(--run_real_inference --no_vis)
    [[ -n "$OUTPUT_JSON" ]] && args+=(--output "$OUTPUT_JSON")
    [[ "$SHOW_SHAPES" == "1" ]] && args+=(--show_shapes)
    [[ -n "$LOGIC_PAR_LAYERNORM" ]] && args+=(--logic_par_layernorm "$LOGIC_PAR_LAYERNORM")
    [[ -n "$LOGIC_PAR_RELU" ]] && args+=(--logic_par_relu "$LOGIC_PAR_RELU")
    [[ -n "$LOGIC_PAR_SPOOL" ]] && args+=(--logic_par_spool "$LOGIC_PAR_SPOOL")
    [[ -n "$LOGIC_PAR_QUANT_DEQUANT" ]] && args+=(--logic_par_quant_dequant "$LOGIC_PAR_QUANT_DEQUANT")
    args+=(--bn_en "$BN_EN")
    args+=(--qd_en "$QD_EN")
    [[ "$EXPORT_MEM_TXN" == "1" ]] && args+=(--export_memory_transactions)
    args+=(--sim_results_dir "$SIM_RESULTS_DIR")
    printf '%s\n' "${args[@]}"
}

echo "========== run_sim_with_real_eval =========="
echo "  CKPT=$CKPT"
echo "  USE_REAL_DATA=$USE_REAL_DATA MNSIM_FULL=$MNSIM_FULL NEUROSIM_FULL=$NEUROSIM_FULL NEUROSIM_CLK_FREQ=${NEUROSIM_CLK_FREQ:-none} NEUROSIM_TECHNODE=${NEUROSIM_TECHNODE:-none} LOGIC_CLK_FREQ=${LOGIC_CLK_FREQ:-none}"
echo "  BATCH_SIZE=$BATCH_SIZE MAX_BATCHES=${MAX_BATCHES:-all} DEVICE=$DEVICE"
echo "  SHOW_SHAPES=$SHOW_SHAPES BN_EN=$BN_EN QD_EN=$QD_EN EXPORT_MEM_TXN=$EXPORT_MEM_TXN"
echo "  SIM_LOG_DIR=$SIM_LOG_DIR SIM_RESULTS_DIR=$SIM_RESULTS_DIR OUTPUT=$OUTPUT_JSON"
echo "==========================================="

python -m pointkan_sim.run_sim_with_real_eval $(build_args)
