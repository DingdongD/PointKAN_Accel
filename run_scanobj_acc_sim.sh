#!/usr/bin/env bash
# ScanObjectNN 分类任务 — ACC 模式 PPA 仿真（真实前向 + NeuroSim/MNSIM CIM 仿真）
#
# 网络:  pointkan-elite-scanobj (1024 pts, 15 类, use_xyz=False)
# 模式:  ACCELERATOR=acc，USE_REAL_DATA=1（ACC 模式强制要求真实数据前向）
#
#   前置条件（ACC 模式需要真实数据路径，当前版本限制如下）:
#   - run_sim_with_real_eval.py 中 load_model_and_dataset 目前硬编码为 ModelNet40 路径。
#   - 若要完整支持 ScanObjectNN 真实数据 ACC 仿真，需在
#     pointkan_sim/run_sim_with_real_eval.py 中添加 ScanObjectNN 数据集分支：
#       1. 在 _ensure_classification_path() 中新增 ScanObjectNN 路径
#       2. 在 load_model_and_dataset() 中新增 pointkan-elite-scanobj 分支（15 类）
#       3. 将 ScanObjectNN 检查点路径赋给 CKPT
#   - 上述修改完成后，本脚本可直接运行。
#   - 若使用 ModelNet40 检查点代替运行（网络架构相同但输出层维度不同），请注意
#     最终分类头 Linear(256→40 vs 256→15) 的 shape 差异会反映在 NeuroSim 结果中。
#
# 使用（在 PointKAN-pytorch 根目录）:
#   chmod +x run_scanobj_acc_sim.sh
#   CKPT=<ScanObjectNN_checkpoint_path> ./run_scanobj_acc_sim.sh
#
# 环境变量（可选）:
#   CKPT                    检查点路径（目录或 .pth），必须指向 ScanObjectNN 检查点
#   NUM_POINTS              输入点数，默认 1024
#   NUM_CLASSES             分类数，默认 15
#   NETWORK                 仿真网络，默认 pointkan-elite-scanobj
#   BATCH_SIZE              测试 batch 大小，默认 1
#   MAX_BATCHES             限制 batch 数（快速测试），例如 MAX_BATCHES=1
#   NEUROSIM_FULL           1=NeuroSim 全网络仿真（默认 1）
#   （已移除 MNSIM 支持）
#   NEUROSIM_CLK_FREQ       NeuroSim 工作频率(Hz)，默认 100 MHz（100000000）
#   NEUROSIM_TECHNODE       工艺节点(nm)，默认 32
#   NEUROSIM_ROOT           NeuroSim 根目录（Inference_pytorch），默认路径见下
#   LOGIC_CLK_FREQ          数字逻辑时钟频率(Hz)，默认 1 GHz
#   SIM_RESULTS_DIR         输出目录，默认 sim_results_scanobj_acc
#   SIM_LOG_DIR             日志目录，默认 sim_log
#   OUTPUT_JSON             报告文件名，默认 report.json
#   DEVICE                  cuda（默认）或 cpu
#   SHOW_SHAPES             1=打印每层 shape，默认 0
#   BN_EN                   1=BN 融合（默认 1）
#   QD_EN                   1=启用量化/反量化（ACC 默认 1）
#   EXPORT_MEM_TXN          1=导出内存事务（默认 1）
#   EXPORT_MEM_CONTRACT     1=导出内存契约（默认 0）
#   FEATURE_BUFFER_PIPELINE_MODE  pipeline | single（ACC 推荐 pipeline）
#   FEATURE_DOUBLE_BUFFERING      1=开启 double buffering（默认 1）
#   KNN_DYNAMIC_THRESHOLD_ROUNDS_MEAN  KNN 动态阈值均值轮数（默认 1）
#   KNN_CANDIDATES_PER_BLOCK    KNN 每个 block coarse 后写入 SRAM 的候选数
#   FPS_BLOCK_TOPK              FPS 后90%阶段每个 block coarse 后保留的候选数
#   ACC_SEGMENT_COUNT       CIM segment 数量
#   RESAMPLER_PARALLELISM   重采样并行度
#   LOGIC_PAR_LAYERNORM / LOGIC_PAR_RELU / LOGIC_PAR_SPOOL / LOGIC_PAR_QUANT_DEQUANT
#   INTER_LAYER_SRAM_BW / INTER_LAYER_SRAM_SIZE_BYTES
#   DRAM_BW_BYTES_PER_CYCLE / DRAM_CHANNELS
#
# 示例：100MHz NeuroSim ACC 仿真（跑 1 batch，开启 pipeline）
#   CKPT=<path> MAX_BATCHES=1 FEATURE_BUFFER_PIPELINE_MODE=pipeline \
#   FEATURE_DOUBLE_BUFFERING=1 KNN_DYNAMIC_THRESHOLD_ROUNDS_MEAN=1 \
#   ./run_scanobj_acc_sim.sh
# 示例：导出内存契约
#   CKPT=<path> EXPORT_MEM_CONTRACT=1 MAX_BATCHES=1 ./run_scanobj_acc_sim.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---- 网络与任务配置 ----
NETWORK="${NETWORK:-pointkan-elite-scanobj}"
NUM_POINTS="${NUM_POINTS:-1024}"
NUM_CLASSES="${NUM_CLASSES:-15}"

# ---- 检查点（ACC 模式必须）----
# 请将此处替换为实际的 ScanObjectNN 检查点路径
CKPT="${CKPT:-PointKAN-elite/classification_ScanObjectNN/checkpoints/pointKANElite-scanobj-exp}"

# ---- 加速器与仿真模式 ----
ACCELERATOR="acc"
USE_REAL_DATA="1"          # ACC 模式强制要求真实数据前向
NEUROSIM_FULL="${NEUROSIM_FULL:-1}"
NEUROSIM_ROOT="${NEUROSIM_ROOT:-}"
NEUROSIM_CLK_FREQ="${NEUROSIM_CLK_FREQ:-100000000}"
NEUROSIM_TECHNODE="${NEUROSIM_TECHNODE:-32}"

# ---- 输出目录 ----
SIM_RESULTS_DIR="${SIM_RESULTS_DIR:-sim_results_scanobj_acc}"
SIM_LOG_DIR="${SIM_LOG_DIR:-sim_log}"
OUTPUT_JSON="${OUTPUT_JSON:-report.json}"

# ---- 运行参数 ----
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_BATCHES="${MAX_BATCHES:-1}"
LOGIC_CLK_FREQ="${LOGIC_CLK_FREQ:-1000000000}"
RUN_REAL_INFERENCE="${RUN_REAL_INFERENCE:-0}"
SHOW_SHAPES="${SHOW_SHAPES:-0}"
BN_EN="${BN_EN:-1}"
QD_EN="${QD_EN:-1}"
EXPORT_MEM_TXN="${EXPORT_MEM_TXN:-1}"
EXPORT_MEM_CONTRACT="${EXPORT_MEM_CONTRACT:-0}"

# ---- ACC 专用参数 ----
FEATURE_BUFFER_PIPELINE_MODE="${FEATURE_BUFFER_PIPELINE_MODE:-pipeline}"
FEATURE_DOUBLE_BUFFERING="${FEATURE_DOUBLE_BUFFERING:-1}"
FEATURE_BUFFER_SLOTS="${FEATURE_BUFFER_SLOTS:-}"
KNN_DYNAMIC_THRESHOLD_ROUNDS_MEAN="${KNN_DYNAMIC_THRESHOLD_ROUNDS_MEAN:-1}"
KNN_CANDIDATES_PER_BLOCK="${KNN_CANDIDATES_PER_BLOCK:-32}"
FPS_BLOCK_TOPK="${FPS_BLOCK_TOPK:-32}"
ACC_SEGMENT_COUNT="${ACC_SEGMENT_COUNT:-}"
RESAMPLER_PARALLELISM="${RESAMPLER_PARALLELISM:-}"

# ---- 逻辑电路并行度（可选）----
LOGIC_PAR_LAYERNORM="${LOGIC_PAR_LAYERNORM:-}"
LOGIC_PAR_RELU="${LOGIC_PAR_RELU:-}"
LOGIC_PAR_SPOOL="${LOGIC_PAR_SPOOL:-}"
LOGIC_PAR_QUANT_DEQUANT="${LOGIC_PAR_QUANT_DEQUANT:-}"

# ---- 内存/DRAM 参数（可选）----
INTER_LAYER_SRAM_BW="${INTER_LAYER_SRAM_BW:-}"
INTER_LAYER_SRAM_SIZE_BYTES="${INTER_LAYER_SRAM_SIZE_BYTES:-}"
DRAM_BW_BYTES_PER_CYCLE="${DRAM_BW_BYTES_PER_CYCLE:-}"
DRAM_CHANNELS="${DRAM_CHANNELS:-}"

build_args() {
    local args=(
        --network    "$NETWORK"
        --checkpoint "$CKPT"
        --num_points "$NUM_POINTS"
        --num_classes "$NUM_CLASSES"
        --accelerator "$ACCELERATOR"
        --sim_log_dir "$SIM_LOG_DIR"
        --device "$DEVICE"
    )
    # ACC 模式：USE_REAL_DATA=1
    args+=(--use_real_data --batch_size "$BATCH_SIZE")
    [[ -n "$MAX_BATCHES" ]] && args+=(--max_batches "$MAX_BATCHES")
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
    [[ "$EXPORT_MEM_CONTRACT" == "1" ]] && args+=(--export_memory_contract)
    [[ -n "$INTER_LAYER_SRAM_BW" ]] && args+=(--inter_layer_sram_bw "$INTER_LAYER_SRAM_BW")
    [[ -n "$INTER_LAYER_SRAM_SIZE_BYTES" ]] && args+=(--inter_layer_sram_size_bytes "$INTER_LAYER_SRAM_SIZE_BYTES")
    [[ -n "$DRAM_BW_BYTES_PER_CYCLE" ]] && args+=(--dram_bw_bytes_per_cycle "$DRAM_BW_BYTES_PER_CYCLE")
    [[ -n "$DRAM_CHANNELS" ]] && args+=(--dram_channels "$DRAM_CHANNELS")
    [[ -n "$FEATURE_BUFFER_PIPELINE_MODE" ]] && args+=(--feature_buffer_pipeline_mode "$FEATURE_BUFFER_PIPELINE_MODE")
    [[ "$FEATURE_DOUBLE_BUFFERING" == "1" ]] && args+=(--feature_double_buffering)
    [[ -n "$FEATURE_BUFFER_SLOTS" ]] && args+=(--feature_buffer_slots "$FEATURE_BUFFER_SLOTS")
    [[ -n "$KNN_DYNAMIC_THRESHOLD_ROUNDS_MEAN" ]] && args+=(--knn_dynamic_threshold_rounds_mean "$KNN_DYNAMIC_THRESHOLD_ROUNDS_MEAN")
    [[ -n "$KNN_CANDIDATES_PER_BLOCK" ]] && args+=(--knn_candidates_per_block "$KNN_CANDIDATES_PER_BLOCK")
    [[ -n "$FPS_BLOCK_TOPK" ]] && args+=(--fps_block_topk "$FPS_BLOCK_TOPK")
    [[ -n "$ACC_SEGMENT_COUNT" ]] && args+=(--acc_segment_count "$ACC_SEGMENT_COUNT")
    [[ -n "$RESAMPLER_PARALLELISM" ]] && args+=(--resampler_parallelism "$RESAMPLER_PARALLELISM")
    args+=(--sim_results_dir "$SIM_RESULTS_DIR")
    printf '%s\n' "${args[@]}"
}

echo "========== ScanObjectNN ACC 仿真 =========="
echo "  NETWORK=$NETWORK  ACCELERATOR=$ACCELERATOR  USE_REAL_DATA=$USE_REAL_DATA"
echo "  CKPT=$CKPT"
echo "  NUM_POINTS=$NUM_POINTS  NUM_CLASSES=$NUM_CLASSES"
echo "  NEUROSIM_FULL=$NEUROSIM_FULL"
echo "  NEUROSIM_CLK_FREQ=${NEUROSIM_CLK_FREQ:-none}  NEUROSIM_TECHNODE=${NEUROSIM_TECHNODE:-none}"
echo "  LOGIC_CLK_FREQ=${LOGIC_CLK_FREQ:-none}"
echo "  BATCH_SIZE=$BATCH_SIZE  MAX_BATCHES=${MAX_BATCHES:-all}  DEVICE=$DEVICE"
echo "  SHOW_SHAPES=$SHOW_SHAPES  BN_EN=$BN_EN  QD_EN=$QD_EN"
echo "  EXPORT_MEM_TXN=$EXPORT_MEM_TXN  EXPORT_MEM_CONTRACT=$EXPORT_MEM_CONTRACT"
echo "  FEATURE_BUFFER_PIPELINE_MODE=${FEATURE_BUFFER_PIPELINE_MODE:-default}"
echo "  FEATURE_DOUBLE_BUFFERING=$FEATURE_DOUBLE_BUFFERING"
echo "  KNN_DYNAMIC_THRESHOLD_ROUNDS_MEAN=${KNN_DYNAMIC_THRESHOLD_ROUNDS_MEAN:-default}"
echo "  KNN_CANDIDATES_PER_BLOCK=${KNN_CANDIDATES_PER_BLOCK:-default}"
echo "  FPS_BLOCK_TOPK=${FPS_BLOCK_TOPK:-default}"
echo "  SIM_RESULTS_DIR=$SIM_RESULTS_DIR  SIM_LOG_DIR=$SIM_LOG_DIR  OUTPUT=$OUTPUT_JSON"
echo "==========================================="
echo ""
echo "⚠  注意：ACC 模式需要真实数据前向。当前 load_model_and_dataset 硬编码为 ModelNet40。"
echo "   如需支持 ScanObjectNN 真实数据，请先在 run_sim_with_real_eval.py 中添加对应数据分支。"
echo ""

python -m pointkan_sim.run_sim_with_real_eval $(build_args) "$@"
