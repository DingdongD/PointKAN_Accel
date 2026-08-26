#!/usr/bin/env bash
# PointCIM (acc) 模式 + neurosim_core pybind 仿真脚本
# =====================================================================
# 与 run_sim_with_real_eval.sh 的区别：
#   - ACCELERATOR=pointcim（RRAM VMM 距离 + acc 流水；脚本内已默认）
#   - NEUROSIM_PYBIND=1 (chip_setup/chip_run_all pybind，无 subprocess，无文件 IO)
#   - NEUROSIM_FULL=0   (不走 subprocess)
#   - 预设合理的 acc 专项参数（KNN 阈值、FPS block、SRAM-CIM、时钟等）
#
# 使用（在 PointKAN-pytorch 根目录）：
#   chmod +x run_sim_pointcim_pybind.sh && ./run_sim_pointcim_pybind.sh
#
# 快速测试（2 batch）：
#   MAX_BATCHES=2 ./run_sim_pointcim_pybind.sh
#
# 切换网络：
#   NETWORK=pointkan-elite-scanobj ./run_sim_pointcim_pybind.sh
#
# 修改时钟/工艺（不影响 pybind 模式，参数透传给 chip_setup）：
#   NEUROSIM_TECHNODE=32 NEUROSIM_CLK_FREQ=100000000 ./run_sim_pointcim_pybind.sh
#
# 打开 cycle-accurate NS 引擎（SRAM-CIM 部分用周期精确仿真）：
#   NS_SIMULATION_MODE=cycle_accurate ./run_sim_pointcim_pybind.sh
#
# 环境变量（全部可选，有合理默认）：
# ─── 网络与检查点 ─────────────────────────────────────────────────────
#   NETWORK           pointkan-elite（默认）| pointkan-elite-scanobj | pointkan-elite-partseg
#   CKPT              检查点路径（随 NETWORK 自动设置）
#   NUM_POINTS        点云点数（随 NETWORK 自动设置，默认 1024）
#   NUM_CLASSES       类别数（随 NETWORK 自动设置，默认 40）
#   INPUT_CHANNELS    输入通道数（S3DIS 等特殊场景使用）
# ─── 运行控制 ─────────────────────────────────────────────────────────
#   BATCH_SIZE        测试 batch 大小（默认 1）
#   MAX_BATCHES       限制 batch 数做快速测试（不设则跑全量）
#   DEVICE            cuda（默认）| cpu
#   RUN_REAL_INFERENCE 1=先跑 inference_and_kan_analysis，0=不跑（默认）
# ─── NeuroSim pybind 参数 ─────────────────────────────────────────────
#   NEUROSIM_ROOT     NeuroSim Inference_pytorch 目录（默认见下方）
#   NEUROSIM_CLK_FREQ  芯片时钟(Hz)，默认不设（由 chip_setup 自行确定关键路径时钟）
#   NEUROSIM_TECHNODE  工艺节点(nm)，默认 32（支持 130/90/65/45/32/22/14/10/7）
# ─── PointCIM (acc) 专项参数 ──────────────────────────────────────────
#   NS_SIMULATION_MODE        event（默认）| cycle_accurate
#   KNN_CONV_PIPELINE_MODE    KNN→Conv 流水模式（默认不设，由 hw_config 决定）
#   KNN_DYNAMIC_THRESHOLD_ROUNDS_MEAN  KNN 动态阈值 refine 轮数（默认不设）
#   KNN_CANDIDATES_PER_BLOCK  KNN 每 block coarse 候选数（默认不设）
#   FPS_BLOCK_TOPK            FPS block coarse 保留候选数（默认不设）
#   ACC_SEGMENT_COUNT         acc speculative FPS segment 数（默认不设）
#   RESAMPLER_PARALLELISM     resampler 并行度（默认不设）
# ─── 时钟与内存 ───────────────────────────────────────────────────────
#   LOGIC_CLK_FREQ    数字逻辑时钟(Hz)，默认 1GHz
#   DRAM_BW_BYTES_PER_CYCLE   DRAM 每周期带宽字节数（默认不设）
#   DRAM_CHANNELS     DRAM 通道数（默认不设）
#   INTER_LAYER_SRAM_BW       层间 SRAM 带宽（默认不设）
#   INTER_LAYER_SRAM_SIZE_BYTES 层间 SRAM 容量（默认不设）
#   FEATURE_BUFFER_PIPELINE_MODE  feature buffer 流水模式（默认不设）
#   FEATURE_DOUBLE_BUFFERING   1=启用 feature double buffering（默认 0）
#   FEATURE_BUFFER_SLOTS       feature buffer slot 数（默认不设）
# ─── 输出控制 ─────────────────────────────────────────────────────────
#   SHOW_SHAPES       1=打印每层 shape 信息（默认 0）
#   BN_EN             1=BN 融合（默认 1）
#   QD_EN             1=量化/反量化事件（acc 默认 1）
#   EXPORT_MEM_TXN    1=导出内存事务序列（默认 1）
#   SIM_LOG_DIR       日志目录（默认 sim_log_pointcim_pybind）
#   SIM_RESULTS_DIR   结果输出目录（默认 sim_results_pointcim_pybind）
#   OUTPUT_JSON       报告 JSON 文件名（默认 report_pointcim_pybind.json）

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── 网络与检查点 ──────────────────────────────────────────────────────
NETWORK="${NETWORK:-pointkan-elite}"

if [ -z "${CKPT+x}" ]; then
  case "$NETWORK" in
    pointkan-elite-partseg)
      CKPT="PointKAN-elite/part_segmentation/checkpoints/pointKANElite_part_exp" ;;
    pointkan-elite-scanobj)
      CKPT="PointKAN-elite/classification_ScanObjectNN/checkpoints/pointKANElite-scan_exp-42" ;;
    *)
      CKPT="PointKAN-elite/classification_ModelNet40/checkpoints/pointKANElite-pointkan_elite_exp-42" ;;
  esac
fi

if [ -z "${NUM_POINTS+x}" ]; then
  case "$NETWORK" in
    pointkan-elite-partseg) NUM_POINTS=2048 ;;
    *)                      NUM_POINTS=1024 ;;
  esac
fi

if [ -z "${NUM_CLASSES+x}" ]; then
  case "$NETWORK" in
    pointkan-elite-partseg) NUM_CLASSES=50  ;;
    pointkan-elite-scanobj) NUM_CLASSES=15  ;;
    *)                      NUM_CLASSES=40  ;;
  esac
fi

INPUT_CHANNELS="${INPUT_CHANNELS:-}"

# ── 运行控制 ──────────────────────────────────────────────────────────
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_BATCHES="${MAX_BATCHES:-}"
DEVICE="${DEVICE:-cuda}"
RUN_REAL_INFERENCE="${RUN_REAL_INFERENCE:-0}"

# ── PointCIM（RRAM 距离 + acc 流水）+ pybind 模式 ───────────────────────
ACCELERATOR="pointcim"
USE_REAL_DATA="1"
NEUROSIM_PYBIND="1"
NEUROSIM_FULL="0"

# ── NeuroSim pybind 参数 ──────────────────────────────────────────────
NEUROSIM_ROOT="${NEUROSIM_ROOT:-/workspace/DNN_NeuroSim_V1.4/Inference_pytorch}"
NEUROSIM_CLK_FREQ="${NEUROSIM_CLK_FREQ:-100000000}" # NeuroSim 系统时钟，默认 100 MHz；NeuroSim 会取 min(用户值, 关键路径)
NEUROSIM_ADC_CLK_FREQ="${NEUROSIM_ADC_CLK_FREQ:-1000000000}" # SAR ADC 时钟，默认 1 GHz
NEUROSIM_TECHNODE="${NEUROSIM_TECHNODE:-32}"         # 32nm RRAM-CIM 默认工艺

# ── PointCIM acc 专项参数 ─────────────────────────────────────────────
NS_SIMULATION_MODE="${NS_SIMULATION_MODE:-event}"
KNN_CONV_PIPELINE_MODE="${KNN_CONV_PIPELINE_MODE:-}"
KNN_DYNAMIC_THRESHOLD_ROUNDS_MEAN="${KNN_DYNAMIC_THRESHOLD_ROUNDS_MEAN:-}"
KNN_CANDIDATES_PER_BLOCK="${KNN_CANDIDATES_PER_BLOCK:-}"
FPS_BLOCK_TOPK="${FPS_BLOCK_TOPK:-}"
ACC_SEGMENT_COUNT="${ACC_SEGMENT_COUNT:-}"
RESAMPLER_PARALLELISM="${RESAMPLER_PARALLELISM:-}"

# ── 时钟与内存 ────────────────────────────────────────────────────────
LOGIC_CLK_FREQ="${LOGIC_CLK_FREQ:-1000000000}"
DRAM_BW_BYTES_PER_CYCLE="${DRAM_BW_BYTES_PER_CYCLE:-}"
DRAM_CHANNELS="${DRAM_CHANNELS:-}"
INTER_LAYER_SRAM_BW="${INTER_LAYER_SRAM_BW:-}"
INTER_LAYER_SRAM_SIZE_BYTES="${INTER_LAYER_SRAM_SIZE_BYTES:-}"
FEATURE_BUFFER_PIPELINE_MODE="${FEATURE_BUFFER_PIPELINE_MODE:-}"
FEATURE_DOUBLE_BUFFERING="${FEATURE_DOUBLE_BUFFERING:-0}"
FEATURE_BUFFER_SLOTS="${FEATURE_BUFFER_SLOTS:-}"

# ── 输出控制 ──────────────────────────────────────────────────────────
SHOW_SHAPES="${SHOW_SHAPES:-0}"
BN_EN="${BN_EN:-1}"
QD_EN="${QD_EN:-1}"              # acc 模式默认开启量化/反量化事件
EXPORT_MEM_TXN="${EXPORT_MEM_TXN:-1}"
EXPORT_MEM_CONTRACT="${EXPORT_MEM_CONTRACT:-0}"
SIM_LOG_DIR="${SIM_LOG_DIR:-sim_log_pointcim_pybind}"
SIM_RESULTS_DIR="${SIM_RESULTS_DIR:-sim_results_pointcim_pybind}"
OUTPUT_JSON="${OUTPUT_JSON:-report_pointcim_pybind.json}"

# ── 组装参数 ──────────────────────────────────────────────────────────
BUILT_ARGS=(
    --checkpoint      "$CKPT"
    --num_points      "$NUM_POINTS"
    --num_classes     "$NUM_CLASSES"
    --accelerator     "$ACCELERATOR"
    --device          "$DEVICE"
    --use_real_data
    --batch_size      "$BATCH_SIZE"
    --neurosim_pybind
    --neurosim_root   "$NEUROSIM_ROOT"
    --neurosim_technode "$NEUROSIM_TECHNODE"
    --logic_clk_freq  "$LOGIC_CLK_FREQ"
    --ns_simulation_mode "$NS_SIMULATION_MODE"
    --bn_en           "$BN_EN"
    --qd_en           "$QD_EN"
    --sim_log_dir     "$SIM_LOG_DIR"
    --sim_results_dir "$SIM_RESULTS_DIR"
    --output          "$OUTPUT_JSON"
    --network         "$NETWORK"
)

[[ -n "$MAX_BATCHES"                       ]] && BUILT_ARGS+=(--max_batches "$MAX_BATCHES")
[[ -n "$INPUT_CHANNELS"                    ]] && BUILT_ARGS+=(--input_channels "$INPUT_CHANNELS")
[[ -n "$NEUROSIM_CLK_FREQ"                 ]] && BUILT_ARGS+=(--neurosim_clk_freq "$NEUROSIM_CLK_FREQ")
[[ -n "$NEUROSIM_ADC_CLK_FREQ"            ]] && BUILT_ARGS+=(--neurosim_adc_clk_freq "$NEUROSIM_ADC_CLK_FREQ")
[[ "$RUN_REAL_INFERENCE" == "1"            ]] && BUILT_ARGS+=(--run_real_inference --no_vis)
[[ "$SHOW_SHAPES" == "1"                   ]] && BUILT_ARGS+=(--show_shapes)
[[ "$EXPORT_MEM_TXN" == "1"               ]] && BUILT_ARGS+=(--export_memory_transactions)
[[ "$EXPORT_MEM_CONTRACT" == "1"           ]] && BUILT_ARGS+=(--export_memory_contract)
[[ -n "$KNN_CONV_PIPELINE_MODE"            ]] && BUILT_ARGS+=(--knn_conv_pipeline_mode "$KNN_CONV_PIPELINE_MODE")
[[ -n "$KNN_DYNAMIC_THRESHOLD_ROUNDS_MEAN" ]] && BUILT_ARGS+=(--knn_dynamic_threshold_rounds_mean "$KNN_DYNAMIC_THRESHOLD_ROUNDS_MEAN")
[[ -n "$KNN_CANDIDATES_PER_BLOCK"          ]] && BUILT_ARGS+=(--knn_candidates_per_block "$KNN_CANDIDATES_PER_BLOCK")
[[ -n "$FPS_BLOCK_TOPK"                   ]] && BUILT_ARGS+=(--fps_block_topk "$FPS_BLOCK_TOPK")
[[ -n "$ACC_SEGMENT_COUNT"                 ]] && BUILT_ARGS+=(--acc_segment_count "$ACC_SEGMENT_COUNT")
[[ -n "$RESAMPLER_PARALLELISM"             ]] && BUILT_ARGS+=(--resampler_parallelism "$RESAMPLER_PARALLELISM")
[[ -n "$DRAM_BW_BYTES_PER_CYCLE"           ]] && BUILT_ARGS+=(--dram_bw_bytes_per_cycle "$DRAM_BW_BYTES_PER_CYCLE")
[[ -n "$DRAM_CHANNELS"                     ]] && BUILT_ARGS+=(--dram_channels "$DRAM_CHANNELS")
[[ -n "$INTER_LAYER_SRAM_BW"              ]] && BUILT_ARGS+=(--inter_layer_sram_bw "$INTER_LAYER_SRAM_BW")
[[ -n "$INTER_LAYER_SRAM_SIZE_BYTES"       ]] && BUILT_ARGS+=(--inter_layer_sram_size_bytes "$INTER_LAYER_SRAM_SIZE_BYTES")
[[ -n "$FEATURE_BUFFER_PIPELINE_MODE"      ]] && BUILT_ARGS+=(--feature_buffer_pipeline_mode "$FEATURE_BUFFER_PIPELINE_MODE")
[[ "$FEATURE_DOUBLE_BUFFERING" == "1"      ]] && BUILT_ARGS+=(--feature_double_buffering)
[[ -n "$FEATURE_BUFFER_SLOTS"              ]] && BUILT_ARGS+=(--feature_buffer_slots "$FEATURE_BUFFER_SLOTS")

# ── 打印配置摘要 ──────────────────────────────────────────────────────
echo "========== PointCIM (acc) + neurosim_core pybind =========="
echo "  NETWORK=$NETWORK  CKPT=$CKPT"
echo "  NUM_POINTS=$NUM_POINTS  NUM_CLASSES=$NUM_CLASSES  DEVICE=$DEVICE"
echo "  BATCH_SIZE=$BATCH_SIZE  MAX_BATCHES=${MAX_BATCHES:-all}"
echo "  backend: neurosim_pybind (chip_setup/chip_run_all, 无 subprocess)"
echo "  NEUROSIM_ROOT=$NEUROSIM_ROOT  NEUROSIM_TECHNODE=$NEUROSIM_TECHNODE"
echo "  NEUROSIM_CLK_FREQ=${NEUROSIM_CLK_FREQ:-auto}  NEUROSIM_ADC_CLK_FREQ=${NEUROSIM_ADC_CLK_FREQ:-auto}  LOGIC_CLK_FREQ=$LOGIC_CLK_FREQ"
echo "  NS_SIMULATION_MODE=$NS_SIMULATION_MODE"
echo "  KNN_CONV_PIPELINE_MODE=${KNN_CONV_PIPELINE_MODE:-auto}"
echo "  KNN_DYNAMIC_THRESHOLD_ROUNDS_MEAN=${KNN_DYNAMIC_THRESHOLD_ROUNDS_MEAN:-default}"
echo "  KNN_CANDIDATES_PER_BLOCK=${KNN_CANDIDATES_PER_BLOCK:-default}"
echo "  FPS_BLOCK_TOPK=${FPS_BLOCK_TOPK:-default}"
echo "  ACC_SEGMENT_COUNT=${ACC_SEGMENT_COUNT:-default}"
echo "  RESAMPLER_PARALLELISM=${RESAMPLER_PARALLELISM:-default}"
echo "  BN_EN=$BN_EN  QD_EN=$QD_EN  EXPORT_MEM_TXN=$EXPORT_MEM_TXN"
echo "  SIM_RESULTS_DIR=$SIM_RESULTS_DIR  OUTPUT=$OUTPUT_JSON"
echo "==========================================================="

python -m pointkan_sim.run_sim_with_real_eval "${BUILT_ARGS[@]}" "$@"
