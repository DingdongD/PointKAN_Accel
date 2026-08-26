#!/usr/bin/env bash
# ShapeNetPart 零件分割任务 — Baseline 模式 PPA 仿真（结构仿真，不需要 checkpoint / 数据集）
#
# 网络:  pointkan-elite-partseg (2048 pts, 50 类, use_xyz=True, 输入 xyz+normal)
# 模式:  ACCELERATOR=baseline（或 base_v2），USE_REAL_DATA=0
#
# 与分类任务的主要区别:
#   - 输入点数 2048（分类 1024），输入通道数 6（xyz + normal）
#   - 4 个编码阶段 + 4 个 PointNetFeaturePropagation 解码阶段
#   - use_xyz=True（GAM/LFP 中 S-Pool 输入宽度 = 2*ch + 3）
#   - 全局上下文：5×gmp_map_list + gmp_map_end
#   - 分类头：Conv1d(256→128)→BN→Conv1d(128→50)
#
# 使用（在 PointKAN-pytorch 根目录）:
#   chmod +x run_partseg_baseline_sim.sh && ./run_partseg_baseline_sim.sh
#
# 环境变量（可选）:
#   ACCELERATOR             baseline（默认）| base_v2
#   NUM_POINTS              输入点数，默认 2048
#   NUM_CLASSES             分割类数，默认 50
#   NETWORK                 仿真网络，默认 pointkan-elite-partseg
#   SIM_RESULTS_DIR         输出目录，默认 sim_results_partseg_baseline
#   SIM_LOG_DIR             日志目录，默认 sim_log
#   OUTPUT_JSON             报告文件名，默认 report.json
#   DEVICE                  cpu（结构仿真不依赖 GPU）或 cuda
#   SHOW_SHAPES             1=打印每层 shape，默认 0
#   BN_EN                   1=BN 融合（默认 1）
#   QD_EN                   0=不启用量化（baseline 默认 0）
#   EXPORT_MEM_TXN          1=导出内存事务（默认 1）
#   EXPORT_MEM_CONTRACT     1=导出内存契约（默认 0）
#   LOGIC_CLK_FREQ          数字逻辑时钟频率(Hz)，默认 1 GHz
#   LOGIC_PAR_LAYERNORM / LOGIC_PAR_RELU / LOGIC_PAR_SPOOL / LOGIC_PAR_QUANT_DEQUANT
#   INTER_LAYER_SRAM_BW / INTER_LAYER_SRAM_SIZE_BYTES
#   DRAM_BW_BYTES_PER_CYCLE / DRAM_CHANNELS
#   USE_SCALESIM_SYSTOLIC   1=用 SCALE-Sim 做 Conv/Linear 周期统计（默认 0）
#   SCALE_SIM_ROOT          SCALE-Sim 根目录（USE_SCALESIM_SYSTOLIC=1 时）
#   SYSTOLIC_ROWS / SYSTOLIC_COLS  脉动阵列行/列（默认 64）
#
# 示例：快速 baseline 仿真
#   ./run_partseg_baseline_sim.sh
# 示例：使用 base_v2 加速器
#   ACCELERATOR=base_v2 SIM_RESULTS_DIR=sim_results_partseg_base_v2 ./run_partseg_baseline_sim.sh
# 示例：打印每层 shape
#   SHOW_SHAPES=1 ./run_partseg_baseline_sim.sh
# 示例：导出内存契约
#   EXPORT_MEM_CONTRACT=1 ./run_partseg_baseline_sim.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---- 网络与任务配置 ----
NETWORK="${NETWORK:-pointkan-elite-partseg}"
NUM_POINTS="${NUM_POINTS:-2048}"
NUM_CLASSES="${NUM_CLASSES:-50}"

# ---- 加速器与仿真模式 ----
ACCELERATOR="${ACCELERATOR:-baseline}"
USE_REAL_DATA="0"          # baseline 结构仿真：不需要 checkpoint 与数据集
NEUROSIM_FULL="0"

# ---- 输出目录 ----
SIM_RESULTS_DIR="${SIM_RESULTS_DIR:-sim_results_partseg_baseline}"
SIM_LOG_DIR="${SIM_LOG_DIR:-sim_log}"
OUTPUT_JSON="${OUTPUT_JSON:-report.json}"

# ---- 运行参数 ----
DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_BATCHES="${MAX_BATCHES:-}"
SHOW_SHAPES="${SHOW_SHAPES:-0}"
BN_EN="${BN_EN:-1}"
QD_EN="${QD_EN:-0}"
EXPORT_MEM_TXN="${EXPORT_MEM_TXN:-1}"
EXPORT_MEM_CONTRACT="${EXPORT_MEM_CONTRACT:-0}"
LOGIC_CLK_FREQ="${LOGIC_CLK_FREQ:-1000000000}"

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

# ---- SCALE-Sim（仅 baseline）----
# baseline 结构仿真中的 Conv/Linear 当前默认通过 SCALE-Sim 建模
USE_SCALESIM_SYSTOLIC="${USE_SCALESIM_SYSTOLIC:-1}"
SCALE_SIM_ROOT="${SCALE_SIM_ROOT:-}"
SYSTOLIC_ROWS="${SYSTOLIC_ROWS:-}"
SYSTOLIC_COLS="${SYSTOLIC_COLS:-}"
SYSTOLIC_SRAM_SIZE_BYTES="${SYSTOLIC_SRAM_SIZE_BYTES:-}"
SYSTOLIC_IFMAP_SRAM_SIZE_BYTES="${SYSTOLIC_IFMAP_SRAM_SIZE_BYTES:-}"
SYSTOLIC_FILTER_SRAM_SIZE_BYTES="${SYSTOLIC_FILTER_SRAM_SIZE_BYTES:-}"
SYSTOLIC_OFMAP_SRAM_SIZE_BYTES="${SYSTOLIC_OFMAP_SRAM_SIZE_BYTES:-}"
SYSTOLIC_SRAM_BLOCK_BYTES="${SYSTOLIC_SRAM_BLOCK_BYTES:-}"
SYSTOLIC_SRAM_NUM_BANKS="${SYSTOLIC_SRAM_NUM_BANKS:-}"
SYSTOLIC_SRAM_BUS_WIDTH_BITS="${SYSTOLIC_SRAM_BUS_WIDTH_BITS:-}"
SYSTOLIC_SRAM_FREQ_HZ="${SYSTOLIC_SRAM_FREQ_HZ:-}"

build_args() {
    local args=(
        --network    "$NETWORK"
        --num_points "$NUM_POINTS"
        --num_classes "$NUM_CLASSES"
        --accelerator "$ACCELERATOR"
        --sim_log_dir "$SIM_LOG_DIR"
        --device "$DEVICE"
    )
    # baseline 不传 --use_real_data（结构仿真路径）
    [[ -n "$MAX_BATCHES" ]] && args+=(--max_batches "$MAX_BATCHES")
    [[ -n "$LOGIC_CLK_FREQ" ]] && args+=(--logic_clk_freq "$LOGIC_CLK_FREQ")
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
    [[ "$USE_SCALESIM_SYSTOLIC" == "1" ]] && args+=(--use_scalesim_systolic)
    [[ -n "$SCALE_SIM_ROOT" ]] && args+=(--scalesim_root "$SCALE_SIM_ROOT")
    [[ -n "$SYSTOLIC_ROWS" ]] && args+=(--systolic_rows "$SYSTOLIC_ROWS")
    [[ -n "$SYSTOLIC_COLS" ]] && args+=(--systolic_cols "$SYSTOLIC_COLS")
    [[ -n "$SYSTOLIC_SRAM_SIZE_BYTES" ]] && args+=(--systolic_sram_size_bytes "$SYSTOLIC_SRAM_SIZE_BYTES")
    [[ -n "$SYSTOLIC_IFMAP_SRAM_SIZE_BYTES" ]] && args+=(--systolic_ifmap_sram_size_bytes "$SYSTOLIC_IFMAP_SRAM_SIZE_BYTES")
    [[ -n "$SYSTOLIC_FILTER_SRAM_SIZE_BYTES" ]] && args+=(--systolic_filter_sram_size_bytes "$SYSTOLIC_FILTER_SRAM_SIZE_BYTES")
    [[ -n "$SYSTOLIC_OFMAP_SRAM_SIZE_BYTES" ]] && args+=(--systolic_ofmap_sram_size_bytes "$SYSTOLIC_OFMAP_SRAM_SIZE_BYTES")
    [[ -n "$SYSTOLIC_SRAM_BLOCK_BYTES" ]] && args+=(--systolic_sram_block_bytes "$SYSTOLIC_SRAM_BLOCK_BYTES")
    [[ -n "$SYSTOLIC_SRAM_NUM_BANKS" ]] && args+=(--systolic_sram_num_banks "$SYSTOLIC_SRAM_NUM_BANKS")
    [[ -n "$SYSTOLIC_SRAM_BUS_WIDTH_BITS" ]] && args+=(--systolic_sram_bus_width_bits "$SYSTOLIC_SRAM_BUS_WIDTH_BITS")
    [[ -n "$SYSTOLIC_SRAM_FREQ_HZ" ]] && args+=(--systolic_sram_freq_hz "$SYSTOLIC_SRAM_FREQ_HZ")
    [[ -n "$INTER_LAYER_SRAM_BW" ]] && args+=(--inter_layer_sram_bw "$INTER_LAYER_SRAM_BW")
    [[ -n "$INTER_LAYER_SRAM_SIZE_BYTES" ]] && args+=(--inter_layer_sram_size_bytes "$INTER_LAYER_SRAM_SIZE_BYTES")
    [[ -n "$DRAM_BW_BYTES_PER_CYCLE" ]] && args+=(--dram_bw_bytes_per_cycle "$DRAM_BW_BYTES_PER_CYCLE")
    [[ -n "$DRAM_CHANNELS" ]] && args+=(--dram_channels "$DRAM_CHANNELS")
    args+=(--sim_results_dir "$SIM_RESULTS_DIR")
    printf '%s\n' "${args[@]}"
}

echo "========== ShapeNetPart 零件分割 Baseline 仿真 =========="
echo "  NETWORK=$NETWORK  ACCELERATOR=$ACCELERATOR  USE_REAL_DATA=$USE_REAL_DATA"
echo "  NUM_POINTS=$NUM_POINTS  NUM_CLASSES=$NUM_CLASSES  (use_xyz=True, 6ch input)"
echo "  DEVICE=$DEVICE  BN_EN=$BN_EN  QD_EN=$QD_EN"
echo "  SHOW_SHAPES=$SHOW_SHAPES  EXPORT_MEM_TXN=$EXPORT_MEM_TXN  EXPORT_MEM_CONTRACT=$EXPORT_MEM_CONTRACT"
echo "  USE_SCALESIM_SYSTOLIC=$USE_SCALESIM_SYSTOLIC"
echo "  SIM_RESULTS_DIR=$SIM_RESULTS_DIR  SIM_LOG_DIR=$SIM_LOG_DIR  OUTPUT=$OUTPUT_JSON"
echo "========================================================"

python -m pointkan_sim.run_sim_with_real_eval $(build_args) "$@"
