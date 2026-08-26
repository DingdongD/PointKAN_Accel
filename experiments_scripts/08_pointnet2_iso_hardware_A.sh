#!/usr/bin/env bash
# =============================================================================
# 实验 8（方案A / iso-hardware）：
#   固定 CM tile/PE size 与 tile 总数预算（按 W8A8 规格），公平对比 W8A8 vs W4A4
#
# 硬件约束（CM / Conventional Mapping）：
#   - tile_size_cm = 1024
#   - pe_size_cm   = 512
#   - num_tiles_cm 预算：按网络固定（cls=60, partseg=70, sem=65）
#
# 时钟口径：
#   默认 capped + 100MHz（与实验5/6一致）；可通过 NEUROSIM_CLK_MODE / NEUROSIM_CLK_FREQ 覆盖。
#
# 使用：
#   cd /workspace/PointKAN-pytorch
#   bash experiments_scripts/08_pointnet2_iso_hardware_A.sh
#
# 可选环境变量：
#   DEVICE        cuda（默认）或 cpu
#   MAX_BATCHES   默认 1
# =============================================================================
set -euo pipefail

ROOT="/workspace/PointKAN-pytorch"
cd "$ROOT"

LOG_DIR="$ROOT/experiments_scripts/logs/08_pointnet2_iso_hardware_A"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
MAX_BATCHES="${MAX_BATCHES:-1}"

# Fixed CM hardware (Scheme A)
FORCE_TILE_SIZE_CM=1024
FORCE_PE_SIZE_CM=512

NEUROSIM_CLK_MODE="${NEUROSIM_CLK_MODE:-capped}"
NEUROSIM_CLK_FREQ="${NEUROSIM_CLK_FREQ:-100000000}"
NEUROSIM_ADC_CLK_FREQ="${NEUROSIM_ADC_CLK_FREQ:-1000000000}"

declare -a RUNS=(
  # net:pts:cls:tile_budget
  "pointnet2-cls-ssg:1024:40:60"
  "pointnet2-partseg-ssg:1024:50:70"
  "pointnet2-sem-ssg:4096:13:65"
)

run_one() {
  local wl_w="$1"
  local wl_a="$2"
  local network="$3"
  local pts="$4"
  local ncls="$5"
  local tile_budget="$6"

  local acc="pointcim"
  local tag="${network}_${acc}_p${pts}_w${wl_w}a${wl_a}bit_isoA"
  local results_dir="$ROOT/sim_results_${tag}"
  local log_file="$LOG_DIR/${tag}.log"

  echo ""
  echo "----------------------------------------------------------"
  echo "  [$(date '+%Y-%m-%d %H:%M:%S')] 运行: $tag"
  echo "  iso-hardware A: tile=${FORCE_TILE_SIZE_CM} pe=${FORCE_PE_SIZE_CM} budget=${tile_budget}"
  echo "  clk_mode=${NEUROSIM_CLK_MODE} neurosim_clk_freq=${NEUROSIM_CLK_FREQ}"
  echo "  results -> $results_dir"
  echo "----------------------------------------------------------"

  local envs=(
    ACCELERATOR="$acc"
    NETWORK="$network"
    NUM_POINTS="$pts"
    NUM_CLASSES="$ncls"
    USE_REAL_DATA=1
    DEVICE="$DEVICE"
    BATCH_SIZE=1
    MAX_BATCHES="$MAX_BATCHES"
    NEUROSIM_FULL=1
    NEUROSIM_PYBIND=0
    NEUROSIM_ROOT=/workspace/DNN_NeuroSim_V1.4/Inference_pytorch
    NEUROSIM_CLK_MODE="$NEUROSIM_CLK_MODE"
    NEUROSIM_CLK_FREQ="$NEUROSIM_CLK_FREQ"
    NEUROSIM_ADC_CLK_FREQ="$NEUROSIM_ADC_CLK_FREQ"
    NEUROSIM_TECHNODE=32
    NEUROSIM_WL_WEIGHT="$wl_w"
    NEUROSIM_WL_ACTIVATION="$wl_a"
    NEUROSIM_NUM_COL_MUXED="$wl_w"
    NEUROSIM_FORCE_TILE_SIZE_CM="$FORCE_TILE_SIZE_CM"
    NEUROSIM_FORCE_PE_SIZE_CM="$FORCE_PE_SIZE_CM"
    NEUROSIM_FORCE_NUM_TILES_CM="$tile_budget"
    LOGIC_CLK_FREQ=1000000000
    PROGRESS_INTERVAL_S=30
    KNN_CONV_PIPELINE_MODE=strict
    BN_EN=1
    QD_EN=1
    EXPORT_MEM_TXN=1
    SIM_RESULTS_DIR="$results_dir"
    SIM_LOG_DIR="$ROOT/sim_log"
    OUTPUT_JSON=report.json
  )

  if [ "$network" = "pointnet2-sem-ssg" ]; then
    envs+=(INPUT_CHANNELS=12)
  fi

  env "${envs[@]}" \
    bash "$ROOT/run_sim_with_real_eval.sh" \
    --pointcim_sa_conv_pipeline \
    2>&1 | tee "$log_file"
}

echo "=========================================================="
echo "  实验 8（iso-hardware A）：W8A8 vs W4A4"
echo "  device=$DEVICE max_batches=$MAX_BATCHES"
echo "  CM tile=$FORCE_TILE_SIZE_CM  CM pe=$FORCE_PE_SIZE_CM"
echo "  clk_mode=$NEUROSIM_CLK_MODE  neurosim_clk_freq=$NEUROSIM_CLK_FREQ"
echo "  开始: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="

for spec in "${RUNS[@]}"; do
  IFS=':' read -r NET PTS CLS BUDGET <<< "$spec"
  run_one 8 8 "$NET" "$PTS" "$CLS" "$BUDGET"
  run_one 4 4 "$NET" "$PTS" "$CLS" "$BUDGET"
done

echo ""
echo "=========================================================="
echo "  实验 8 全部完成：$(date '+%Y-%m-%d %H:%M:%S')"
echo "  日志目录: $LOG_DIR"
echo "=========================================================="

