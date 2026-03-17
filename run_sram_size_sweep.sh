#!/usr/bin/env bash
set -euo pipefail
#
# SRAM 容量扫描：对多组 SRAM 大小批量跑仿真。
# 支持两种扫描目标（SWEEP_TARGET）：
#   feature   - 层间 feature buffer SRAM（INTER_LAYER_SRAM_SIZE_BYTES），默认
#   systolic  - 脉动阵列 SRAM（SYSTOLIC_SRAM_SIZE_BYTES）
#
# 用法：
#   ./run_sram_size_sweep.sh                                    # 默认：feature SRAM 扫描
#   SWEEP_TARGET=systolic ./run_sram_size_sweep.sh               # 脉动阵列 SRAM 扫描
#   SRAM_SIZE_KB_LIST="32 64 128 256 512" SWEEP_TARGET=systolic ./run_sram_size_sweep.sh
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# feature=层间 feature SRAM，systolic=脉动阵列 SRAM
SWEEP_TARGET="${SWEEP_TARGET:-feature}"
SRAM_SIZE_KB_LIST="${SRAM_SIZE_KB_LIST:-512 1024 2048 4096}"
# 脉动阵列 SRAM 常用较小容量，可单独指定默认
if [[ "$SWEEP_TARGET" == "systolic" ]]; then
  SRAM_SIZE_KB_LIST="${SRAM_SIZE_KB_LIST:-32 64 128 256 512}"
  BASE_DIR="${SWEEP_ROOT_DIR:-sim_results_systolic_sram_sweep}"
  SUBDIR_PREFIX="systolic_sram"
else
  BASE_DIR="${SWEEP_ROOT_DIR:-sim_results_sram_size_sweep}"
  SUBDIR_PREFIX="feature_sram"
fi

ACCELERATOR="${ACCELERATOR:-baseline}"
USE_SCALESIM_SYSTOLIC="${USE_SCALESIM_SYSTOLIC:-1}"
USE_REAL_DATA="${USE_REAL_DATA:-1}"
MAX_BATCHES="${MAX_BATCHES:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"

mkdir -p "$BASE_DIR"

for kb in $SRAM_SIZE_KB_LIST; do
  bytes=$((kb * 1024))
  out_dir="$BASE_DIR/${SUBDIR_PREFIX}_${bytes}B"
  log_dir="${SIM_LOG_DIR:-sim_log}/sram_sweep_${SUBDIR_PREFIX}_${bytes}B"
  mkdir -p "$out_dir" "$log_dir"

  echo "=================================================="
  echo "SRAM size sweep ($SWEEP_TARGET): ${kb} KB (${bytes} B) -> $out_dir"
  echo "=================================================="

  if [[ "$SWEEP_TARGET" == "systolic" ]]; then
    SIM_RESULTS_DIR="$out_dir" \
    SIM_LOG_DIR="$log_dir" \
    OUTPUT_JSON="report.json" \
    SYSTOLIC_SRAM_SIZE_BYTES="$bytes" \
    ACCELERATOR="$ACCELERATOR" \
    USE_SCALESIM_SYSTOLIC="$USE_SCALESIM_SYSTOLIC" \
    USE_REAL_DATA="$USE_REAL_DATA" \
    MAX_BATCHES="$MAX_BATCHES" \
    BATCH_SIZE="$BATCH_SIZE" \
    ./run_sim_with_real_eval.sh
  else
    SIM_RESULTS_DIR="$out_dir" \
    SIM_LOG_DIR="$log_dir" \
    OUTPUT_JSON="report.json" \
    INTER_LAYER_SRAM_SIZE_BYTES="$bytes" \
    ACCELERATOR="$ACCELERATOR" \
    USE_SCALESIM_SYSTOLIC="$USE_SCALESIM_SYSTOLIC" \
    USE_REAL_DATA="$USE_REAL_DATA" \
    MAX_BATCHES="$MAX_BATCHES" \
    BATCH_SIZE="$BATCH_SIZE" \
    ./run_sim_with_real_eval.sh
  fi
done

echo "=================================================="
echo "SRAM size sweep ($SWEEP_TARGET) 完成。汇总与绘图请运行："
if [[ "$SWEEP_TARGET" == "systolic" ]]; then
  echo "  python scripts/aggregate_sram_sweep_and_plot.py --sweep_dir $BASE_DIR --sweep_type systolic"
else
  echo "  python scripts/aggregate_sram_sweep_and_plot.py --sweep_dir $BASE_DIR --sweep_type feature"
fi
echo "=================================================="
