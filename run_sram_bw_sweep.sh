#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

BWS="${SRAM_BW_LIST:-128 256 512 1024}"
BASE_DIR="${SWEEP_ROOT_DIR:-sim_results_sram_bw_sweep}"
ACCELERATOR="${ACCELERATOR:-baseline}"
USE_SCALESIM_SYSTOLIC="${USE_SCALESIM_SYSTOLIC:-1}"

mkdir -p "$BASE_DIR"

for bw in $BWS; do
  out_dir="$BASE_DIR/${ACCELERATOR}_bw_${bw}Bpc"
  log_dir="$out_dir/logs"
  mkdir -p "$out_dir" "$log_dir"

  echo "=================================================="
  echo "SRAM bandwidth sweep: bw=${bw} B/cycle -> $out_dir"
  echo "=================================================="

  SIM_RESULTS_DIR="$out_dir" \
  SIM_LOG_DIR="$log_dir" \
  OUTPUT_JSON="report_bw_${bw}.json" \
  INTER_LAYER_SRAM_BW="$bw" \
  ACCELERATOR="$ACCELERATOR" \
  USE_SCALESIM_SYSTOLIC="$USE_SCALESIM_SYSTOLIC" \
  ./run_sim_with_real_eval.sh

done
