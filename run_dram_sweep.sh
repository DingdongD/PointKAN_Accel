#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

BWS="${DRAM_BW_LIST:-16 26 32 64}"
CHANNELS="${DRAM_CHANNEL_LIST:-1 2 4}"
OUT_ROOT="${DRAM_SWEEP_OUT_ROOT:-sim_results_dram_sweep}"
ACCELERATOR="${ACCELERATOR:-baseline}"
USE_SCALESIM_SYSTOLIC="${USE_SCALESIM_SYSTOLIC:-1}"
USE_REAL_DATA="${USE_REAL_DATA:-1}"
MAX_BATCHES="${MAX_BATCHES:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"

mkdir -p "$OUT_ROOT"

for ch in $CHANNELS; do
  for bw in $BWS; do
    out_dir="$OUT_ROOT/${ACCELERATOR}_dram_bw_${bw}Bpc_ch${ch}"
    echo "DRAM sweep: channels=${ch} bw=${bw} B/cycle -> $out_dir"
    DRAM_CHANNELS="$ch" \
    DRAM_BW_BYTES_PER_CYCLE="$bw" \
    SIM_RESULTS_DIR="$out_dir" \
    USE_SCALESIM_SYSTOLIC="$USE_SCALESIM_SYSTOLIC" \
    USE_REAL_DATA="$USE_REAL_DATA" \
    MAX_BATCHES="$MAX_BATCHES" \
    BATCH_SIZE="$BATCH_SIZE" \
    ACCELERATOR="$ACCELERATOR" \
    ./run_sim_with_real_eval.sh
  done
done
