#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
SIZES_KB=(256)
OUT_ROOT="sim_results_filter_sram_sweep"
mkdir -p "$OUT_ROOT"
SUMMARY_CSV="$OUT_ROOT/filter_sram_dram_sweep.csv"
echo "filter_sram_kb,dram_energy_nj,dram_share_pct,total_energy_nj,weight_dram_trace_bytes" > "$SUMMARY_CSV"
for sz in "${SIZES_KB[@]}"; do
  bytes=$((sz * 1024))
  echo "[filter_sram_sweep] Running filter SRAM ${sz}KB"
  ACCELERATOR=baseline USE_SCALESIM_SYSTOLIC=1 MAX_BATCHES="${MAX_BATCHES:-1}" BATCH_SIZE="${BATCH_SIZE:-1}" \
  SYSTOLIC_SRAM_SIZE_BYTES=32768 \
  SYSTOLIC_FILTER_SRAM_SIZE_BYTES="$bytes" \
  SYSTOLIC_IFMAP_SRAM_SIZE_BYTES="${SYSTOLIC_IFMAP_SRAM_SIZE_BYTES:-32768}" \
  SYSTOLIC_OFMAP_SRAM_SIZE_BYTES="${SYSTOLIC_OFMAP_SRAM_SIZE_BYTES:-32768}" \
  SIM_RESULTS_DIR="${OUT_ROOT}/run_${sz}kb" ./run_sim_with_real_eval.sh >/tmp/filter_sram_${sz}kb.log 2>&1
  run_dir="${OUT_ROOT}/run_${sz}kb"
  python3 - <<PY
import csv, json
from pathlib import Path
sz = ${sz}
base = Path('$SCRIPT_DIR') / '$OUT_ROOT' / f'run_{sz}kb'
report = json.loads((base / 'report.json').read_text())
rows = list(csv.DictReader((base / 'scalesim_conv_linear_layers.csv').open()))
weight_trace = sum(int(r.get('weight_dram_trace_bytes') or 0) for r in rows)
dram = float(report.get('full_operator_breakdown', {}).get('DRAM', 0.0) or 0.0)
total = float(report.get('energy_nj', 0.0) or 0.0)
share = (dram / total * 100.0) if total > 0 else 0.0
with open(Path('$SCRIPT_DIR') / '$SUMMARY_CSV', 'a') as f:
    f.write(f"{sz},{dram:.6f},{share:.6f},{total:.6f},{weight_trace}\n")
print(f"filter_sram_kb={sz} dram_energy_nj={dram:.6f} dram_share_pct={share:.6f} total_energy_nj={total:.6f} weight_dram_trace_bytes={weight_trace}")
PY
done
