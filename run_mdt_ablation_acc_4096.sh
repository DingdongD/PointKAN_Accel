#!/usr/bin/env bash
set -euo pipefail

cd /workspace/PointKAN-pytorch

summary="/workspace/PointKAN-pytorch/ablation_acc_scanobj_4096_mdt_summary.csv"
sizes=("$@")
if [ ${#sizes[@]} -eq 0 ]; then
  sizes=(4 8 16 32)
fi

if [ ! -f "$summary" ] || [ "${RESET_SUMMARY:-0}" = "1" ]; then
  echo "mdt_kb,dram_nj,sram_nj,mdt_dram_nj,spill_read_events,spill_write_events,spill_read_bytes,spill_write_bytes" > "$summary"
fi

for kb in "${sizes[@]}"; do
  out="/workspace/PointKAN-pytorch/ablation_acc_scanobj_4096_mdt_${kb}kb_unified"
  rm -rf "$out"

  python -m pointkan_sim.run_sim_with_real_eval \
    --checkpoint PointKAN-elite/classification_ScanObjectNN/checkpoints/pointKANElite-scan_exp-42 \
    --num_points 4096 \
    --num_classes 15 \
    --sim_log_dir sim_log \
    --accelerator acc \
    --device cuda \
    --use_real_data \
    --batch_size 1 \
    --max_batches 1 \
    --neurosim_full \
    --neurosim_technode 32 \
    --logic_clk_freq 1000000000 \
    --output report.json \
    --bn_en 1 \
    --qd_en 1 \
    --export_memory_transactions \
    --ns_simulation_mode event \
    --fps_min_distance_table_sram_size_bytes $((kb * 1024)) \
    --network pointkan-elite-scanobj \
    --sim_results_dir "$out"

  KB="$kb" OUT="$out" python - <<'PY' >> "$summary"
import csv
import os

p = os.path.join(os.environ["OUT"], "runtime_memory_summary.csv")
kb = int(os.environ["KB"])
dram = 0.0
sram = 0.0
mdt = 0.0
spill_read_events = 0
spill_write_events = 0
spill_read_bytes = 0
spill_write_bytes = 0

with open(p, newline="") as f:
    r = csv.reader(f)
    next(r)
    for row in r:
        level, op, buf, layer, phase, res, events, bytes_, energy, lat = row
        energy = float(energy)
        if level == "DRAM":
            dram += energy
            if buf == "fps_min_distance_table_sram":
                mdt += energy
                if phase == "spill_read":
                    spill_read_events += int(events)
                    spill_read_bytes += int(bytes_)
                elif phase == "spill_write":
                    spill_write_events += int(events)
                    spill_write_bytes += int(bytes_)
        elif level == "SRAM":
            sram += energy

print(f"{kb},{dram:.6f},{sram:.6f},{mdt:.6f},{spill_read_events},{spill_write_events},{spill_read_bytes},{spill_write_bytes}")
PY
done
