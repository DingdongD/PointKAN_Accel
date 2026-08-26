#!/bin/bash
# Compare Vanilla GRF-KAN vs v2 PWL (Part Segmentation)
set -euo pipefail
IFS=$'\n\t'

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/pointkan/bin/python}"
WORKDIR="/workspace/PointKAN-pytorch/PointKAN-elite/part_segmentation"

OUT_JSON="${1:-${PWD}/my_pwl_compare_out/partseg_pwl_compare.json}"
mkdir -p "$(dirname "${OUT_JSON}")"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${OUT_JSON%.json}_${TS}.log"

cd "${WORKDIR}"
echo "Running PWL compare in ${WORKDIR}"

"${PYTHON_BIN}" run_inference_pwl_compare.py --out_json "${OUT_JSON}" 2>&1 | tee "${LOG_FILE}"

echo "Done. out_json: ${OUT_JSON}"
echo "Log: ${LOG_FILE}"

