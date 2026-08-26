#!/bin/bash
# 一键跑三数据集：PWL（v2）推理对比 Vanilla GRF-KAN
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/run_pwl_compare_modelnet40.sh"
bash "${SCRIPT_DIR}/run_pwl_compare_scanobjectnn.sh"
bash "${SCRIPT_DIR}/run_pwl_compare_partseg.sh"

echo "All PWL compare finished."

