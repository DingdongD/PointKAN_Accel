#!/bin/bash
# 一键跑三任务：baseline(W8A8+RTN) vs PWL v2 + W8A8+RTN
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/run_pwl_v2_w8a8_rtn_modelnet40.sh" "${1:-}"
bash "${SCRIPT_DIR}/run_pwl_v2_w8a8_rtn_scanobjectnn.sh" "${2:-}"
bash "${SCRIPT_DIR}/run_pwl_v2_w8a8_rtn_partseg.sh" "${3:-}"

echo "All PWL+W8A8 evaluations finished."

