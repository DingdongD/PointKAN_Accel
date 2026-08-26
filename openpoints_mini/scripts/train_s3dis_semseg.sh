#!/usr/bin/env bash
# S3DIS 室内 HDF5 语义分割（ISEDA data_config），可选 PointNeXt / PointVector。
# 用法: bash openpoints_mini/scripts/train_s3dis_semseg.sh --test_area 5 --backbone pointvector --variant l
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
exec python -m openpoints_mini.train_s3dis_pointnext_iseda26_data "$@"
