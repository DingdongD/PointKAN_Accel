#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/PointKAN-pytorch"
MODE="${1:-all}"

run_one() {
  local network="$1"
  local num_points="$2"
  local num_classes="$3"
  shift 3

  cd "$ROOT"
  python scripts/sweep_pointnet2_pointcim_subarray_auto.py \
    --network "$network" \
    --num-points "$num_points" \
    --num-classes "$num_classes" \
    --subarray-rows 128 \
    --subarray-cols 128 \
    --wl-weight 8 \
    --wl-activation 8 \
    --max-batches 1 \
    --device cuda \
    --force \
    "$@"
}

case "$MODE" in
  cls)
    run_one pointnet2-cls-ssg 1024 40
    ;;
  partseg)
    run_one pointnet2-partseg-ssg 1024 50
    ;;
  semseg)
    run_one pointnet2-sem-ssg 4096 13
    ;;
  all)
    run_one pointnet2-cls-ssg 1024 40
    run_one pointnet2-partseg-ssg 1024 50
    run_one pointnet2-sem-ssg 4096 13
    ;;
  *)
    echo "Usage: $0 [cls|partseg|semseg|all]" >&2
    exit 1
    ;;
esac
