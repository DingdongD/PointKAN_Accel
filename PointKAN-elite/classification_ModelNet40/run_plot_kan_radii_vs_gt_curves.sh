#!/bin/bash
# GT radii 曲线 vs KAN 预测曲线（横轴 FPS 步），自动扫描 one-click 目录。
# 非 MSE–prefix_ratio 图。
#
#   BASE=./runs/fps_radii_oneclick_all PR=0.10 K=512 ./run_plot_kan_radii_vs_gt_curves.sh
#
set -e
cd "$(dirname "$0")"
BASE="${BASE:-./runs/fps_radii_oneclick_all}"
PR="${PR:-0.10}"
K="${K:-512}"
MT="${MODEL_TYPE:-kan}"
OUT="${OUT:-}"
ND="${NO_DETAIL:-0}"
extra=()
[[ -n "$OUT" ]] && extra+=(--out_dir "$OUT")
[[ "$ND" == "1" ]] && extra+=(--no_detail)

exec python plot_kan_radii_vs_gt_curves.py \
  --auto_from "$BASE" \
  --auto_pr "$PR" \
  --auto_k "$K" \
  --auto_model_type "$MT" \
  "${extra[@]}" \
  "$@"
