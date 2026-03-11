#!/usr/bin/env bash
# 根据 outputs/ 下已有 profile 结果绘制 Latency / GPU·CPU Energy 柱状图与功耗曲线
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "$ROOT/scripts/plot_profile_results.py" --out-dir "$ROOT/outputs" --out "$ROOT/outputs/profile_summary.png" --curves "$ROOT/outputs/profile_curves.png"
