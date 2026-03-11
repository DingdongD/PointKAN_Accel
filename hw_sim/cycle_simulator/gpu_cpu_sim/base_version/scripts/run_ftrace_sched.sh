#!/usr/bin/env bash
set -euo pipefail

# 必须由外部传入待分析命令
CMD="${CMD:?Please set CMD='python infer.py ...'}"

# 输出目录
OUT_DIR="${OUT_DIR:-./outputs/ftrace}"
mkdir -p "$OUT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"

# 使用 trace-cmd 抓取调度相关事件
# - sched_switch     : 线程切换
# - sched_wakeup     : 线程被唤醒
# - sched_wakeup_new : 新线程唤醒
sudo trace-cmd record \
  -o "$OUT_DIR/trace_${TS}.dat" \
  -e sched_switch \
  -e sched_wakeup \
  -e sched_wakeup_new \
  bash -lc "$CMD"

echo "trace saved: $OUT_DIR/trace_${TS}.dat"