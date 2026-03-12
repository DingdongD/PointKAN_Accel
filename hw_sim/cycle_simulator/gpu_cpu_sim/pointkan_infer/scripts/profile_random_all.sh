#!/usr/bin/env bash
# 仅 model(data) 推理：三任务 GPU + CPU 各跑一遍，获取 推理 latency 与 energy（不依赖 checkpoint/数据集）
# 配置：跑 2 次 run，每次 20 批次，每批次 1 样本；GPU 仅统计单卡（GPU_INDEX=0）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_BREAKDOWN="${RUN_BREAKDOWN:-0}"
WARMUP="${WARMUP:-1}"
REPEAT="${REPEAT:-2}"
MAX_BATCHES="${MAX_BATCHES:-20}"
BATCH="${BATCH:-1}"
GPU_INDEX="${GPU_INDEX:-0}"
export WARMUP REPEAT MAX_BATCHES BATCH GPU_INDEX

echo ">>> profile_random_all: 三任务 × (GPU + CPU)，仅 model 推理 latency + energy"
echo ">>> 配置: REPEAT=${REPEAT} 次 run，每次 ${MAX_BATCHES} 批次，每批次 ${BATCH} 样本；GPU 单卡 index=${GPU_INDEX}"
for TASK in modelnet40 scanobjectnn part_seg; do
  export TASK
  echo "========== Random GPU: $TASK =========="
  bash "$ROOT/scripts/profile_random_gpu.sh" || true
  echo "========== Random CPU: $TASK =========="
  bash "$ROOT/scripts/profile_random_cpu.sh" || true
done

echo ">>> 全部完成。推理 latency/energy 在 outputs/random_gpu_<task>/ 与 outputs/random_cpu_<task>/ 下"
if python3 -c "import matplotlib" 2>/dev/null; then
  echo ">>> 生成 CPU vs GPU 对比图..."
  python3 "$ROOT/scripts/plot_cpu_gpu_compare.py" --out-dir "$ROOT/outputs" --source random --out "$ROOT/outputs/cpu_gpu_compare.png" || true
  echo ">>> 对比图: $ROOT/outputs/cpu_gpu_compare.png"
fi
