#!/usr/bin/env bash
# PointKAN 随机数据推理最小 demo：一键在 CPU/GPU 上跑 2 次 × 20 批次 × batch_size=1，并统计推理时延与能耗
set -euo pipefail

DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# PointKAN-elite：优先 mini_demo 内拷贝，否则环境变量，否则仓库相对路径
if [[ -d "$DEMO_ROOT/PointKAN-elite" ]]; then
  export POINTKAN_ELITE_ROOT="$DEMO_ROOT/PointKAN-elite"
else
  REPO_ROOT="$(cd "$DEMO_ROOT/../../../../" 2>/dev/null && pwd || true)"
  export POINTKAN_ELITE_ROOT="${POINTKAN_ELITE_ROOT:-${REPO_ROOT}/PointKAN-elite}"
fi
# pointnet2_ops：若 mini_demo 内已拷贝 pointnet2_ops_lib，加入 PYTHONPATH 以便直接 import
if [[ -d "$DEMO_ROOT/pointnet2_ops_lib" ]]; then
  export PYTHONPATH="$DEMO_ROOT/pointnet2_ops_lib${PYTHONPATH:+:$PYTHONPATH}"
fi

if [[ ! -d "${POINTKAN_ELITE_ROOT}" ]]; then
  echo "错误: 未找到 PointKAN-elite 目录: ${POINTKAN_ELITE_ROOT}"
  echo "请设置环境变量: export POINTKAN_ELITE_ROOT=/path/to/PointKAN-elite 或将 PointKAN-elite 拷贝到 mini_demo 下"
  exit 1
fi

REPEAT="${REPEAT:-2}"
MAX_BATCHES="${MAX_BATCHES:-20}"
BATCH="${BATCH:-1}"
GPU_INDEX="${GPU_INDEX:-0}"
# 三任务：modelnet40, scanobjectnn, part_seg；若只跑单个任务可设置 TASK=modelnet40
TASKS=("modelnet40" "scanobjectnn" "part_seg")
if [[ -n "${TASK:-}" ]]; then
  TASKS=("$TASK")
fi

export REPEAT MAX_BATCHES BATCH GPU_INDEX

echo ">>> PointKAN random 推理 mini_demo：三任务 × (GPU + CPU)，${REPEAT} 次 run × ${MAX_BATCHES} 批 × batch=${BATCH}"
echo ">>> POINTKAN_ELITE_ROOT=${POINTKAN_ELITE_ROOT}"

for TASK in "${TASKS[@]}"; do
  export TASK
  echo "========== Random GPU: $TASK =========="
  python3 "$DEMO_ROOT/profiler.py" \
    --cmd "python3 $DEMO_ROOT/run_infer_random.py --task $TASK --device cuda --batch-size $BATCH --max-batches $MAX_BATCHES" \
    --mode gpu \
    --out-dir "$DEMO_ROOT/outputs/random_gpu_${TASK}" \
    --gpu-index "$GPU_INDEX" \
    --warmup 1 \
    --repeat "$REPEAT" \
    --print-inference-only || true

  echo "========== Random CPU: $TASK =========="
  python3 "$DEMO_ROOT/profiler.py" \
    --cmd "python3 $DEMO_ROOT/run_infer_random.py --task $TASK --device cpu --batch-size $BATCH --max-batches $MAX_BATCHES" \
    --mode cpu \
    --out-dir "$DEMO_ROOT/outputs/random_cpu_${TASK}" \
    --warmup 1 \
    --repeat "$REPEAT" \
    --print-inference-only || true
done

echo ">>> 完成。结果在 $DEMO_ROOT/outputs/random_gpu_<task>/ 与 random_cpu_<task>/ 下"
# 若已安装 matplotlib，生成 CPU vs GPU 对比图
if python3 -c "import matplotlib" 2>/dev/null; then
  echo ">>> 生成 CPU vs GPU 对比图..."
  python3 "$DEMO_ROOT/plot_cpu_gpu_compare.py" --out-dir "$DEMO_ROOT/outputs" --source random --out "$DEMO_ROOT/outputs/cpu_gpu_compare.png" || true
  echo ">>> 对比图: $DEMO_ROOT/outputs/cpu_gpu_compare.png"
fi
