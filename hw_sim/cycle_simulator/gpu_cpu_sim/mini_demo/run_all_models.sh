#!/usr/bin/env bash
# 一键跑齐 mini_demo 支持的全部任务：PointKAN 三任务 + ISEDA 九模型 + openpoints 三任务 × 两种 backbone。
# 每项各跑 GPU 与 CPU 侧 profiler（与 run.sh / run_isda.sh 同构）。单项失败不中断（|| true）。
#
# 环境变量（可选）：
#   REPEAT=2 MAX_BATCHES=20 BATCH=1 GPU_INDEX=0
#   只跑子集：ONLY=pointkan|isda|openpoints（默认全跑）
#   OPENPOINTS_VARIANT=l|xl — openpoints_sem + pointvector 时用（默认 l）
#   跳过 CPU：SKIP_CPU=1
#   跳过 GPU：SKIP_GPU=1
#
set -euo pipefail

DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -d "$DEMO_ROOT/PointKAN-elite" ]]; then
  export POINTKAN_ELITE_ROOT="$DEMO_ROOT/PointKAN-elite"
else
  REPO_ROOT="$(cd "$DEMO_ROOT/../../../../" 2>/dev/null && pwd || true)"
  export POINTKAN_ELITE_ROOT="${POINTKAN_ELITE_ROOT:-${REPO_ROOT}/PointKAN-elite}"
fi
if [[ -d "$DEMO_ROOT/pointnet2_ops_lib" ]]; then
  export PYTHONPATH="$DEMO_ROOT/pointnet2_ops_lib${PYTHONPATH:+:$PYTHONPATH}"
fi

REPEAT="${REPEAT:-2}"
MAX_BATCHES="${MAX_BATCHES:-20}"
BATCH="${BATCH:-1}"
GPU_INDEX="${GPU_INDEX:-0}"
ONLY="${ONLY:-}"
SKIP_CPU="${SKIP_CPU:-0}"
SKIP_GPU="${SKIP_GPU:-0}"
OPENPOINTS_VARIANT="${OPENPOINTS_VARIANT:-l}"

export REPEAT MAX_BATCHES BATCH GPU_INDEX

run_pointkan() {
  [[ -d "${POINTKAN_ELITE_ROOT}" ]] || {
    echo ">>> 跳过 PointKAN：未找到 POINTKAN_ELITE_ROOT=${POINTKAN_ELITE_ROOT}"
    return 0
  }
  local TASK
  for TASK in modelnet40 scanobjectnn part_seg; do
    if [[ "$SKIP_GPU" != "1" ]]; then
      echo "========== [ALL] PointKAN GPU: $TASK =========="
      python3 "$DEMO_ROOT/profiler.py" \
        --cmd "python3 $DEMO_ROOT/run_infer_random.py --task $TASK --device cuda --batch-size $BATCH --max-batches $MAX_BATCHES" \
        --mode gpu \
        --out-dir "$DEMO_ROOT/outputs/random_gpu_${TASK}" \
        --gpu-index "$GPU_INDEX" \
        --warmup 1 \
        --repeat "$REPEAT" \
        --print-inference-only || true
    fi
    if [[ "$SKIP_CPU" != "1" ]]; then
      echo "========== [ALL] PointKAN CPU: $TASK =========="
      python3 "$DEMO_ROOT/profiler.py" \
        --cmd "python3 $DEMO_ROOT/run_infer_random.py --task $TASK --device cpu --batch-size $BATCH --max-batches $MAX_BATCHES" \
        --mode cpu \
        --out-dir "$DEMO_ROOT/outputs/random_cpu_${TASK}" \
        --warmup 1 \
        --repeat "$REPEAT" \
        --print-inference-only || true
    fi
  done
}

run_isda() {
  local MODELS=(
    dgcnn pn2_ssg pn2_msg
    dgcnn_partseg dgcnn_semseg
    pn2_part_ssg pn2_part_msg pn2_sem_ssg pn2_sem_msg
  )
  local M
  for M in "${MODELS[@]}"; do
    if [[ "$SKIP_GPU" != "1" ]]; then
      echo "========== [ALL] ISDA GPU: $M =========="
      python3 "$DEMO_ROOT/profiler.py" \
        --cmd "python3 $DEMO_ROOT/run_infer_random.py --task isda_$M --device cuda --batch-size $BATCH --max-batches $MAX_BATCHES" \
        --mode gpu \
        --out-dir "$DEMO_ROOT/outputs/isda_gpu_${M}" \
        --gpu-index "$GPU_INDEX" \
        --warmup 1 \
        --repeat "$REPEAT" \
        --print-inference-only || true
    fi
    if [[ "$SKIP_CPU" != "1" ]]; then
      echo "========== [ALL] ISDA CPU: $M =========="
      python3 "$DEMO_ROOT/profiler.py" \
        --cmd "python3 $DEMO_ROOT/run_infer_random.py --task isda_$M --device cpu --batch-size $BATCH --max-batches $MAX_BATCHES" \
        --mode cpu \
        --out-dir "$DEMO_ROOT/outputs/isda_cpu_${M}" \
        --warmup 1 \
        --repeat "$REPEAT" \
        --print-inference-only || true
    fi
  done
}

run_openpoints() {
  local T BB base_cmd
  for T in openpoints_cls openpoints_part openpoints_sem; do
    for BB in pointnext pointvector; do
      base_cmd="python3 $DEMO_ROOT/run_infer_random.py --task $T --backbone $BB"
      if [[ "$T" == openpoints_sem && "$BB" == pointvector ]]; then
        base_cmd+=" --variant $OPENPOINTS_VARIANT"
      fi
      if [[ "$SKIP_GPU" != "1" ]]; then
        echo "========== [ALL] OpenPoints GPU: $T $BB =========="
        python3 "$DEMO_ROOT/profiler.py" \
          --cmd "$base_cmd --device cuda --batch-size $BATCH --max-batches $MAX_BATCHES" \
          --mode gpu \
          --out-dir "$DEMO_ROOT/outputs/all_openpoints_gpu_${T}_${BB}" \
          --gpu-index "$GPU_INDEX" \
          --warmup 1 \
          --repeat "$REPEAT" \
          --print-inference-only || true
      fi
      if [[ "$SKIP_CPU" != "1" ]]; then
        echo "========== [ALL] OpenPoints CPU: $T $BB =========="
        python3 "$DEMO_ROOT/profiler.py" \
          --cmd "$base_cmd --device cpu --batch-size $BATCH --max-batches $MAX_BATCHES" \
          --mode cpu \
          --out-dir "$DEMO_ROOT/outputs/all_openpoints_cpu_${T}_${BB}" \
          --warmup 1 \
          --repeat "$REPEAT" \
          --print-inference-only || true
      fi
    done
  done
}

echo ">>> run_all_models.sh：PointKAN + ISDA + OpenPoints（ONLY=${ONLY:-all}）"
echo ">>> POINTKAN_ELITE_ROOT=${POINTKAN_ELITE_ROOT}"
echo ">>> REPEAT=$REPEAT MAX_BATCHES=$MAX_BATCHES BATCH=$BATCH SKIP_GPU=$SKIP_GPU SKIP_CPU=$SKIP_CPU"

case "$ONLY" in
  pointkan)
    run_pointkan
    ;;
  isda)
    run_isda
    ;;
  openpoints)
    run_openpoints
    ;;
  "")
    run_pointkan
    run_isda
    run_openpoints
    ;;
  *)
    echo "未知 ONLY=$ONLY（可用 pointkan | isda | openpoints 或留空）" >&2
    exit 1
    ;;
esac

echo ">>> 全部调度结束。PointKAN/ISDA 结果与 run.sh、run_isda.sh 相同目录前缀 random_* / isda_*；OpenPoints 在 all_openpoints_*。"
if python3 -c "import matplotlib" 2>/dev/null; then
  echo ">>> 生成 PointKAN / ISDA 汇总图（与单独跑 run.sh / run_isda.sh 后相同）..."
  python3 "$DEMO_ROOT/plot_cpu_gpu_compare.py" --out-dir "$DEMO_ROOT/outputs" --source random --out "$DEMO_ROOT/outputs/cpu_gpu_compare.png" 2>/dev/null || true
  python3 "$DEMO_ROOT/plot_cpu_gpu_compare.py" --out-dir "$DEMO_ROOT/outputs" --source isda --out "$DEMO_ROOT/outputs/cpu_gpu_compare_isda.png" 2>/dev/null || true
fi
