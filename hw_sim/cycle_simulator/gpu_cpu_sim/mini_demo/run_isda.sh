#!/usr/bin/env bash
# ISEDA 移植网络（DGCNN / PointNet++）× (GPU + CPU) 推理 profiling，与 run.sh 同构。
set -euo pipefail

DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# pointnet2_ops：供 PointKAN-elite 等使用；isda_models 本路径不依赖该扩展
if [[ -d "$DEMO_ROOT/pointnet2_ops_lib" ]]; then
  export PYTHONPATH="$DEMO_ROOT/pointnet2_ops_lib${PYTHONPATH:+:$PYTHONPATH}"
fi

REPEAT="${REPEAT:-2}"
MAX_BATCHES="${MAX_BATCHES:-20}"
BATCH="${BATCH:-1}"
GPU_INDEX="${GPU_INDEX:-0}"
MODELS=(
  "dgcnn" "pn2_ssg" "pn2_msg"
  "dgcnn_partseg" "dgcnn_semseg"
  "pn2_part_ssg" "pn2_part_msg" "pn2_sem_ssg" "pn2_sem_msg"
)
if [[ -n "${ISDA_MODEL:-}" ]]; then
  MODELS=("$ISDA_MODEL")
fi

export REPEAT MAX_BATCHES BATCH GPU_INDEX

echo ">>> ISEDA 移植模型 random 推理：${#MODELS[@]} 个模型 × (GPU + CPU)，${REPEAT} 次 × ${MAX_BATCHES} 批 × batch=${BATCH}"

for M in "${MODELS[@]}"; do
  echo "========== ISDA GPU: $M =========="
  python3 "$DEMO_ROOT/profiler.py" \
    --cmd "python3 $DEMO_ROOT/run_infer_random.py --task isda_$M --device cuda --batch-size $BATCH --max-batches $MAX_BATCHES" \
    --mode gpu \
    --out-dir "$DEMO_ROOT/outputs/isda_gpu_${M}" \
    --gpu-index "$GPU_INDEX" \
    --warmup 1 \
    --repeat "$REPEAT" \
    --print-inference-only || true

  echo "========== ISDA CPU: $M =========="
  python3 "$DEMO_ROOT/profiler.py" \
    --cmd "python3 $DEMO_ROOT/run_infer_random.py --task isda_$M --device cpu --batch-size $BATCH --max-batches $MAX_BATCHES" \
    --mode cpu \
    --out-dir "$DEMO_ROOT/outputs/isda_cpu_${M}" \
    --warmup 1 \
    --repeat "$REPEAT" \
    --print-inference-only || true
done

echo ">>> 完成。结果在 $DEMO_ROOT/outputs/isda_gpu_<model>/ 与 isda_cpu_<model>/ 下"
if python3 -c "import matplotlib" 2>/dev/null; then
  echo ">>> 生成 ISEDA 模型 CPU vs GPU 对比图..."
  python3 "$DEMO_ROOT/plot_cpu_gpu_compare.py" --out-dir "$DEMO_ROOT/outputs" --source isda --out "$DEMO_ROOT/outputs/cpu_gpu_compare_isda.png" || true
fi
