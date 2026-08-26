#!/usr/bin/env bash
# =============================================================================
# 实验 8B：PointCIM 模式在 PointKAN-Elite 全模型上的推理仿真
#
# 覆盖模型：
#   - pointkan-elite         (ModelNet40 分类,  1024 pts, 40 类)
#   - pointkan-elite-scanobj (ScanObjectNN 分类, 1024 pts, 15 类)
#   - pointkan-elite-partseg (ShapeNet  零件分割, 2048 pts, 50 类)
#
# 使用：
#   cd /workspace/PointKAN-pytorch
#   bash experiments_scripts/08b_pointkan_pointcim_all_models.sh
#
# 可选环境变量：
#   MAX_BATCHES   限制每个 run 的 batch 数（默认 1）
#   DEVICE        cuda（默认）或 cpu
#   LIGHTWEIGHT_EXPORT  1=减少超大体积导出（默认 1）
# =============================================================================
set -euo pipefail

ROOT="/workspace/PointKAN-pytorch"
cd "$ROOT"

LOG_DIR="$ROOT/experiments_scripts/logs/08b_pointkan_pointcim_all_models"
mkdir -p "$LOG_DIR"

MAX_BATCHES="${MAX_BATCHES:-1}"
DEVICE="${DEVICE:-cuda}"
LIGHTWEIGHT_EXPORT="${LIGHTWEIGHT_EXPORT:-1}"

COMMON_ENV=(
    ACCELERATOR=pointcim
    USE_REAL_DATA=1
    DEVICE="$DEVICE"
    BATCH_SIZE=1
    MAX_BATCHES="$MAX_BATCHES"
    NEUROSIM_FULL=1
    NEUROSIM_PYBIND=0
    NEUROSIM_ROOT=/workspace/DNN_NeuroSim_V1.4/Inference_pytorch
    NEUROSIM_CLK_FREQ=100000000
    NEUROSIM_ADC_CLK_FREQ=1000000000
    NEUROSIM_TECHNODE=32
    LOGIC_CLK_FREQ=1000000000
    PROGRESS_INTERVAL_S=30
    KNN_CONV_PIPELINE_MODE=strict
    BN_EN=1
    QD_EN=1
    EXPORT_MEM_TXN=1
    LIGHTWEIGHT_EXPORT="$LIGHTWEIGHT_EXPORT"
)

run_model() {
    local network="$1"
    local num_points="$2"
    local tag="$3"
    local results_dir="$ROOT/sim_results_${tag}"
    local log_file="$LOG_DIR/${tag}.log"

    echo ""
    echo "=========================================================="
    echo "  [$(date '+%Y-%m-%d %H:%M:%S')] 开始: $tag"
    echo "  network=$network  num_points=$num_points  accelerator=pointcim"
    echo "  results -> $results_dir"
    echo "=========================================================="

    env "${COMMON_ENV[@]}" \
        NETWORK="$network" \
        NUM_POINTS="$num_points" \
        SIM_RESULTS_DIR="$results_dir" \
        SIM_LOG_DIR="$ROOT/sim_log" \
        OUTPUT_JSON=report.json \
        bash "$ROOT/run_sim_with_real_eval.sh" \
        --pointcim_sa_conv_pipeline \
        2>&1 | tee "$log_file"

    echo "  [$(date '+%Y-%m-%d %H:%M:%S')] 完成: $tag"
}

run_model "pointkan-elite"         1024 "pointkan-elite_pointcim_p1024"
run_model "pointkan-elite-scanobj" 1024 "pointkan-elite-scanobj_pointcim_p1024"
run_model "pointkan-elite-partseg" 2048 "pointkan-elite-partseg_pointcim_p2048"

echo ""
echo "=========================================================="
echo "  实验 8B 全部完成：$(date '+%Y-%m-%d %H:%M:%S')"
echo "  日志目录: $LOG_DIR"
echo "=========================================================="
