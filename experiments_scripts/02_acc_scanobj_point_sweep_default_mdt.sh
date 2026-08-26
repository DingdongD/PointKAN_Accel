#!/usr/bin/env bash
# =============================================================================
# 实验 2：ACC 模式 × pointkan-elite-scanobj × 点云规模 Sweep
#         【MDT SRAM Buffer 保持默认（4 KB）】
#
# 覆盖点数：1024 / 2048 / 4096 / 8192 / 16384 / 32768
#
# 使用：
#   cd /workspace/PointKAN-pytorch
#   bash experiments_scripts/02_acc_scanobj_point_sweep_default_mdt.sh
#
# 可选环境变量：
#   POINTS      空格分隔的点数列表（默认：1024 2048 4096 8192 16384 32768）
#   MAX_BATCHES 限制每个 run 的 batch 数（默认 1）
#   DEVICE      cuda（默认）或 cpu
# =============================================================================
set -euo pipefail

ROOT="/workspace/PointKAN-pytorch"
cd "$ROOT"

LOG_DIR="$ROOT/experiments_scripts/logs/02_acc_scanobj_sweep_default_mdt"
mkdir -p "$LOG_DIR"

SWEEP_POINTS=(${POINTS:-1024 2048 4096 8192 16384 32768})
MAX_BATCHES="${MAX_BATCHES:-1}"
DEVICE="${DEVICE:-cuda}"

NETWORK="pointkan-elite-scanobj"

echo "=========================================================="
echo "  实验 2：ACC + scanobj + 点云 Sweep（MDT 默认 4KB）"
echo "  点数列表: ${SWEEP_POINTS[*]}"
echo "  max_batches=$MAX_BATCHES  device=$DEVICE"
echo "  开始: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="

for NUM_POINTS in "${SWEEP_POINTS[@]}"; do
    TAG="${NETWORK}_acc_p${NUM_POINTS}_mdt_default"
    RESULTS_DIR="$ROOT/sim_results_${TAG}"
    LOG_FILE="$LOG_DIR/${TAG}.log"

    echo ""
    echo "----------------------------------------------------------"
    echo "  [$(date '+%Y-%m-%d %H:%M:%S')] 运行: num_points=$NUM_POINTS"
    echo "  results -> $RESULTS_DIR"
    echo "----------------------------------------------------------"

    env \
        ACCELERATOR=acc \
        NETWORK="$NETWORK" \
        NUM_POINTS="$NUM_POINTS" \
        USE_REAL_DATA=1 \
        DEVICE="$DEVICE" \
        BATCH_SIZE=1 \
        MAX_BATCHES="$MAX_BATCHES" \
        NEUROSIM_FULL=1 \
        NEUROSIM_PYBIND=0 \
        NEUROSIM_ROOT=/workspace/DNN_NeuroSim_V1.4/Inference_pytorch \
        NEUROSIM_CLK_FREQ=100000000 \
        NEUROSIM_ADC_CLK_FREQ=1000000000 \
        NEUROSIM_TECHNODE=32 \
        LOGIC_CLK_FREQ=1000000000 \
        PROGRESS_INTERVAL_S=30 \
        KNN_CONV_PIPELINE_MODE=strict \
        BN_EN=1 \
        QD_EN=1 \
        EXPORT_MEM_TXN=1 \
        SIM_RESULTS_DIR="$RESULTS_DIR" \
        SIM_LOG_DIR="$ROOT/sim_log" \
        OUTPUT_JSON=report.json \
        bash "$ROOT/run_sim_with_real_eval.sh" \
        --pointcim_sa_conv_pipeline \
        2>&1 | tee "$LOG_FILE"

    echo "  [$(date '+%Y-%m-%d %H:%M:%S')] 完成: num_points=$NUM_POINTS"
done

echo ""
echo "=========================================================="
echo "  实验 2 全部完成：$(date '+%Y-%m-%d %H:%M:%S')"
echo "  日志目录: $LOG_DIR"
echo "=========================================================="
