#!/usr/bin/env bash
# =============================================================================
# 实验 5：baseline / base_v2 / acc / pointcim 四种模式
#         × PointNet++ 全部模型推理仿真（8-bit 权重与激活）
#
# 覆盖网络：
#   - pointnet2-cls-ssg       ModelNet40 分类       1024 pts  40 类
#   - pointnet2-partseg-ssg   ShapeNet 零件分割     1024 pts  50 类
#   - pointnet2-sem-ssg       S3DIS 语义分割        4096 pts  13 类
#
# 覆盖模式：baseline  base_v2  acc  pointcim
#
# 使用：
#   cd /workspace/PointKAN-pytorch
#   bash experiments_scripts/05_three_modes_pointnet2_all_models.sh
#
# 可选环境变量：
#   ACCELERATORS 空格分隔的模式列表（默认：baseline base_v2 acc pointcim）
#   MAX_BATCHES  限制每个 run 的 batch 数（默认 1）
#   DEVICE       cuda（默认）或 cpu
#   LIGHTWEIGHT_EXPORT  1=保留最终 PPA 与关键 breakdown，关闭超大体积明细导出（默认 1）
#   ALLOW_DUPLICATE_RUNS  1=允许相同 results_dir 的任务重复启动（默认 0，即检测到重复则跳过）
# =============================================================================
set -euo pipefail

ROOT="/workspace/PointKAN-pytorch"
cd "$ROOT"

LOG_DIR="$ROOT/experiments_scripts/logs/05_four_modes_pointnet2"
mkdir -p "$LOG_DIR"

SWEEP_MODES=(${ACCELERATORS:-baseline base_v2 acc pointcim})
MAX_BATCHES="${MAX_BATCHES:-1}"
DEVICE="${DEVICE:-cuda}"
LIGHTWEIGHT_EXPORT="${LIGHTWEIGHT_EXPORT:-1}"
ALLOW_DUPLICATE_RUNS="${ALLOW_DUPLICATE_RUNS:-0}"

# 模型列表：network:num_points:num_classes
declare -a MODELS=(
    "pointnet2-cls-ssg:1024:40"
    "pointnet2-partseg-ssg:1024:50"
    "pointnet2-sem-ssg:4096:13"
)

echo "=========================================================="
echo "  实验 5：四种模式 + PointNet++ 全部模型（8-bit）"
echo "  模式列表: ${SWEEP_MODES[*]}"
echo "  max_batches=$MAX_BATCHES  device=$DEVICE"
echo "  开始: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="

is_sim_results_dir_running() {
    local results_dir="$1"
    ps -eo args= | grep -F "python -m pointkan_sim.run_sim_with_real_eval" | grep -F -- "--sim_results_dir $results_dir" >/dev/null 2>&1
}

run_one() {
    local acc="$1"
    local network="$2"
    local num_pts="$3"
    local num_cls="$4"

    local tag="${network}_${acc}_p${num_pts}"
    local results_dir="$ROOT/sim_results_${tag}"
    local log_file="$LOG_DIR/${tag}.log"

    echo ""
    echo "----------------------------------------------------------"
    echo "  [$(date '+%Y-%m-%d %H:%M:%S')] 运行: network=$network  accelerator=$acc  points=$num_pts"
    echo "  results -> $results_dir"
    echo "----------------------------------------------------------"

    if [ "$ALLOW_DUPLICATE_RUNS" != "1" ] && is_sim_results_dir_running "$results_dir"; then
        echo "  [$(date '+%Y-%m-%d %H:%M:%S')] 检测到同一 results_dir 已有运行中任务，跳过: $results_dir"
        return 0
    fi

    local base_env=(
        ACCELERATOR="$acc"
        NETWORK="$network"
        NUM_POINTS="$num_pts"
        NUM_CLASSES="$num_cls"
        USE_REAL_DATA=1
        DEVICE="$DEVICE"
        BATCH_SIZE=1
        MAX_BATCHES="$MAX_BATCHES"
        LOGIC_CLK_FREQ=1000000000
        PROGRESS_INTERVAL_S=30
        KNN_CONV_PIPELINE_MODE=strict
        BN_EN=1
        EXPORT_MEM_TXN=1
        SIM_RESULTS_DIR="$results_dir"
        SIM_LOG_DIR="$ROOT/sim_log"
        OUTPUT_JSON=report.json
    )

    # S3DIS 附加通道数
    if [ "$network" = "pointnet2-sem-ssg" ]; then
        base_env+=(INPUT_CHANNELS=12)
    fi

    if [ "$acc" = "acc" ]; then
        env "${base_env[@]}" \
            LIGHTWEIGHT_EXPORT="$LIGHTWEIGHT_EXPORT" \
            NEUROSIM_FULL=1 \
            NEUROSIM_PYBIND=0 \
            NEUROSIM_ROOT=/workspace/DNN_NeuroSim_V1.4/Inference_pytorch \
            NEUROSIM_CLK_FREQ=100000000 \
            NEUROSIM_ADC_CLK_FREQ=1000000000 \
            NEUROSIM_TECHNODE=32 \
            QD_EN=1 \
            bash "$ROOT/run_sim_with_real_eval.sh" \
            --pointcim_sa_conv_pipeline \
            2>&1 | tee "$log_file"

    elif [ "$acc" = "pointcim" ]; then
        env "${base_env[@]}" \
            LIGHTWEIGHT_EXPORT="$LIGHTWEIGHT_EXPORT" \
            NEUROSIM_FULL=1 \
            NEUROSIM_PYBIND=0 \
            NEUROSIM_ROOT=/workspace/DNN_NeuroSim_V1.4/Inference_pytorch \
            NEUROSIM_CLK_FREQ=100000000 \
            NEUROSIM_ADC_CLK_FREQ=1000000000 \
            NEUROSIM_TECHNODE=32 \
            QD_EN=1 \
            bash "$ROOT/run_sim_with_real_eval.sh" \
            --pointcim_sa_conv_pipeline \
            2>&1 | tee "$log_file"

    else
        env "${base_env[@]}" \
            LIGHTWEIGHT_EXPORT="$LIGHTWEIGHT_EXPORT" \
            NEUROSIM_FULL=0 \
            USE_SCALESIM_SYSTOLIC=1 \
            SCALE_SIM_ROOT=/workspace/SCALE-Sim \
            QD_EN=0 \
            bash "$ROOT/run_sim_with_real_eval.sh" \
            2>&1 | tee "$log_file"
    fi

    echo "  [$(date '+%Y-%m-%d %H:%M:%S')] 完成: network=$network  accelerator=$acc"
}

for ACC in "${SWEEP_MODES[@]}"; do
    for MODEL_SPEC in "${MODELS[@]}"; do
        IFS=':' read -r NET PTS CLS <<< "$MODEL_SPEC"
        run_one "$ACC" "$NET" "$PTS" "$CLS"
    done
done

echo ""
echo "=========================================================="
echo "  实验 5 全部完成：$(date '+%Y-%m-%d %H:%M:%S')"
echo "  日志目录: $LOG_DIR"
echo "=========================================================="
