#!/usr/bin/env bash
# =============================================================================
# 实验 6：acc / pointcim 两种模式
#         × PointNet++ 全部模型推理仿真
#         【NeuroSim 权重与输入激活均设为 4-bit】
#
# 说明：
#   4-bit 量化仅对 NeuroSim 仿真路径（acc / pointcim）有意义；
#   baseline / base_v2 使用 SCALE-Sim，无 NeuroSim bit-width 参数，
#   因此本脚本仅覆盖 acc 和 pointcim 两种模式。
#
# 覆盖网络：
#   - pointnet2-cls-ssg       ModelNet40 分类       1024 pts  40 类
#   - pointnet2-partseg-ssg   ShapeNet 零件分割     1024 pts  50 类
#   - pointnet2-sem-ssg       S3DIS 语义分割        4096 pts  13 类
#
# 覆盖模式：acc  pointcim
#
# 使用：
#   cd /workspace/PointKAN-pytorch
#   bash experiments_scripts/06_four_modes_pointnet2_4bit.sh
#
# 可选环境变量：
#   ACCELERATORS  空格分隔的模式列表（默认：acc pointcim）
#   MAX_BATCHES   限制每个 run 的 batch 数（默认 1）
#   DEVICE        cuda（默认）或 cpu
#   WL_WEIGHT     权重位宽（默认 4）
#   WL_ACTIVATION 激活位宽（默认 4）
# =============================================================================
set -euo pipefail

ROOT="/workspace/PointKAN-pytorch"
cd "$ROOT"

LOG_DIR="$ROOT/experiments_scripts/logs/06_acc_pointcim_pointnet2_4bit"
mkdir -p "$LOG_DIR"

SWEEP_MODES=(${ACCELERATORS:-acc pointcim})
MAX_BATCHES="${MAX_BATCHES:-1}"
DEVICE="${DEVICE:-cuda}"
WL_WEIGHT="${WL_WEIGHT:-4}"
WL_ACTIVATION="${WL_ACTIVATION:-4}"

# 模型列表：network:num_points:num_classes
declare -a MODELS=(
    "pointnet2-cls-ssg:1024:40"
    "pointnet2-partseg-ssg:1024:50"
    "pointnet2-sem-ssg:4096:13"
)

# 校验：只允许 acc / pointcim
for _m in "${SWEEP_MODES[@]}"; do
    if [ "$_m" != "acc" ] && [ "$_m" != "pointcim" ]; then
        echo "ERROR: 本脚本仅支持 acc / pointcim 模式，不支持 '$_m'（4-bit 对 SCALE-Sim 路径无效）" >&2
        exit 1
    fi
done

echo "=========================================================="
echo "  实验 6：${WL_WEIGHT}-bit 权重 / ${WL_ACTIVATION}-bit 激活 + PointNet++ 全部模型"
echo "  模式列表: ${SWEEP_MODES[*]}"
echo "  max_batches=$MAX_BATCHES  device=$DEVICE"
echo "  开始: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="

run_one() {
    local acc="$1"
    local network="$2"
    local num_pts="$3"
    local num_cls="$4"

    local tag="${network}_${acc}_p${num_pts}_w${WL_WEIGHT}a${WL_ACTIVATION}bit"
    local results_dir="$ROOT/sim_results_${tag}"
    local log_file="$LOG_DIR/${tag}.log"

    echo ""
    echo "----------------------------------------------------------"
    echo "  [$(date '+%Y-%m-%d %H:%M:%S')] 运行: network=$network  accelerator=$acc  points=$num_pts"
    echo "  weight=${WL_WEIGHT}bit  activation=${WL_ACTIVATION}bit"
    echo "  results -> $results_dir"
    echo "----------------------------------------------------------"

    local base_env=(
        ACCELERATOR="$acc"
        NETWORK="$network"
        NUM_POINTS="$num_pts"
        NUM_CLASSES="$num_cls"
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
        NEUROSIM_WL_WEIGHT="$WL_WEIGHT"
        NEUROSIM_WL_ACTIVATION="$WL_ACTIVATION"
        NEUROSIM_NUM_COL_MUXED="$WL_WEIGHT"
        LOGIC_CLK_FREQ=1000000000
        PROGRESS_INTERVAL_S=30
        KNN_CONV_PIPELINE_MODE=strict
        BN_EN=1
        QD_EN=1
        EXPORT_MEM_TXN=1
        SIM_RESULTS_DIR="$results_dir"
        SIM_LOG_DIR="$ROOT/sim_log"
        OUTPUT_JSON=report.json
    )

    # S3DIS 附加通道数
    if [ "$network" = "pointnet2-sem-ssg" ]; then
        base_env+=(INPUT_CHANNELS=12)
    fi

    env "${base_env[@]}" \
        bash "$ROOT/run_sim_with_real_eval.sh" \
        --pointcim_sa_conv_pipeline \
        2>&1 | tee "$log_file"

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
echo "  实验 6 全部完成：$(date '+%Y-%m-%d %H:%M:%S')"
echo "  日志目录: $LOG_DIR"
echo "=========================================================="
