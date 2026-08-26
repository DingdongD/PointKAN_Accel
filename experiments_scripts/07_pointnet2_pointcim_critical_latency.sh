#!/usr/bin/env bash
# =============================================================================
# 实验 7：PointNet++ × pointcim（可选 acc）
#         基于 NeuroSim 关键路径 clkPeriod 估算 latency（critical clock）
#
# 说明：
#   - 通过 NEUROSIM_CLK_MODE=critical 让仿真在汇总/注入时按 clkPeriod（最坏关键路径周期）
#     估算 Conv/Linear 的时序，从而得到一份“去掉 NEUROSIM_CLK_FREQ 限速”的 latency 口径。
#   - energy 不随时钟模式改变（仍是同一份能耗估算）。
#
# 覆盖网络：
#   - pointnet2-cls-ssg       ModelNet40 分类       1024 pts  40 类
#   - pointnet2-partseg-ssg   ShapeNet 零件分割     1024 pts  50 类
#   - pointnet2-sem-ssg       S3DIS 语义分割        4096 pts  13 类
#
# 默认只跑 pointcim；如需同时跑 acc，可设置：ACCELERATORS="acc pointcim"
#
# 使用：
#   cd /workspace/PointKAN-pytorch
#   bash experiments_scripts/07_pointnet2_pointcim_critical_latency.sh
#
# 可选环境变量：
#   ACCELERATORS  空格分隔的模式列表（默认：pointcim）
#   MAX_BATCHES   限制每个 run 的 batch 数（默认 1）
#   DEVICE        cuda（默认）或 cpu
#   WL_WEIGHT     权重位宽（默认 8；4 表示 4-bit）
#   WL_ACTIVATION 激活位宽（默认 8；4 表示 4-bit）
# =============================================================================
set -euo pipefail

ROOT="/workspace/PointKAN-pytorch"
cd "$ROOT"

LOG_DIR="$ROOT/experiments_scripts/logs/07_pointnet2_critical_latency"
mkdir -p "$LOG_DIR"

SWEEP_MODES=(${ACCELERATORS:-pointcim})
MAX_BATCHES="${MAX_BATCHES:-1}"
DEVICE="${DEVICE:-cuda}"
WL_WEIGHT="${WL_WEIGHT:-8}"
WL_ACTIVATION="${WL_ACTIVATION:-8}"

# 模型列表：network:num_points:num_classes
declare -a MODELS=(
    "pointnet2-cls-ssg:1024:40"
    "pointnet2-partseg-ssg:1024:50"
    "pointnet2-sem-ssg:4096:13"
)

# 校验：只允许 acc / pointcim（critical clock 口径仅对 NeuroSim 路径有意义）
for _m in "${SWEEP_MODES[@]}"; do
    if [ "$_m" != "acc" ] && [ "$_m" != "pointcim" ]; then
        echo "ERROR: 本脚本仅支持 acc / pointcim 模式，不支持 '$_m'（critical clock 仅对 NeuroSim 路径有意义）" >&2
        exit 1
    fi
done

echo "=========================================================="
echo "  实验 7：PointNet++ + critical latency (clkPeriod)"
echo "  模式列表: ${SWEEP_MODES[*]}"
echo "  wl_weight=${WL_WEIGHT}  wl_activation=${WL_ACTIVATION}"
echo "  max_batches=$MAX_BATCHES  device=$DEVICE"
echo "  开始: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="

run_one() {
    local acc="$1"
    local network="$2"
    local num_pts="$3"
    local num_cls="$4"

    local tag="${network}_${acc}_p${num_pts}_w${WL_WEIGHT}a${WL_ACTIVATION}bit_critclk"
    local results_dir="$ROOT/sim_results_${tag}"
    local log_file="$LOG_DIR/${tag}.log"

    echo ""
    echo "----------------------------------------------------------"
    echo "  [$(date '+%Y-%m-%d %H:%M:%S')] 运行: network=$network  accelerator=$acc  points=$num_pts"
    echo "  critical clock: enabled (NEUROSIM_CLK_MODE=critical)"
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
        NEUROSIM_ADC_CLK_FREQ=1000000000
        NEUROSIM_TECHNODE=32
        NEUROSIM_WL_WEIGHT="$WL_WEIGHT"
        NEUROSIM_WL_ACTIVATION="$WL_ACTIVATION"
        NEUROSIM_NUM_COL_MUXED="$WL_WEIGHT"
        NEUROSIM_CLK_MODE=critical
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
echo "  实验 7 全部完成：$(date '+%Y-%m-%d %H:%M:%S')"
echo "  日志目录: $LOG_DIR"
echo "=========================================================="
