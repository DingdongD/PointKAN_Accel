#!/usr/bin/env bash
# =============================================================================
# 实验 4：baseline / base_v2 / acc / pointcim 四种模式
#         × pointkan-elite-scanobj × 点云规模 Sweep
#         【MDT SRAM Buffer 保持默认 4 KB】
#
# 覆盖点数：1024 / 2048 / 4096 / 8192 / 16384 / 32768
# 覆盖模式：baseline  base_v2  acc  pointcim
#
# 使用：
#   cd /workspace/PointKAN-pytorch
#   bash experiments_scripts/04_three_modes_scanobj_point_sweep.sh
#   bash experiments_scripts/04_three_modes_scanobj_point_sweep.sh --mode acc --points "1024 2048" --parallel 2
#
# 可选环境变量：
#   POINTS       空格分隔的点数列表（默认：1024 2048 4096 8192 16384 32768）
#   ACCELERATORS 空格分隔的模式列表（默认：baseline base_v2 acc pointcim）
#   MAX_BATCHES  限制每个 run 的 batch 数（默认 1）
#   DEVICE       cuda（默认）或 cpu
#   LIGHTWEIGHT_EXPORT  1=acc/pointcim 保留最终 PPA 与关键 breakdown，关闭超大体积明细导出（默认 1）
#   ALLOW_DUPLICATE_RUNS  1=允许相同 results_dir 的任务重复启动（默认 0，即检测到重复则跳过）
# =============================================================================
set -euo pipefail

ROOT="/workspace/PointKAN-pytorch"
cd "$ROOT"

LOG_DIR="$ROOT/experiments_scripts/logs/04_four_modes_scanobj_sweep"
mkdir -p "$LOG_DIR"

usage() {
    cat <<'EOF'
Usage:
  bash experiments_scripts/04_three_modes_scanobj_point_sweep.sh [options]

Options:
  --mode MODE           追加单个模式；可重复传入，如 --mode baseline --mode acc
  --modes "A B C"       直接指定模式列表（空格或逗号分隔）
  --points "1024 2048"  指定点数列表（空格或逗号分隔）
  --parallel N          同时运行的任务数（默认 1）
  --device DEV          覆盖设备，如 cuda / cpu
  --max-batches N       覆盖 MAX_BATCHES
  --lightweight-export 0|1
  --allow-duplicate-runs 0|1
  --help                显示本帮助并退出
EOF
}

MAX_BATCHES="${MAX_BATCHES:-1}"
DEVICE="${DEVICE:-cuda}"
LIGHTWEIGHT_EXPORT="${LIGHTWEIGHT_EXPORT:-1}"
ALLOW_DUPLICATE_RUNS="${ALLOW_DUPLICATE_RUNS:-0}"
PARALLEL_JOBS="${PARALLEL_JOBS:-1}"
CLI_MODES=()
CLI_POINTS_RAW=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --mode)
            CLI_MODES+=("$2")
            shift 2
            ;;
        --modes)
            IFS=', ' read -r -a CLI_MODES <<< "$2"
            shift 2
            ;;
        --points)
            CLI_POINTS_RAW="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --max-batches)
            MAX_BATCHES="$2"
            shift 2
            ;;
        --lightweight-export)
            LIGHTWEIGHT_EXPORT="$2"
            shift 2
            ;;
        --allow-duplicate-runs)
            ALLOW_DUPLICATE_RUNS="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

SWEEP_POINTS=(${POINTS:-1024 2048 4096 8192 16384 32768})
SWEEP_MODES=(${ACCELERATORS:-baseline base_v2 acc pointcim})

NETWORK="pointkan-elite-scanobj"
MDT_SRAM_BYTES=4096   # 4 KB（acc/pointcim 模式默认值）

if [ "${#CLI_MODES[@]}" -gt 0 ]; then
    SWEEP_MODES=("${CLI_MODES[@]}")
fi
if [ -n "$CLI_POINTS_RAW" ]; then
    IFS=', ' read -r -a SWEEP_POINTS <<< "$CLI_POINTS_RAW"
fi
if ! [[ "$PARALLEL_JOBS" =~ ^[0-9]+$ ]] || [ "$PARALLEL_JOBS" -lt 1 ]; then
    echo "ERROR: --parallel must be a positive integer, got '$PARALLEL_JOBS'" >&2
    exit 1
fi

echo "=========================================================="
echo "  实验 4：四种模式 + scanobj + 点云 Sweep（MDT 4KB）"
echo "  模式列表: ${SWEEP_MODES[*]}"
echo "  点数列表: ${SWEEP_POINTS[*]}"
echo "  max_batches=$MAX_BATCHES  device=$DEVICE  parallel=$PARALLEL_JOBS"
echo "  开始: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="

is_sim_results_dir_running() {
    local results_dir="$1"
    ps -eo args= | grep -F "python -m pointkan_sim.run_sim_with_real_eval" | grep -F -- "--sim_results_dir $results_dir" >/dev/null 2>&1
}

run_one() {
    local acc="$1"
    local num_pts="$2"

    local tag="${NETWORK}_${acc}_p${num_pts}_mdt_4kb"
    local results_dir="$ROOT/sim_results_${tag}"
    local log_file="$LOG_DIR/${tag}.log"

    echo ""
    echo "----------------------------------------------------------"
    echo "  [$(date '+%Y-%m-%d %H:%M:%S')] 运行: accelerator=$acc  num_points=$num_pts"
    echo "  results -> $results_dir"
    echo "----------------------------------------------------------"

    if [ "$ALLOW_DUPLICATE_RUNS" != "1" ] && is_sim_results_dir_running "$results_dir"; then
        echo "  [$(date '+%Y-%m-%d %H:%M:%S')] 检测到同一 results_dir 已有运行中任务，跳过: $results_dir"
        return 0
    fi

    local base_env=(
        ACCELERATOR="$acc"
        NETWORK="$NETWORK"
        NUM_POINTS="$num_pts"
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

    if [ "$acc" = "acc" ]; then
        # ACC 模式：NeuroSim 全网络仿真，不含 RRAM 距离计算
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
            --fps_min_distance_table_sram_size_bytes "$MDT_SRAM_BYTES" \
            2>&1 | tee "$log_file"

    elif [ "$acc" = "pointcim" ]; then
        # PointCIM 模式：ACC 架构 + RRAM VMM 距离计算
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
            --fps_min_distance_table_sram_size_bytes "$MDT_SRAM_BYTES" \
            2>&1 | tee "$log_file"

    else
        # baseline / base_v2：SCALE-Sim 脉动阵列建模
        env "${base_env[@]}" \
            LIGHTWEIGHT_EXPORT="$LIGHTWEIGHT_EXPORT" \
            NEUROSIM_FULL=0 \
            USE_SCALESIM_SYSTOLIC=1 \
            SCALE_SIM_ROOT=/workspace/SCALE-Sim \
            QD_EN=0 \
            bash "$ROOT/run_sim_with_real_eval.sh" \
            2>&1 | tee "$log_file"
    fi

    echo "  [$(date '+%Y-%m-%d %H:%M:%S')] 完成: accelerator=$acc  num_points=$num_pts"
}

wait_for_available_slot() {
    while [ "$(jobs -pr | wc -l | tr -d ' ')" -ge "$PARALLEL_JOBS" ]; do
        wait -n
    done
}

for ACC in "${SWEEP_MODES[@]}"; do
    for NUM_POINTS in "${SWEEP_POINTS[@]}"; do
        if [ "$PARALLEL_JOBS" -gt 1 ]; then
            wait_for_available_slot
            run_one "$ACC" "$NUM_POINTS" &
        else
            run_one "$ACC" "$NUM_POINTS"
        fi
    done
done

if [ "$PARALLEL_JOBS" -gt 1 ]; then
    wait
fi

echo ""
echo "=========================================================="
echo "  实验 4 全部完成：$(date '+%Y-%m-%d %H:%M:%S')"
echo "  日志目录: $LOG_DIR"
echo "=========================================================="
