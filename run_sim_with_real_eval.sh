#!/usr/bin/env bash
# 使用真实 checkpoint 与 ModelNet40 数据运行 PPA 仿真（真实前向 + 按样本串行仿真）
#
# 使用（在 PointKAN-pytorch 根目录）:
#   chmod +x run_sim_with_real_eval.sh && ./run_sim_with_real_eval.sh
#
# NeuroSim 环境变量与 CLI 对照、子进程 argv 约定、iso/自适应说明见:
#   docs/NEUROSIM_INTEGRATION_USAGE.md
#
# 环境变量（可选）:
#   CKPT              检查点路径，默认 PointKAN-elite/.../pointKANElite-pointkan_elite_exp-42
#   NUM_POINTS        点云点数，默认 1024
#   BATCH_SIZE        测试 batch 大小，默认 32
#   MAX_BATCHES       限制 batch 数（快速测试），例如 MAX_BATCHES=2
#   MAX_SAMPLES       限制样本数（更直接），例如 MAX_SAMPLES=1
#   USE_REAL_DATA     1=真实数据+NeuroSim 全网络/单层估算（默认），0=仅结构仿真
#   NEUROSIM_FULL     1=显式启用 NeuroSim 全网络 subprocess（acc/pointcim 默认已启用全网络，可不设）
#   NEUROSIM_PYBIND   1=启用 neurosim_core pybind 模式（chip_setup/chip_run_all，无 subprocess，更快），0=不启用（默认）；与 NEUROSIM_FULL 互斥，NEUROSIM_FULL 优先
#   NEUROSIM_ROOT    NeuroSim 根目录（Inference_pytorch），默认 /workspace/DNN_NeuroSim_V1.4/Inference_pytorch
#   NEUROSIM_CLK_FREQ  NeuroSim 工作频率(Hz)，由脚本传入、可调，不限于 100M（如 100000000=100MHz、200000000=200MHz）；不设则用 NeuroSim 关键路径频率
#   NEUROSIM_TECHNODE  NeuroSim 工艺节点(nm)，如 32；仅支持 130/90/65/45/32/22/14/10/7
#   NEUROSIM_TIMEOUT_S NeuroSim 全网络 main 超时秒数；默认不设(无限等待)。需要防卡死时可设 300/900 等
#   NEUROSIM_PYBIND_SO_DIR neurosim_core(.so) 所在目录（用于 PointCIM 距离链的 neurosim_dist.py import）
#   NEUROSIM_SUBARRAY        NeuroSim SubArray 行/列尺寸（正方形，兼容旧接口），例如 128 表示 128x128；同时覆盖 numRowSubArray / numColSubArray / numRowParallel
#   NEUROSIM_SUBARRAY_ROWS   NeuroSim numRowSubArray（优先于 NEUROSIM_SUBARRAY）
#   NEUROSIM_SUBARRAY_COLS   NeuroSim numColSubArray（优先于 NEUROSIM_SUBARRAY）
#   NEUROSIM_TILE_BUFFER_CORE_SIZE_ROW  NeuroSim tileBufferCoreSizeRow（tile buffer core 行尺寸，单位 bit 行）
#   NEUROSIM_TILE_BUFFER_CORE_SIZE_COL  NeuroSim tileBufferCoreSizeCol（tile buffer core 列尺寸，同时影响 interface width）
#   LOGIC_CLK_FREQ    数字逻辑时钟频率(Hz)，接口可配、默认 1 GHz（如 1000000000）；不设则 hw_config 使用默认 1 GHz
#   RUN_REAL_INFERENCE 1=先跑 inference_and_kan_analysis 再仿真，0=不跑（默认）
#   DEVICE            cuda（默认）或 cpu
#   SIM_LOG_DIR       日志目录，默认 sim_log
#   OUTPUT_JSON       报告 JSON 文件名，默认 report.json
#   SHOW_SHAPES       1=打印 LayerDef 与真实前向每层 conv/linear 的 shape 与 indata/rdata（默认 0）
#   LOGIC_PAR_LAYERNORM / LOGIC_PAR_RELU / LOGIC_PAR_SPOOL / LOGIC_PAR_QUANT_DEQUANT  逻辑电路各模块并行度（≥1，不设则用默认）
#   BN_EN             1=启用 BN 融合（默认 1，BN 不单独计入）；0=关闭融合（保留 BN 事件）
#   QD_EN             1=启用量化/反量化事件（默认 0）
#   EXPORT_MEM_TXN    1=导出内存事务序列文件（默认 1）
#   LIGHTWEIGHT_EXPORT 1=保留最终 PPA/breakdown，但关闭 operator_sequence/timeline/runtime memory events 等重型明细导出
#   SIM_RESULTS_DIR   PPA 报告 JSON 与 breakdown 图表输出目录，默认 sim_results
#   PROGRESS_INTERVAL_S            NeuroSim full 心跳间隔秒数（同时也可作为 DRAMSim3 心跳默认值），例如 30
#   DRAMSIM3_PROGRESS_INTERVAL_S   DRAMSim3 心跳间隔秒数（优先级高于 PROGRESS_INTERVAL_S）
#   DRAM_TRANSFER_CACHE            1=启用 DRAM 单次 transfer 结果缓存（默认 1，显著减少 DRAMSim3 次数），0=关闭用于 A/B 对照
#   USE_SCALESIM_SYSTOLIC  1=base_v1/base_v2 下用 SCALE-Sim 统计 Conv/Linear 周期与 MAC/访问（能耗仍用本工程内存建模），0=不启用（默认）
#   SCALE_SIM_ROOT    SCALE-Sim 根目录（USE_SCALESIM_SYSTOLIC=1 时），默认 /workspace/SCALE-Sim
#   SYSTOLIC_ROWS / SYSTOLIC_COLS  脉动阵列行/列（仅 baseline+SCALE-Sim），默认 64；例 16×16：SYSTOLIC_ROWS=16 SYSTOLIC_COLS=16
#   KNN_DYNAMIC_THRESHOLD_ROUNDS_MEAN  KNN 全局候选动态阈值 refine 轮数
#   KNN_CANDIDATES_PER_BLOCK  KNN 每个 block coarse 后写入 SRAM 的候选数
#   FPS_BLOCK_TOPK            FPS 后90%阶段每个 block coarse 后保留的候选数
#   ACC_SEGMENT_COUNT         acc speculative FPS 的 segment 数
#   RESAMPLER_PARALLELISM     resampler 并行度
#
# 示例：快速跑 2 个 batch
#   MAX_BATCHES=2 ./run_sim_with_real_eval.sh
# 示例：先跑真实推理再仿真
#   RUN_REAL_INFERENCE=1 ./run_sim_with_real_eval.sh
# 示例：打印 LayerDef 与真实前向每层 shape/indata/rdata（核对 36 vs 20 层）
#   SHOW_SHAPES=1 ./run_sim_with_real_eval.sh
# 示例：全网络 NeuroSim 仿真（真实数据 + 打印 shape + BN 不融合 + 量化 + 导出内存事务）
#   SHOW_SHAPES=1 BN_EN=0 QD_EN=1 EXPORT_MEM_TXN=1 NEUROSIM_FULL=1 ./run_sim_with_real_eval.sh
# 示例：NeuroSim 100MHz + 32nm 工艺
#   NEUROSIM_FULL=1 NEUROSIM_CLK_FREQ=100000000 NEUROSIM_TECHNODE=32 ./run_sim_with_real_eval.sh
# 示例：base_v1 用 SCALE-Sim 做 Conv/Linear 周期统计
#   ACCELERATOR=baseline USE_SCALESIM_SYSTOLIC=1 ./run_sim_with_real_eval.sh
# 示例：base_v2 用 SCALE-Sim 做 Conv/Linear 周期统计
#   ACCELERATOR=base_v2 USE_SCALESIM_SYSTOLIC=1 ./run_sim_with_real_eval.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

NETWORK="${NETWORK:-pointkan-elite}"
# 检查点/点数/类别数默认值会随 NETWORK 切换
if [ -z "${CKPT+x}" ]; then
  case "$NETWORK" in
    pointkan-elite-partseg)
      CKPT="PointKAN-elite/part_segmentation/checkpoints/pointKANElite_part_exp"
      ;;
    pointkan-elite-scanobj)
      CKPT="PointKAN-elite/classification_ScanObjectNN/checkpoints/pointKANElite-scan_exp-42"
      ;;
    pointnet2-cls-ssg)
      CKPT="/workspace/ISEDA26/ckpts/modelnet_pointnet2ssg_p1024_k20/checkpoints/best_model.pth"
      ;;
    pointnet2-partseg-ssg)
      CKPT="/workspace/ISEDA26/ckpts/shapenet_pointnet2ssg_p1024_k20/checkpoints/best_model.pth"
      ;;
    pointnet2-sem-ssg)
      CKPT="/workspace/ISEDA26/ckpts/s3dis_pointnet2ssg_p4096_k40/checkpoints/best_model.pth"
      ;;
    *)
      CKPT="PointKAN-elite/classification_ModelNet40/checkpoints/pointKANElite-pointkan_elite_exp-42"
      ;;
  esac
fi
if [ -z "${NUM_POINTS+x}" ]; then
  case "$NETWORK" in
    pointkan-elite-partseg)
      NUM_POINTS=2048
      ;;
    pointnet2-partseg-ssg)
      NUM_POINTS=1024
      ;;
    pointnet2-sem-ssg)
      NUM_POINTS=4096
      ;;
    *)
      NUM_POINTS=1024
      ;;
  esac
fi
if [ -z "${NUM_CLASSES+x}" ]; then
  case "$NETWORK" in
    pointkan-elite-partseg|pointnet2-partseg-ssg)
      NUM_CLASSES=50
      ;;
    pointkan-elite-scanobj)
      NUM_CLASSES=15
      ;;
    pointnet2-sem-ssg)
      NUM_CLASSES=13
      ;;
    *)
      NUM_CLASSES=40
      ;;
  esac
fi
if [ -z "${INPUT_CHANNELS+x}" ] && [ "$NETWORK" = "pointnet2-sem-ssg" ]; then
  INPUT_CHANNELS=12
fi
BATCH_SIZE="${BATCH_SIZE:-1}"
# 限 batch 数做快速测试（不设则跑全量测试集）
MAX_BATCHES="${MAX_BATCHES:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
PROGRESS_INTERVAL_S="${PROGRESS_INTERVAL_S:-}"
DRAMSIM3_PROGRESS_INTERVAL_S="${DRAMSIM3_PROGRESS_INTERVAL_S:-}"
DRAM_TRANSFER_CACHE="${DRAM_TRANSFER_CACHE:-}"
SIM_LOG_DIR="${SIM_LOG_DIR:-sim_log}"
OUTPUT_JSON="${OUTPUT_JSON:-report.json}"
SIM_RESULTS_DIR="${SIM_RESULTS_DIR:-sim_results}"
DEVICE="${DEVICE:-cuda}"
ACCELERATOR="${ACCELERATOR:-baseline}"

# 默认：真实数据模式 + NeuroSim（按层真实 indata/rdata 驱动全网络）
USE_REAL_DATA="${USE_REAL_DATA:-1}"
NEUROSIM_FULL="${NEUROSIM_FULL:-0}"
NEUROSIM_PYBIND="${NEUROSIM_PYBIND:-0}"
NEUROSIM_ROOT="${NEUROSIM_ROOT:-}"
NEUROSIM_CLK_FREQ="${NEUROSIM_CLK_FREQ:-}"
NEUROSIM_ADC_CLK_FREQ="${NEUROSIM_ADC_CLK_FREQ:-}"
NEUROSIM_TECHNODE="${NEUROSIM_TECHNODE:-32}"
NEUROSIM_TIMEOUT_S="${NEUROSIM_TIMEOUT_S:-}"
NEUROSIM_PYBIND_SO_DIR="${NEUROSIM_PYBIND_SO_DIR:-}"
LOGIC_CLK_FREQ="${LOGIC_CLK_FREQ:-1000000000}"
RUN_REAL_INFERENCE="${RUN_REAL_INFERENCE:-0}"
SHOW_SHAPES="${SHOW_SHAPES:-0}"
LOGIC_PAR_LAYERNORM="${LOGIC_PAR_LAYERNORM:-}"
LOGIC_PAR_RELU="${LOGIC_PAR_RELU:-}"
LOGIC_PAR_SPOOL="${LOGIC_PAR_SPOOL:-}"
LOGIC_PAR_QUANT_DEQUANT="${LOGIC_PAR_QUANT_DEQUANT:-}"
BN_EN="${BN_EN:-1}"
if [ -z "${QD_EN+x}" ]; then
  if [ "$ACCELERATOR" = "acc" ]; then
    QD_EN=1
  else
    QD_EN=0
  fi
fi
EXPORT_MEM_TXN="${EXPORT_MEM_TXN:-1}"
LIGHTWEIGHT_EXPORT="${LIGHTWEIGHT_EXPORT:-0}"
EXPORT_MEM_CONTRACT="${EXPORT_MEM_CONTRACT:-0}"
# baseline/base_v2 的 Conv/Linear 当前要求通过 SCALE-Sim 建模；
# 若用户未显式指定，则为 baseline 系列默认开启。
if [ -z "${USE_SCALESIM_SYSTOLIC+x}" ]; then
  if [ "$ACCELERATOR" = "baseline" ] || [ "$ACCELERATOR" = "base_v2" ] || [ "$ACCELERATOR" = "base_v3" ]; then
    USE_SCALESIM_SYSTOLIC=1
  else
    USE_SCALESIM_SYSTOLIC=0
  fi
fi
SCALE_SIM_ROOT="${SCALE_SIM_ROOT:-}"
SYSTOLIC_ROWS="${SYSTOLIC_ROWS:-}"
SYSTOLIC_COLS="${SYSTOLIC_COLS:-}"
SYSTOLIC_ARRAY_AREA_UM2="${SYSTOLIC_ARRAY_AREA_UM2:-}"
SYSTOLIC_SRAM_SIZE_BYTES="${SYSTOLIC_SRAM_SIZE_BYTES:-}"
SYSTOLIC_IFMAP_SRAM_SIZE_BYTES="${SYSTOLIC_IFMAP_SRAM_SIZE_BYTES:-}"
SYSTOLIC_FILTER_SRAM_SIZE_BYTES="${SYSTOLIC_FILTER_SRAM_SIZE_BYTES:-}"
SYSTOLIC_OFMAP_SRAM_SIZE_BYTES="${SYSTOLIC_OFMAP_SRAM_SIZE_BYTES:-}"
SYSTOLIC_SRAM_BLOCK_BYTES="${SYSTOLIC_SRAM_BLOCK_BYTES:-}"
SYSTOLIC_SRAM_NUM_BANKS="${SYSTOLIC_SRAM_NUM_BANKS:-}"
SYSTOLIC_SRAM_BUS_WIDTH_BITS="${SYSTOLIC_SRAM_BUS_WIDTH_BITS:-}"
SYSTOLIC_SRAM_FREQ_HZ="${SYSTOLIC_SRAM_FREQ_HZ:-}"
INTER_LAYER_SRAM_BW="${INTER_LAYER_SRAM_BW:-}"
INTER_LAYER_SRAM_SIZE_BYTES="${INTER_LAYER_SRAM_SIZE_BYTES:-}"
DRAM_BW_BYTES_PER_CYCLE="${DRAM_BW_BYTES_PER_CYCLE:-}"
DRAM_CHANNELS="${DRAM_CHANNELS:-}"
FEATURE_BUFFER_PIPELINE_MODE="${FEATURE_BUFFER_PIPELINE_MODE:-}"
FEATURE_DOUBLE_BUFFERING="${FEATURE_DOUBLE_BUFFERING:-0}"
FEATURE_BUFFER_SLOTS="${FEATURE_BUFFER_SLOTS:-}"
KNN_DYNAMIC_THRESHOLD_ROUNDS_MEAN="${KNN_DYNAMIC_THRESHOLD_ROUNDS_MEAN:-}"
KNN_CANDIDATES_PER_BLOCK="${KNN_CANDIDATES_PER_BLOCK:-}"
FPS_BLOCK_TOPK="${FPS_BLOCK_TOPK:-}"
ACC_SEGMENT_COUNT="${ACC_SEGMENT_COUNT:-}"
RESAMPLER_PARALLELISM="${RESAMPLER_PARALLELISM:-}"
KNN_CONV_PIPELINE_MODE="${KNN_CONV_PIPELINE_MODE:-}"
NS_SIMULATION_MODE="${NS_SIMULATION_MODE:-event}"
INPUT_CHANNELS="${INPUT_CHANNELS:-}"
NEUROSIM_WL_WEIGHT="${NEUROSIM_WL_WEIGHT:-}"
NEUROSIM_WL_ACTIVATION="${NEUROSIM_WL_ACTIVATION:-}"
NEUROSIM_CLK_MODE="${NEUROSIM_CLK_MODE:-}"
NEUROSIM_SUBARRAY="${NEUROSIM_SUBARRAY:-}"
NEUROSIM_SUBARRAY_ROWS="${NEUROSIM_SUBARRAY_ROWS:-}"
NEUROSIM_SUBARRAY_COLS="${NEUROSIM_SUBARRAY_COLS:-}"
NEUROSIM_TILE_BUFFER_CORE_SIZE_ROW="${NEUROSIM_TILE_BUFFER_CORE_SIZE_ROW:-}"
NEUROSIM_TILE_BUFFER_CORE_SIZE_COL="${NEUROSIM_TILE_BUFFER_CORE_SIZE_COL:-}"
NEUROSIM_NUM_COL_MUXED="${NEUROSIM_NUM_COL_MUXED:-}"
NEUROSIM_FORCE_TILE_SIZE_CM="${NEUROSIM_FORCE_TILE_SIZE_CM:-}"
NEUROSIM_FORCE_PE_SIZE_CM="${NEUROSIM_FORCE_PE_SIZE_CM:-}"
NEUROSIM_FORCE_NUM_TILES_CM="${NEUROSIM_FORCE_NUM_TILES_CM:-}"

build_args() {
    BUILT_ARGS=(
        --checkpoint "$CKPT"
        --num_points "$NUM_POINTS"
        --num_classes "$NUM_CLASSES"
        --sim_log_dir "$SIM_LOG_DIR"
        --accelerator "$ACCELERATOR"
        --device "$DEVICE"
    )
    [[ "$USE_REAL_DATA" == "1" ]] && BUILT_ARGS+=(--use_real_data --batch_size "$BATCH_SIZE")
    [[ -n "$MAX_BATCHES" ]] && BUILT_ARGS+=(--max_batches "$MAX_BATCHES")
    [[ -n "$MAX_SAMPLES" ]] && BUILT_ARGS+=(--max_samples "$MAX_SAMPLES")
    [[ -n "$PROGRESS_INTERVAL_S" ]] && BUILT_ARGS+=(--progress_interval_s "$PROGRESS_INTERVAL_S")
    [[ -n "$DRAMSIM3_PROGRESS_INTERVAL_S" ]] && export DRAMSIM3_PROGRESS_INTERVAL_S
    [[ -n "$DRAM_TRANSFER_CACHE" ]] && export DRAM_TRANSFER_CACHE
    [[ "$NEUROSIM_FULL" == "1" ]] && BUILT_ARGS+=(--neurosim_full)
    [[ "$NEUROSIM_PYBIND" == "1" ]] && BUILT_ARGS+=(--neurosim_pybind)
    [[ -n "$NEUROSIM_ROOT" ]] && BUILT_ARGS+=(--neurosim_root "$NEUROSIM_ROOT")
    if [[ -n "$NEUROSIM_PYBIND_SO_DIR" ]]; then
        export PYTHONPATH="${NEUROSIM_PYBIND_SO_DIR}:${PYTHONPATH:-}"
    fi
    [[ -n "$NEUROSIM_CLK_FREQ" ]] && BUILT_ARGS+=(--neurosim_clk_freq "$NEUROSIM_CLK_FREQ")
    [[ -n "$NEUROSIM_CLK_MODE" ]] && BUILT_ARGS+=(--neurosim_clk_mode "$NEUROSIM_CLK_MODE")
    [[ -n "$NEUROSIM_ADC_CLK_FREQ" ]] && BUILT_ARGS+=(--neurosim_adc_clk_freq "$NEUROSIM_ADC_CLK_FREQ")
    [[ -n "$NEUROSIM_NUM_COL_MUXED" ]] && BUILT_ARGS+=(--neurosim_num_col_muxed "$NEUROSIM_NUM_COL_MUXED")
    [[ -n "$NEUROSIM_SUBARRAY" ]] && BUILT_ARGS+=(--neurosim_subarray "$NEUROSIM_SUBARRAY")
    [[ -n "$NEUROSIM_SUBARRAY_ROWS" ]] && BUILT_ARGS+=(--neurosim_subarray_rows "$NEUROSIM_SUBARRAY_ROWS")
    [[ -n "$NEUROSIM_SUBARRAY_COLS" ]] && BUILT_ARGS+=(--neurosim_subarray_cols "$NEUROSIM_SUBARRAY_COLS")
    [[ -n "$NEUROSIM_TILE_BUFFER_CORE_SIZE_ROW" ]] && BUILT_ARGS+=(--neurosim_tile_buffer_core_size_row "$NEUROSIM_TILE_BUFFER_CORE_SIZE_ROW")
    [[ -n "$NEUROSIM_TILE_BUFFER_CORE_SIZE_COL" ]] && BUILT_ARGS+=(--neurosim_tile_buffer_core_size_col "$NEUROSIM_TILE_BUFFER_CORE_SIZE_COL")
    [[ -n "$NEUROSIM_FORCE_TILE_SIZE_CM" ]] && BUILT_ARGS+=(--neurosim_force_tile_size_cm "$NEUROSIM_FORCE_TILE_SIZE_CM")
    [[ -n "$NEUROSIM_FORCE_PE_SIZE_CM" ]] && BUILT_ARGS+=(--neurosim_force_pe_size_cm "$NEUROSIM_FORCE_PE_SIZE_CM")
    [[ -n "$NEUROSIM_FORCE_NUM_TILES_CM" ]] && BUILT_ARGS+=(--neurosim_force_num_tiles_cm "$NEUROSIM_FORCE_NUM_TILES_CM")
    [[ -n "$NEUROSIM_TECHNODE" ]] && BUILT_ARGS+=(--neurosim_technode "$NEUROSIM_TECHNODE")
    [[ -n "$NEUROSIM_TIMEOUT_S" ]] && BUILT_ARGS+=(--neurosim_timeout_s "$NEUROSIM_TIMEOUT_S")
    [[ -n "$LOGIC_CLK_FREQ" ]] && BUILT_ARGS+=(--logic_clk_freq "$LOGIC_CLK_FREQ")
    [[ "$RUN_REAL_INFERENCE" == "1" ]] && BUILT_ARGS+=(--run_real_inference --no_vis)
    [[ -n "$OUTPUT_JSON" ]] && BUILT_ARGS+=(--output "$OUTPUT_JSON")
    [[ "$SHOW_SHAPES" == "1" ]] && BUILT_ARGS+=(--show_shapes)
    [[ -n "$LOGIC_PAR_LAYERNORM" ]] && BUILT_ARGS+=(--logic_par_layernorm "$LOGIC_PAR_LAYERNORM")
    [[ -n "$LOGIC_PAR_RELU" ]] && BUILT_ARGS+=(--logic_par_relu "$LOGIC_PAR_RELU")
    [[ -n "$LOGIC_PAR_SPOOL" ]] && BUILT_ARGS+=(--logic_par_spool "$LOGIC_PAR_SPOOL")
    [[ -n "$LOGIC_PAR_QUANT_DEQUANT" ]] && BUILT_ARGS+=(--logic_par_quant_dequant "$LOGIC_PAR_QUANT_DEQUANT")
    BUILT_ARGS+=(--bn_en "$BN_EN")
    BUILT_ARGS+=(--qd_en "$QD_EN")
    [[ "$EXPORT_MEM_TXN" == "1" ]] && BUILT_ARGS+=(--export_memory_transactions)
    [[ "$LIGHTWEIGHT_EXPORT" == "1" ]] && BUILT_ARGS+=(--lightweight_export)
    [[ "$EXPORT_MEM_CONTRACT" == "1" ]] && BUILT_ARGS+=(--export_memory_contract)
    [[ "$USE_SCALESIM_SYSTOLIC" == "1" ]] && BUILT_ARGS+=(--use_scalesim_systolic)
    [[ -n "$SCALE_SIM_ROOT" ]] && BUILT_ARGS+=(--scalesim_root "$SCALE_SIM_ROOT")
    [[ -n "$SYSTOLIC_ROWS" ]] && BUILT_ARGS+=(--systolic_rows "$SYSTOLIC_ROWS")
    [[ -n "$SYSTOLIC_COLS" ]] && BUILT_ARGS+=(--systolic_cols "$SYSTOLIC_COLS")
    [[ -n "$SYSTOLIC_SRAM_SIZE_BYTES" ]] && BUILT_ARGS+=(--systolic_sram_size_bytes "$SYSTOLIC_SRAM_SIZE_BYTES")
    [[ -n "$SYSTOLIC_IFMAP_SRAM_SIZE_BYTES" ]] && BUILT_ARGS+=(--systolic_ifmap_sram_size_bytes "$SYSTOLIC_IFMAP_SRAM_SIZE_BYTES")
    [[ -n "$SYSTOLIC_FILTER_SRAM_SIZE_BYTES" ]] && BUILT_ARGS+=(--systolic_filter_sram_size_bytes "$SYSTOLIC_FILTER_SRAM_SIZE_BYTES")
    [[ -n "$SYSTOLIC_OFMAP_SRAM_SIZE_BYTES" ]] && BUILT_ARGS+=(--systolic_ofmap_sram_size_bytes "$SYSTOLIC_OFMAP_SRAM_SIZE_BYTES")
    [[ -n "$SYSTOLIC_SRAM_BLOCK_BYTES" ]] && BUILT_ARGS+=(--systolic_sram_block_bytes "$SYSTOLIC_SRAM_BLOCK_BYTES")
    [[ -n "$SYSTOLIC_SRAM_NUM_BANKS" ]] && BUILT_ARGS+=(--systolic_sram_num_banks "$SYSTOLIC_SRAM_NUM_BANKS")
    [[ -n "$SYSTOLIC_SRAM_BUS_WIDTH_BITS" ]] && BUILT_ARGS+=(--systolic_sram_bus_width_bits "$SYSTOLIC_SRAM_BUS_WIDTH_BITS")
    [[ -n "$SYSTOLIC_SRAM_FREQ_HZ" ]] && BUILT_ARGS+=(--systolic_sram_freq_hz "$SYSTOLIC_SRAM_FREQ_HZ")
    [[ -n "$INTER_LAYER_SRAM_BW" ]] && BUILT_ARGS+=(--inter_layer_sram_bw "$INTER_LAYER_SRAM_BW")
    [[ -n "$INTER_LAYER_SRAM_SIZE_BYTES" ]] && BUILT_ARGS+=(--inter_layer_sram_size_bytes "$INTER_LAYER_SRAM_SIZE_BYTES")
    [[ -n "$DRAM_BW_BYTES_PER_CYCLE" ]] && BUILT_ARGS+=(--dram_bw_bytes_per_cycle "$DRAM_BW_BYTES_PER_CYCLE")
    [[ -n "$DRAM_CHANNELS" ]] && BUILT_ARGS+=(--dram_channels "$DRAM_CHANNELS")
    [[ -n "$FEATURE_BUFFER_PIPELINE_MODE" ]] && BUILT_ARGS+=(--feature_buffer_pipeline_mode "$FEATURE_BUFFER_PIPELINE_MODE")
    [[ "$FEATURE_DOUBLE_BUFFERING" == "1" ]] && BUILT_ARGS+=(--feature_double_buffering)
    [[ -n "$FEATURE_BUFFER_SLOTS" ]] && BUILT_ARGS+=(--feature_buffer_slots "$FEATURE_BUFFER_SLOTS")
    [[ -n "$KNN_DYNAMIC_THRESHOLD_ROUNDS_MEAN" ]] && BUILT_ARGS+=(--knn_dynamic_threshold_rounds_mean "$KNN_DYNAMIC_THRESHOLD_ROUNDS_MEAN")
    [[ -n "$KNN_CANDIDATES_PER_BLOCK" ]] && BUILT_ARGS+=(--knn_candidates_per_block "$KNN_CANDIDATES_PER_BLOCK")
    [[ -n "$FPS_BLOCK_TOPK" ]] && BUILT_ARGS+=(--fps_block_topk "$FPS_BLOCK_TOPK")
    [[ -n "$ACC_SEGMENT_COUNT" ]] && BUILT_ARGS+=(--acc_segment_count "$ACC_SEGMENT_COUNT")
    [[ -n "$RESAMPLER_PARALLELISM" ]] && BUILT_ARGS+=(--resampler_parallelism "$RESAMPLER_PARALLELISM")
    [[ -n "$KNN_CONV_PIPELINE_MODE" ]] && BUILT_ARGS+=(--knn_conv_pipeline_mode "$KNN_CONV_PIPELINE_MODE")
    [[ -n "$NS_SIMULATION_MODE" ]] && BUILT_ARGS+=(--ns_simulation_mode "$NS_SIMULATION_MODE")
    [[ -n "$NETWORK" ]] && BUILT_ARGS+=(--network "$NETWORK")
    [[ -n "$INPUT_CHANNELS" ]] && BUILT_ARGS+=(--input_channels "$INPUT_CHANNELS")
    [[ -n "$NEUROSIM_WL_WEIGHT" ]] && BUILT_ARGS+=(--neurosim_wl_weight "$NEUROSIM_WL_WEIGHT")
    [[ -n "$NEUROSIM_WL_ACTIVATION" ]] && BUILT_ARGS+=(--neurosim_wl_activation "$NEUROSIM_WL_ACTIVATION")
    BUILT_ARGS+=(--sim_results_dir "$SIM_RESULTS_DIR")
}

echo "========== run_sim_with_real_eval =========="
echo "  CKPT=$CKPT ACCELERATOR=$ACCELERATOR NS_SIMULATION_MODE=$NS_SIMULATION_MODE"
echo "  USE_REAL_DATA=$USE_REAL_DATA NEUROSIM_FULL=$NEUROSIM_FULL NEUROSIM_PYBIND=$NEUROSIM_PYBIND NEUROSIM_CLK_FREQ=${NEUROSIM_CLK_FREQ:-none} NEUROSIM_TECHNODE=${NEUROSIM_TECHNODE:-none} NEUROSIM_TIMEOUT_S=${NEUROSIM_TIMEOUT_S:-none} LOGIC_CLK_FREQ=${LOGIC_CLK_FREQ:-none}"
echo "  USE_SCALESIM_SYSTOLIC=$USE_SCALESIM_SYSTOLIC SCALE_SIM_ROOT=${SCALE_SIM_ROOT:-default} SYSTOLIC_ROWS=${SYSTOLIC_ROWS:-64} SYSTOLIC_COLS=${SYSTOLIC_COLS:-64} SYSTOLIC_ARRAY_AREA_UM2=${SYSTOLIC_ARRAY_AREA_UM2:-default} SYSTOLIC_SRAM_SIZE_BYTES=${SYSTOLIC_SRAM_SIZE_BYTES:-default} SYSTOLIC_IFMAP_SRAM_SIZE_BYTES=${SYSTOLIC_IFMAP_SRAM_SIZE_BYTES:-default} SYSTOLIC_FILTER_SRAM_SIZE_BYTES=${SYSTOLIC_FILTER_SRAM_SIZE_BYTES:-default} SYSTOLIC_OFMAP_SRAM_SIZE_BYTES=${SYSTOLIC_OFMAP_SRAM_SIZE_BYTES:-default} SYSTOLIC_SRAM_BLOCK_BYTES=${SYSTOLIC_SRAM_BLOCK_BYTES:-default} SYSTOLIC_SRAM_NUM_BANKS=${SYSTOLIC_SRAM_NUM_BANKS:-default} SYSTOLIC_SRAM_BUS_WIDTH_BITS=${SYSTOLIC_SRAM_BUS_WIDTH_BITS:-default} SYSTOLIC_SRAM_FREQ_HZ=${SYSTOLIC_SRAM_FREQ_HZ:-default} INTER_LAYER_SRAM_SIZE_BYTES=${INTER_LAYER_SRAM_SIZE_BYTES:-default} INTER_LAYER_SRAM_BW=${INTER_LAYER_SRAM_BW:-default} DRAM_BW_BYTES_PER_CYCLE=${DRAM_BW_BYTES_PER_CYCLE:-default} DRAM_CHANNELS=${DRAM_CHANNELS:-default} FEATURE_BUFFER_PIPELINE_MODE=${FEATURE_BUFFER_PIPELINE_MODE:-default} FEATURE_DOUBLE_BUFFERING=${FEATURE_DOUBLE_BUFFERING:-0} FEATURE_BUFFER_SLOTS=${FEATURE_BUFFER_SLOTS:-auto} KNN_DYNAMIC_THRESHOLD_ROUNDS_MEAN=${KNN_DYNAMIC_THRESHOLD_ROUNDS_MEAN:-default} KNN_CANDIDATES_PER_BLOCK=${KNN_CANDIDATES_PER_BLOCK:-default} FPS_BLOCK_TOPK=${FPS_BLOCK_TOPK:-default} ACC_SEGMENT_COUNT=${ACC_SEGMENT_COUNT:-default} RESAMPLER_PARALLELISM=${RESAMPLER_PARALLELISM:-default} KNN_CONV_PIPELINE_MODE=${KNN_CONV_PIPELINE_MODE:-auto} NETWORK=${NETWORK:-pointkan-elite} INPUT_CHANNELS=${INPUT_CHANNELS:-default}"
echo "  BATCH_SIZE=$BATCH_SIZE MAX_BATCHES=${MAX_BATCHES:-all} DEVICE=$DEVICE"
echo "  SHOW_SHAPES=$SHOW_SHAPES BN_EN=$BN_EN QD_EN=$QD_EN EXPORT_MEM_TXN=$EXPORT_MEM_TXN LIGHTWEIGHT_EXPORT=$LIGHTWEIGHT_EXPORT EXPORT_MEM_CONTRACT=$EXPORT_MEM_CONTRACT"
echo "  SIM_LOG_DIR=$SIM_LOG_DIR SIM_RESULTS_DIR=$SIM_RESULTS_DIR OUTPUT=$OUTPUT_JSON"
echo "==========================================="

build_args
python -m pointkan_sim.run_sim_with_real_eval "${BUILT_ARGS[@]}" "$@"
