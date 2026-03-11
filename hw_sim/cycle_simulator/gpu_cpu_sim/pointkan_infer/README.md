# PointKAN 推理 Profiling 与 Breakdown 分析

基于 `gpu_cpu_sim/base_version` 的 profiler，对 PointKAN-elite 三个任务（ModelNet40 分类、ScanObjectNN 分类、Part Segmentation）做**统一推理入口**、**能耗/延迟 profile** 与 **阶段级 breakdown** 分析。

## 目录结构

```
pointkan_infer/
├── run_infer.py             # 统一推理入口（需 checkpoint + 数据集，供 profiler 调用）
├── run_infer_random.py      # 随机数据+随机权重推理（无需 checkpoint/数据集，采 latency+energy）
├── rapl_inference.py        # model(x) 计时（GPU: cuda.Event，CPU: perf_counter 前后打点）+ pyRAPL 仅包推理循环
├── METRICS.md               # 各指标含义与测量方式（仅推理 vs 进程级）
├── breakdown_profiler.py    # 按阶段统计前向耗时的 breakdown 分析
├── configs/
│   └── tasks.json           # 各任务默认 checkpoint 与参数
├── scripts/
│   ├── profile_gpu.sh         # 单任务 GPU profiling
│   ├── profile_cpu.sh         # 单任务 CPU profiling
│   ├── profile_random_gpu.sh # 随机数据 GPU profiling（不依赖 checkpoint/数据）
│   ├── profile_random_cpu.sh # 随机数据 CPU profiling
│   ├── profile_random_all.sh # 三任务 × (GPU+CPU) 随机数据 profiling（结束后自动出 CPU vs GPU 对比图）
│   ├── plot_cpu_gpu_compare.py # CPU vs GPU Latency/Energy 对比图
│   ├── breakdown_all.sh      # 三任务各跑一次 breakdown
│   ├── profile_all.sh        # 三任务 profile（默认 GPU；MODE=cpu 为 CPU）+ 可选 breakdown
│   ├── plot_profile_results.py  # 聚合 outputs 并绘制 Latency/Energy 柱状图与功耗曲线
│   └── run_plot.sh           # 调用 plot_profile_results.py
└── README.md
```

## 快速运行
```bash
cd /workspace/PointKAN-pytorch/hw_sim/cycle_simulator/gpu_cpu_sim/pointkan_infer
bash scripts/profile_random_all.sh

```

## 依赖与路径

- **Python 环境**：需与 PointKAN-elite 一致，即安装 PyTorch、`pointnet2_ops`、`kat_rational` 等；建议在能成功运行各任务 `inference_and_kan_analysis.py` 的同一环境中执行本仓脚本。
- **CPU 推理能耗（仅推理阶段）**：需安装 **pyRAPL**（`pip install pyRAPL`）。推理脚本在进程内用 `pyRAPL.Measurement.begin/end` 只包住模型前向循环，得到「仅模型推理」的 CPU 能耗并写入 `*_inference_only.json`；绘图与汇总优先使用该值。未安装 pyRAPL 时 CPU 能耗项为空，不影响延迟与 GPU 能耗。
- **PointKAN-elite 路径**：脚本默认从 `pointkan_infer` 向上一级级推导到 `PointKAN-pytorch`，再取 `PointKAN-elite`；也可通过环境变量 `POINTKAN_ELITE_ROOT` 显式指定。
- **数据与权重**（仅 `run_infer.py` 需要；`run_infer_random.py` 不需要）：
  - 三个任务的数据集需已放在对应任务目录下（与各任务 `inference_and_kan_analysis.py` 一致）。
  - 权重目录（可自行替换）：
    - ModelNet40: `PointKAN-elite/classification_ModelNet40/checkpoints/pointKANElite-pointkan_elite_exp-42`
    - ScanObjectNN: `PointKAN-elite/classification_ScanObjectNN/checkpoints/pointKANElite-ScanObjectNN`
    - Part Seg: `PointKAN-elite/part_segmentation/checkpoints/pointKANElite_part_exp`

## Profile 方案概览

**本仓只关心「model(data) 推理」的 latency 与 energy**，不统计进程级 wall time。Latency 为单次前向时间（GPU 用 cuda.Event，CPU 用 perf_counter）；CPU 能耗为 pyRAPL 仅包推理循环；GPU 能耗为 NVML/dmon 在推理运行期间的采样积分。

| 层级 | 工具 | 输出 | 用途 |
|------|------|------|------|
| model 推理 latency/能耗 | `base_version/profiler.py` + `run_infer*.py` | `summary.json`（inference_only_*、gpu_energy、cpu_energy_inference_*）, `*_inference_only.json` | 单次前向时间、GPU 能耗、仅推理阶段 CPU 能耗 |
| 阶段级 breakdown | `breakdown_profiler.py` | `breakdown_*.json` | 各阶段（embedding / GAM / LFP / GFP / classifier）耗时占比、data load vs forward |

### 1. model 推理 Profiling（latency + GPU/CPU energy）

- **入口**：`run_infer.py` 或 `run_infer_random.py`，由 base_version 的 `profiler.py` 通过 `--cmd` 调用。
- **流程**：warmup → repeat 次正式跑，每次执行 CMD；推理脚本在 model(data) 前后打点（GPU 用 cuda.Event，CPU 用 perf_counter），并写 `*_inference_only.json`；profiler 采 GPU 功耗、读推理结果。
- **输出目录**：`outputs/gpu_<task>/<timestamp>/` 或 `outputs/cpu_<task>/...`，内含：
  - `summary.json`：**推理** latency 均值（inference_only_per_forward_s_mean）、GPU energy、仅推理 CPU energy（cpu_energy_inference_*）；
  - `run_<i>_inference_only.json`：单次 run 的推理时间与 CPU 能耗；
  - `run_<i>_gpu_samples.csv` 等：GPU 功耗采样曲线；
  - `meta.json`：本次 profile 参数。

**示例（GPU，单任务）**：

```bash
cd pointkan_infer
export TASK=modelnet40
export CHECKPOINT=/path/to/PointKAN-elite/classification_ModelNet40/checkpoints/pointKANElite-pointkan_elite_exp-42
bash scripts/profile_gpu.sh
```

**示例（CPU）**：

```bash
TASK=modelnet40 bash scripts/profile_cpu.sh
```

**直接调用 profiler（等价于脚本中的 CMD）**：

```bash
python3 ../base_version/profiler.py \
  --cmd "python3 run_infer.py --task modelnet40 --checkpoint $CHECKPOINT --batch-size 1 --max-batches 20" \
  --mode gpu --out-dir outputs/gpu_modelnet40 --warmup 1 --repeat 3
```

### 2. Breakdown 分析（阶段级耗时）

- **目的**：得到「数据准备 vs 前向」以及「embedding / GAM_i / LFP_i / GFP_i / classifier」各阶段耗时与占比，便于分析 PointKAN 推理中 KAN（LFP/GFP）与几何模块（GAM）的开销分布。
- **工具**：`breakdown_profiler.py`，通过在各阶段注册 `forward_pre_hook` / `forward_hook` 打时间戳（必要时 `torch.cuda.synchronize()`），汇总多 batch 后写出 JSON。
- **输出字段**（节选）：
  - `stages_ms`：各阶段总耗时（ms）；
  - `stages_ms_mean`：单 batch 平均；
  - `stage_pct`：各阶段占比（%）；
  - `data_load_ms_mean` / `forward_ms_mean`：数据搬到 device + 前向平均时间。

**单任务示例**：

```bash
python3 breakdown_profiler.py \
  --task modelnet40 \
  --checkpoint /path/to/PointKAN-elite/classification_ModelNet40/checkpoints/pointKANElite-pointkan_elite_exp-42 \
  --max-batches 5 \
  --out outputs/breakdown_modelnet40.json
```

**三任务各跑一次**：

```bash
bash scripts/breakdown_all.sh
# 结果：outputs/breakdown_modelnet40.json, breakdown_scanobjectnn.json, breakdown_part_seg.json
```

### 3. 一键：多任务 GPU Profile + Breakdown

```bash
bash scripts/profile_all.sh
# 会依次跑 profile_gpu.sh（modelnet40 / scanobjectnn / part_seg），再跑 breakdown_all.sh
# CPU 推理：MODE=cpu bash scripts/profile_all.sh
# 关闭 breakdown：RUN_BREAKDOWN=0 bash scripts/profile_all.sh
```

### 随机数据+随机权重推理（无需 checkpoint/数据集）

若**不依赖真实数据集与权重**，仅用相同模型结构 + 随机输入在 CPU/GPU 上采 **latency 与 energy**，可使用：

- **run_infer_random.py**：三种模型用随机权重初始化，输入为 `torch.randn` 生成的随机张量（shape 与各任务一致），仍走完整前向。
- 与现有 profiler 用法一致，仅 CMD 改为 `run_infer_random.py`，无需 `--checkpoint`。

```bash
# 单任务 GPU
TASK=modelnet40 bash scripts/profile_random_gpu.sh

# 单任务 CPU（CPU 能耗由 pyRAPL 仅包推理循环）
TASK=modelnet40 bash scripts/profile_random_cpu.sh

# 三任务 × (GPU + CPU) 各跑一遍
bash scripts/profile_random_all.sh
```

结果目录：`outputs/random_gpu_<task>/`、`outputs/random_cpu_<task>/`，内含与 `profile_gpu.sh`/`profile_cpu.sh` 相同的 `summary.json`、`run_*_gpu_samples.csv` 等。适合在新平台无数据/无权重时先测吞吐与能耗。

**本仓仅统计 model 推理**：不报进程 wall time。推理时间与能耗来自 `*_inference_only.json` 与 summary 的 `inference_only_per_forward_s_mean`、`cpu_energy_inference_*`；GPU 用 cuda.Event，CPU 用 **model(x) 前后**的 perf_counter，由 `rapl_inference.time_forward_seconds` 封装。各字段含义与测量方式见 **[METRICS.md](METRICS.md)**。

**CPU vs GPU 对比图**：跑完 `profile_random_all.sh` 后会自动生成 `outputs/cpu_gpu_compare.png`：左为三任务 **单次 model 推理 Latency (s)**（batch=1，多轮平均），右为三任务 **推理阶段 Energy (J)**（GPU 为运行期间采样积分，CPU 为 pyRAPL 仅包推理循环）。也可手动执行：

```bash
# 随机数据结果
python3 scripts/plot_cpu_gpu_compare.py --out-dir outputs --source random --out outputs/cpu_gpu_compare.png

# checkpoint 跑的结果（先跑过 profile_all.sh 或 MODE=gpu/cpu 各一遍）
python3 scripts/plot_cpu_gpu_compare.py --out-dir outputs --source checkpoint
```

## Breakdown 结果解读

- **embedding**：首层点云卷积，通常较快。
- **GAM_i**：几何聚合模块（含 FPS、ball query 等），与点数和采样数相关，常占一定比例。
- **LFP_i / GFP_i**：局部/全局 KAN 主体，有理函数与 MLP；若占比高，可重点做量化或算子优化。
- **classifier**：末尾分类头。
- **data_load_ms_mean**：从 DataLoader 取 batch 到搬到 device 的时间；若过大可检查 `num_workers`、数据路径或 IO。
- **forward_ms_mean**：纯前向时间；与 summary 的 `inference_only_per_forward_s_mean` 对应（多次 batch 平均）。

结合 `summary.json` 的 GPU 功耗/能耗与 `breakdown_*.json` 的占比，可得到「谁在耗时间、谁在耗能」的结论，用于写报告或优化优先级。

### 4. 输出可视化（三任务 Latency / GPU·CPU Energy 曲线）

在跑完 `profile_gpu.sh` 或 `profile_all.sh` 后，可用脚本聚合 `outputs/gpu_*/` 下最新一次运行的 `summary.json` 与 `run_*_gpu_samples.csv`、`run_*_cpu_samples.csv`，生成：

- **profile_summary.png**：三任务并列的柱状图，分别为 **单次 model 推理 Latency (s)**、**GPU Energy (J)**、**推理阶段 CPU Energy (J)**（pyRAPL 仅包推理循环；未安装则为 0）。
- **profile_curves.png**：各任务 run_0 的 **GPU 功耗 (W) 随时间**、**CPU 利用率 (%) 随时间** 曲线（3×2 子图）。

```bash
python3 scripts/plot_profile_results.py --out-dir outputs
# 或
bash scripts/run_plot.sh
```

依赖：`matplotlib`。输出默认在 `outputs/profile_summary.png` 与 `outputs/profile_curves.png`。

### 5. 输出文件与已知问题

- **returncode**：`run_<i>.json` 中 `returncode` 为 1 表示当次运行异常退出（如路径错误）。修复后重跑得到 returncode 0；本仓图表仅使用 model 推理 latency 与 energy，不使用进程 wall time。
- **POINTKAN_ELITE_ROOT**：脚本中从 `pointkan_infer` 推导 workspace 时使用 4 层上级目录（`../../../../`）指向 `PointKAN-pytorch`，再拼接 `PointKAN-elite`。若仓库布局不同，请设置环境变量 `POINTKAN_ELITE_ROOT`。
- **CPU Energy**：由推理脚本内 pyRAPL 仅包推理循环写入 `cpu_energy_inference_*`；未安装 pyRAPL 时图中为 0。详见下节。

### 6. 为何 CPU 没有抓到 energy / latency？

**Latency**  
- 本仓**只使用 model 推理时间**：由推理脚本在 model(data) 前后打点写入 `*_inference_only.json`，summary 中为 `inference_only_per_forward_s_mean`。不使用进程 wall time（`latency_s`）。
- GPU 用 cuda.Event，CPU 用 time.perf_counter，由 `rapl_inference.time_forward_seconds` 统一封装。

**CPU Energy**  
- 本仓**只使用「仅推理阶段」CPU 能耗**：推理脚本内用 pyRAPL 的 `Measurement.begin/end` 只包住 model 前向循环，结果写入 `*_inference_only.json` 的 `cpu_energy_inference_*`，summary 中为 `cpu_energy_inference_*_mean`。
- 需安装 `pip install pyRAPL`，且硬件支持 RAPL（Intel/AMD 常见）。未安装或不可用时图中 CPU Energy 为 0。
- 不启用进程级 `--enable-perf`（默认 ENABLE_PERF=0），不依赖 perf/sysfs 整进程 RAPL。

## 参数速查

| 参数 | 说明 | 默认 |
|------|------|------|
| `TASK` | 任务：modelnet40 / scanobjectnn / part_seg | modelnet40 |
| `CHECKPOINT` | 检查点目录或 .pth 路径 | 见 scripts 内默认 |
| `POINTKAN_ELITE_ROOT` | PointKAN-elite 根目录 | workspace/PointKAN-elite |
| `BATCH` / `--batch-size` | 推理 batch size | 1 |
| `MAX_BATCHES` / `--max-batches` | 最多跑多少个 batch（0=全量） | 20（profile）/ 5（breakdown） |
| `WARMUP` / `REPEAT` | profiler warmup 与正式 repeat 次数 | 1 / 3 |
| `ENABLE_PERF` | 默认 0；本仓 CPU 能耗由 pyRAPL 仅包推理循环提供，不依赖 perf | 0 |
| `MODE` | profile_all.sh 用：gpu 或 cpu | gpu |
| `OUT_DIR` | profiler 输出目录 | outputs/gpu_<task> 或 cpu_<task> |

## 与 inference_and_kan_analysis.py 的关系

- 数据加载、模型构建与 checkpoint 解析逻辑与各任务目录下的 `inference_and_kan_analysis.py` 保持一致（数据集类、`best_checkpoint.pth` / `best_insiou_model.pth`、state_dict 的 `net`/`model` 与 `module.` 前缀处理）。
- 本仓只做「推理 + 测速/能耗/breakdown」，不做 KAN 曲线可视化或校准；完整分析与可视化请直接运行各任务的 `inference_and_kan_analysis.py`。
