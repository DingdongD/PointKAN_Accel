# run_sim_with_real_eval 完整运行指令与可调版本

在 **PointKAN-pytorch 根目录** 下执行。所有控制项均可组合使用。

---

## 一、Shell 方式（环境变量 + 脚本）

```bash
# 基础：使用默认配置（真实数据 + MNSIM 全 NetStruct）
./run_sim_with_real_eval.sh

# 带环境变量：  VAR1=value1 VAR2=value2 ./run_sim_with_real_eval.sh
```

### 环境变量一览

| 变量 | 含义 | 默认值 |
|------|------|--------|
| **CKPT** | 检查点路径（相对 classification_ModelNet40 或绝对） | `PointKAN-elite/.../pointKANElite-pointkan_elite_exp-42` |
| **NUM_POINTS** | 点云点数 | 1024 |
| **NUM_CLASSES** | 分类数 | 40 |
| **BATCH_SIZE** | 测试 batch 大小 | 1 |
| **MAX_BATCHES** | 最多跑多少 batch（不设=全量） | — |
| **USE_REAL_DATA** | 1=真实数据+仿真，0=仅结构仿真 | 1 |
| **MNSIM_FULL** | 1=Conv/Linear 用 MNSIM 全 NetStruct | 1 |
| **NEUROSIM_FULL** | 1=Conv/Linear 用 NeuroSim 全网络（与 MNSIM 二选一，优先 NeuroSim） | 0 |
| **NEUROSIM_ROOT** | NeuroSim 根目录 | — |
| **NEUROSIM_CLK_FREQ** | NeuroSim 工作频率 (Hz)，可调，不限于 100M | 不设=关键路径频率 |
| **NEUROSIM_TECHNODE** | NeuroSim 工艺节点 (nm)，如 32 | — |
| **LOGIC_CLK_FREQ** | 数字逻辑时钟频率 (Hz)，默认 1 GHz | 不设=1e9 |
| **RUN_REAL_INFERENCE** | 1=先跑 inference 再仿真 | 0 |
| **DEVICE** | cuda / cpu | cuda |
| **SIM_LOG_DIR** | 日志目录 | sim_log |
| **OUTPUT_JSON** | 报告 JSON 文件名 | report.json |
| **SHOW_SHAPES** | 1=打印 Conv/Linear + 非线性算子 shape | 0 |
| **LOGIC_PAR_LAYERNORM** | LayerNorm 并行度 (≥1) | — |
| **LOGIC_PAR_RELU** | ReLU 并行度 | — |
| **LOGIC_PAR_SPOOL** | SPool 并行度 | — |
| **LOGIC_PAR_QUANT_DEQUANT** | 量化/反量化并行度 | — |
| **BN_EN** | 1=BN 融合（不单独计），0=保留 BN 事件 | 1 |
| **QD_EN** | 1=插入量化/反量化事件 | 0 |
| **EXPORT_MEM_TXN** | 1=导出内存事务文件 | 1 |
| **SIM_RESULTS_DIR** | PPA 报告与图表输出目录 | sim_results |
| **ACCELERATOR** | 加速器配置名 | baseline |

---

## 二、Python 直接调用（完整参数）

```bash
python -m pointkan_sim.run_sim_with_real_eval [选项]
```

### 参数一览（与环境变量对应）

| 参数 | 含义 | 默认 |
|------|------|------|
| `--checkpoint` | 检查点路径 | 见 CKPT 默认 |
| `--num_points` | 点云点数 | 1024 |
| `--num_classes` | 分类数 | 40 |
| `--run_real_inference` | 先跑真实推理 | 否 |
| `--accelerator` | 加速器名 | baseline |
| `--mnsim_full` | 启用 MNSIM 全 NetStruct | 否 |
| `--neurosim_full` | 启用 NeuroSim 全网络 | 否 |
| `--cim_backend` | mnsim / neurosim / fallback | — |
| `--neurosim_root` | NeuroSim 根目录 | — |
| `--neurosim_clk_freq` | NeuroSim 频率 (Hz) | — |
| `--neurosim_technode` | NeuroSim 工艺 (nm) | — |
| `--logic_clk_freq` | 数字逻辑频率 (Hz) | 1e9 |
| `--output` | 报告 JSON 名 | — |
| `--sim_log_dir` | 日志目录 | sim_log |
| `--no_vis` | 推理时禁用可视化 | — |
| `--use_real_data` | 真实数据 + 仿真 | 否 |
| `--batch_size` | batch 大小 | 32 |
| `--max_batches` | 最多 batch 数 | — |
| `--device` | cuda / cpu | cuda |
| `--show_shapes` | 打印 shape 分析 | 否 |
| `--sim_results_dir` | 结果目录 | sim_results |
| `--bn_en` | 1=BN 融合，0=不融合 | 1 |
| `--qd_en` | 1=量化事件，0=不插入 | 0 |
| `--export_memory_transactions` | 导出内存事务 | 否 |
| `--logic_par_layernorm` | LayerNorm 并行度 | — |
| `--logic_par_relu` | ReLU 并行度 | — |
| `--logic_par_spool` | SPool 并行度 | — |
| `--logic_par_quant_dequant` | 量化/反量化并行度 | — |

---

## 三、按“内容版本”的示例指令

### 1. 仅完整 shape 分析（不跑 MNSIM/NeuroSim）

```bash
USE_REAL_DATA=1 SHOW_SHAPES=1 MNSIM_FULL=0 ./run_sim_with_real_eval.sh
# 或
python -m pointkan_sim.run_sim_with_real_eval --use_real_data --show_shapes --batch_size 2
```

### 2. 仅 MNSIM 全 NetStruct（无 NeuroSim）

```bash
USE_REAL_DATA=1 MNSIM_FULL=1 NEUROSIM_FULL=0 ./run_sim_with_real_eval.sh
```

### 3. NeuroSim 全网络 + 100MHz + 32nm

```bash
NEUROSIM_FULL=1 NEUROSIM_CLK_FREQ=100000000 NEUROSIM_TECHNODE=32 ./run_sim_with_real_eval.sh
```

### 4. NeuroSim 200MHz + 数字逻辑 500MHz

```bash
NEUROSIM_FULL=1 NEUROSIM_CLK_FREQ=200000000 NEUROSIM_TECHNODE=32 \
LOGIC_CLK_FREQ=500000000 ./run_sim_with_real_eval.sh
```

### 5. 打印 shape + NeuroSim + BN 不融合 + 量化事件

```bash
SHOW_SHAPES=1 BN_EN=0 QD_EN=1 NEUROSIM_FULL=1 MNSIM_FULL=0 \
EXPORT_MEM_TXN=1 ./run_sim_with_real_eval.sh
```

### 6. 数字逻辑并行度（如 LayerNorm=4）

```bash
NEUROSIM_FULL=1 NEUROSIM_CLK_FREQ=100000000 \
LOGIC_PAR_LAYERNORM=4 ./run_sim_with_real_eval.sh
```

### 7. 快速试跑（2 个 batch）

```bash
MAX_BATCHES=2 ./run_sim_with_real_eval.sh
```

### 8. 先跑真实推理再仿真

```bash
RUN_REAL_INFERENCE=1 ./run_sim_with_real_eval.sh
```

### 9. 指定 checkpoint 与结果目录

```bash
CKPT=path/to/checkpoint SIM_RESULTS_DIR=my_results ./run_sim_with_real_eval.sh
# 或
python -m pointkan_sim.run_sim_with_real_eval --use_real_data \
  --checkpoint path/to/checkpoint --sim_results_dir my_results
```

### 10. CPU + 仅结构仿真（无真实数据）

```bash
USE_REAL_DATA=0 DEVICE=cpu ./run_sim_with_real_eval.sh
```

---

## 四、等价写法对照

| 目的 | Shell | Python |
|------|--------|--------|
| 默认真实数据+MNSIM | `./run_sim_with_real_eval.sh` | `python -m pointkan_sim.run_sim_with_real_eval --use_real_data --mnsim_full` |
| 开 shape | `SHOW_SHAPES=1 ...` | `--show_shapes` |
| NeuroSim 100M 32nm | `NEUROSIM_FULL=1 NEUROSIM_CLK_FREQ=100000000 NEUROSIM_TECHNODE=32 ...` | `--neurosim_full --neurosim_clk_freq 100000000 --neurosim_technode 32` |
| 数字逻辑 500MHz | `LOGIC_CLK_FREQ=500000000 ...` | `--logic_clk_freq 500000000` |
| BN 不融合 | `BN_EN=0 ...` | `--bn_en 0` |
| 启用量化事件 | `QD_EN=1 ...` | `--qd_en 1` |
| LayerNorm 并行度 4 | `LOGIC_PAR_LAYERNORM=4 ...` | `--logic_par_layernorm 4` |

以上组合可任意叠加，得到你需要的“内容版本”。
