# PointKAN CIM 硬件仿真器 — 架构分析文档

## 0. 运行指令（快速开始）

在工程根目录 `/workspace/PointKAN-pytorch` 下执行：

```bash
# 演示（PointKAN + baseline）
python -m pointkan_sim.main --demo

# 指定网络与加速器，并输出结构化报告
python -m pointkan_sim.main --network pointkan --accelerator baseline --show_layers --output report.json

# PointKAN-Elite + acc 配置，输出 CSV
python -m pointkan_sim.main --network pointkan-elite --accelerator acc --output report.csv

# 启用 MNSIM 全 NetStruct（Tile/NoC 级）用于 conv1d/linear（需 MNSIM-2.0 与 SimConfig.ini）
python -m pointkan_sim.main --network pointkan --accelerator baseline --mnsim_full --output report.json

# 查看支持的加速器
python -m pointkan_sim.main --list_accelerators

# 打印每层激活 shape（input_shape → output_shape）与数据量
python -m pointkan_sim.main --network pointkan --show_layers --show_shapes

# 禁用层间 buffer（仅仿真计算单元，与旧行为一致）
python -m pointkan_sim.main --network pointkan --no_layer_buffer
```

- `--output`：按扩展名自动选择 JSON 或 CSV，报告包含 PPA 汇总、breakdown 与 **activation_shapes**（每层 input/output shape 与字节数）。  
- `--mnsim_full`：对 conv1d/linear 使用 MNSIM 完整 NetStruct 的 Model_latency/Model_energy；未指定或 MNSIM 不可用时回退到 PE 级 mnsim_adapter。  
- `--show_shapes`：控制台打印每层激活 shape，便于确认 pipeline 下的激活交互次数。  
- `--no_layer_buffer`：关闭层间 buffer 读写与 buffer size 约束。

---

## 1. 工程目标与范围

### 1.1 目标

在 CIM（Compute-in-Memory）硬件仿真器上对 **PointKAN / PointKAN-Elite** 进行端到端评估，产出：

- **Latency**：总延迟及按层/按算子/按电路模块的分解
- **Energy**：总能耗及按层/按算子/按模块的分解
- **Breakdown 分析**：各阶段（GAM / LFP / GFP / Classifier）、各算子类型、各硬件模块的周期与能耗占比
- **端到端性能**：单次推理的周期、能耗、平均/峰值功率等

### 1.2 评估策略分工

| 计算类型 | 评估方式 | 说明 |
|----------|----------|------|
| **特征计算（Conv/Linear）** | **按加速器区分** | **base_v1/base_v2**：im2col + N×N 脉动阵列（`systolic_array`、`im2col` 配置）；**acc**：MNSIM/NeuroSim（RRAM-CIM，全 NetStruct 或 PE 级）。 |
| **BN / KAN / 点云与逻辑** | **自定义或 MNSIM** | BN 融合进 Conv/Linear 或单独逻辑电路；KAN 用 RRAM-LUT；FPS、KNN、S-Pool、排序、AffineNorm、Add、MaxPool 等用自定义算子 + `hw_config`。 |

最终报告合并上述两部分的 latency、energy，并做统一 breakdown 与端到端汇总。

### 1.3 内存架构（标准 SRAM buffer / SRAM-II / RRAM-CIM / RRAM-LUT）

- **标准 SRAM buffer**：层间激活缓存（`inter_layer_buffer`），容量/带宽/DRAM spill 见 `hw_config.inter_layer_buffer`。
- **SRAM-II**：距离计算专用，存 key 点云；每次激活 128 行，2bit 距离。周期公式：`(bits_per_axis/bit_slice)*ceil(N/128)` 每查询。配置见 `hw_config.sram_ii`。
- **特征计算**：**base_v1/base_v2** 用 im2col + N×N 脉动阵列（`hw_config.systolic_array`、`im2col`）；**acc** 用 RRAM-CIM，由 MNSIM/NeuroSim 仿真（Tile/NoC），输入/权 8bit，可选量化/反量化（`enable_w8a8_quant_dequant`）。  
  **脉动仅用 SCALE-Sim**：base_v1/base_v2 下 Conv/Linear 严格通过 SCALE-Sim 脉动阵列仿真（FXP16，无量化/反量化）；未启用或失败则 raise。acc 模式用 RRAM-CIM/MNSIM，可含量化/反量化；其他配置 raise not supported。
- **RRAM-LUT**：KAN 等由 roofline/功耗模型手动定义（`rram_lut_macro`）。
- **Phase1**：系统启动，从 DRAM 加载 1×N×3 点云（12bit/特征）到 SRAM-II，roofline + DRAM 功耗（`hw_config.dram`，可接 CACTI）。见 `gen_first_layer_input_load_events`（首层/Phase1 唯一入口，调用方传入 input_bytes）。
- **Phase2**：距离计算（每查询 1×1×3 从 DRAM 取，与 key 在 SRAM-II 上算距离）；算完 N 个距离后：(a) 喂入 FPS 引擎做最远点选择，或 (b) 喂入 KNN 引擎做 top-k 并写回 DRAM。Baseline 下 (a)(b) 串行（共享 bitonic）。
- **FPS 引擎**：`distance_in_phase2=True` 时，stage1–4 归入 Phase2，FPS 只做 bitonic 部分；dispatcher 中每迭代先插 Phase2 距离事件再插 FPS bitonic。
- **KNN 引擎**：同上，Phase2 距离 + bitonic top-k；fuse 模式（复用距离、粗筛+细筛）待扩展。
- **BN 融合**：`bn_fuse_with_conv_linear=True` 且 BN 紧跟 Conv/Linear 时，BN 不计 cycle/能耗（`dispatch_layer` 中跳过）。

### 1.4 Breakdown 覆盖范围（算子与内存）

**算子级 breakdown**（`operator_breakdown` / `latency_breakdown_by_operator.png` / `energy_breakdown_by_operator.png`）由仿真事件按 `e.operator` 聚合，完整涵盖网络中会产事件的算子：

| 类别 | 算子名（饼图中） | 说明 |
|------|------------------|------|
| 计算 | Conv1d, Linear, KAN, FPS, KNN, LayerNorm, ReLU, ADD, AdaptiveMaxPool1d, AffineNorm, SPool | 与网络定义一致；BN 在融合时无单独事件，不出现。 |
| 内存 | **LayerBuffer**（层间 SRAM） | 每层输入从 buffer 读、输出写回 buffer（容量/带宽/能耗见 `inter_layer_buffer`）。 |
| 内存 | **DRAM** | 首层点云从 DRAM 加载；若启用 `spill_to_dram`，超出 buffer 的读写也计为 DRAM 事件。 |

- **Conv/Linear**：baseline 下 SCALE-Sim 单事件能耗已含 MAC + 该层 SRAM 访问（IFMAP/FILTER/OFMAP）+ 可选 DRAM（per_access 或 trace→DRAMSim3），不再单独列 SRAM/DRAM 行。
- **内存汇总**：`memory_transactions_summary.txt` 与 `--export_memory_transactions` 导出按 src/dst/kind 的访存次数；能耗与周期已并入上述 **LayerBuffer** 与 **DRAM** 两档，饼图中可区分 DRAM 与 SRAM（LayerBuffer）。
- **DRAM/SRAM 统一接口**（base_v1/base_v2）：层间 SRAM、首层 DRAM、spill DRAM 的周期与能耗均经 `pointkan_sim.integration.memory_model_adapter` 的 `query_sram_cycles_energy` / `query_dram_cycles_energy` 计算，仅使用 CACTI/DRAMSim3 回填的访问级模型（无带宽功率回退）；需先调用 `fill_inter_layer_buffer_from_cacti` 等回填。baseline 中 SCALE-Sim 的 trace 调度可参考 `SCALE-Sim/demo_with_accelergy.py`。

### 1.5 状态更新（2026-03-05）

以下能力已在当前代码中落地：

- `dispatcher.py` 已实现 `conv1d/linear/bn/layernorm/relu/s-pool/kan/affine_norm/adaptive_max_pool1d/add/mul` 的事件生成路径。
- `conv1d/linear` 已接入 `operators/mnsim_adapter.py`，通过调用 MNSIM 的 PE latency/power 接口进行 RRAM-CIM 估算（异常时自动回退到保守模型）。
- `kan` 已按 RRAM-LUT 进行建模（区分 `standard`/`elite` 变体），并加入片上 SRAM buffer 读写事件。
- `add` 支持残差路径的 off-chip DRAM 读写事件（可通过参数关闭/调整）。
- `pointkan_net.py` 与 `pointkan_elite_net.py` 已为归一化、KAN、add、adaptive_max_pool 等算子补齐 `num_points/num_features` 等参数，支持 pipeline 级别能耗与延迟统计。

> 注：本文后续章节部分“未完成项”描述基于较早版本，以上“状态更新”为最新实现状态。

---

## 2. 工程架构总览

### 2.1 目录结构

```
pointkan_sim/
├── network/                    # 网络与算子流水线定义
│   ├── pointkan_net.py         # PointKAN LayerDef + OperatorRef 序列
│   └── pointkan_elite_net.py   # PointKAN-Elite 定义（复用 GAM/GFP，自研 LFP）
├── configs/
│   └── hw_config.py            # 硬件配置（baseline / acc）、电路模块、power/cycles
├── operators/                  # 算子 → 事件 桥接与实现
│   ├── base_operator.py        # BaseOperator、Stage、SubStage
│   ├── dispatcher.py           # op_type → handler，dispatch_layer
│   ├── fps.py                  # FPSOperator（已实现）
│   └── knn.py                  # KNNOperator（已实现）
├── simulator/                  # 周期级仿真
│   ├── event.py                # Event、EventQueue
│   └── scheduler.py             # Scheduler、SimulationResult
├── analysis/
│   └── breakdown.py            # 按模块/阶段/算子的 breakdown 分析
├── report/
│   └── reporter.py             # PPAResult、Reporter、多格式输出
├── utils/
│   ├── logger.py
│   └── units.py
├── integration/                # 外部工具集成
│   └── mnsim_netstruct.py     # 从 LayerDef 构建 MNSIM NetStruct，跑 Model_latency/Model_energy
├── main.py                     # CLI 入口、仿真流程、breakdown、Reporter 报告
├── SIMULATION_CHAIN.md        # 仿真链路说明
├── OPERATOR_DEV_GUIDE.md      # 新算子开发指南
└── ARCHITECTURE.md            # 本文档
```

### 2.2 数据流与仿真链

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Network Definition (pointkan_net / pointkan_elite_net)│
│  List[LayerDef]  →  each LayerDef.operators: List[OperatorRef]              │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Dispatcher (operators/dispatcher.py)                                        │
│  dispatch_layer(layer.operators, hw_config, context?)                       │
│    → 每个 OperatorRef 按 op_type 调用对应 handler                             │
│    → conv1d/linear：若有 mnsim_context 则用 MNSIM 全 NetStruct 结果，否则 PE 级 MNSIM 适配 │
│    → 其余 op_type 调用 common_ops / FPS/KNN 等生成 Event                    │
│  Output: List[Event]（带 cycle/duration/power；metadata 在 main 中补 layer_name）│
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Scheduler (simulator/scheduler.py)                                          │
│  event_queue.add_event(...)  →  run_simulation() / run_simulation_fast()     │
│  - 事件驱动：按 (cycle, start/end) 推进，叠加 power，统计 peak/avg            │
│  - total_energy = sum(event.energy)，event.energy = power×duration×T_clk/1e3│
│  Output: SimulationResult (total_cycles, total_energy, peak_power, events)   │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Analysis & Report                                                           │
│  - breakdown.py: analyze_simulation_events_breakdown(events) → by_layer/by_operator/by_module │
│  - reporter.py:  PPAResult、Reporter、JSON/CSV（--output）                    │
│  - main.py:      仿真后调用 breakdown → create_ppa_result → Reporter 写 JSON/CSV │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 硬件配置抽象（configs/hw_config.py）

- **单芯片、扁平化**：无 Chip→Tile→PE 层级，由 5 类宏模块组成：
  1. **SRAM-CIM Macro**：array、shift_add、input_buffer、output_buffer、dist2_merge_adder（距离计算：交叉项 + 自平方项累加）；FPS 最小距离表归属排序电路
  2. **RRAM-LUT Macro**：rram_lut（+ acc 的 rram_lut_speculative）
  3. **RRAM-CIM Macro**：占位，计划与 MNSIM 对接，用于 Conv/Linear 等
  4. **Sorting Circuit**：min_distance_table（FPS 最小距离表更新）、baseline 为 BitonicSort（FPS MAX / KNN MIN），acc 为 coarse+fine filter
  5. **Logic Circuit**：batchnorm1d、layernorm、relu、spool、quant_dequant（acc 另有 interpolation）

- **当前状态**：各子模块的 `power`、`cycles`、`area` 多为 `0` 或 `TODO`，需填表或由 MNSIM 接口回填。

### 2.4 各模块设计原则

| 模块 | 设计原则 |
|------|----------|
| **network/** | 与原始 PyTorch 推理 pipeline **严格一致**：仅描述“做什么”（LayerDef + OperatorRef 及 params），不包含硬件细节；GAM/LFP/GFP/Classifier 的算子顺序、归一化/残差/ReLU 等与参考实现一一对应。 |
| **configs/** | **扁平宏模块**：单芯片、无 Chip→Tile→PE 层级；由 SRAM-CIM、RRAM-LUT、RRAM-CIM、Sorting、Logic 等几类宏组成，每类下为具名子模块（power/cycles/area）。标定后可被 operators 与 MNSIM 共用。 |
| **operators/** | **op_type → Event**：每个 op_type 对应一种“如何做”的实现（handler）；只负责从 params + hw_config 生成带 cycle/duration/power 的 Event 列表，不关心层序或调度。conv1d/linear 可走 PE 级 MNSIM 或（可选）全 NetStruct 预算结果。 |
| **dispatcher** | **纯桥接**：不包含 cycle/power/事件逻辑；只做 op_type → handler 查找、调用、并按层顺序做 cycle 偏移；可选传入 mnsim_context 供 conv1d/linear 使用 MNSIM 全 NetStruct 的 (latency_ns, energy_nj)。 |
| **simulator/** | **事件驱动**：EventQueue 按 cycle 管理事件；Scheduler 推进时间线、叠加功率、统计 total_cycles/total_energy/peak_power；不解释算子语义，只消费 Event。 |
| **analysis/** | **事件聚合**：对仿真后的 Event 列表按 metadata（layer_name）、operator、circuit_module 聚合，得到 by_layer / by_operator / by_module 的 cycles 与 energy_nj，供 PPAResult 与报告使用。 |
| **integration/** | **MNSIM NetStruct + 一次跑通**：从 LayerDef 中抽取 conv1d/linear 构建 MNSIM NetStruct，调用 TCG、Model_latency、Model_energy，得到每层 latency(ns) 与 energy(nJ)；结果通过 dispatcher 的 context 注入，不替代其余算子。 |
| **report/** | **PPA + breakdown 输出**：PPAResult 承载总 latency/energy 与 breakdown 字典；Reporter 负责多格式（Console/JSON/CSV），`--output` 指定路径时自动按扩展名写 JSON 或 CSV，形成端到端评估产物。 |

---

## 3. 模块说明与完成度

### 3.1 网络定义（network/）

| 文件 | 内容 | 状态 |
|------|------|------|
| pointkan_net.py | PointKAN 的 LayerDef 序列；GAM/LFP/GFP/Classifier 的 OperatorRef 列表；affine_norm、adaptive_max_pool1d、add 等已补全 | ✅ 与原始推理 pipeline 一致 |
| pointkan_elite_net.py | PointKAN-Elite 的 LayerDef；LFP 为 Pre-Norm + Efficient_KAN，含 residual add、adaptive_max_pool1d、fusion add | ✅ 与原始推理 pipeline 一致 |

- **GAM**：FPS → KNN → affine_norm → S-Pool（`_build_gam_operators` 在 pointkan_net，Elite 复用）。
- **GFP**：Conv1d→BN→ReLU→Conv1d→BN→add→relu（pointkan_net 已含残差与末段 ReLU，Elite 复用）。

### 3.2 算子与 Dispatcher（operators/）

| op_type | Handler | 实现方式 | 状态 |
|---------|---------|----------|------|
| fps | FPSOperator | fps.py，SRAM-CIM + 排序电路 | ⚠️ 已实现，hw_config 中 sorting 等未填时可能报错或能量为 0 |
| knn | KNNOperator | knn.py | ⚠️ 同上 |
| conv1d | _gen_conv1d | common_ops + MNSIM；可选 mnsim_context 用全 NetStruct | ✅ 已实现 |
| linear | _gen_linear | 同上 | ✅ 已实现 |
| bn / batchnorm1d | _gen_bn | common_ops | ✅ 已实现 |
| layernorm | _gen_layernorm | common_ops | ✅ 已实现 |
| relu | _gen_relu | common_ops | ✅ 已实现 |
| s-pool | _gen_spool | common_ops | ✅ 已实现 |
| kan | _gen_kan | common_ops，RRAM-LUT + SRAM buffer | ✅ 已实现 |
| affine_norm | _gen_affine_norm | common_ops | ✅ 已实现 |
| adaptive_max_pool1d | _gen_adaptive_max_pool | common_ops | ✅ 已实现 |
| add | _gen_add | common_ops，支持残差 DRAM | ✅ 已实现 |
| mul / element_multiply | _gen_mul | common_ops | ✅ 已实现 |

- **conv1d/linear**：默认走 PE 级 MNSIM 适配；`--mnsim_full` 且 MNSIM 可用时用 `integration/mnsim_netstruct.py` 全 NetStruct 的 Model_latency/Model_energy 结果。

### 3.3 仿真器（simulator/）

| 组件 | 功能 | 状态 |
|------|------|------|
| Event | cycle、duration、power、energy、operator/stage/circuit_module | ✅ 已用 |
| EventQueue | 按 cycle 排序、总能量计算 | ✅ |
| Scheduler | 事件驱动仿真、peak/avg power、SimulationResult | ✅ 已用；当前 main 用 run_simulation |

- **注意**：`total_energy` 来自 `event_queue.calculate_total_energy()`，即各 event 的 `power×duration×clock_period_ns/1e3` 之和；若各模块 power 为 0，则总能量为 0。

### 3.4 分析与报告（analysis/、report/）

| 模块 | 功能 | 状态 |
|------|------|------|
| breakdown.py | analyze_simulation_events_breakdown(events, clock_period_ns) → by_layer / by_operator / by_module（cycles、energy_nj、event_count） | ✅ 已实现，main 仿真后调用 |
| reporter.py | PPAResult、Reporter、JSON/CSV；create_ppa_result 支持 latency/energy_breakdown | ✅ 已实现 |

- **main.py**：仿真结束后调用 `_attach_breakdown_and_ppa`（breakdown + create_ppa_result），`--output` 时用 Reporter 写 JSON 或 CSV，报告含 PPA 汇总与 breakdown_detail。

### 3.5 硬件配置（configs/hw_config.py）

- **baseline / acc**：结构完整（模块组成、类型区分明确）。
- **未完成**：各子模块的 `power`、`cycles`、`area` 多为 0 或 TODO，需：
  - 自定义模块：通过文献/设计填表或由单独脚本标定；
  - RRAM-CIM / Conv/Linear 等：由 MNSIM 接口在运行时回填或生成等价 Event。

---

## 4. 待修改 / 待完成（To-Do）

### 4.1 必须完成（才能得到有意义的 Latency/Energy）

1. **hw_config 标定**  
   - 为 SRAM-CIM、ShiftAdd、MinDistanceTable、Buffer、Sorting（comp_tree、bitonic 或 two_stage_reconf）、Logic（BN、LN、ReLU、SPool 等）等填入合理 `power`（mW）与 `cycles`，否则 FPS/KNN 可能报错或能量/延迟为 0。  
   - **待修改位置**：`configs/hw_config.py` 中各宏模块的 `power`、`cycles`、`area` 字典。

2. **MNSIM 全 NetStruct 默认策略**  
   - 当前为可选 `--mnsim_full`；若希望默认启用 Tile/NoC 级评估，需在 main 中改为默认调用 `build_netstruct_from_layers` + `run_mnsim_full`，并在 MNSIM 不可用时静默回退到 PE 级。  
   - **待修改位置**：`main.py` 中 `run_network_simulation` 内 mnsim_context 的构建逻辑。

### 4.2 已实现：层间 Buffer 与激活 Shape（2026-03-05）

5. **层间 buffer 交互**  
   - 已实现：`configs/hw_config.py` 中 **InterLayerBufferConfig**（size_bytes、bandwidth_bytes_per_cycle、read/write_power_mw、spill_to_dram、dram 带宽与功率）。  
   - `operators/layer_buffer_ops.py`：每层开始前可插入「从 buffer（及 DRAM spill）读入输入」事件，每层结束后可插入「将输出写回 buffer（及 spill）」事件；首层前可插入「点云 N×3 从 DRAM 加载」事件。  
   - 当单层数据量超过 `size_bytes` 时，超出部分按 DRAM 读写建模（周期与能耗），模拟实际内存行为。  
   - 使用 `--no_layer_buffer` 可关闭层间 buffer，仅仿真计算单元（与旧行为一致）。

6. **激活 shape 与点云输入 N×3**  
   - **点云输入**：每个样本固定为 **N×3**（N=点数，3=xyz），在 `compute_activation_shapes` 中第一层 input_shape=(input_points, 3)，input_bytes=N×3×bytes_per_element。  
   - **network/activation_shape.py**：`compute_activation_shapes(layers, input_points, input_channels=3)` 逐层推导每层 input_shape/output_shape、input_bytes/output_bytes，供 buffer 事件与报告使用。  
   - **显示**：`--show_shapes` 时在控制台打印每层「in shape → out shape (bytes)」；报告 JSON/CSV 的 **activation_shapes** 小节包含每层 input_shape、output_shape、input_bytes、output_bytes，便于确定当前 pipeline 下的激活交互次数与数据量。

### 4.3 建议完成（用于完整评估与报告）

3. **Adapter 层升级**  
   - 用户可下一步在 `operators/mnsim_adapter.py` 或 `integration/mnsim_netstruct.py` 中增强：更细粒度 Tile/NoC 映射、Pipeline 模式、或与 MNSIM 的 Inputindex/Outputindex 等字段的完整对齐。  
   - **待修改位置**：`integration/mnsim_netstruct.py`（NetStruct 构建）、`operators/mnsim_adapter.py`（PE 级单 op 估算）。

4. **端到端多配置对比**  
   - 单次运行已输出总 Latency/Energy 与 breakdown；可增加脚本或 `--batch` 对多 accelerator、多 network 跑多次并输出对比表（CSV/JSON）。  
   - **待修改位置**：可新增 `scripts/run_sweep.py` 或扩展 main 的 `--output` 为目录时写入多文件。

---

## 5. MNSIM-2.0 对接要点

### 5.1 MNSIM-2.0 能力

- **输入**：`NetStruct`（由 `TrainTestInterface.get_structure()` 得到，依赖网络定义 + 权重量化）、`SimConfig_path`（如 SimConfig.ini）。  
- **输出**：  
  - **Model_latency**：per-layer 与 total 的 latency（含 xbar/ADC/DAC/buffer 等）。  
  - **Model_energy**：per-layer 与 total 的 energy（xbar、ADC、DAC、adder、buffer、pooling 等）。  
  - **Model_inference_power**：per-layer power，可与 latency 相乘得 energy。

### 5.2 对接思路

1. **NetStruct 来源**  
   - 若 MNSIM 侧已支持 PointKAN：在 MNSIM 的 network 中注册 PointKAN，用 `get_structure()` 得到完整 NetStruct，再在 pointkan_sim 中按层只取 Conv/Linear/BN 等层交给 MNSIM 算 latency/energy。  
   - 若暂不支持：在 pointkan_sim 中根据 `LayerDef` 与 `OperatorRef` 为“仅含 Conv1d/Linear/BN 的子图”手写或生成简化 NetStruct（仅用于调用 MNSIM 的 Model_latency / Model_energy），保证 in_channels、out_channels、kernel_size、input_size 等与 PointKAN 一致。

2. **调用方式**  
   - 在 **dispatcher** 中，对 `conv1d`、`linear`、`bn` 等 op_type：  
     - 调用 MNSIM 适配器：输入 (op_ref, hw_config)，输出 `List[Event]`（或 (cycles, energy_nJ) 再转 Event）。  
   - 每个这样的 Event 的 `cycle` 需要与前后算子做 **cycle offset**（与现有 `dispatch_layer` 一致），以便 Scheduler 得到正确的总周期和重叠关系。

3. **统一单位**  
   - MNSIM 的 latency 单位需与 pointkan_sim 一致（如 ns 或 cycles，用 hw_config.clock_period 换算）；energy 统一为 nJ，便于与自定义模块的 event.energy 相加。

### 5.3 Conv/Linear 矩阵乘与 W8A8、量化/反量化

- **矩阵乘形式与 Tile/NoC 映射**  
  - **Conv1d**：在 `integration/mnsim_netstruct.py` 中映射为 MNSIM 的 conv 层，空间为 (1, L)；计算等价于 (out_ch × (in_ch×kernel)) 与 (in_ch×kernel × L) 的矩阵乘，通过 TCG 做 **Tile/NoC 级** 仿真。  
  - **Linear**：映射为 MNSIM 的 fc 层，即 Infeature×Outfeature 矩阵乘，同样走 Tile/NoC 仿真。
- **W8A8 与 MNSIM 输出**  
  - 二者在 NetStruct 中均使用 **Inputbit/Weightbit/outputbit = 8**（W8A8）；MNSIM 侧为 **INT** 计算，输出为整数量化结果。
- **量化和反量化节点**  
  - 非线性（BN、ReLU 等）在仿真中按 **FP32** 处理，因此需在 Conv/Linear **前** 插入 **量化**（FP32→INT8）、**后** 插入 **反量化**（INT8→FP32）。  
  - 在 `hw_config` 中设置 `enable_w8a8_quant_dequant = True`，并在 `logic_circuit.quant_dequant` 中配置 `power`、`cycles` 后，`operators/common_ops.py` 会在生成 conv1d/linear 事件时自动在计算前后插入 quant/dequant 事件，参与 cycle 与能耗仿真。

---

## 6. 最终评估输出需求（对照实现情况）

| 需求 | 当前状态 | 说明 |
|------|----------|------|
| **总 Latency**（cycles / ns / ms） | ✅ | main 打印 + PPAResult + Reporter（JSON/CSV） |
| **总 Energy**（nJ / uJ） | ✅ | 同上 |
| **按层 Breakdown**（每层 cycles、energy） | ✅ | analyze_simulation_events_breakdown → by_layer，写入报告 breakdown_detail + results.breakdown |
| **按算子类型 Breakdown** | ✅ | by_operator，同上 |
| **按电路模块 Breakdown** | ✅ | by_module，同上 |
| **端到端性能**（单次推理） | ✅ | 总 cycles/energy/power + breakdown；`--output` 生成结构化 JSON/CSV |
| **Conv/Linear 用 MNSIM** | ✅ | PE 级 mnsim_adapter 已接；可选 `--mnsim_full` 用全 NetStruct Model_latency/Model_energy |
| **其他模块用自定义接口** | ✅ | FPS/KNN/BN/LN/ReLU/SPool/KAN/affine_norm/adaptive_max_pool/add/mul 已实现；hw_config 标定仍待填 |

---

## 7. 建议实现顺序（当前大部分已完成）

1. ~~**hw_config 填表**~~：仍待完成；至少为 FPS/KNN 依赖的 sorting、sram_cim 等填 power/cycles。  
2. ~~**MNSIM 适配**~~：PE 级 `operators/mnsim_adapter.py` 已接；全 NetStruct `integration/mnsim_netstruct.py` 已实现，`--mnsim_full` 可选启用。  
3. ~~**Dispatcher Conv/Linear 走 MNSIM**~~：已接；可选 mnsim_context 注入全 NetStruct 结果。  
4. ~~**Breakdown 与 Reporter 接入 main**~~：已接；`_attach_breakdown_and_ppa` + `--output` 写 JSON/CSV。  
5. ~~**其余算子**~~：relu、layernorm、s-pool、kan、affine_norm、add、adaptive_max_pool1d、mul 已实现。  
6. **多配置与端到端对比**：可新增 sweep 脚本或扩展 `--output` 为目录时多文件输出。

---

## 8. 文档与代码引用

- 仿真链路：`SIMULATION_CHAIN.md`  
- 新算子开发：`OPERATOR_DEV_GUIDE.md`  
- 网络与算子列表：`network/pointkan_net.py`、`network/pointkan_elite_net.py`  
- 硬件模块枚举与配置：`configs/hw_config.py`  
- MNSIM 入口与结构：`/workspace/MNSIM-2.0/main.py`、`Model_latency`、`Model_energy`、`get_structure()`
