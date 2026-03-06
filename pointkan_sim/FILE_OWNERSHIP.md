# pointkan_sim 模块与文件归属

各目录/文件的职责与归属，便于维护与扩展。

---

## 顶层入口

| 文件 | 职责 |
|------|------|
| **main.py** | CLI 入口（`python -m pointkan_sim.main`）。解析参数、加载网络、运行仿真、写 JSON/CSV 报告（`--output` 时用 `report.reporter.Reporter` + `create_ppa_result`）。**归属**：主流程与“仅仿真”路径。 |
| **run_sim_with_real_eval.py** | 真实数据评估入口（`run_sim_with_real_eval.sh` 调用）。`--use_real_data`：加载 ModelNet40 + checkpoint、真实前向收集 conv/linear、MNSIM、按批仿真、写 **sim_results**（report.json + 算子级 breakdown 饼图 + operator_sequence.txt + 可选 memory_transactions.txt）。非 use_real_data：仅跑仿真，与 main 类似。**归属**：真实数据评估与脚本对接。 |
| **constants.py** | 算子序列键名常量（KEY_*、OP_NAME_*）、表头（OPERATOR_SEQUENCE_HEADER 等）。**归属**：与 operator_sequence.txt / 报告输出一致化。 |

---

## configs/

| 文件 | 职责 |
|------|------|
| **hw_config.py** | 硬件配置：baseline/acc 加速器、五大宏模块（SRAM-CIM、RRAM-LUT、RRAM-CIM、排序、逻辑）、层间 buffer、DRAM、SRAM-II、BN 融合与 W8A8 量化开关。`get_accelerator_config(name)`、`list_accelerator_configs()`。**归属**：所有与硬件参数相关的配置。 |

---

## network/

| 文件 | 职责 |
|------|------|
| **pointkan_net.py** | PointKAN 网络结构定义：LayerDef、OperatorRef、GAM/LFP/GFP 算子列表。`create_pointkan_classification()`。**归属**：PointKAN（非 Elite）拓扑。 |
| **pointkan_elite_net.py** | PointKAN-Elite 网络结构：embedding、GAM、LFP、stage、classifier 及 conv1d/linear 展开。`create_pointkan_elite_classification()`。**归属**：PointKAN-Elite 拓扑。 |
| **activation_shape.py** | 按层推算激活 shape 与字节数（input_bytes/output_bytes）。`compute_activation_shapes(layers, ...)`，供层间 buffer 与报告使用。**归属**：激活维度与数据量。 |

---

## operators/

| 文件 | 职责 |
|------|------|
| **base_operator.py** | 算子基类 BaseOperator、Stage、SubStage、OperatorParams。**归属**：算子抽象接口。 |
| **common_ops.py** | 通用算子事件：量化/反量化、BN、LayerNorm、ReLU、SPool、AffineNorm、Add/Mul、Conv1d/Linear 的 PE 级 RRAM-CIM。依赖 **utils.math_utils.safe_div_ceil**。**归属**：逻辑电路与 Conv/Linear 回退路径。 |
| **dispatcher.py** | op_type → handler 派发，按层链式调用并累加 cycle。Conv1d/Linear 使用 constants.OP_NAME_*。**归属**：算子派发与事件顺序。 |
| **fps.py** | FPS 算子：距离计算 + Bitonic 排序，生成 Event（operator="FPS"）。**归属**：FPS 周期与功耗模型。 |
| **knn.py** | KNN 算子：距离计算 + SRAM 写回 + Bitonic 取 k 近邻。**归属**：KNN 周期与功耗模型。 |
| **layer_buffer_ops.py** | 层间 buffer 读/写、spill、首层输入加载（**gen_first_layer_input_load_events**）、Phase2 距离事件。依赖 **utils.math_utils.safe_div_ceil**。**归属**：内存与首层/距离阶段事件。 |
| **mnsim_adapter.py** | MNSIM PE 级延迟/能耗估算，供 common_ops 中 Conv1d/Linear 无全 NetStruct 时使用。**归属**：MNSIM 对接。 |

---

## simulator/

| 文件 | 职责 |
|------|------|
| **event.py** | Event、EventQueue、EventStatus。单条事件的 cycle/duration/power/energy 及 metadata（如 layer_name）。**归属**：事件模型与队列。 |
| **scheduler.py** | 按周期推进仿真、汇总 total_cycles/total_energy、返回 SimulationResult（events 等）。**归属**：周期级调度与结果。 |

---

## analysis/

| 文件 | 职责 |
|------|------|
| **breakdown.py** | **analyze_simulation_events_breakdown(events)**：按层/算子/电路模块聚合 cycles 与 energy_nj，供 main 的 breakdown 与 PPA 报告使用。单算子细分的 analyze_operator_breakdown、format_breakdown_table 保留供调试或扩展。**不提供**网络级 breakdown（已移除）。**归属**：仿真事件聚合与算子级 breakdown。 |
| **timeline.py** | 按周期构建模块活动时间线、分段、导出 CSV。供 main 的 `--output_timeline_csv`。**归属**：时间线分析与导出。 |

---

## integration/

| 文件 | 职责 |
|------|------|
| **mnsim_netstruct.py** | 从 LayerDef 的 conv1d/linear 构建 MNSIM NetStruct，调用 MNSIM 跑全 NetStruct 得 latency/energy。**build_netstruct_from_layers**、**run_mnsim_full**。**归属**：MNSIM 网络结构与全 NetStruct 评估。 |
| **real_data_eval.py** | 加载 checkpoint 与 ModelNet40、真实前向收集 conv/linear、real_data_overrides、**run_real_inference_and_sim_over_dataset**。**归属**：真实数据评估流程。 |
| **ppa_report.py** | **build_ppa_report**：端到端 PPA + 算子 breakdown + operator_sequence 等字典。**write_ppa_report_and_plots**：写 report.json + latency/energy 饼图（算子级 breakdown 可视化）。**归属**：真实数据路径下的报告与图表输出。 |

---

## report/

| 文件 | 职责 |
|------|------|
| **reporter.py** | PPAResult、Reporter。**create_ppa_result**：从仿真结果与 breakdown 构建 PPAResult。Reporter 写 JSON/CSV（main 的 `--output` 使用）。**归属**：main 路径的报告格式与写出。 |

---

## utils/

| 文件 | 职责 |
|------|------|
| **math_utils.py** | **safe_div_ceil(a, b)**：安全整除上取整，供 operators 与 layer_buffer_ops 复用。**归属**：公共数学工具。 |
| **logger.py** | setup_logger、log_info、log_section、log_separator、LogLevel 等。**归属**：日志配置与输出。 |
| **units.py** | 能耗/功率/周期/单位换算（当前主流程未引用，预留）。**归属**：单位换算工具。 |

---

## 报告输出两条路径

| 路径 | 谁写 | 写出内容 |
|------|------|----------|
| **main.py --output** | report.reporter.Reporter + create_ppa_result | JSON/CSV（PPAResult 格式），写入 sim_log。 |
| **run_sim_with_real_eval --use_real_data** | integration.ppa_report.build_ppa_report + write_ppa_report_and_plots | report.json + latency/energy 饼图（算子级 breakdown）+ operator_sequence.txt（+ 可选 memory_transactions.txt），写入 sim_results_dir。 |

两者分工明确：main 侧重“仅仿真”的报告；run_sim_with_real_eval 侧重“真实数据 + 准确率 + PPA”的汇总与图表。
