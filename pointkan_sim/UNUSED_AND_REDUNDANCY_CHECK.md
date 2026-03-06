# pointkan_sim 未使用与冗余接口检查报告

对 `pointkan_sim/` 的静态检查结果：未被调用的函数/变量、重复或冗余接口。

**已完成的优化（2025-03）**：  
- 已将 `_safe_div_ceil` 抽到 **utils/math_utils.safe_div_ceil**，common_ops 与 layer_buffer_ops 改为引用。  
- 已删除 **gen_phase1_point_cloud_load_events**，仅保留 **gen_first_layer_input_load_events** 作为首层/Phase1 加载唯一入口。  
- 已移除网络级 breakdown：**analyze_network_breakdown / format_network_breakdown / print_breakdown / print_network_breakdown**；保留算子级（analyze_simulation_events_breakdown + ppa_report 饼图）。  
- 已删除 **\_get_op_type_from_netstruct**（ppa_report）、**build_netstruct_from_conv_linear_list**（real_data_eval）。  
- 模块与文件归属见 **FILE_OWNERSHIP.md**。

---

## 一、未被使用的函数 / 接口（当前仍存在）

### 1. report/reporter.py

| 符号 | 说明 |
|------|------|
| `create_report_from_simulation` | 根据仿真结果创建 Reporter 并填充 PPAResult。**全仓库无调用**。main 使用 `create_ppa_result` + `Reporter` 直接写 JSON/CSV，未走该封装。 |

---

### 3. network/

| 符号 | 说明 |
|------|------|
| `get_pointkan_info` (pointkan_net.py) | 返回 PointKAN 网络元信息字典。**无调用**。 |
| `get_pointkan_elite_info` (pointkan_elite_net.py) | 返回 PointKAN-Elite 网络元信息字典。**无调用**。 |

---

### 6. utils/logger.py（仅定义、无调用）

| 符号 | 说明 |
|------|------|
| `log_dict` | 递归打印字典，带标题与缩进。**无调用**。 |
| `log_table` | 打印表格。**无调用**。 |
| `log_ppa_summary` | 打印 PPA 汇总。**无调用**。 |
| `SimulationLogger` | 上下文管理器，用于仿真日志。**无调用**。 |
| `get_log_file_path` | 返回当前日志文件路径。**无调用**。 |
| `set_log_level` | 动态设置日志级别。**无调用**。 |
| `log_debug` / `log_warning` / `log_error` / `log_critical` | 各级别日志。**当前仅见 `log_info`、`log_section`、`log_separator` 被 main/run_sim_with_real_eval 使用**。 |

---

### 7. utils/units.py（整模块未被引用）

**全模块**：`energy_nJ_from_ns`, `energy_to_nj`, `power_from_energy_nJ`, `power_to_mw`, `cycles_to_ns`, `cycles_to_ms`, `time_to_ns`, `freq_mhz_from_period_ns`, `area_to_mm2` 等均在 `utils/__init__.py` 中导出，但 **除 utils 自身外无任何 `from pointkan_sim.utils import ...` 或 `from pointkan_sim.utils.units import ...`**。能耗/周期/单位换算目前均在各自模块内手写或使用 hw_config 的 clock_period 等。

---

### 8. operators/dispatcher.py

| 符号 | 说明 |
|------|------|
| `register_op_type` | 动态注册 op_type → handler。**无调用**。当前所有 handler 均在模块内写死到 `_HANDLERS` 字典。 |

---

## 二、冗余 / 重复接口（已处理）

- **_safe_div_ceil**：已统一为 **utils/math_utils.safe_div_ceil**，common_ops 与 layer_buffer_ops 已改为引用。  
- **首层点云加载**：已删除 gen_phase1_point_cloud_load_events，仅保留 gen_first_layer_input_load_events 作为唯一入口。

---

## 三、建议汇总

| 类型 | 建议 |
|------|------|
| 未使用且可删 | `create_report_from_simulation`（reporter）可考虑删除或标注为遗留；已删 `_get_op_type_from_netstruct`、`build_netstruct_from_conv_linear_list` 及网络级 breakdown 四函数。 |
| 未使用但保留 | `register_op_type`、`get_pointkan_info`、`get_pointkan_elite_info` 可作为扩展/脚本用 API 保留，在文档中标明“当前主流程未使用”。 |
| 日志与单位 | `log_dict`、`log_table`、`log_ppa_summary`、`SimulationLogger`、`get_log_file_path`、`set_log_level`、各级别 log_*：若短期无使用计划，可仅做文档说明；`utils/units.py` 注明“预留单位换算”。 |
| 去重 | 已完成：`safe_div_ceil` 在 utils/math_utils；首层加载仅保留 `gen_first_layer_input_load_events`。 |

---

## 四、检查方法说明

- 通过 `grep` 对 `pointkan_sim` 内所有 `def ` / `class ` 的符号做“定义处 + 调用处”检索。
- “未使用”指：除定义与 `__all__`/导出外，无其它文件或同文件其它位置调用。
- “冗余”指：两处实现相同或高度相似逻辑，或两条接口服务同一用途且其中一条无调用。

如需对某一类（例如仅删除明确无用的死代码、或仅做 _safe_div_ceil 与首层加载的合并）做具体补丁级修改，可指定范围后按上述建议逐项改。
