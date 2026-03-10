# reconfig_sim_proj 接口分析（中文）

本文档对现有代码做**中文注释分析**，标出**未完善的接口**、**有问题的接口**（已修复或待修）以及**未使用的接口**，便于后续扩展与维护。

---

## 1. 未完善的接口

以下接口已实现但功能不完整、或与上下游未打通，建议后续完善。

### 1.1 仿真器与 NoC trace 未打通

| 位置 | 接口/模块 | 说明 |
|------|-----------|------|
| `reconfig_sim/simulator.py` | `ReconfigSimulator` | 仅提供 `run_gemm` / `run_conv2d`，返回周期/能耗与 trace 字典；**未提供**从内部 schedule（schedulers 的 TileDesc 列表）导出 **NoC 流量 trace** 的接口。若要从「仿真器一次运行」直接得到 BookSim 用 trace，需在仿真器内调用 dispatch 链路或增加 `export_noc_trace()` 之类方法。 |
| `reconfig_sim/schedulers/` | `GemmTiler` / `Conv2DTiler` / `Dispatcher` | 使用的 `TileDesc` 来自 `models/tile.py`（字段：m0, k0, n0, m_tile, k_tile, n_tile, mode, op_type 等），与 **dispatch 的 TileDesc**（`dispatch/dispatcher.py`：op_name, op_type, mode, m0, n0, k0, mt, nt, kt）**不是同一类型**。两套分块/调度体系并行，仿真器内部分块结果目前不能直接喂给 `build_traffic_trace`（其需要 dispatch 的 TileDesc + TileStats）。 |

**建议**：  
- 要么在仿真器内增加「将 schedulers 的 tile 序列 + 周期信息」转换为 `dispatch.TileDesc` + `TileStats` 的适配层，再调用 `build_traffic_trace`；  
- 要么在文档中明确：**NoC trace 仅来自 dispatch 链路**（OpDesc → ModeAwareDispatcher → build_traffic_trace / dispatch_to_traffic_trace），仿真器仅做周期/能耗估计。

### 1.2 `merge_traces` 无调用方

| 位置 | 接口 | 说明 |
|------|------|------|
| `reconfig_sim/noc/traffic_trace.py` | `merge_traces(traces: Iterable[TrafficTrace]) -> TrafficTrace` | 用于将多段 TrafficTrace 按 cycle 排序合并为一条 trace。当前**无任何业务代码或示例调用**，多 op 场景通常用单次 `dispatch(ops)` 得到一条 schedule 再 `build_traffic_trace`。若后续支持「多 run 合并」「多策略对比合并」可启用此接口。 |

### 1.3 配置与 dispatch 参数未统一

| 位置 | 说明 |
|------|------|
| `configs/defaults.py` | `SimConfig` 含 `timing.mode_switch_cycles`、`array.arr_r/arr_c` 等； |
| `dispatch/dispatcher.py` | `HardwareConfig` 单独定义 `mode_switch_cycles`、`arr_r`、`arr_c`、`os_clear_cycles` 等。两处**未共享**，修改默认值时需改两处，易不一致。 |

**建议**：提供从 `SimConfig` 生成 `HardwareConfig` 的工厂函数，或统一从同一配置源读取。

### 1.4 BookSim 实际对接

| 位置 | 说明 |
|------|------|
| `reconfig_sim/noc/booksim_adapter.py` | 仅生成 **BookSim 可读的文本/JSON/CSV**（如 packet trace 行格式），**未实现**调用 BookSim 可执行文件或库、读回延迟/吞吐结果。`noc_model.py` 的 `why_not_booksim()` 也仅说明「当前为解析模型」。 |

**建议**：若需「仿真器 → trace → BookSim 运行 → 回填周期」的闭环，需在 adapter 或单独脚本中增加调用 BookSim 的接口（如 subprocess 或 C++ 扩展）。

---

## 2. 有问题的接口（已修复或需注意）

### 2.1 已修复

| 问题 | 位置 | 修复说明 |
|------|------|----------|
| **traffic_trace.py 重复定义** | `reconfig_sim/noc/traffic_trace.py` | 文件中曾存在**两段几乎相同的代码**（TrafficEvent/TrafficTrace/build_traffic_trace/dispatch_to_traffic_trace/merge_traces 被定义两次），后者会覆盖前者，且第二段缺少 `build_traffic_trace`，易导致行为混乱。**已删除第二段重复代码**，仅保留一份完整实现。 |
| **CalibrationResult.mode 语义错误** | `reconfig_sim/noc/calibration.py` | `evaluate_policy_traffic` 中原先用 `policy.__class__.__name__` 作为 `mode`，对 `FixedModePolicy("OS")` 与 `FixedModePolicy("WS")` 都会得到 `"FixedModePolicy"`，无法区分 OS/WS。**已改为**对带 `mode` 属性的策略（如 FixedModePolicy）使用 `policy.mode`，否则回退到类名。 |

### 2.2 需注意（未改代码）

| 位置 | 说明 |
|------|------|
| **dispatch.dispatcher.ModeAwareDispatcher.dispatch** | 返回的 `DispatchResult` 中 `schedule` 与 `tile_stats` 一一对应；`build_traffic_trace` 依赖 `tile_stats` 提供每块周期以推进 `cycle_cursor`。若外部构造 schedule 时未提供或错位提供 `tile_stats`，trace 的时间轴会错。调用方应保证成对传入。 |
| **schedulers.models.tile.TileDesc** | 可变（非 frozen），`Dispatcher.assign_modes` 会原地改 `t.mode`。若多处共享同一 tile 对象，可能产生副作用。 |

---

## 3. 未使用的接口

以下接口在代码库中**未被任何模块或测试调用**，仅定义并导出。保留可用于未来扩展，但需在文档中标明「当前未用」。

### 3.1 策略类 OverridePolicy

| 位置 | 接口 | 说明 |
|------|------|------|
| `reconfig_sim/dispatch/policies.py` | `OverridePolicy(MappingPolicy)` | 在某一策略基础上按 op 名称覆盖模式（per_op_mode）。**当前无任何示例、测试或脚本使用**；policy_demo 仅使用 `FixedModePolicy` 与 `RuleBasedPolicy`。若需「默认规则 + 少量 op 覆盖」可启用。 |

### 3.2 merge_traces

| 位置 | 接口 | 说明 |
|------|------|------|
| `reconfig_sim/noc/traffic_trace.py` | `merge_traces(traces: Iterable[TrafficTrace]) -> TrafficTrace` | 见 1.2：已实现且已导出，**无调用方**。 |

### 3.3 包级导出但仅测试使用的接口

| 接口 | 说明 |
|------|------|
| `BookSimAdapter.write_trace_json` / `write_trace_csv` | 在 `tests/test_noc_trace_adapter.py` 中有使用；示例脚本 `dispatcher_trace_minimal.py` 仅调用 `write_packet_trace`，未写 JSON/CSV。若希望示例也产出 JSON/CSV，可在 minimal 或 run_sims.sh 中增加一次调用。 |

---

## 4. 接口与数据流小结（中文注释）

```
【仿真器链路】仅周期/能耗，不产 NoC trace
  SimConfig → ReconfigSimulator → run_gemm / run_conv2d
    → schedulers (GemmTiler / Conv2DTiler / Dispatcher)
    → models.tile.TileDesc
    → CycleEnergyModel.estimate_tile → 报告 JSON

【NoC trace 链路】独立于仿真器，产 trace 与 BookSim 文本
  OpDesc 列表 + HardwareConfig + MappingPolicy
    → dispatch.ModeAwareDispatcher.dispatch → DispatchResult (schedule + tile_stats)
    → build_traffic_trace(schedule, hw, tile_stats=..., total_cycles=...)
      或 dispatch_to_traffic_trace(dispatch_result, hw)
    → TrafficTrace
    → BookSimAdapter.write_packet_trace / write_trace_json / write_trace_csv
      或 export_trace（内存 payload）

【标定】
  compare_fixed_modes(ops, hw) → Dict["OS"|"WS", CalibrationResult]
  内部使用 FixedModePolicy、build_traffic_trace、BookSimAdapter.summarize。
```

两套链路的 **TileDesc 不同**：  
- 仿真器：`reconfig_sim.models.tile.TileDesc`（m_tile, k_tile, n_tile, op_type, batch_idx 等）；  
- NoC trace：`reconfig_sim.dispatch.dispatcher.TileDesc`（mt, nt, kt, op_name, op_type, mode 等）。  
二者不可混用，需通过适配或文档明确分工。

---

## 5. 使用 run_sims.sh 执行不同仿真与 NoC trace

工程已提供 **run_sims.sh**，用于统一执行：

- 周期仿真 demo（run_demo.py，可选 --viz）
- 策略对比 demo（policy_demo.py）
- NoC trace 最小示例（dispatcher_trace_minimal.py）
- NoC trace + 标定测试（test_noc_trace_adapter.py）
- 冒烟测试（smoke_test、test_policy_dispatch）

详见脚本内注释与 `./run_sims.sh --help`（若实现）。
