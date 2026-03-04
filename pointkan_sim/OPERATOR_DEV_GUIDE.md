# Custom Operator Development Guide

How to implement a new hardware operator for the CIM PPA simulator.

---

## Architecture Overview

```
hw_config.py          base_operator.py         your_operator.py
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│HardwareConfig│     │  BaseOperator    │     │  YourOperator    │
│  .clock_period│────▶│  (abstract)      │◀────│  (concrete)      │
│  .sram_cim_.. │     │  _define_stages  │     │  _define_stages  │
│  .sorting_..  │     │  calculate_cycles│     │  calculate_cycles│
│  .logic_..    │     │  generate_events │     │  generate_events │
└──────────────┘     └──────────────────┘     └──────────────────┘
                                                       │
                                                       ▼
                                                  List[Event]
                                                       │
                                               ┌───────▼───────┐
                                               │   Scheduler   │
                                               │   .run()      │
                                               └───────────────┘
```

Operator 负责两件事:
1. **描述硬件处理流程** — 哪些 stage、使用哪些电路模块 (`_define_stages`)
2. **生成仿真事件** — 按处理流程逐 cycle 生成 Event 列表 (`generate_events`)

---

## Step-by-Step Implementation

### Step 1: Define Params Dataclass

继承 `OperatorParams`（提供 `input_points`），添加算子特有的参数。

```python
from dataclasses import dataclass
from .base_operator import OperatorParams

@dataclass
class MyParams(OperatorParams):
    """
    Inherits from OperatorParams:
        input_points: int = 1024

    Custom fields below:
    """
    output_points: int = 256        # task-specific parameter
    some_width: int = 16            # bit-width for some computation
    bit_slice: int = 2              # bits processed per CIM cycle
    fixed_sort_cycles: int = 10     # fixed-cycle stage
```

**Guidelines:**
- 只放影响 cycle/event 计算的参数，不放纯描述性字段
- 硬件 power/area 参数不放这里，统一从 `hw_config` 读取
- 固定常数（如 bitonic sort cycles）直接设为默认值

---

### Step 2: Define the Operator Class

```python
from typing import List, Optional, Dict, Any, TYPE_CHECKING
import math

from .base_operator import BaseOperator, Stage, SubStage, OperatorParams
from ..configs.hw_config import CircuitModule
from ..simulator.event import Event

if TYPE_CHECKING:
    from ..configs.hw_config import HardwareConfig


class MyOperator(BaseOperator):

    def __init__(
        self,
        params: Optional[MyParams] = None,
        hw_config: Optional['HardwareConfig'] = None
    ):
        if params is None:
            params = MyParams()
        self.my_params: MyParams = params          # typed reference
        super().__init__("MyOp", params, hw_config) # triggers _define_stages()
```

**Key pattern:**
- `self.my_params` 保存具体类型的引用，避免后续 cast
- `super().__init__()` 会自动调用 `_define_stages()`

---

### Step 3: Implement `_define_stages()`

定义硬件处理流程中的每个 stage 和 sub-stage，并关联 `CircuitModule`。

```python
    def _define_stages(self) -> None:
        # Stage 1: CIM computation
        stage1 = Stage(
            name="cim_compute",
            is_pipeline=False,
            description="SRAM-CIM array bit-serial MAC"
        )
        stage1.add_sub_stage(SubStage(
            name="cim_mac",
            circuit_modules=[
                CircuitModule.SRAM_CIM_ARRAY,
                CircuitModule.SHIFT_ADD,
            ],
            is_pipeline=False,
            description="Bit-serial MAC via SRAM-CIM array + shift-and-add"
        ))
        self.stages.append(stage1)

        # Stage 2: Post-processing
        stage2 = Stage(
            name="post_process",
            is_pipeline=False,
            description="Result post-processing"
        )
        stage2.add_sub_stage(SubStage(
            name="sort_result",
            circuit_modules=[CircuitModule.COMP_TREE_1024_MAX],
            is_pipeline=False,
            description="Find maximum via comparator tree"
        ))
        self.stages.append(stage2)
```

**CircuitModule 选择参考:**

| CircuitModule enum | hw_config 位置 | 典型用途 |
|---|---|---|
| `SRAM_CIM_ARRAY` | `sram_cim_macro.array` | CIM 阵列读/MAC |
| `SHIFT_ADD` | `sram_cim_macro.shift_add` | bit-serial 累加 |
| `MIN_DISTANCE_TABLE` | `sram_cim_macro.min_distance_table` | FPS 最小距离表 |
| `BUFFER` | `sram_cim_macro.buffer` | SRAM 写回/暂存 |
| `RRAM_LUT` | `rram_lut_macro.rram_lut` | RRAM 查找表 |
| `COMP_TREE_1024_MAX` | `sorting_circuit.comp_tree_1024_max` | 1024 取最大 (FPS) |
| `COMP_BITONIC_1024_MIN32` | `sorting_circuit.comp_bitonic_1024_min32` | 1024 取 k 小 (KNN) |
| `BATCHNORM1D` | `logic_circuit.batchnorm1d` | BatchNorm |
| `LAYERNORM` | `logic_circuit.layernorm` | LayerNorm |
| `RELU` | `logic_circuit.relu` | ReLU 激活 |
| `SPOOL` | `logic_circuit.spool` | 空间池化 |
| `QUANT_DEQUANT` | `logic_circuit.quant_dequant` | 量化/反量化 |

如果现有枚举不包含你的硬件模块，需要先在 `hw_config.py` 中:
1. 添加新的 `XxxConfig` dataclass
2. 将其加入对应的 macro config（如 `LogicCircuitConfig`）
3. 在 `CircuitModule` enum 中添加对应成员
4. 在 `_build_baseline()` / `_build_acc()` 中实例化
5. 在 `get_all_modules()` 中注册

---

### Step 4: Implement Derived Cycle Helpers

将每个 stage 的 cycle 数封装为 helper 方法，便于复用。

```python
    def _get_compute_cycles(self) -> int:
        """Cycles for CIM compute stage per batch."""
        p = self.my_params
        return p.some_width // p.bit_slice

    def _get_per_batch_cycles(self) -> int:
        """Total cycles per batch (all batch-level stages)."""
        return self._get_compute_cycles() + 1  # +1 for post-processing
```

---

### Step 5: Implement `calculate_cycles()`

纯数学计算，不生成 Event，用于快速估算。

```python
    def calculate_cycles(self, params: Optional[MyParams] = None) -> int:
        if params is not None:
            self.my_params = params

        p = self.my_params
        per_batch = self._get_per_batch_cycles()

        total_cycles = 0
        for i in range(p.output_points):
            candidates = p.input_points - i
            batches = math.ceil(candidates / 128)  # cim_parallelism
            total_cycles += batches * per_batch + p.fixed_sort_cycles

        return total_cycles
```

**Important:** `calculate_cycles()` 和 `generate_events()` 的总 cycle 数必须一致。

---

### Step 6: Implement `calculate_energy()`

统一模式 — 从 events 累加 energy:

```python
    def calculate_energy(self, params: Optional[MyParams] = None) -> float:
        """Calculate total energy (nJ). Computed from generated events."""
        events = self.generate_events(params)
        return sum(e.energy for e in events)
```

---

### Step 7: Implement `generate_events()`

这是核心方法。按硬件处理流程生成 Event 列表。

```python
    def generate_events(self, params: Optional[MyParams] = None) -> List[Event]:
        if params is not None:
            self.my_params = params

        p = self.my_params

        # --- Read power from hw_config ---
        hw = self.hw_config
        cim_power = hw.sram_cim_macro.array.cim_power
        sort_power = hw.sorting_circuit.comp_tree_1024_max.power
        clock_period = hw.clock_period

        compute_cycles = self._get_compute_cycles()

        events: List[Event] = []
        event_id = 0
        cycle = 0

        for i in range(p.output_points):
            candidates = p.input_points - i
            batches = math.ceil(candidates / 128)

            for b in range(batches):
                # Stage 1: CIM compute (per batch)
                events.append(Event(
                    event_id=event_id,
                    cycle=cycle,
                    operator="MyOp",
                    stage="cim_compute",
                    sub_stage="cim_mac",
                    circuit_module="sram_cim_array",
                    operation="bit_serial_mac",
                    power=cim_power,
                    duration=compute_cycles,
                    clock_period_ns=clock_period,
                    metadata={"iteration": i, "batch": b},
                ))
                event_id += 1
                cycle += compute_cycles

            # Stage 2: Sort (once per iteration, after all batches)
            events.append(Event(
                event_id=event_id,
                cycle=cycle,
                operator="MyOp",
                stage="post_process",
                sub_stage="sort_result",
                circuit_module="comp_tree_1024_max",
                operation="find_max",
                power=sort_power,
                duration=p.fixed_sort_cycles,
                clock_period_ns=clock_period,
                metadata={"iteration": i},
            ))
            event_id += 1
            cycle += p.fixed_sort_cycles

        return events
```

---

## Event Fields Reference

每个 Event 的字段含义:

| Field | Type | Description |
|---|---|---|
| `event_id` | int | 自增 ID，在算子内从 0 开始 |
| `cycle` | int | Event 开始的 cycle（相对于算子起点，从 0 开始） |
| `operator` | str | 算子名称，如 `"FPS"`, `"KNN"` |
| `stage` | str | Stage 名称，与 `_define_stages()` 中一致 |
| `sub_stage` | str | SubStage 名称，与 `_define_stages()` 中一致 |
| `circuit_module` | str | 电路模块名称（小写下划线），如 `"sram_cim_array"` |
| `operation` | str | 具体操作描述，如 `"k2_bit_serial"` |
| `power` | float | 功耗 (mW)，从 `hw_config` 读取 |
| `duration` | int | 持续 cycle 数 |
| `clock_period_ns` | float | 时钟周期 (ns)，从 `hw_config.clock_period` 读取 |
| `metadata` | dict | 附加信息，如 `{"iteration": 0, "batch": 3}` |

**Energy 自动计算:** `event.energy = power(mW) * duration(cycles) * clock_period_ns / 1000 → nJ`

---

## Event Generation Patterns

### Pattern A: Per-batch stages + per-iteration stage (FPS / KNN)

```
For each iteration:
    For each batch:
        Stage 1 event   (per batch)
        Stage 2 event   (per batch)
        ...
    Stage N event       (once per iteration, after all batches)
```

FPS 和 KNN 都使用这种模式:
- Stages 1-4 在每个 batch 内顺序执行
- Stage 5 (sort) 在所有 batch 完成后执行一次

### Pattern B: Fixed-count iteration (KNN)

KNN 的 batches 数在所有 query 中恒定:
```python
batches = math.ceil(p.input_points / p.cim_parallelism)  # constant
for q in range(p.query_points):
    for b in range(batches):
        ...  # stages 1-4
    ...  # stage 5
```

### Pattern C: Decreasing iteration (FPS)

FPS 每次迭代减少候选点数:
```python
for i in range(p.sampled_points_num):
    candidate_points = p.input_points - i    # decreasing
    batches = math.ceil(candidate_points / p.cim_parallelism)
    for b in range(batches):
        ...  # stages 1-4
    ...  # stage 5
```

---

## Registering with Dispatcher (Optional)

如果算子需要通过 network 层定义被调用，需在 `dispatcher.py` 中注册。

### 1. Add handler function

```python
def _gen_myop(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    **ctx
) -> List[Event]:
    from .my_operator import MyOperator, MyParams

    input_points = ctx.get('num_points', 1024)
    output_points = op_ref.params.get('output_points', 256)

    params = MyParams(
        input_points=input_points,
        output_points=output_points,
    )
    op = MyOperator(params=params, hw_config=hw_config)
    return op.generate_events()
```

### 2. Register in `_OP_HANDLERS`

```python
_OP_HANDLERS: Dict[str, Callable] = {
    ...
    'myop': _gen_myop,
}
```

---

## Checklist

- [ ] `MyParams` 继承 `OperatorParams`，所有影响 cycle 的参数都有默认值
- [ ] `_define_stages()` 中每个 stage 的 `circuit_modules` 对应 `CircuitModule` enum 中的值
- [ ] `generate_events()` 中 `stage`/`sub_stage` 字符串与 `_define_stages()` 中的 name 一致
- [ ] `generate_events()` 中 `circuit_module` 字符串是 enum 成员的小写形式
- [ ] power 从 `self.hw_config` 对应的 config 读取，不写死默认值
- [ ] `clock_period_ns` 统一从 `self.hw_config.clock_period` 读取
- [ ] `calculate_cycles()` 和 `generate_events()` 返回的总 cycle 数一致
- [ ] Event 的 `cycle` 字段从 0 开始递增，每个 event 的 cycle += 前一个 event 的 duration
- [ ] 如需 dispatcher 集成，在 `_OP_HANDLERS` 注册

---

## Complete Example: FPS Operator (Simplified)

以下是 FPS 算子的核心结构，展示完整的实现模式:

```
File: operators/fps.py

Module docstring
    ├── Execution flow description (stages, formula)
    └── Cycle formula

FPSParams(OperatorParams)
    ├── sampled_points_num: int = 512
    ├── cim_parallelism: int = 128
    ├── bit_slice: int = 2
    ├── k2_width: int = 28
    ├── two_qk_width: int = 28
    └── bitonic_sort_cycles: int = 2

FPSOperator(BaseOperator)
    ├── __init__(params, hw_config)
    │   ├── self.fps_params = params
    │   └── super().__init__("FPS", params, hw_config)
    │
    ├── _get_k2_cycles()           # helper: k2_width // bit_slice
    ├── _get_two_qk_cycles()       # helper: two_qk_width // bit_slice
    ├── _get_per_batch_cycles()    # helper: sum of stage 1-4
    │
    ├── _define_stages()           # 5 stages with CircuitModule
    │   ├── Stage 1: k_square         → [SRAM_CIM_ARRAY, SHIFT_ADD]
    │   ├── Stage 2: cross_term       → [SRAM_CIM_ARRAY, SHIFT_ADD]
    │   ├── Stage 3: accumulate       → [SHIFT_ADD]
    │   ├── Stage 4: min_distance_update → [MIN_DISTANCE_TABLE]
    │   └── Stage 5: bitonic_sort     → [COMP_TREE_1024_MAX]
    │
    ├── calculate_cycles()         # pure math, no events
    │   └── sum over iterations: batches * per_batch + sort_cycles
    │
    ├── calculate_energy()         # sum(e.energy for e in generate_events())
    │
    ├── generate_events()          # core: produces List[Event]
    │   ├── Read power from hw_config (no fallback defaults)
    │   ├── Nested loop: iterations × batches
    │   │   ├── Stage 1 event (duration = k2_cycles)
    │   │   ├── Stage 2 event (duration = tqk_cycles)
    │   │   ├── Stage 3 event (duration = 1)
    │   │   └── Stage 4 event (duration = 1)
    │   └── Stage 5 event per iteration (duration = bitonic_sort_cycles)
    │
    └── get_cycle_breakdown()      # optional reporting helper
```
