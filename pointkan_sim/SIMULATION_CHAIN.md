# Simulation Chain Design

CIM PPA Simulator 的端到端仿真链路说明。

---

## Overall Flow

```
Network Definition          Dispatcher              Operator            Scheduler
───────────────           ────────────           ──────────           ─────────

List[LayerDef]
  ├─ LayerDef              dispatch_layer()
  │   ├─ OperatorRef  ───▶ dispatch_operator()
  │   │   op_type="fps"       │
  │   │   params={...}        │  lookup op_type
  │   │                       ▼
  │   │                  _gen_fps()
  │   │                       │
  │   │                       │  extract params
  │   │                       │  from OperatorRef
  │   │                       ▼
  │   │                  FPSOperator(
  │   │                    FPSParams(...),    ◀── hw_config
  │   │                    hw_config
  │   │                  )
  │   │                       │
  │   │                       ▼
  │   │                  .generate_events()
  │   │                       │
  │   │                       ▼
  │   │                  List[Event]  ────────────────────▶  EventQueue
  │   │                                                       │
  │   ├─ OperatorRef  ───▶ (next operator, cycle offset)      │
  │   └─ ...                                                  │
  │                                                           │
  ├─ LayerDef         ───▶ (next layer, cycle offset)         │
  └─ ...                                                      ▼
                                                         Scheduler
                                                         .run()
                                                           │
                                                           ▼
                                                     SimulationResult
                                                       total_cycles
                                                       total_energy
                                                       peak_power
                                                       statistics
```

---

## Data Structures

### 1. Network Layer: `LayerDef`

```python
# Defined in: network/pointkan_net.py

@dataclass
class LayerDef:
    name: str                        # e.g., "GAM-stage0"
    layer_type: str                  # e.g., "GAM", "LFP", "classifier"
    operators: List[OperatorRef]     # ordered operator sequence
    description: str
```

### 2. Operator Reference: `OperatorRef`

```python
# Defined in: network/pointkan_net.py

@dataclass
class OperatorRef:
    op_type: str                     # e.g., "fps", "knn", "conv1d"
    params: Dict[str, Any]           # operator-specific parameters
    description: str                 # human-readable name
```

`params` 内容由 op_type 决定:

| op_type | params 示例 |
|---------|-------------|
| fps | `{input_points: 1024, output_points: 512}` |
| knn | `{input_points: 1024, query_points: 512, k: 24}` |
| conv1d | `{in_channels: 64, out_channels: 128, num_points: 512}` |
| bn | `{num_points: 512}` |
| relu | `{num_points: 512}` |
| kan | `{in_features: 64, out_features: 128, num_points: 512}` |

### 3. Operator Params: `XxxParams`

```python
# Defined in: operators/fps.py, operators/knn.py, etc.

@dataclass
class FPSParams(OperatorParams):
    input_points: int = 1024         # inherited from OperatorParams
    sampled_points_num: int = 512
    cim_parallelism: int = 128
    bit_slice: int = 2
    k2_width: int = 28
    two_qk_width: int = 28
    bitonic_sort_cycles: int = 2
```

### 4. Event

```python
# Defined in: simulator/event.py

@dataclass
class Event:
    event_id: int
    cycle: int                       # start cycle (relative, from 0)
    operator: str                    # operator name
    stage: str                       # stage name
    sub_stage: str                   # sub-stage name
    circuit_module: str              # hardware module (lowercase)
    operation: str                   # operation description
    power: float                     # mW (from hw_config)
    duration: int                    # cycles
    clock_period_ns: float           # ns (from hw_config)
    metadata: Dict[str, Any]

    # Auto-computed:
    # energy = power * duration * clock_period_ns / 1000  (nJ)
```

### 5. Hardware Config: `HardwareConfig`

```python
# Defined in: configs/hw_config.py

@dataclass
class HardwareConfig:
    accelerator_type: AcceleratorType    # BASELINE or ACC
    clock_period: float                  # ns
    sram_cim_macro: SramCimMacroConfig   # array + shift_add + mdt + buffer
    rram_lut_macro: RramLutMacroConfig   # rram_lut (+ speculative for acc)
    rram_cim_macro: RramCimMacroConfig   # placeholder
    sorting_circuit: SortingCircuitConfig # bitonic sort / two-stage comparator
    logic_circuit: LogicCircuitConfig    # bn, ln, relu, spool, quant, interp
```

---

## Dispatcher Role

Dispatcher 是纯粹的桥接层，不包含任何 cycle/power 计算逻辑:

```python
# operators/dispatcher.py

def dispatch_operator(op_ref, hw_config) -> List[Event]:
    """
    1. Look up op_type → handler
    2. Handler extracts params from op_ref.params
    3. Handler builds XxxParams
    4. Handler instantiates XxxOperator(params, hw_config)
    5. Returns operator.generate_events()
    """

def dispatch_layer(layer_operators, hw_config) -> List[Event]:
    """
    Calls dispatch_operator for each OperatorRef sequentially.
    Applies cycle offsets so operators run one after another.
    """
```

Handler 示例 (FPS):

```python
def _gen_fps(op_ref, hw_config):
    params = op_ref.params
    fps_params = FPSParams(
        input_points=params.get('input_points', 1024),
        sampled_points_num=params.get('output_points', 512),
    )
    return FPSOperator(params=fps_params, hw_config=hw_config).generate_events()
```

---

## Cycle Offset Chain

Layer 内的算子顺序执行，由 `dispatch_layer` 管理 cycle offset:

```
Layer: GAM-stage0
  ├─ FPS events:   cycle 0 ── 24999
  ├─ KNN events:   cycle 25000 ── 230000   (offset = FPS end cycle)
  └─ SPool events: cycle 230001 ── 232449  (offset = KNN end cycle)

Layer: LFP-stage0
  ├─ LN events:    cycle 232450 ── 233431  (offset = GAM end cycle)
  ├─ KAN events:   cycle 233432 ── ...
  └─ ...
```

Layer 间的 offset 由 `main.py: simulate_network()` 管理:

```python
for layer in layers:
    layer_events = dispatch_layer(layer.operators, hw_config)
    for event in layer_events:
        event.cycle += current_cycle          # layer-level offset
        scheduler.event_queue.add_event(...)
    current_cycle += max(e.cycle + e.duration for e in layer_events)
```

---

## Scheduler Processing

Scheduler 采用事件驱动仿真，只处理 event 起止时间点:

```
Input: EventQueue (all events with absolute cycles)

Step 1: Build timeline
    For each event → (start_cycle, event_id, START), (end_cycle, event_id, END)
    Sort by cycle

Step 2: Walk timeline
    For each time point:
        - Calculate power × interval for previous interval
        - Update active event set
        - Track peak power

Step 3: Aggregate
    total_cycles = max(end_cycle) across all events
    total_energy = sum(event.energy) for all events
    peak_power = max simultaneous power

Output: SimulationResult
    - total_cycles, total_energy, peak_power
    - events (with status COMPLETED)
    - statistics (by operator, by stage)
```

---

## File Map

```
pointcloud_ppa_sim_new/
├── network/
│   └── pointkan_net.py          # OperatorRef, LayerDef, network definitions
│
├── configs/
│   └── hw_config.py             # HardwareConfig, CircuitModule, power/cycles
│
├── operators/
│   ├── base_operator.py         # BaseOperator (abstract), Stage, SubStage
│   ├── fps.py                   # FPSOperator ← BaseOperator
│   ├── knn.py                   # KNNOperator ← BaseOperator
│   └── dispatcher.py            # op_type → Operator bridge
│
├── simulator/
│   ├── event.py                 # Event dataclass, EventQueue
│   └── scheduler.py             # Scheduler, SimulationResult
│
└── main.py                      # Entry point, simulate_network()
```

---

## Adding a New Operator

1. **定义 Operator 类** — 参考 `OPERATOR_DEV_GUIDE.md`
2. **在 dispatcher 注册** — 添加 handler 到 `_OP_TYPE_HANDLERS`
3. **在 network 中使用** — 创建 `OperatorRef(op_type="xxx", params={...})`

Handler 只做参数桥接:

```python
def _gen_xxx(op_ref, hw_config):
    from .xxx import XxxOperator, XxxParams
    params = op_ref.params
    xxx_params = XxxParams(
        input_points=params.get('input_points', 1024),
        # ... extract other params
    )
    return XxxOperator(params=xxx_params, hw_config=hw_config).generate_events()
```

---

## Pending Operators (TODO)

以下算子在 dispatcher 中已注册但尚未实现 Operator 类（调用会抛出 NotImplementedError）:

| op_type | 目标 Operator 类 | 使用的硬件模块 |
|---------|-----------------|---------------|
| conv1d | Conv1dOperator | rram-cim macro？ |
| linear | LinearOperator | rram-cim macro？ |
| bn / batchnorm1d | BNOperator | BATCHNORM1D |
| layernorm | LayerNormOperator | LAYERNORM |
| relu | ReLUOperator | RELU |
| s-pool | SPoolOperator | SPOOL |
| kan | KANOperator | RRAM_LUT |

实现步骤:
1. 创建 `operators/xxx.py`，实现 `XxxOperator(BaseOperator)`（参考 `OPERATOR_DEV_GUIDE.md`）
2. 更新 `dispatcher.py` 中对应 handler，实例化 XxxOperator 并调用 `generate_events()`
