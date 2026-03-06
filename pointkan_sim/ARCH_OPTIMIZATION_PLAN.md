# PointKAN 仿真器架构符合性分析与优化方案

## 1. 结论先行（对应你给出的 8 点）

当前 `pointkan_sim` 已经具备你提出架构的大部分“骨架能力”，但还没有形成“严格状态机 + 统一资源冲突仲裁 + 可切换 baseline/fuse 的完整执行语义”。

- **已对齐**：
  - DRAM→SRAM-II 的 Phase1 事件与 roofline 周期模型已存在。
  - Phase2 距离计算 `(12/2)*ceil(N/128)` 已显式建模。
  - FPS/KNN 的“距离计算剥离到 Phase2、引擎只做 bitonic”的路径已实现。
  - BN 融合开关、W8A8 量化/反量化 + MNSIM 接口已存在。
- **主要缺口**：
  - baseline 与 fuse 的**共享比较器冲突模型**未做统一资源锁（现在主要靠顺序 dispatch）。
  - KNN fuse（复用距离 + 粗筛细筛）仍是注释级“待扩展”。
  - 标量算子统一接口覆盖不全（如 maxpooling/adaptive pool 分类、spool 命名与语义耦合）。
  - 功耗参数大量为 0/TODO，难以得到可信 PPA。

---

## 2. 与目标架构逐条对照

## 2.1 内存架构分层

- **标准 SRAM buffer**：已由 `inter_layer_buffer` 建模，含容量、带宽和 DRAM spill。 
- **SRAM-II（距离计算）**：已有 `SRAM_IIConfig`，含 `active_rows=128`、`bit_slice=2`、`bits_per_axis=12`。 
- **RRAM-CIM（MNSIM）**：`conv1d/linear` 已对接 PE 级适配和可选 full-NetStruct。 
- **RRAM-LUT**：KAN 已通过手动 LUT roofline 模型建事件。

## 2.2 Phase1：启动加载

现有 `gen_phase1_point_cloud_load_events()` 已按字节数和 DRAM 带宽给出周期，能耗由 `DRAM.read_power_mw` 计算，符合“roofline+DRAM功耗”思路。

**建议增强**：把“Phase1 结束触发状态2”显式化为状态机（而不仅是事件顺序）。

## 2.3 Phase2：距离计算 + 后续分流（FPS/KNN）

当前 dispatcher 在 `distance_in_phase2=True` 下，已经把每查询距离事件插入到 FPS/KNN 之前；并采用 `per_query_cycles=(bits/slice)*ceil(N/active_rows)`。这与你的核心公式一致。

**现状问题**：
- 只实现了“先距离后排序”的串行拼接；
- 缺少统一“共享 bitonic 资源”的仲裁层（baseline 的 a/b 不并行只在逻辑上成立，未提升为硬约束模型）。

## 2.4 FPS 引擎（baseline/fuse）

- baseline：已支持“stage1-4 归 Phase2，FPS 只算 bitonic”。
- fuse：当前代码没有独立 fuse 行为分支（本质仍同 baseline 路径）。

## 2.5 KNN 引擎（baseline/fuse）

- baseline：bitonic top-k 已实现。
- fuse：代码注释提到“复用距离，待扩展”，尚未实现 coarse/fine 两阶段筛选事件。

## 2.6 BN_EN 融合

`bn_fuse_with_conv_linear=True` 时，BN 紧跟 Conv/Linear 会被 dispatcher 跳过，符合推理融合设想。

## 2.7 量化→RRAM-CIM→反量化流水

已支持 `enable_w8a8_quant_dequant` 在 Conv/Linear 事件前后插入 quant/dequant，满足“输入激活量化、权重量化、计算后反量化”的基本闭环。

**建议增强**：增加“是否写回 DRAM”条件判定（按后继层位置/buffer容量/流水并发）并做事件级决策。

## 2.8 统一接口（特征计算 + 标量计算）

- 特征计算接口目前以 `op_type` dispatch 统一，但仅明确支持 `conv1d/linear`，`conv2d` 尚未并入。
- 标量接口已有 `bn/layernorm/relu/spool/adaptive_max_pool1d/add/mul`，但池化类可再统一抽象。

---

## 3. 优化蓝图（建议按 4 个里程碑推进）

## M1：先补“执行语义正确性”（优先级最高）

1. 引入显式状态机 `SimState`：
   - `BOOT_LOAD`（Phase1）
   - `DISTANCE`（Phase2）
   - `FPS_SELECT`
   - `KNN_TOPK`
   - `FEATURE_COMPUTE`
2. 每个状态定义 entry/exit 条件：
   - 例如 `BOOT_LOAD` 完成后才能进入 `DISTANCE`。
3. 将 baseline 的共享比较器建模为**资源锁**：
   - `resource_id="bitonic_shared"`，同一时段只允许一个事件占用。

> 这样可避免“靠 dispatch 顺序凑串行”的隐式行为，报告可解释性也更强。

## M2：补全 baseline/fuse 差异化路径

1. FPS：新增 `mode in {baseline, fuse}` 参数化生成。
2. KNN：实现 fuse 两级候选过滤：
   - coarse filter（低精度/块级筛选）
   - fine filter（精排 top-k）
3. distance 复用：在 fuse 路径中显式复用 Phase2 距离缓存，避免重复距离事件。

## M3：统一“内存-计算流水”接口

定义统一任务描述 `TensorTask`：
- `src`（DRAM/SRAM buffer/SRAM-II）
- `dst`
- `dtype`（fp32/int8）
- `shape`
- `producer/consumer`

由统一编排器自动生成：
- load（DRAM→SRAM）
- quant
- compute（RRAM-CIM / RRAM-LUT / logic）
- dequant
- writeback（按依赖判断是否落 DRAM）

这样第 6 点的流水与写回策略就不是“人工分散在各 op 里”，而是统一规划。

## M4：参数标定与报告可信度建设

1. 将 `hw_config` 中 0/TODO 功耗参数替换为可追溯标定值（CACTI/MNSIM/文献）。
2. 报告新增“参数来源字段”：`source={cacti,mnsim,paper,assumed}`。
3. 在 breakdown 报告中拆分：
   - memory energy（DRAM/SRAM）
   - compute energy（RRAM-CIM/RRAM-LUT/logic）
   - control/sorting energy。

---

## 4. 建议的数据结构改造（可直接落地）

## 4.1 事件层新增资源占用描述

给 `Event.metadata` 增加可选字段：
- `resource_id`：如 `bitonic_shared`
- `exclusive`：是否互斥
- `priority`：调度优先级

Scheduler 在推进时检测冲突并自动顺延事件起始周期。

## 4.2 统一距离核接口

新增 `DistanceKernelConfig`：
- `bits_per_axis`
- `bit_slice`
- `active_rows`
- `query_fetch_bytes`
- `query_fetch_from_dram`

把 FPS/KNN 的距离阶段统一调用同一个 kernel 生成器（当前虽接近统一，但仍分散在 dispatcher/FPS/KNN）。

## 4.3 统一特征计算接口

新增 `FeatureComputeOp`：
- `kind in {linear, conv1d, conv2d}`
- `backend in {mnsim_rram_cim, analytical}`
- `quant_policy in {none, w8a8}`

dispatch 只做映射，不直接携带太多分支逻辑。

---

## 5. 快速收益清单（两周内可完成）

- [ ] 增加 `mode`（baseline/fuse）总开关并贯穿 FPS/KNN。
- [ ] 为 bitonic 比较器增加互斥资源建模。
- [ ] 实现 KNN fuse 的粗筛+细筛事件路径（先用解析模型）。
- [ ] 把 DRAM query 读取也纳入 Phase2 事件（当前主要建了距离计算核本身）。
- [ ] 为报告增加“Phase1/Phase2/FPS/KNN/Feature”分阶段占比。

---

## 6. 最终建议

你的思路与当前代码方向一致，建议不要“大重写”，而是采用“**状态机 + 资源仲裁 + 模式分支补齐 + 参数标定**”的增量改造。

最关键的是先把“baseline 串行共享比较器”和“fuse 复用距离”做成可验证的事件级语义；只要这两点落地，后续 roofline、能耗和流水优化结果才有可信度。
