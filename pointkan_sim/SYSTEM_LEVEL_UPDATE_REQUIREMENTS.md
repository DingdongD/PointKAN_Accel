# PointKAN Simulator 系统级建模更新需求（草案）

> 目标：先统一“事件/流水/访存/并行度/周期与功耗面积公式”的建模口径，再进入代码实现。
> 本文是 **需求冻结文档**，用于指导 `pointkan_sim` 与 `hw_sim/cycle_simulator/memory_model/demo` 的后续实现。

## 1. 总体建模原则

1. **时钟统一**：当前所有给定模块频率均为 1GHz，默认时钟周期 `1ns`。
2. **并行度参数化**：所有可并行模块统一引入 `P`，除非模块定义了固定并行度（如 `P=32` 或 `128`）。
3. **单位规则**：
   - 功耗：mW（工作态）
   - 面积：um^2（可在报表里再换算 mm^2）
   - 周期：cycle
4. **事件建模层级**：
   - Operator Event（算子级）
   - Stage Event（阶段级）
   - Memory Event（SRAM/DRAM 读写）
5. **资源互斥与并发**：
   - 同一物理资源（bitonic/coarse-fine comparator/resampler 等）使用 `resource_id` 串行化。
   - 可 overlap 阶段在 metadata 标记 `overlap_group`，由 scheduler 按资源和依赖图求解。

---

## 2. 单元级 PPA + 周期模型（需更新）

## 2.1 基础算术与激活

- **FP32 MUL**：`cycle=1`，`power=2.5mW`，`area=2024um^2`，并行度 `P`
- **FP32 DIV**：`cycle=10`，`power=2.59mW`，`area=6307um^2`，并行度 `P`
- **FP32 ADD**：`cycle=1`，`power=1.16mW`，`area=1273um^2`，并行度 `P`
- **ReLU**：`cycle=1(逐元素)`，`power=0.45mW`，`area=222um^2`，并行度 `P`

> 说明：上述 cycle 均指“一对操作数/一个元素”。

## 2.2 归一化/池化/量化

- **BatchNorm**（输入 shape `(B,N,S)`，按 `N` 归一化）
  - 约束：`N mod P = 0`
  - 周期：`T = 1 + N*S/P`
  - 功耗：`power_total = 3.1718 * P (mW)`
  - 面积：`area_total = 3228 * P (um^2)`

- **AffineNorm**
  - 公式：`T = 2*(L/P + 1) + 46`，`L=S*K*D`，固定 `P=32`
  - 功耗：`212.7465mW`
  - 面积：`304976um^2`

- **S-pool**
  - 公式：`T = 56 + (N - 1)*47`
  - 其中通常 `N=B*D*Ns`（单样本时 `B=1`）
  - 功耗：`26.82mW`
  - 面积：`54926um^2`

- **Quant Engine**（坐标12bit / 激活8bit）
  - 输入展开元素数 `N`，周期：`T=N+6`
  - 几何坐标量化功耗：`2.28mW`
  - 激活特征量化功耗：`2.234mW`
  - 面积：`2460um^2`
  - 支持并行度 `P`

## 2.3 GRF-KAN

- **base 模式（数字 GRF）**
  - 每数据周期：`19`
  - 功耗：`23.7833mW`
  - 面积：`47877um^2`
  - 并行度 `P` 可绑定 `KAT_Group` 分组并行
- **acc 模式（RRAM-LUT）**
  - 当前仅保留占位，不实现行为，仅保留接口与报表字段

---

## 3. FPS/KNN 建模需求

## 3.1 距离计算

### baseline 模式（3x64 脉动阵列）

- 公式：`T = M + R + C - 2`
- 例：64点批次 -> `64+3+64-2=129` cycle
- 1024点 -> `1024+3+64-2=1089` cycle
- 功耗：`94mW`
- 面积：`89884um^2`

### acc 模式（SRAM_CIM_array）

- 先计算自平方项（仅启动时一次），后续主要计算交叉项
- 周期（自平方/交叉项同式）：
  - `T = (num_input_points / num_array_parallelism) * (num_xyz_quant_bit / num_bit_slice)`
  - 参数：`num_xyz_quant_bit=12`，`num_bit_slice=2`，`num_array_parallelism=128`
  - 含义：每次 128 距离值，位串行开销 `6` 周期
- 功耗（自平方/交叉项）：`6 + 1.3*128 mW`
- 面积：暂置 `0`

## 3.2 dist2_merge_adder（acc）

- SRAM-CIM 输出交叉项后，执行 1 周期完成“自平方+交叉项”合并
- 资源：128 个 adder（与 array 吞吐对齐）
- 单 adder 功耗：`1.3mW` -> 总 `1.3*128mW`
- 面积：`2179*128um^2`

## 3.3 baseline 排序（FPS/KNN 共享）

共享模块：`bitonic_sort_1024_max_32min`

- FPS（max 模式）：`10` cycle
- KNN（min32 模式）：
  - 512->top32：17 cycle
  - 256->top32：15 cycle
  - 128->top32：13 cycle
  - 64->top32：11 cycle
- 功耗：`2097.4mW`
- 面积：`2157561um^2`

## 3.4 acc 模式 coarse/fine + resampler

- coarse filter：`38mW`，`644686um^2`
- fine filter：`30mW`，`53261um^2`

- resampler：`T=N+2`
  - 功耗：`3.95mW`
  - 面积：`4719um^2`
  - 额外 LUT：
    - 下采样 LUT：`13485um^2`
    - 上采样 LUT：`85331um^2`
  - 仅第一层 FPS 启动一次

- 下采样 overlap 规则：
  - 可与 `SRAM-CIM distance` + `FPS` overlap
  - 条件：一旦拿到两最远 center 可开始插值

- 上采样规则：
  - 需等待 Efficient-KAN 输出 64 点后启动
  - 不能 overlap，采用并行 resampler 压缩：`T = N/P + 2`

---

## 4. acc 模式 FPS/KNN 时序细化

1. **第一层 FPS/KNN 距离复用**：共享 SRAM-CIM 输出距离（KNN 走片上 buffer 读写）。
2. **后续层**：FPS 可由第一层子集推导；KNN 仍需重算距离。
3. **FPS（acc）**：
   - 每 128 候选距离为一组
   - 距离计算 6cycle + merge 1cycle
   - 前10%使用 `fps_mode1` 完整比较
   - 后90%使用 `fps_mode0`（coarse阈值筛 + fine max）
   - coarse 每组 3cycle（可与6cycle距离计算 overlap）
   - fine max 2cycle（产生下一轮查询点）
4. **KNN（acc）**：
   - 粗筛粒度 512 距离
   - 先等待 24cycle 距离积累
   - 单轮粗筛开销 `4+1` cycle，按阈值自适应多轮
   - 粗筛后候选送入 fine filter（2cycle top-k）
5. **串并行约束**：
   - FPS 与下一组 128 距离计算可并行
   - FPS 与 KNN 因共享 coarse filter 串行

---

## 5. DRAM + SRAM 建模改造要求

1. 保持与 `hw_sim/cycle_simulator/memory_model/demo` 一致的事务粒度：
   - `read_req`, `write_req`, `bank_conflict`, `burst_len`, `queue_delay`
2. 在 `pointkan_sim` 新增 memory event 字段：
   - `mem_type` (`sram`/`dram`)
   - `bytes`
   - `bank_id`（可选）
   - `reuse_tag`（复用距离缓存）
3. KNN 第一层距离复用必须显式计入：
   - 写入 buffer 周期 + 功耗
   - 读取 buffer 周期 + 功耗
4. 增加可配置容量与带宽：
   - `distance_buffer_size`
   - `distance_buffer_bw_bytes_per_cycle`
   - `dram_bw_bytes_per_cycle`
5. 若容量溢出，触发 spill 到 DRAM 并计入额外时延/功耗。

---

## 6. 对 pointkan_sim 的最终系统级修改清单

1. **配置层 (`configs/hw_config.py`)**
   - 为所有模块补齐 `power/area/cycles/parallelism` 新参数。
   - 新增 acc 专属模块配置：`dist2_merge_adder`, `coarse_filter`, `fine_filter`, `resampler`, `resampler_lut`。
   - 量化拆分为两类配置：`xyz_quant` 与 `act_quant`。

2. **算子层 (`operators/common_ops.py`, `operators/fps.py`, `operators/knn.py`)**
   - 按本文公式重写周期模型。
   - 新增 baseline/acc 双模式分支，确保参数可切换。
   - 增加 overlap 事件标记与资源互斥标记。

3. **调度层 (`simulator/scheduler.py`)**
   - 由“仅按 cycle 串接”升级为“依赖 + 资源约束调度”。
   - 支持 overlap group 与 shared resource（bitonic/coarse_filter/resampler）。

4. **存储层 (`memory/transactions.py`)**
   - 引入距离缓存事务模型。
   - 加入首层 FPS/KNN 距离复用策略。
   - 输出可核验的 memory trace（按层/按算子）。

5. **报告层 (`report/reporter.py`, `analysis/*`)**
   - 新增每模块 PPA 占比表。
   - 新增 overlap 有效率（理论周期 vs 实际调度周期）。
   - baseline/acc 并列比较报告。

6. **校验与回归**
   - 增加最小网络回归：检查关键层 `FPS/KNN/BN/SPool/AffineNorm` 周期是否符合公式。
   - 增加 memory 回归：检查第一层距离复用确实减少 DRAM 访问。

---

## 7. 建议的分阶段实施

- **Phase A（参数冻结）**：先冻结所有公式与默认参数（本文）。
- **Phase B（算子级落地）**：先改 `common_ops/fps/knn`，不改调度器。
- **Phase C（调度器升级）**：加入 overlap + resource mutex。
- **Phase D（内存耦合）**：接入 DRAM/SRAM 事务和距离缓存复用。
- **Phase E（报表与回归）**：形成可对比的 baseline/acc 全链路报告。

---

<<<<<<< ours
## 8. 待你确认的关键开关（进入代码前）

1. BatchNorm 中 `N` 与 `S` 在当前网络映射中的精确定义（避免轴歧义）。
2. AffineNorm 固定 `P=32` 是否允许在 DSE 中放开。
3. acc KNN 粗筛“轮数”是否采用统计分布输入，或先用固定均值。
4. resampler 下/上采样 LUT 的功耗是否先按静态泄漏建模，还是先只记面积。
5. 第一层后 FPS 子集推导的精确规则（决定后续层周期缩减比）。

> 建议：你确认上述 5 个开关后，再执行代码改造，能显著降低返工风险。
=======
## 8. 已确认口径（用于落地实现）

1. **BatchNorm 维度约定**：激活可为 `(B,N,S)` 或 `(B,N)`；其中 `B=1`（单样本），`N` 为通道数，`S` 为特征维。2D 情况按 `S=1` 处理。
2. **AffineNorm 并行度**：周期公式中固定 `P=32`，不参与 DSE；引擎并行数默认 `1`，并且该模块给定面积/功耗不再额外乘 `32`。
3. **acc-KNN 粗筛轮数**：先采用固定均值建模（配置项可调）。
4. **Resampler LUT 功耗**：上下采样 LUT 总功耗按 `0.094mW` 建模（面积按既有上下采样 LUT 面积分别计入）。
5. **第一层后 FPS 子集规则**：若第一层采样点序列为 `512→256→128→64`，后续层直接取第一层 512 索引集合的前 256/128/64；并显式计入对应 DRAM/SRAM 访存开销。
>>>>>>> theirs
