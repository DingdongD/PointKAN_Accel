# Feature Compute RTL 工程说明

本目录为 PointKAN 硬件仿真中的**特征计算 RTL** 工程，实现可重构脉动阵列（Systolic Array）矩阵运算与 im2col 数据重排，用于加速 GEMM（矩阵乘）及卷积前处理。

---

## 1. 工程概述

- **功能**：支持多种模式的矩阵乘法（GEMM）与逐元素乘（Element-Wise），以及 1D/2D 卷积的 im2col 展开。
- **核心结构**：可配置的 PE（Processing Element）脉动阵列，通过 `mode` 选择 **Output Stationary (OS)**、**Weight Stationary Preload (WS_PRELOAD)**、**Weight Stationary Diagonal (WS_DIAG)** 或 **Element-Wise (EW)**。
- **数据位宽**：激活/权重量化字长 `WORD_SIZE=16`，累加器 `ACC_WIDTH=64`，有符号运算。

---

## 2. 目录与文件结构

```
feature_compute_rtl/
├── readme.md                    # 本说明文档
├── generate_os_mem.py           # 生成 OS 模式测试用 A/B/C 内存文件
├── A.mem, B.mem, C_full.mem     # 预生成的 8x8 测试数据（十六进制）
│
├── 顶层与阵列（脉动 GEMM）
├── top_reconfig_systolic_v2.v   # 顶层 v2：OS + WS_PRELOAD + WS_DIAG，含完整控制器
├── top_reconfig_systolic_v3.v   # 顶层 v3：轻量级，当前仅支持 MODE_EW 示意
├── PE_array_reconfig_v2.v       # 4x4 PE 阵列（v2），配合 controller_reconfig_v2
├── PE_array_reconfig_v3.v       # 4x4 PE 阵列（v3/v4），含 a_local 用于 EW
├── PE_reconfig_v2.v            # 单 PE（v2）：OS/WS_PRELOAD/WS_DIAG
├── PE_reconfig_v3.v            # 单 PE（v4）：在 v2 基础上增加 MODE_EW 与 a_local
│
├── 控制器（仅 v2 拓扑使用）
├── controller_reconfig_v2.v     # 多模式控制器，按 mode 选子控制器
├── controller_os_v2.v           # OS 模式控制器
├── controller_ws_preload_v2.v   # WS 预加载模式控制器
├── controller_ws_diag_v2.v      # WS 对角线注入模式控制器
│
├── im2col（卷积展开）
├── unified_im2col.v             # 统一 1D/2D im2col 模块（模块名为 im2col）
│
└── 测试平台
    ├── tb_reconfig_v2_os.v      # 顶层 v2 + OS 模式仿真
    ├── tb_reconfig_ws_preload.v # WS 预加载模式仿真
    ├── tb_reconfig_v2_ws_diag.v # WS 对角线模式仿真
    ├── tb_reconfig_ew_v3.v      # PE 阵列 v3/v4 的 EW 模式仿真（注：文件名 ew_v3，内部实例为 v4）
    ├── tb_array_os.v            # 仅 PE 阵列 OS 模式
    ├── tb_im2col_1d.v           # 1D im2col 测试
    └── tb_im2col_2d.v           # 2D im2col 测试
```

---

## 3. 架构说明

### 3.1 脉动阵列与数据流

- **阵列规模**：默认 `ARR_R x ARR_C`（如 4x4 或 8x8），可参数化。
- **水平方向**：激活 `a` 从左边界进入，每拍向右传递一列。
- **垂直方向**：
  - **MODE_OS**：从上向下流式输入**权重**，每个 PE 做 `a * weight` 并累加部分和。
  - **MODE_WS_***：垂直通道传递**部分和（psum）**；权重通过 `w_local` 预加载或对角线注入。
- **输出**：
  - OS：可从 `psum_snapshot_bus` 直接读出各 PE 的累加结果（需阵列规模 ≥ M×N 时一次出整矩阵）。
  - WS 模式：由控制器从 `vert_bottom_bus` 收集并重组为 `C_out_flat`。

### 3.2 工作模式（mode）

| mode 值 | 名称           | 说明 |
|--------|----------------|------|
| `2'd0` | MODE_OS        | Output Stationary：权重重用、沿列向下流动，适合权重复用高的场景。 |
| `2'd1` | MODE_WS_PRELOAD| Weight Stationary 预加载：先向各 PE 预加载权重，再流式输入激活与 psum。 |
| `2'd2` | MODE_WS_DIAG   | Weight Stationary 对角线：每周期按对角线向 PE 注入权重。 |
| `2'd3` | MODE_EW        | Element-Wise：每个 PE 用 `a_local` × `w_local` 得到逐元素乘，无累加。 |

- **v2 拓扑**：`top_reconfig_systolic_v2` + `controller_reconfig_v2` 完整支持 OS / WS_PRELOAD / WS_DIAG；输出在 OS 时部分来自 `psum_snapshot`，其余来自控制器。
- **v3 拓扑**：`top_reconfig_systolic_v3` 为轻量级顶层，当前仅对 MODE_EW 做使能/清零，无完整控制器与 C 输出收集，多用于 EW 行为或扩展开发。

### 3.3 数据布局（扁平化）

- **A_flat**：矩阵 A(M×K)，行优先，`A_flat[(i*K+j)*WORD_SIZE +: WORD_SIZE]` = A[i][j]。
- **B_flat**：矩阵 B(K×N)，行优先，`B_flat[(i*N+j)*WORD_SIZE +: WORD_SIZE]` = B[i][j]。
- **C_out_flat**：矩阵 C(M×N)，行优先，64 位有符号，`C_out_flat[(i*N+j)*ACC_WIDTH +: ACC_WIDTH]` = C[i][j]。

### 3.4 im2col 模块（unified_im2col.v）

- **用途**：将 1D/2D 卷积输入展开为列矩阵，便于后续用 GEMM 核做卷积运算。
- **参数**：
  - `DIMENSION`：1 或 2（1D 卷积要求 `INPUT_H=1`, `KERNEL_H=1`）。
  - `INPUT_H/W/C`、`KERNEL_H/W`、`STRIDE`、`PADDING`。
  - 输出列矩阵尺寸由 `OUTPUT_H/W`、`COL_HEIGHT`、`COL_WIDTH` 推导。
- **接口**：IFM 读地址/使能/数据，列缓冲写地址/数据/使能；`start`/`done`/`busy` 控制一次完整 im2col。
- **数据布局**：IFM 线性地址 `(h*W+w)*C+c`，即 [H][W][C]。

---

## 4. 主要模块与参数

### 4.1 顶层 top_reconfig_systolic_v2

| 参数        | 默认 | 说明 |
|-------------|------|------|
| M, K, N     | 8,8,8 | 矩阵维度 A(M×K) * B(K×N) = C(M×N) |
| ARR_R, ARR_C| 4,4   | 脉动阵列行/列 PE 数 |
| WORD_SIZE   | 16    | 激活/权重大小（bit） |
| ACC_WIDTH   | 64    | 累加/输出位宽（bit） |

**接口要点**：

- `clk`, `rst_n`, `start`, `mode[1:0]`
- `A_flat`, `B_flat`：扁平化输入
- `C_out_flat`：扁平化输出（64 位有符号）
- `done`：计算完成标志

### 4.2 PE_reconfig_v2 / PE_reconfig_v3

- 单 PE：水平传激活，垂直传权重（OS）或 psum（WS），本地权重 `w_local`/`w_local_valid`。
- v3 文件内模块名为 **PE_reconfig_v4**，增加 `a_local`/`a_local_valid` 与 **MODE_EW**（逐元素乘）。

### 4.3 测试数据生成（generate_os_mem.py）

用于生成 OS 模式仿真所需的 `A.mem`、`B.mem`、`C_full.mem`（64 位 golden）。

```bash
cd /workspace/PointKAN-pytorch/hw_sim/rtl/feature_compute_rtl
python generate_os_mem.py
```

脚本内可修改：

- `row, col, k`：矩阵维度（默认 8×8×8）
- `word_size`：16
- `signed`：True
- `seed`：随机种子
- `out_dir`：输出目录（当前脚本内为 Windows 路径，可改为 `"."` 输出到当前目录）

修改示例（输出到当前目录）：

```python
if __name__ == "__main__":
    generate_os_mem(
        row=8, col=8, k=8,
        word_size=16,
        signed=True,
        seed=42,
        out_dir="."   # 改为当前目录
    )
```

---

## 5. 仿真与测试

### 5.1 环境要求

- Verilog 仿真器：**iverilog**（Icarus Verilog）或 **VCS/ModelSim** 等。
- 运行 OS 模式测试前，需在同一目录下存在 `A.mem`、`B.mem`、`C_full.mem`（可由 `generate_os_mem.py` 生成）。

### 5.2 OS 模式（top v2）

```bash
cd /workspace/PointKAN-pytorch/hw_sim/rtl/feature_compute_rtl
# 若没有 .mem 文件，先生成
python generate_os_mem.py   # 注意先把 out_dir 改为 "."

# 使用 iverilog 编译与仿真
iverilog -o tb_v2_os.vvp tb_reconfig_v2_os.v
vvp tb_v2_os.vvp
```

- 测试台会拉高 `start`、`mode=MODE_OS`，等待 `done` 后比对 `C_out_flat` 与 `C_full.mem` 中的 golden。
- 通过时打印 `*** TOP V2 OS TEST PASSED ***`，否则报告错误个数。
- 波形可查看生成的 `tb_reconfig_v2_os.vcd`。

### 5.3 WS 预加载 / WS 对角线

- `tb_reconfig_ws_preload.v`、`tb_reconfig_v2_ws_diag.v` 分别对应两种 WS 模式，需在仿真时提供对应的 A/B 及 golden（若有单独 .mem 或 testbench 内嵌数据）。
- 编译方式类似，顶层为 `top_reconfig_systolic_v2`，注意 `include` 链会拉入对应 controller 与 PE 阵列。

### 5.4 Element-Wise（v3/v4 阵列）

- `tb_reconfig_ew_v3.v` 直接例化 `PE_array_reconfig_v4`（来自 `PE_array_reconfig_v3.v`），不经过 top_reconfig_systolic_v3。
- 需要 `A_ew.mem`、`B_ew.mem`、`C_ew.mem`（格式与 testbench 中 `$readmemh` 约定一致）。
- 运行方式（示例）：

```bash
iverilog -o tb_ew.vvp tb_reconfig_ew_v3.v
vvp tb_ew.vvp
```

### 5.5 im2col 测试

- **1D**：`tb_im2col_1d.v`，例化 `unified_im2col.v`，`DIMENSION=1`，`INPUT_H=1`，`KERNEL_H=1`。
- **2D**：`tb_im2col_2d.v`，`DIMENSION=2`。
- 编译与运行示例：

```bash
iverilog -o tb_im2col_1d.vvp tb_im2col_1d.v
vvp tb_im2col_1d.vvp
```

---

## 6. 使用流程小结

1. **确定模式**：OS / WS_PRELOAD / WS_DIAG 用 **top_reconfig_systolic_v2**；仅做 EW 或扩展可参考 **top_reconfig_systolic_v3** 与 PE_array_reconfig_v3。
2. **准备数据**：OS 用 `generate_os_mem.py` 生成 A/B/C；其它 testbench 按需准备对应 .mem 文件。
3. **参数匹配**：顶层与 testbench 的 M、K、N、ARR_R、ARR_C 需一致；OS 下若用 8×8 全矩阵输出，需 `ARR_R=ARR_C=8`（见 tb_reconfig_v2_os 注释）。
4. **仿真**：从对应 tb_*.v 编译并运行，根据 `done` 与打印结果判断通过与否，必要时查看 .vcd 波形。
5. **im2col**：按 1D/2D 选 testbench，确认 INPUT/KERNEL/STRIDE/PADDING 与 DIMENSION 符合 `unified_im2col.v` 约束（如 DIMENSION=1 时 INPUT_H=1、KERNEL_H=1）。

---

## 7. 注意事项

- **命名不一致**：`PE_reconfig_v3.v` 中实际模块名为 `PE_reconfig_v4`；`PE_array_reconfig_v3.v` 中阵列模块名为 `PE_array_reconfig_v4`，例化时以模块名为准。
- **v3 顶层**：`top_reconfig_systolic_v3` 未实现完整控制器与 C 输出收集，仅对 MODE_EW 做使能/清零，不能直接替代 v2 做 OS/WS 验证。
- **OS 阵列规模**：当 M、N 大于 ARR_R、ARR_C 时，需分块或多次运行，输出选择逻辑在 top_reconfig_systolic_v2 的 generate 块中已按 (r,c) 与 ARR_R、ARR_C 做了区分。
- **符号与位宽**：所有 PE 与控制器按有符号 16 位输入、64 位累加设计；im2col 的 `DATA_WIDTH` 可配置，与后续 GEMM 接口时需注意位宽一致。

---

## 8. 与 PointKAN 的关系

本 RTL 作为 **特征计算** 的硬件实现，可与 PointKAN-pytorch 中的模型结构对应：  
- GEMM 用于全连接或 1×1 卷积等线性变换；  
- im2col + GEMM 用于常规 1D/2D 卷积；  
- EW 可用于激活或逐元素操作。  

具体与 PyTorch 层的映射与量化约定需结合上层 hw_sim 与算法脚本使用。
