# FP32 线性插值重采样 — 用法与变量说明

将 **N_IN** 个 FP32 采样点通过一维线性插值重采样为 **N_OUT** 个点，同时支持下采样（N_IN > N_OUT）和上采样（N_IN < N_OUT）。Python 提供参考模型与 hex 生成，Verilog 为可综合 RTL，二者算法一致，便于比对。

---

## 一、文件一览

| 文件 | 说明 |
|------|------|
| **ref_resampler.py** | 参考模型：输入→线性插值→输出，可打印/写出 FP32 hex，并生成 t LUT、golden、in.hex |
| **gen_t_lut.py** | 仅生成 t 权重 LUT 的 hex 文件（与 ref_resampler 的 t 公式一致） |
| **fp32_resampler.v** | RTL 顶层：含 fp32_lerp、mem、t_lut、定点步长与 FSM；依赖 fp32_adder、fp32_mul |
| **tb_fp32_resampler.v** | 测试台：单次 N_IN/N_OUT 或 RUN_ALL 三组用例，用 `$readmemh` 载入 in/t_lut/gold |
| **run_51_32_64_461.bat** | 批处理：先 51→32、再 64→461，每次用 Python 生成 hex 后 iverilog+vvp |

---

## 二、插值公式与变量对应

与 PyTorch/常见实现一致：

- 输出索引 `j = 0..N_OUT-1` 对应源位置：  
  **src_pos[j] = j × (N_IN−1) / (N_OUT−1)**（在 [0, N_IN−1] 上均匀分布）
- **left = floor(src_pos)**，**right = min(left+1, N_IN−1)**
- 小数权重 **w = src_pos − left**，即 **t_lut[j]**（范围 [0, 1)）
- 插值结果：**y[j] = (1−w)×x[left] + w×x[right] = x[left] + (x[right]−x[left])×w**

RTL 中：

- **pos**：定点累加器，整数部分为 **left** 索引，小数部分未用于 t（t 来自 LUT）
- **step**：每输出一个点 pos 的递增量，**step = floor((N_IN−1)×2^FRAC_W / (N_OUT−1))**
- **t_lut[out_cnt]**：第 `out_cnt` 个输出对应的权重 w（FP32），由 Python 预生成 hex 写入

---

## 三、Python 用法

### 3.1 ref_resampler.py（参考 + 生成 hex）

**基本：按 N_IN / N_OUT 生成默认输入 1.0, 2.0, …, L，并写出 golden 与 t LUT**

```bash
python ref_resampler.py <N_IN> <N_OUT> -o gold.hex --t-lut-out t_lut.hex
```

示例：

```bash
# 5→3，生成 gold.hex、t_lut.hex（默认输入 1,2,3,4,5）
python ref_resampler.py 5 3 -o gold_5_3.hex --t-lut-out t_lut_5_3.hex

# 同时把当前输入写出为 in.hex，供 TB 的 $readmemh("in.hex", mem)
python ref_resampler.py 51 32 -o gold.hex --t-lut-out t_lut.hex --in-out in.hex --quiet
```

**常用参数**

| 参数 | 说明 |
|------|------|
| `L` / 第一参数 | 输入长度 N_IN |
| `out_len` / 第二参数 | 输出长度 N_OUT |
| `-o`, `--output` | Golden 输出 hex 路径（每行 8 位 hex） |
| `--t-lut-out FILE` | t 权重 LUT 的 hex 路径（可被 `$readmemh` 读入） |
| `--t-lut-pad N` | t LUT 填到 N 行（0=不填，默认行数=out_len） |
| `--t-lut-only` | 只生成 t LUT，不跑插值（需同时给 out_len 与 -o/--t-lut-out） |
| `--in-float "1.0 2.0 ..."` | 自定义输入（空格分隔浮点） |
| `--in-hex FILE` | 从 hex 文件读入输入（每行 8 位 hex） |
| `--in-out FILE` | 把当前输入向量写出为 hex（供 TB 的 in.hex） |
| `--quiet` | 不打印逐项 hex，只写文件与摘要 |
| `--gold-pad N` | Golden 文件填充到 N 行（0 不填充） |

**仅生成 t LUT**

```bash
python ref_resampler.py 5 3 --t-lut-only -o t_lut_5_3.hex
```

### 3.2 gen_t_lut.py（只生成 t LUT）

与 `ref_resampler.py` 的 t 公式一致，仅输出 t 权重的 hex 文件。

```bash
python gen_t_lut.py <N_IN> <N_OUT> -o t_lut.hex
python gen_t_lut.py 5 3 -o t_lut_5_3.hex
python gen_t_lut.py 32 460 -o t_lut.hex --pad-to 512   # RTL 深度 512 时填满消告警
```

| 参数 | 说明 |
|------|------|
| `L` | 输入长度 N_IN |
| `out_len` | 输出长度 N_OUT |
| `-o`, `--output` | 输出 hex 路径（必填） |
| `--pad-to N` | 填到 N 行（0=不填，行数=out_len） |

---

## 四、Verilog 用法

### 4.1 编译与运行（单次 N_IN / N_OUT）

TB 通过 `tb_fp32_resampler.v` 的 `include "fp32_resampler.v"` 拉入 RTL，编译时只需指定 TB 文件（依赖由 include 带入）：

```bash
# 1) 生成 hex（固定名 in.hex, t_lut.hex, gold.hex）
python ref_resampler.py 51 32 -o gold.hex --t-lut-out t_lut.hex --in-out in.hex --quiet

# 2) 编译：传入 N_IN、N_OUT，与 hex 行数一致
iverilog -D N_IN=51 -D N_OUT=32 -o sim.vvp -s tb_fp32_resampler tb_fp32_resampler.v

# 3) 运行
vvp sim.vvp
```

单次验证时 TB 会：

- `$readmemh("in.hex", u_dut.mem)`
- `$readmemh("t_lut.hex", u_dut.t_lut)`
- `$readmemh("gold.hex", gold_ref)`  
然后拉高 `start`，按 `out_valid` 收 `out_data`，与 `gold_ref` 逐项比较。

**允许 1 ULP 误差（长链 64→461 等建议加上）：**

```bash
iverilog -D N_IN=64 -D N_OUT=461 -D GOLDEN_ULP_TOL -o sim.vvp -s tb_fp32_resampler tb_fp32_resampler.v
vvp sim.vvp
```

### 4.2 RUN_ALL（一次跑三组用例）

DUT 深度固定为 5/5，需要事先准备好对应 hex 文件（见 TB 注释或下节）。

```bash
iverilog -D RUN_ALL -D N_IN_MAX_TB=5 -D N_OUT_MAX_TB=5 -o sim_runall.vvp -s tb_fp32_resampler tb_fp32_resampler.v
vvp sim_runall.vvp
```

三组用例：5→3、3→5、2→4；使用 `in_5.hex`、`in_3.hex`、`in_2.hex`，以及 `t_lut_5_3.hex`、`t_lut_3_5.hex`、`t_lut_2_4.hex`，`gold_5_3.hex`、`gold_3_5.hex`、`gold_2_4.hex`。

### 4.3 批处理脚本

```bash
run_51_32_64_461.bat
```

依次执行：51→32（生成 in/t_lut/gold.hex 并仿真）、64→461（同上），每次覆盖 `in.hex`、`t_lut.hex`、`gold.hex`；两次都带 `GOLDEN_ULP_TOL`。

---

## 五、RTL 与 TB 变量说明

### 5.1 fp32_resampler 模块参数

| 参数 | 含义 |
|------|------|
| **N_IN_MAX** | 输入缓冲区最大点数（mem 深度） |
| **N_OUT_MAX** | 输出/t_lut 最大点数（t_lut 深度） |
| **ADDR_W** | 地址位宽（mem、t_lut 地址） |
| **FRAC_W** | 定点步长小数位宽（step 为 (N_IN−1)×2^FRAC_W/(N_OUT−1) 的整数部分） |

### 5.2 配置与写端口（start 前锁存/写入）

| 信号 | 方向 | 说明 |
|------|------|------|
| **cfg_n_in** | 输入 | 本次输入样本数 N_IN（≥2） |
| **cfg_n_out** | 输入 | 本次输出样本数 N_OUT（≥1） |
| **cfg_step** | 输入 | 定点步长 step = floor((N_IN−1)×2^FRAC_W/(N_OUT−1))，用于 pos 累加得到 left 索引 |
| **wr_en** | 输入 | 写使能：为 1 时把 wr_data 写入 mem[wr_addr] |
| **wr_addr** | 输入 | 写 mem 的地址 |
| **wr_data** | 输入 | 写入 mem 的 FP32 数据 |
| **wr_t_en** | 输入 | 写使能：为 1 时把 wr_t_data 写入 t_lut[wr_t_addr] |
| **wr_t_addr** | 输入 | 写 t_lut 的地址 |
| **wr_t_data** | 输入 | 写入 t_lut 的 FP32 权重 w |
| **start** | 输入 | 单周期脉冲：锁存 cfg_*，开始一次重采样 |

上板/集成时用 **wr_en/wr_addr/wr_data** 灌入输入数据，用 **wr_t_en/wr_t_addr/wr_t_data** 灌入 t LUT（或由 Python 生成 hex 后通过总线写入）。**当前 TB 用 `$readmemh` 载入 mem/t_lut，写端口在 TB 中接 0，不在波形中体现。**

### 5.3 输出与状态

| 信号 | 方向 | 说明 |
|------|------|------|
| **out_data** | 输出 | 当前输出点 FP32 值 |
| **out_valid** | 输出 | 为 1 时 out_data 有效，应在此拍采样 |
| **busy** | 输出 | 正在运算 |
| **done** | 输出 | 单周期脉冲，表示本次 N_OUT 个点全部输出完毕 |

### 5.4 内部关键变量（便于查波形/调试）

| 变量 | 说明 |
|------|------|
| **mem[]** | 输入样本寄存器堆，mem[left]、mem[right] 参与插值 |
| **t_lut[]** | t 权重 LUT，t_lut[out_cnt] 为当前输出的 w（FP32） |
| **pos** | 定点位置累加器；pos += step 每输出一点；整数部分 = left 索引 |
| **idx** | pos 的整数部分，即当前 left |
| **idx_eff / idx1_eff** | 对最后一拍做右端点约束，与 Python 的 right = min(left+1, L−1) 及最后一格对齐 |
| **out_cnt** | 当前输出序号 0..N_OUT−1，用于查 t_lut[out_cnt] |
| **n_in_r, n_out_r, step_r** | start 时锁存的 cfg_n_in、cfg_n_out、cfg_step |

### 5.5 TB 宏与 hex 文件（单次）

| 宏/文件 | 说明 |
|--------|------|
| **N_IN** | 编译时 -D N_IN=51 等，单次测试的输入点数 |
| **N_OUT** | 编译时 -D N_OUT=32 等，单次测试的输出点数 |
| **in.hex** | N_IN 行，每行 8 位 hex，对应输入序列 |
| **t_lut.hex** | N_OUT 行，每行 8 位 hex，对应 t 权重 |
| **gold.hex** | N_OUT 行，参考输出，与 out_data 逐项比对 |
| **GOLDEN_ULP_TOL** | 可选 -D，允许与 golden 差 1 ULP（长链 64→461 等建议开） |

---

## 六、RUN_ALL 所需 hex 文件

| 用例 | 输入 hex | t LUT hex | Golden hex |
|------|----------|-----------|------------|
| 5→3 | in_5.hex | t_lut_5_3.hex | gold_5_3.hex |
| 3→5 | in_3.hex | t_lut_3_5.hex | gold_3_5.hex |
| 2→4 | in_2.hex | t_lut_2_4.hex | gold_2_4.hex |

可用 ref_resampler 生成，例如：

```bash
python ref_resampler.py 5 3 -o gold_5_3.hex --t-lut-out t_lut_5_3.hex --in-out in_5.hex
python ref_resampler.py 3 5 -o gold_3_5.hex --t-lut-out t_lut_3_5.hex --in-out in_3.hex
python ref_resampler.py 2 4 -o gold_2_4.hex --t-lut-out t_lut_2_4.hex --in-out in_2.hex
```

（in_3.hex、in_2.hex 若行数不足 5，可手工补 0 或按 TB 的 mem 深度补足，避免 $readmemh 告警。）


## 七、注意事项

- **64→461** 等长链在边界上可能出现个别点与 Python 差约 1.0（非 1 ULP），因 RTL 用整数 step 累加与 Python 的浮点 src_pos 在边界索引上可能差 1；可开 **GOLDEN_ULP_TOL** 接受 1 ULP，更大差异需查 pos/idx。
- **写端口**：RTL 保留 wr_* / wr_t_* 供上板或集成使用；本 TB 一律用 `$readmemh` 加载，写端口接 0。
- 波形文件：TB 中为 **fp32_resampler.vcd**，可用 GTKWave 等查看。

## 八、快速运行
```bash 
# 一键复制
python ref_resampler.py 51 32 -o gold.hex --t-lut-out t_lut.hex --in-out in.hex --quiet
iverilog -D N_IN=51 -D N_OUT=32 -D GOLDEN_ULP_TOL -o sim.vvp -s tb_fp32_resampler tb_fp32_resampler.v
vvp sim.vvp

python ref_resampler.py 64 461 -o gold.hex --t-lut-out t_lut.hex --in-out in.hex --quiet
iverilog -D N_IN=64 -D N_OUT=461 -D GOLDEN_ULP_TOL -o sim.vvp -s tb_fp32_resampler tb_fp32_resampler.v
vvp sim.vvp

