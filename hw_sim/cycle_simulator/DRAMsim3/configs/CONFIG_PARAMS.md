# DRAMsim3 配置文件参数说明与 INI 差异

以 `DDR4_8Gb_x4_3200.ini` 为例说明各段含义，并对比同目录下其他 INI 的差异。

---

## 一、DDR4_8Gb_x4_3200.ini 参数含义

### [dram_structure] — DRAM 结构

| 参数 | 值 | 含义 |
|------|-----|------|
| **protocol** | DDR4 | 协议类型：DDR3/DDR4/LPDDR3/LPDDR4/GDDR5/GDDR6/HBM/HBM2/HMC |
| **bankgroups** | 4 | Bank 组数（DDR4 引入；DDR3 无 bank group，此处等效为 1） |
| **banks_per_group** | 4 | 每组内 bank 数；总 bank 数 = 4×4 = 16 |
| **rows** | 131072 | 每 bank 行数（2^17） |
| **columns** | 1024 | 每行列数（每列 = device_width 位） |
| **device_width** | 4 | 单颗芯片数据位宽（bit），x4 表示 4 位宽 |
| **BL** | 8 | Burst Length：一次列访问传输的 beat 数；DDR 每次传 BL/2 个周期，即 8/2=4 个时钟周期 |

**容量**：单 die 8Gb，组织为 4 个 bank group × 4 bank × 131072 row × 1024 col × 4 bit = 8Gbit。  
**文件名含义**：DDR4、单颗 8Gb、数据位宽 x4、等效数据率 3200 MT/s。

---

### [timing] — 时序（单位：除 tCK 为 ns 外，其余多为时钟周期数）

| 参数 | 值 | 含义 |
|------|-----|------|
| **tCK** | 0.63 | 时钟周期 (ns)。3200 MT/s 对应 1.6 GHz 时钟，tCK = 1/1.6 ≈ 0.625 ns |
| **AL** | 0 | Additive Latency（附加延迟，周期） |
| **CL** | 22 | CAS Latency：读命令到首数据延迟（周期） |
| **CWL** | 16 | CAS Write Latency：写命令到首数据延迟（周期） |
| **tRCD** | 22 | ACT 到读/写允许的延迟（周期），行选通到列选通 |
| **tRP** | 22 | PRE 到下一次 ACT 的延迟（周期），预充电时间 |
| **tRAS** | 52 | ACT 到 PRE 的最短间隔（周期），行活跃时间 |
| **tRFC** | 560 | 刷新周期（单次 refresh 时长，周期）；tRFC2/tRFC4 为 2/4 bank 刷新变体 |
| **tREFI** | 12480 | 两次刷新命令之间的最大间隔（周期） |
| **tRPRE/tWPRE** | 1 | 读/写前导（preamble）周期数 |
| **tRRD_S / tRRD_L** | 4 / 8 | 同 bank group / 不同 bank group 的 ACT→ACT 间隔 |
| **tWTR_S / tWTR_L** | 4 / 12 | 写后读：同/不同 bank group |
| **tFAW** | 16 | 四激活窗口：在 tFAW 内最多 4 次 ACT |
| **tWR** | 24 | 写恢复时间（写结束到 PRE）；tWR2 为相关变体 |
| **tRTP** | 12 | 读后预充电延迟（读结束到 PRE） |
| **tCCD_S / tCCD_L** | 4 / 8 | 列到列延迟：同/不同 bank group |
| **tCKE / tCKESR** | 8 / 9 | 时钟使能与自刷新退出相关 |
| **tXS / tXP** | 576 / 10 | 自刷新/掉电退出时间 |
| **tRTRS** | 1 | Rank 切换时间（周期） |

---

### [power] — 功耗（JEDEC 风格电流，单位 mA；VDD 为 V）

| 参数 | 值 | 含义 |
|------|-----|------|
| **VDD** | 1.2 | 工作电压 (V)，DDR4 典型 1.2 V |
| **IDD0** | 52 | 预充电空闲电流（所有 bank 关闭） |
| **IPP0** | 3.0 | VDD 端输入漏电流等 |
| **IDD2P / IDD2N** | 25 / 37 | 空闲（无操作，CKE 高/低） |
| **IDD3P / IDD3N** | 38 / 47 | 活跃空闲（有 bank 打开） |
| **IDD4W / IDD4R** | 130 / 143 | 写/读操作电流 |
| **IDD5AB** | 250 | 全 bank 刷新电流 |
| **IDD6x** | 30 | 自刷新电流 |

仿真用这些电流与 VDD、tCK 等换算成能耗（如 mW、nJ/access）。

---

### [system] — 系统与控制器

| 参数 | 值 | 含义 |
|------|-----|------|
| **channel_size** | 32768 | 单 channel 容量 (MB) |
| **channels** | 1 | 通道数 |
| **bus_width** | 64 | 内存总线位宽 (bit)，与 device_width、devices_per_rank 一致 |
| **address_mapping** | rochrababgco | 地址映射顺序：r=row, o=other/offset, c=channel, h=high bits, ra=rank, ba=bank, bg=bankgroup, co=column。顺序决定 row/column/bank 在地址中的位置 |
| **queue_structure** | PER_BANK | 命令队列按 bank 组织 |
| **refresh_policy** | RANK_LEVEL_STAGGERED | 按 rank 错峰刷新 |
| **row_buf_policy** | OPEN_PAGE | 行缓冲策略：保持页打开以利用局部性 |
| **cmd_queue_size** | 8 | 每 bank 命令队列深度 |
| **trans_queue_size** | 32 | 事务队列深度 |

---

### [other]

| 参数 | 值 | 含义 |
|------|-----|------|
| **epoch_period** | 1587301 | 统计/分段周期（周期数），用于周期级统计 |
| **output_level** | 1 | 输出详细程度 |

---

## 二、与其他 INI 的差异概览

### 1. 同协议、不同数据率：DDR4 2133 vs 3200（x4）

| 项目 | DDR4_8Gb_x4_2133 | DDR4_8Gb_x4_3200 |
|------|-------------------|-------------------|
| **tCK** | 0.94 ns (≈1066 MHz) | 0.63 ns (≈1588 MHz) |
| **CL / CWL** | 16 / 11 | 22 / 16 |
| **tRCD / tRP / tRAS** | 16 / 16 / 36 | 22 / 22 / 52 |
| **tRFC / tREFI** | 374 / 8328 | 560 / 12480 |
| **tWR / tRTP** | 16 / 8 | 24 / 12 |
| **IDD0 / IDD4R** | 40 / 100 | 52 / 143 |
| **epoch_period** | 1063829 | 1587301 |

结论：3200 时序更紧、周期更短、延迟（周期数）更大、电流略高；2133 更慢但单周期更长。

---

### 2. 同协议同数据率、不同位宽：x4 vs x8（DDR4 8Gb 3200）

| 项目 | DDR4_8Gb_x4_3200 | DDR4_8Gb_x8_3200 |
|------|-------------------|-------------------|
| **rows** | 131072 | 65536（x8 行数通常减半） |
| **device_width** | 4 | 8 |
| **channel_size** | 32768 MB | 16384 MB（同 bus_width=64 时，x8 每 rank 容量不同） |
| **tFAW** | 16 | 34（x8 常见更大） |
| **IDD0 / IDD4W / IDD4R** | 52 / 130 / 143 | 57 / 150 / 168 |

结论：x8 单颗位宽大，行数/容量组织不同，电流略高；系统 channel_size 随 devices_per_rank 与 page 计算变化。

---

### 3. 不同协议：DDR3 vs DDR4（均为 8Gb x8 档位）

| 项目 | DDR3_8Gb_x8_1866 | DDR4_8Gb_x4_3200 |
|------|-------------------|-------------------|
| **protocol** | DDR3 | DDR4 |
| **bankgroups** | 1 | 4（DDR3 无 bank group） |
| **banks_per_group** | 8 | 4 → 总 bank 8 vs 16 |
| **columns** | 2048 | 1024 |
| **VDD** | 1.35 V | 1.2 V |
| **tCK** | 1.07 ns | 0.63 ns |
| **REFI** | REFI (7290) | tREFI (12480)；DDR3 无 tRFC2/tRFC4 |
| **tRPRE/tWPRE** | 0 | 1（DDR4 有 preamble） |
| **IDD0** | 69 | 52 |

结论：DDR3 无 bank group、电压更高、时序命名/结构不同；DDR4 更密、更快、电压更低。

---

### 4. 移动/高带宽协议：LPDDR4、HBM2

**LPDDR4_8Gb_x16_2400**

- **protocol**: LPDDR4；**bankgroups** = 2，**banks_per_group** = 4；**BL** = 16；**device_width** = 16。
- **channel_size** 常更小（如 8192 MB），**tCK** = 0.83 ns。
- 有 **tPPD** 等 LPDDR4 特有时序；功耗电流档位与 DDR4 不同。

**HBM2_8Gb_x128**

- **protocol** = HBM（代码里 HBM2）；**device_width** = 128，**BL** = 4；**num_dies** = 4。
- **rows** = 32768，**columns** = 64（HBM 列定义不同，代码中会 columns*=2）。
- **channels** = 8，**bus_width** = 128；**channel_size** = 1024 MB（每 channel 小、多 channel）。
- 时序命名不同：**tRCDRD / tRCDWR** 代替 tRCD；**tREFIb**、**tRTP_L/tRTP_S**、**tCKSRE** 等；无 tRFC2/tRFC4。
- **address_mapping** = rorabgbachco（HBM 典型映射）；可选 **unified_queue**。

---

## 三、总结表（按协议区分）

| 协议 | bankgroups | 典型 BL | VDD | 特点 |
|------|------------|--------|-----|------|
| DDR3 | 1 | 8 | 1.35 V 或 1.5 V | 无 bank group，REFI 命名 |
| DDR4 | 4 | 8 | 1.2 V | bank group，tRFC2/4，preamble |
| LPDDR4 | 2 | 16 | 1.2 V | 小 channel、tPPD 等 |
| HBM2 | 4 | 4 | 1.2 V | 宽位宽、多 channel、tRCDRD/WR、tREFIb |

同一协议下，不同 INI 主要差在：**数据率（tCK、CL 等）**、**位宽与容量（device_width、rows、channel_size）**、**功耗电流值**；系统段（address_mapping、queue、refresh_policy、row_buf_policy）多数配置共用或仅少数变体。

---

## 四、带宽 / 能耗 / 延迟对比（DRAMsim3 configs 同口径）

基于当前 INI 的 **单通道、单 rank** 配置，用统一公式估算（仅作相对比较）。

**峰值带宽** ≈ (bus_width/8) × (2/tCK) GB/s（DDR 双沿，tCK 单位 ns）。  
**读延迟（row hit）** ≈ CL × tCK (ns)。  
**读延迟（row miss）** ≈ (tRCD + CL) × tCK (ns)。  
**能耗相对** 用 VDD×IDD4R 作读写活跃电流、VDD×IDD0 作空闲粗略比较（未乘时间）。

| 配置 | 峰值带宽 (GB/s) | 读延迟 row hit (ns) | 读延迟 row miss (ns) | VDD | IDD0 | IDD4R | 相对能耗 |
|------|-----------------|----------------------|-----------------------|-----|------|-------|----------|
| DDR4_8Gb_x4_3200 | ~25.4 | ~13.9 | ~27.7 | 1.2 | 52 | 143 | 中 |
| DDR4_8Gb_x4_2133 | ~17.0 | ~15.0 | ~30.1 | 1.2 | 40 | 100 | **低** |
| LPDDR4_8Gb_x16_2400 | ~19.3 | ~14.1 | ~26.6 | 1.2 | 80 | 243 | 中高 |
| DDR3_8Gb_x8_1866 | ~15.0 | **~13.9** | ~27.8 | 1.35 | 69 | 135 | 中 |
| GDDR6_8Gb_x16 | ~48.5 | ~15.8 | ~31.7 | 1.35 | 71 | 248 | 高 |
| HBM2_8Gb_x128 (8ch) | **~256** | ~14 | ~28 | 1.2 | 65 | 390 | 高 |

结论（仅就本仓库 INI 而言）：

- **带宽最高**：**HBM2**（8 通道 × 128 bit，总带宽远高于单通道 DDR/GDDR）；单通道里 **GDDR6** 最高。
- **能耗最低**：**DDR4_8Gb_x4_2133**（1.2 V + IDD0/IDD4R 最低），适合对功耗敏感、带宽要求不极端的场景。
- **延迟最低**：**DDR3_8Gb_x8_1866** 与 **DDR4_8Gb_x4_3200** 的 row hit 延迟相当（约 13.9 ns），略优于 LPDDR4/HBM2/GDDR6；若只看 CL×tCK，DDR3 略优。

**不存在“带宽最高且能耗最低且延迟最低”的单一样本**：高带宽（HBM2/GDDR6）通常电流与总功耗更大；低延迟（DDR3/高速 DDR4）与低功耗（DDR4 2133）也不在同一配置上。按需求取舍：要带宽选 HBM2/GDDR6，要能效选 DDR4 2133，要延迟选 DDR3 或 DDR4 3200。
