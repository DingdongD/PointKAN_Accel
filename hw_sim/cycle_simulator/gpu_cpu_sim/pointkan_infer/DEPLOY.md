# 在其他计算平台上复现推理性能测试

在**新平台**上跑与当前环境一致的 PointKAN 推理 profiling，需要准备下面几类内容。

---

## 1. 必须拷贝的代码与脚本（无 output）

从本仓库拷贝以下**两个目录**（不要带 `outputs/`，新平台会重新生成）：

| 目录 | 说明 |
|------|------|
| **gpu_cpu_sim/base_version/** | 通用 profiler：`profiler.py`、`profile_infer.py`、`readme.md`，以及 `scripts/` 下脚本。可只拷贝 `profiler.py` + `scripts/profile_*_infer.sh` 若不需要 profile_infer。 |
| **gpu_cpu_sim/pointkan_infer/** | 推理入口与 profile 脚本：`run_infer.py`、`breakdown_profiler.py`、`configs/`、`scripts/`、`README.md`、`DEPLOY.md`。**不要拷贝** `pointkan_infer/outputs/`。 |

要求**目录相对关系**保持不变，即新平台上存在：

```
某根目录/
├── base_version/
│   ├── profiler.py
│   └── scripts/
│       ├── profile_gpu_infer.sh
│       └── profile_cpu_infer.sh
└── pointkan_infer/
    ├── run_infer.py
    ├── breakdown_profiler.py
    ├── configs/
    ├── scripts/
    ├── README.md
    └── DEPLOY.md
```

即 `pointkan_infer` 与 `base_version` 处于**同一父目录**（例如都放在 `gpu_cpu_sim/` 下）。

---

## 2. PointKAN-elite 与权重、数据

在新平台上需要一套能跑通各任务 `inference_and_kan_analysis.py` 的 **PointKAN-elite**，并让 profiling 能找到它。

- **拷贝或克隆 PointKAN-elite**  
  - 至少包含三个任务目录：`classification_ModelNet40`、`classification_ScanObjectNN`、`part_segmentation`，以及各自依赖的 `models/`、`data` 加载代码等（与当前仓库一致即可）。

- **权重（checkpoints）**  
  - 把在当前平台用到的权重目录或 `.pth` 拷到新平台，路径可任意，只要在运行前通过环境变量或脚本参数指过去即可：
    - ModelNet40：例如 `classification_ModelNet40/checkpoints/pointKANElite-pointkan_elite_exp-42/`
    - ScanObjectNN：例如 `classification_ScanObjectNN/checkpoints/pointKANElite-ScanObjectNN/`
    - Part Seg：例如 `part_segmentation/checkpoints/pointKANElite_part_exp/`
  - 若新平台路径与上面不同，用环境变量 **`CHECKPOINT`** 或各任务单独指定（见 README 参数表）。

- **数据集**  
  - 三个任务的数据集需在新平台存在，且**路径与各任务代码中的一致**（如 `data.py`、`ScanObjectNN.py`、`util/data_util.py` 里的路径或环境变量）。
  - 若新平台路径不同，需在 PointKAN-elite 对应任务里改数据加载路径（或通过环境变量/配置文件），保证能跑通 `inference_and_kan_analysis.py`。

- **指定 PointKAN-elite 根目录**  
  - 若新平台上 PointKAN-elite 不在「从 pointkan_infer 往上 4 层再加 PointKAN-elite」的默认路径下，必须设置：
    - **`POINTKAN_ELITE_ROOT=/新平台/PointKAN-elite 绝对路径`**
  - 再运行 profile 脚本（见下）。

---

## 3. 新平台环境

- **Python**  
  - 与当前环境一致：Python 3.8+，且能跑通 PointKAN-elite 的依赖（PyTorch、CUDA 若测 GPU、`pointnet2_ops`、`kat_rational` 等）。建议先用各任务目录下的 `inference_and_kan_analysis.py` 跑通一次。

- **可选依赖（用于 profiling）**  
  - `psutil`：CPU 采样（建议安装）。  
  - `pynvml`：GPU 功耗/利用率（可选，无则用 nvidia-smi dmon）。  
  - `matplotlib`：仅在做 `plot_profile_results.py` 出图时需要。  
  - **perf**（Linux）：若需 CPU 能耗（RAPL），要装 `perf` 且通常需 root/cap_sys_admin；CPU 模式默认会尝试用 perf（`ENABLE_PERF=1`）。

- **GPU**  
  - 若测 GPU：安装与当前平台一致的 CUDA/驱动；若测 CPU 只需 `MODE=cpu`，无需 GPU。

---

## 4. 在新平台上如何跑

1. **摆好目录**  
   - 把 `base_version/` 与 `pointkan_infer/` 放在同一父目录下；把 PointKAN-elite 放在某路径，并设好 `POINTKAN_ELITE_ROOT`（若与默认不同）。

2. **（可选）指定 checkpoint**  
   - 若权重路径与脚本内默认不同，可对单任务设置，例如：  
     `CHECKPOINT=/新平台/路径/pointKANElite-pointkan_elite_exp-42 TASK=modelnet40 bash scripts/profile_gpu.sh`

3. **一键三任务 + breakdown**  
   - GPU：  
     `MODE=gpu bash pointkan_infer/scripts/profile_all.sh`  
   - CPU：  
     `MODE=cpu bash pointkan_infer/scripts/profile_all.sh`  
   - 在 `pointkan_infer` 目录下执行，或写绝对路径调用脚本（脚本会自己找 ROOT）。

4. **出图（可选）**  
   - `python3 pointkan_infer/scripts/plot_profile_results.py --out-dir pointkan_infer/outputs`  
   - 会读取 `outputs/gpu_*/` 或 `outputs/cpu_*/` 下最新一次运行的 `summary.json` 与采样 CSV，生成 `profile_summary.png`、`profile_curves.png`。

---

## 5. 最小文件清单（拷贝用）

若只想列文件、用 rsync 或 tar 打包，可只带这些（不含 outputs）：

- **base_version**：`profiler.py`，`scripts/profile_gpu_infer.sh`，`scripts/profile_cpu_infer.sh`（以及用到的 `run_perf_stat.sh` 等）。
- **pointkan_infer**：`run_infer.py`，`breakdown_profiler.py`，`configs/tasks.json`，`scripts/*.sh`，`scripts/plot_profile_results.py`，`scripts/run_plot.sh`，`README.md`，`DEPLOY.md`。

再加上 **PointKAN-elite** 三个任务目录及各自 checkpoints、以及新平台所需的数据集路径配置即可。

---

## 6. 打包示例（在当前仓库根目录执行）

在 **gpu_cpu_sim** 目录下执行（即与 `base_version`、`pointkan_infer` 同级）：

```bash
cd /path/to/gpu_cpu_sim
tar -czvf pointkan_profile_package.tar.gz \
  --exclude='pointkan_infer/outputs' \
  --exclude='base_version/outputs' \
  base_version/profiler.py \
  base_version/scripts/ \
  pointkan_infer/run_infer.py \
  pointkan_infer/breakdown_profiler.py \
  pointkan_infer/configs/ \
  pointkan_infer/scripts/ \
  pointkan_infer/README.md \
  pointkan_infer/DEPLOY.md
```

新平台解压后需**单独准备** PointKAN-elite（含权重与数据），并设好 `POINTKAN_ELITE_ROOT`。
