# PointKAN 随机数据推理最小 Demo

在 **CPU / GPU** 上一键跑 PointKAN 随机数据推理，统计**模型推理时延**与 **GPU 功耗/能耗**、**CPU 能耗**（仅推理阶段）。不依赖 checkpoint 与真实数据集。

---

## 打包与部署（zip 下载到其他工程）

**mini_demo 已内含 PointKAN-elite 与 pointnet2_ops_lib**。请用本目录下的 **make_zip.sh** 打包，会自动排除 .pt、runs/、outputs、build 等大文件，只保留运行所需：

```bash
cd /path/to/mini_demo
./make_zip.sh          # 生成上级目录下的 mini_demo.zip
./make_zip.sh my.zip   # 或指定输出文件名
```

解压到目标工程目录后，在目标机安装依赖并一键运行（见下）。

## 最小文件清单（若只拷贝部分文件）

| 文件/目录 | 说明 |
|-----------|------|
| `run.sh` | 一键运行脚本（必选） |
| `run_infer_random.py` | 随机数据推理入口（必选） |
| `rapl_inference.py` | 推理计时 + CPU 能耗测量（必选） |
| `profiler.py` | 命令级 profiler，采集 GPU/CPU 延时与能耗（必选） |
| **PointKAN-elite/** | 已拷贝在 mini_demo 内则无需单独指定；否则通过 `POINTKAN_ELITE_ROOT` 指定 |
| **pointnet2_ops_lib/** | 已拷贝在 mini_demo 内时，run.sh 会自动加入 PYTHONPATH；目标机首次建议 `pip install -e pointnet2_ops_lib` 或执行 `./install_env.sh` 以编译扩展 |
| **requirements.txt** | 一键安装依赖：`pip install -r requirements.txt` |
| **install_env.sh** | 新服务器上一键安装环境（requirements + pointnet2_ops） |

目标机还需安装：PyTorch（见 requirements 说明）；其余依赖见 requirements.txt。

---

## 1. 依赖

- **Python 3.8+**
- **PyTorch**（GPU 推理需 CUDA）
- **PointKAN-elite**：已内嵌在 mini_demo 中，内含模型与 **kat_rational**（KAT_Group），无需单独安装 **rational_kat_cu**；run.sh 会优先使用内嵌的 PointKAN-elite。
- **pointnet2_ops**：本目录下 `pointnet2_ops_lib`，需 `pip install -e pointnet2_ops_lib` 或执行 `./install_env.sh`（含 CPU FPS）。
- **GPU**：`pynvml`（pip 安装）用于采集单卡功耗/能耗
- **CPU 能耗（可选）**：`pyRAPL`（pip 安装）；不装则 CPU 能耗为 null

---

## 2. 部署到其他服务器

**方式 A：整包拷贝（推荐）**  
将 **mini_demo** 整个目录（含内嵌的 `PointKAN-elite/`、`pointnet2_ops_lib/`）zip 后拷到目标机，解压到任意工程目录。无需再设置 `POINTKAN_ELITE_ROOT`，run.sh 会自动使用 mini_demo 内的 PointKAN-elite 与 pointnet2_ops_lib。

**方式 B：仅拷贝脚本**  
若只拷贝 run.sh、run_infer_random.py、rapl_inference.py、profiler.py、README.md，则需将 **PointKAN-elite** 放在 mini_demo 同目录或设置：

```bash
export POINTKAN_ELITE_ROOT=/path/to/PointKAN-elite
```

**目标机依赖**：PyTorch、pynvml；可选 pyRAPL。若使用打包内的 pointnet2_ops_lib，首次建议在 mini_demo 目录下执行 `pip install -e pointnet2_ops_lib` 以编译 CUDA/CPU 扩展；若已用 conda 等安装过 pointnet2_ops，可依赖 run.sh 设置的 PYTHONPATH 或忽略。

---

## 3. 新服务器上一键安装环境

解压 mini_demo 后，在 **mini_demo** 目录下执行（需已安装 Python 3.8+，跑 GPU 则需 CUDA/驱动）：

```bash
# 若尚未安装 PyTorch，请先安装：pip install torch（或从 https://pytorch.org 按 CUDA 安装）
chmod +x install_env.sh
./install_env.sh
```

该脚本会：安装 `requirements.txt` 中的依赖，并执行 `pip install -e pointnet2_ops_lib` 编译安装 pointnet2_ops。

## 4. 一键运行

在 **mini_demo** 目录下执行：

```bash
chmod +x run.sh
./run.sh
```

默认行为：

- **任务**：`modelnet40`
- **2 次 run**，每次 **20 批次**，**batch_size=1**
- **GPU**：单卡（`GPU_INDEX=0`），统计推理时延 + 单卡功耗/能耗
- **CPU**：统计推理时延 + 推理阶段 CPU 能耗（若已装 pyRAPL）

可通过环境变量覆盖，例如：

```bash
export TASK=scanobjectnn
export REPEAT=4
export MAX_BATCHES=10
export GPU_INDEX=0
./run.sh
```

---

## 5. 输出

- 终端会打印各次 run 的 **inference_only** 汇总（含 `inference_only_per_forward_s_mean`、`gpu_energy_j_mean`、`gpu_avg_power_w_mean`、`gpu_peak_power_w_max`、CPU 能耗等）。
- 详细结果目录：
  - `mini_demo/outputs/random_gpu_<TASK>/<timestamp>/`
  - `mini_demo/outputs/random_cpu_<TASK>/<timestamp>/`
- 每个目录下有 `summary.json`、各次 run 的 stdout/stderr 与（GPU 时）功耗采样 CSV。

---

## 6. 仅跑推理、不跑 profiler

若只想快速跑通推理、不采集功耗/能耗，可直接：

```bash
export POINTKAN_ELITE_ROOT=/path/to/PointKAN-elite   # 若需要
python3 run_infer_random.py --task modelnet40 --device cuda --max-batches 20
python3 run_infer_random.py --task modelnet40 --device cpu --max-batches 20
```

---

## 7. 三任务说明

| task         | 说明              |
|-------------|-------------------|
| modelnet40  | 分类，默认 1024 点 |
| scanobjectnn| 分类，15 类       |
| part_seg    | 部件分割，2048 点 |

均使用随机输入与随机权重，不读 checkpoint，仅验证前向与统计时延/能耗。
