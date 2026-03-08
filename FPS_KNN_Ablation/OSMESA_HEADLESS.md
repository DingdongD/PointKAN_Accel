# 无显示环境下用 Open3D 离屏渲染

本脚本 3D 可视化逻辑：

- **无 DISPLAY 且未设置 `KITTI_FPS_USE_OPEN3D=1`**  
  直接使用 **matplotlib**，不调用任何 Open3D，避免 `eglInitialize failed` / Segmentation fault（无头环境下 OffscreenRenderer 走 EGL，EGL 常不可用）。  
- **有 DISPLAY 或已设 `KITTI_FPS_USE_OPEN3D=1`**  
  依次尝试：**OffscreenRenderer** → **Visualizer 离屏** → **matplotlib**。

因此在你当前无显示环境下，**无需任何配置**，脚本会自动用 matplotlib 出 3D 图且不会崩溃。若希望在该环境下仍用 Open3D，需先解决 EGL/OSMesa（见下文）。

## 1. 安装 OSMesa（系统）

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y libosmesa6-dev

# 验证
ls /usr/lib/x86_64-linux-gnu/libOSMesa* 2>/dev/null || ls /usr/lib/libOSMesa* 2>/dev/null
```

若系统 OSMesa 过旧（不支持 GL 3.3 Core），会报错：`OSMesa: Failed to create context`，需从源码编译较新版本 Mesa，见 [Open3D Headless 文档](https://www.open3d.org/docs/latest/tutorial/visualization/headless_rendering.html)。

### 1b. 用 Conda 安装 OSMesa（可选）

可以用 conda-forge 的 **mesalib** 提供 OSMesa，无需 sudo：

```bash
# 仅安装 < 25.1 的 mesalib：25.1 起上游 Mesa 移除了 OSMesa，conda-forge 同步后不再包含
conda install -c conda-forge "mesalib<25.1"
```

安装后，当前环境里会有 `libOSMesa.so`（例如在 `$CONDA_PREFIX/lib`）。若之后**从源码编译 Open3D**，可在同一 conda 环境下编译，并让 CMake 使用该 OSMesa：

```bash
# 在已安装 mesalib<25.1 的 conda 环境下
cd Open3D/build
cmake -DENABLE_HEADLESS_RENDERING=ON \
      -DBUILD_GUI=OFF \
      -DBUILD_WEBRTC=OFF \
      -DUSE_SYSTEM_GLEW=OFF \
      -DUSE_SYSTEM_GLFW=OFF \
      -DOSMESA_INCLUDE_DIR=$CONDA_PREFIX/include \
      -DOSMESA_LIBRARY=$CONDA_PREFIX/lib/libOSMesa.so \
      ..
make -j$(nproc)
make install-pip-package
```

说明：**PyPI 的 `pip install open3d` 未开启 headless**，仅装 conda 的 mesalib 不会让现有 Open3D 自动用上 OSMesa，仍需按上文从源码编译 Open3D 并链接到 OSMesa（系统或 conda 的均可）。

## 2. 从源码编译带 OSMesa 的 Open3D

```bash
# 依赖
sudo apt-get install -y build-essential cmake libglew-dev libglfw3-dev libosmesa6-dev

# 克隆 Open3D（按你需要的版本换 tag）
git clone --recursive https://github.com/isl-org/Open3D
cd Open3D
mkdir build && cd build

# 关键：开启 headless，并用内置 glew/glfw 以链接 OSMesa
cmake -DENABLE_HEADLESS_RENDERING=ON \
      -DBUILD_GUI=OFF \
      -DBUILD_WEBRTC=OFF \
      -DUSE_SYSTEM_GLEW=OFF \
      -DUSE_SYSTEM_GLFW=OFF \
      ..

make -j$(nproc)
# 安装到当前环境（在对应 venv/conda 下执行）
make install-pip-package
```

编译完成后，在当前 Python 环境里用的就是带 OSMesa 的 Open3D。

## 3. 本脚本中启用 Open3D（无 DISPLAY 时）

安装好上述 Open3D 后，在**无 DISPLAY** 的终端里可强制用 Open3D 离屏（走 OSMesa）：

```bash
export KITTI_FPS_USE_OPEN3D=1
python kitti_fps_exp.py --data_dir /path/to/velodyne --out_dir ./out
```

若仍报 `OSMesa: Library not found` 或 segfault，请确认：

- `libOSMesa.so` 在系统路径或 `LD_LIBRARY_PATH` 中；
- 当前用的 `open3d` 确为上述自编译版本（`python -c "import open3d as o3d; print(o3d.__file__)"`）。

## 4. 不装 OSMesa / 不编译时

不加任何环境变量、无 DISPLAY 时，脚本会**自动用 matplotlib 做 3D**，不会调用 Open3D，避免崩溃。若想显式禁用 Open3D 可用：

```bash
python kitti_fps_exp.py --no_open3d
```

参考： [Open3D Headless rendering](https://www.open3d.org/docs/latest/tutorial/visualization/headless_rendering.html)
