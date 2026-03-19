 ---
  实现总结

  新增文件

  /workspace/PointKAN-pytorch/pointkan_sim/network/pointnet2_net.py

  实现了三个 PointNet++ 硬件算子网络定义，完整映射自 /workspace/ISEDA26/model_back.py：

  ┌───────────────────────┬─────────────────────┬──────────────┬──────┬────────┐
  │         网络          │       模型类        │    数据集    │ 层数 │ 算子数 │
  ├───────────────────────┼─────────────────────┼──────────────┼──────┼────────┤
  │ pointnet2-cls-ssg     │ PointNet2ClsSSG     │ ModelNet40   │ 4    │ 41     │
  ├───────────────────────┼─────────────────────┼──────────────┼──────┼────────┤
  │ pointnet2-partseg-ssg │ PointNet2PartSegSSG │ ShapeNetPart │ 7    │ 62     │
  ├───────────────────────┼─────────────────────┼──────────────┼──────┼────────┤
  │ pointnet2-sem-ssg     │ PointNet2SemSSG     │ S3DIS        │ 9    │ 83     │
  └───────────────────────┴─────────────────────┴──────────────┴──────┴────────┘

  算子映射策略：
  - SA Conv2d(k=1) on [B, C, nsample, npoint] → conv1d with num_points = npoint * nsample（等效 MACs）
  - SA BallQuery/KNN → knn(k=nsample)
  - SA MaxPool(nsample→1) → adaptive_max_pool1d(num_points=npoint*nsample)
  - SA3 GroupAll → 跳过 FPS/KNN，直接 conv1d(num_points=N_in)
  - FP 插值(3-NN) → knn(k=3, query=N_target, ref=N_src)
  - FP Conv1d MLP → conv1d(num_points=N_target)

  - network/__init__.py：导出三个新网络
  - pointkan_sim/main.py：load_network() 新增三个网络，--network 选项扩展，新增 --input_channels 参数
  - run_sim_with_real_eval.py：新增 --network（支持五个选项）和 --input_channels 参数；两条路径（--use_real_data
  和默认仿真路径）均使用 args.network；修复 conv_linear_list 未定义 bug

  调用示例

  # PointNet2 分类 (ModelNet40)
  python -m pointkan_sim.run_sim_with_real_eval \
    --network pointnet2-cls-ssg \
    --num_points 1024 --num_classes 40 \
    --accelerator baseline --use_scalesim_systolic

  # PointNet2 分割 (ShapeNetPart)
  python -m pointkan_sim.run_sim_with_real_eval \
    --network pointnet2-partseg-ssg \
    --num_points 2048 --num_classes 50 \
    --accelerator baseline --use_scalesim_systolic

  # PointNet2 语义分割 (S3DIS)
  python -m pointkan_sim.run_sim_with_real_eval \
    --network pointnet2-sem-ssg \
    --num_points 4096 --num_classes 13 --input_channels 12 \
    --accelerator baseline --use_scalesim_systolic