# 绘图风格模板统一
1. DPI设置600，导出格式PNG/PDF，保存参数bbox_inches='tight'，pad_inches=0.02，背景transparent
2. 字体为Arial，xtick.direction = 'in'，ytick.direction = 'in', grid.linestyle = '--', grid.alpha = 0.4
3. Avoid bold except axis labels if necessary, No oversized titles
4. All four spines must be visible,Tick direction: inward, Spine linewidth: 1.5 pt,  No excessive minor ticks
5. 字体在100%所放下可阅读，legend不能覆盖关键数据

# FPS
1. 基于数据集构建层次采样不同类别样本的校准集合
2. 构建FPS迭代更新时的最小距离阈值曲线radii （每个样本都会有）构建这个样本，统计校准集样本下radii曲线分布，横坐标是采样点迭代次数，如果输入点云数目为1024，可以模拟完整的1024次FPS采样迭代次数下的最小距离radii。
3. 分析每一层最远点采样


# FPS-KNN Optimization
1. /workspace/PointKAN-pytorch/PointKAN/classification_ModelNet40/opt_fps/opt_fps_knn.py中定义了fps_with_knn_adaptive函数，这里是我对原有pointkan中FPS-KNN pipeline的优化，需要增加一个接口对比这个函数和vanilla FPS-KNN生成的索引点的准确率/iou分析