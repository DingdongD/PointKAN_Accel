import torch
from typing import Tuple

@torch.no_grad()  # 禁用梯度计算，提高性能
def fps_with_knn_adaptive(
    X: torch.Tensor,  # 输入点云，形状为 [B, N, D]
    K: int,  # 要采样的点数量
    k_neighbors: int = 16,  # 每个采样点的 KNN 邻居数
    init_threshold: float = 0.1,  # 初始阈值
    max_iter: int = 10,  # 最大迭代次数
    tol: float = 0.05,  # 容差
    seed: str = "first",  # 种子点选择方式
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    FPS + 动态阈值KNN
    返回：
        - idx: FPS采样索引
        - radii: FPS半径曲线
        - knn_idx: 动态阈值下的KNN索引
        - distances_matrix: 距离矩阵
        - knn_radii: 第k近邻半径
        - thresholds: 最终阈值
        - iters_used: 每个采样点的迭代次数
    """
    assert X.ndim == 3  # 确保输入是3维张量
    B, N, D = X.shape  # 获取批量大小、点数量和特征维度
    device, dtype = X.device, X.dtype  # 获取设备和数据类型
    K = min(K, N)  # 确保采样点数不超过总点数

    x2 = (X * X).sum(dim=-1)  # (B, N)，计算每个点的平方范数
    idx = torch.zeros((B, K), dtype=torch.long, device=device)  # 存储FPS采样索引
    radii = torch.zeros((B, K), dtype=dtype, device=device)  # 存储FPS半径曲线
    min_dist = torch.full((B, N), float("inf"), dtype=dtype, device=device)  # 存储每个点到最近采样点的距离
    distances_matrix = torch.zeros((B, K, N), dtype=dtype, device=device)  # 存储每个采样点到所有点的距离
    thresholds = torch.zeros((B, K), dtype=dtype, device=device)  # 存储每个采样点的最终阈值
    iters_used = torch.zeros((B, K), dtype=torch.int32, device=device)  # 存储每个采样点的迭代次数

    # 初始化种子点
    if seed == "first":  # 选择第一个点作为种子
        first = torch.zeros(B, dtype=torch.long, device=device)
    elif seed == "random":  # 随机选择种子点
        first = torch.randint(0, N, (B,), device=device)
    elif seed == "max_norm":  # 选择范数最大的点作为种子
        first = torch.argmax(x2, dim=1)
    else:
        raise ValueError(f"Unknown seed: {seed}")  # 抛出未知种子类型的错误

    
    idx[:, 0] = first  # 存储第一个采样点的索引
    C = X[torch.arange(B), first]  # 获取第一个采样点的坐标
    c2 = (C * C).sum(dim=-1, keepdim=True)  # 计算第一个采样点的平方范数
    dot = (X @ C.unsqueeze(-1)).squeeze(-1)  # 计算每个点与第一个采样点的点积
    d2 = (x2 + c2 - 2.0 * dot).clamp_min_(0.0).sqrt()  # 计算每个点与第一个采样点的距离
    distances_matrix[:, 0, :] = d2  # 存储第一个采样点到所有点的距离
    min_dist.copy_(d2)  # 初始化最小距离为到第一个采样点的距离
    radii[:, 0], _ = torch.max(min_dist, dim=1)  # 计算第一个采样点的半径（最大距离）
    min_dist[torch.arange(B), first] = 0.0  # 将第一个采样点的距离设置为0

    # FPS主循环
    for t in range(1, K):  # 从第二个采样点开始
        farthest = torch.argmax(min_dist, dim=1)  # 选择距离最远的点作为下一个采样点
        idx[:, t] = farthest  # 存储当前采样点的索引

        
        C = X[torch.arange(B), farthest]  # 获取当前采样点的坐标
        c2 = (C * C).sum(dim=-1, keepdim=True)  # 计算当前采样点的平方范数
        dot = (X @ C.unsqueeze(-1)).squeeze(-1)  # 计算每个点与当前采样点的点积
        d2 = (x2 + c2 - 2.0 * dot).clamp_min_(0.0).sqrt()  # 计算每个点与当前采样点的距离

        distances_matrix[:, t, :] = d2  # 存储当前采样点到所有点的距离
        torch.minimum(min_dist, d2, out=min_dist)  # 更新每个点到最近采样点的距离
        radii[:, t], _ = torch.max(min_dist, dim=1)  # 计算当前采样点的半径（最大距离）
        min_dist[torch.arange(B), farthest] = 0.0  # 将当前采样点的距离设置为0

    # ===== 动态阈值KNN选择 =====
    B_idx = torch.arange(B, device=device)  # 批量索引
    knn_idx = torch.zeros((B, K, k_neighbors), dtype=torch.long, device=device)  # 存储KNN索引
    knn_radii = torch.zeros((B, K), dtype=dtype, device=device)  # 存储第k近邻的半径

    for t in range(K):  # 对每个采样点
        d = distances_matrix[:, t, :]  # (B, N)，当前采样点到所有点的距离
        threshold = torch.full((B,), init_threshold, dtype=dtype, device=device)  # 初始化阈值

        for i in range(max_iter):  # 动态调整阈值
            mask = d < threshold.unsqueeze(-1)  # 生成距离小于阈值的掩码
            count = mask.sum(dim=1).float()  # 计算每个样本中距离小于阈值的点数量

            # 检查是否在容差范围内
            done = (count >= k_neighbors * (1 - tol)) & (count <= k_neighbors * (1 + tol))
            if done.all():  # 如果所有样本都在容差范围内
                iters_used[:, t] = i + 1  # 记录迭代次数
                break  # 停止迭代

            # 计算调整量
            too_many = (count > (k_neighbors * (1 + tol))).float()  # 邻居数量过多
            too_few = (count < (k_neighbors * (1 - tol))).float()  # 邻居数量过少
            adjust = torch.where(too_many.bool(), -0.1, torch.where(too_few.bool(), 0.1, 0.0))  # 计算调整系数
            threshold = threshold * (1 + adjust)  # 调整阈值

            # 最后一次迭代仍未收敛
            if i == max_iter - 1:
                iters_used[:, t] = max_iter  # 记录最大迭代次数

        thresholds[:, t] = threshold  # 存储最终阈值

        # 应用最终阈值
        mask = d < threshold.unsqueeze(-1)  # 生成距离小于最终阈值的掩码
        masked_d = d.clone()  # 复制距离矩阵
        masked_d[~mask] = float("inf")  # 将距离大于等于阈值的点设置为无穷大
        knn_vals, knn_idx_t = torch.topk(masked_d, k=min(k_neighbors, N), dim=1, largest=False)  # 选择距离最小的k个点
        knn_idx[:, t, :] = knn_idx_t  # 存储KNN索引
        knn_radii[:, t] = knn_vals[:, -1]  # 存储第k近邻的半径

    return idx, radii, knn_idx, distances_matrix, knn_radii, thresholds, iters_used  # 返回所有结果