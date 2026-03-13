import heapq
import numpy as np


# ============================================================
# 1. Vanilla KNN
# ============================================================
def vanilla_knn_single_query(query_xyz, ref_xyz, k):
    """
    query_xyz: [3]
    ref_xyz:   [N, 3]
    """
    diff = ref_xyz - query_xyz[None, :]
    dist2 = np.sum(diff * diff, axis=1)
    k = min(k, len(dist2))
    idx = np.argpartition(dist2, k - 1)[:k]
    idx = idx[np.argsort(dist2[idx])]
    return idx, dist2[idx]


# ============================================================
# 2. Bit helpers
# ============================================================
def leading_bit_pos(v: int):
    """
    return floor(log2(v)), require v > 0
    """
    return v.bit_length() - 1


def msb_pos_of_int(v: int):
    """
    MSB position for positive integer; for v <= 0 return -1
    """
    if v <= 0:
        return -1
    return v.bit_length() - 1


# ============================================================
# 3. SAFE lower-bound version (not original FastNN, but lossless-friendly)
# ============================================================
def mss_lower_bound_sq(d: int):
    """
    d: absolute integer difference, d >= 0
    return: certified lower bound of d^2 from top-2 bits (MSS)
    """
    if d == 0:
        return 0

    p = leading_bit_pos(d)  # highest set bit
    second = (d >> (p - 1)) & 1 if p >= 1 else 0

    if second == 0:
        # MSS = 10...
        # d >= 2^p
        lb = 1 << (2 * p)
    else:
        # MSS = 11...
        # d >= 3*2^(p-1)
        base = 3 << (p - 1)
        lb = base * base

    return lb


def precompare_lower_bound_sq(q_int, p_int):
    """
    q_int, p_int: integer xyz arrays of shape [3]
    Safe lower bound of 3D squared distance.
    We use max per-dim lower bound as a certified lower bound.
    """
    dx = abs(int(q_int[0]) - int(p_int[0]))
    dy = abs(int(q_int[1]) - int(p_int[1]))
    dz = abs(int(q_int[2]) - int(p_int[2]))

    lbx = mss_lower_bound_sq(dx)
    lby = mss_lower_bound_sq(dy)
    lbz = mss_lower_bound_sq(dz)

    return max(lbx, lby, lbz)


def precomp_knn_single_query_safe(query_xyz, ref_xyz, ref_xyz_int, q_int, k, scale_sq=1.0):
    """
    SAFE mode:
      - lower-bound based discard
      - designed to be much closer to lossless filtering

    query_xyz:   [3], float
    ref_xyz:     [N, 3], float
    ref_xyz_int: [N, 3], int quantized coords
    q_int:       [3], int quantized query
    scale_sq:    float_sq_dist = int_sq_dist / scale_sq
    """
    N = ref_xyz.shape[0]
    k = min(k, N)

    heap = []  # max-heap by negative distance: store (-dist2, idx)
    full_compute_cnt = 0
    filtered_cnt = 0

    # warmup first k exact
    for i in range(k):
        d = ref_xyz[i] - query_xyz
        dist2 = float(np.dot(d, d))
        heapq.heappush(heap, (-dist2, i))
        full_compute_cnt += 1

    if N <= k:
        items = sorted([(-d, i) for d, i in heap])
        idx = np.array([i for _, i in items], dtype=np.int64)
        dist = np.array([d for d, _ in items], dtype=np.float64)
        return idx, dist, full_compute_cnt, filtered_cnt

    tau_k = -heap[0][0]  # current kth distance in float squared units

    for i in range(k, N):
        lb = precompare_lower_bound_sq(q_int, ref_xyz_int[i])  # int squared domain

        # safe discard: lb / scale_sq >= tau_k
        if lb >= tau_k * scale_sq:
            filtered_cnt += 1
            continue

        # exact distance
        d = ref_xyz[i] - query_xyz
        dist2 = float(np.dot(d, d))
        full_compute_cnt += 1

        if dist2 < tau_k:
            heapq.heapreplace(heap, (-dist2, i))
            tau_k = -heap[0][0]

    items = sorted([(-d, i) for d, i in heap])
    idx = np.array([i for _, i in items], dtype=np.int64)
    dist = np.array([d for d, _ in items], dtype=np.float64)
    return idx, dist, full_compute_cnt, filtered_cnt


# ============================================================
# 4. FastNN-faithful-ish version
#    Paper-like logic:
#      - derive P(MSS) from top-2 bits
#      - keep b = MSB of current kth true squared distance
#      - if P(MSS) < b => do full distance
#        else discard
#
#    Note:
#      The short paper does not fully formalize 3D aggregation.
#      Here we use max(Px, Py, Pz) as the final estimate.
# ============================================================
def fastnn_p_mss_1d(d: int):
    """
    Paper-style 1D P(MSS).

    For absolute difference d:
      MSS=10 -> P(MSS)=2p-1
      MSS=11 -> P(MSS)=2p

    We treat d=0 as a very small score so it always tends to pass.
    """
    if d == 0:
        return -10**9  # effectively "very small", always eligible

    p = leading_bit_pos(d)
    second = (d >> (p - 1)) & 1 if p >= 1 else 0

    if second == 0:
        return 2 * p - 1
    else:
        return 2 * p


def fastnn_precompare_score_xyz(q_int, p_int):
    """
    3D aggregation for FastNN pre-comparator.
    Since the short paper doesn't fully spell out xyz aggregation,
    we use the 'max-dimension domination' style completion.
    """
    dx = abs(int(q_int[0]) - int(p_int[0]))
    dy = abs(int(q_int[1]) - int(p_int[1]))
    dz = abs(int(q_int[2]) - int(p_int[2]))

    sx = fastnn_p_mss_1d(dx)
    sy = fastnn_p_mss_1d(dy)
    sz = fastnn_p_mss_1d(dz)

    return max(sx, sy, sz)


def precomp_knn_single_query_fastnn(query_xyz, ref_xyz, ref_xyz_int, q_int, k, scale_sq=1.0):
    """
    FastNN-style approximate pre-comparator KNN.

    Implementation logic:
      1) warmup first k points with full exact distance
      2) current kth neighbor true distance^2 -> convert to integer squared distance
      3) b = MSB position of kth true distance^2
      4) candidate pre-score P(MSS) from xyz
      5) if P(MSS) < b: do full exact distance
         else discard

    scale_sq:
      int_sq_dist = float_sq_dist * scale_sq
    """
    N = ref_xyz.shape[0]
    k = min(k, N)

    heap = []
    full_compute_cnt = 0
    filtered_cnt = 0

    # warmup first k exact
    for i in range(k):
        d = ref_xyz[i] - query_xyz
        dist2 = float(np.dot(d, d))
        heapq.heappush(heap, (-dist2, i))
        full_compute_cnt += 1

    if N <= k:
        items = sorted([(-d, i) for d, i in heap])
        idx = np.array([i for _, i in items], dtype=np.int64)
        dist = np.array([d for d, _ in items], dtype=np.float64)
        return idx, dist, full_compute_cnt, filtered_cnt

    tau_k = -heap[0][0]  # float squared distance
    tau_k_int = int(round(tau_k * scale_sq))
    b = msb_pos_of_int(tau_k_int)

    for i in range(k, N):
        p_score = fastnn_precompare_score_xyz(q_int, ref_xyz_int[i])

        # FastNN paper-like decision
        if p_score >= b:
            filtered_cnt += 1
            continue

        # exact distance
        d = ref_xyz[i] - query_xyz
        dist2 = float(np.dot(d, d))
        full_compute_cnt += 1

        if dist2 < tau_k:
            heapq.heapreplace(heap, (-dist2, i))
            tau_k = -heap[0][0]
            tau_k_int = int(round(tau_k * scale_sq))
            b = msb_pos_of_int(tau_k_int)

    items = sorted([(-d, i) for d, i in heap])
    idx = np.array([i for _, i in items], dtype=np.int64)
    dist = np.array([d for d, _ in items], dtype=np.float64)
    return idx, dist, full_compute_cnt, filtered_cnt


# ============================================================
# 5. Legacy coarse score (optional)
# ============================================================
def mss_score(d: int):
    """
    Coarse legacy score, kept for compatibility.
    Not the exact FastNN formula.
    """
    if d == 0:
        return -1
    p = leading_bit_pos(d)
    second = (d >> (p - 1)) & 1 if p >= 1 else 0
    return 2 * p + second


def precompare_score(q_int, p_int):
    dx = abs(int(q_int[0]) - int(p_int[0]))
    dy = abs(int(q_int[1]) - int(p_int[1]))
    dz = abs(int(q_int[2]) - int(p_int[2]))
    return max(mss_score(dx), mss_score(dy), mss_score(dz))


# ============================================================
# 6. Metrics
# ============================================================
def knn_set_iou(idx_a, idx_b):
    a, b = set(idx_a.tolist()), set(idx_b.tolist())
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 1.0


def knn_recall_at_k(idx_gt, idx_pred, k):
    a, b = set(idx_gt.tolist()), set(idx_pred.tolist())
    return len(a & b) / k


# ============================================================
# 7. Evaluation driver
# ============================================================
def evaluate_knn_module(queries_xyz, refs_xyz, queries_int, refs_int, k, mode="safe", scale_sq=1.0):
    """
    mode:
      - "safe"   : strict lower-bound filtering
      - "fastnn" : paper-like approximate pre-comparator
    """
    ious = []
    recalls = []
    full_counts = []
    filtered_counts = []

    for qi in range(queries_xyz.shape[0]):
        gt_idx, gt_dist = vanilla_knn_single_query(queries_xyz[qi], refs_xyz, k)

        if mode == "safe":
            pd_idx, pd_dist, full_cnt, filt_cnt = precomp_knn_single_query_safe(
                queries_xyz[qi], refs_xyz, refs_int, queries_int[qi], k, scale_sq=scale_sq
            )
        elif mode == "fastnn":
            pd_idx, pd_dist, full_cnt, filt_cnt = precomp_knn_single_query_fastnn(
                queries_xyz[qi], refs_xyz, refs_int, queries_int[qi], k, scale_sq=scale_sq
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        ious.append(knn_set_iou(gt_idx, pd_idx))
        recalls.append(knn_recall_at_k(gt_idx, pd_idx, k))
        full_counts.append(full_cnt)
        filtered_counts.append(filt_cnt)

    return {
        "mean_iou": float(np.mean(ious)),
        "mean_recall": float(np.mean(recalls)),
        "std_iou": float(np.std(ious)),
        "std_recall": float(np.std(recalls)),
        "avg_full_compute": float(np.mean(full_counts)),
        "avg_filtered": float(np.mean(filtered_counts)),
        "avg_filter_ratio": float(np.mean(np.array(filtered_counts) / refs_xyz.shape[0])),
        "ious": ious,
        "recalls": recalls,
    }


def evaluate_knn_module_compare_all(queries_xyz, refs_xyz, queries_int, refs_int, k, scale_sq=1.0):
    """
    一次遍历同时跑 Vanilla（仅作 GT）、Safe、FastNN，返回三者统计便于对比。
    Vanilla 无过滤，每 query 精确计算数 = N_ref，过滤数 = 0。
    """
    n_ref = refs_xyz.shape[0]
    ious_safe, recalls_safe, full_safe, filt_safe = [], [], [], []
    ious_fastnn, recalls_fastnn, full_fastnn, filt_fastnn = [], [], [], []

    for qi in range(queries_xyz.shape[0]):
        gt_idx, _ = vanilla_knn_single_query(queries_xyz[qi], refs_xyz, k)

        pd_safe, _, fc_safe, ff_safe = precomp_knn_single_query_safe(
            queries_xyz[qi], refs_xyz, refs_int, queries_int[qi], k, scale_sq=scale_sq
        )
        pd_fastnn, _, fc_fastnn, ff_fastnn = precomp_knn_single_query_fastnn(
            queries_xyz[qi], refs_xyz, refs_int, queries_int[qi], k, scale_sq=scale_sq
        )

        ious_safe.append(knn_set_iou(gt_idx, pd_safe))
        recalls_safe.append(knn_recall_at_k(gt_idx, pd_safe, k))
        full_safe.append(fc_safe)
        filt_safe.append(ff_safe)

        ious_fastnn.append(knn_set_iou(gt_idx, pd_fastnn))
        recalls_fastnn.append(knn_recall_at_k(gt_idx, pd_fastnn, k))
        full_fastnn.append(fc_fastnn)
        filt_fastnn.append(ff_fastnn)

    def _agg(ious, recalls, full, filt):
        return {
            "mean_iou": float(np.mean(ious)),
            "std_iou": float(np.std(ious)),
            "mean_recall": float(np.mean(recalls)),
            "std_recall": float(np.std(recalls)),
            "avg_full_compute": float(np.mean(full)),
            "avg_filtered": float(np.mean(filt)),
            "avg_filter_ratio": float(np.mean(np.array(filt) / n_ref)),
        }

    return {
        "vanilla": {
            "mean_iou": 1.0,
            "std_iou": 0.0,
            "mean_recall": 1.0,
            "std_recall": 0.0,
            "avg_full_compute": float(n_ref),
            "avg_filtered": 0.0,
            "avg_filter_ratio": 0.0,
        },
        "safe": _agg(ious_safe, recalls_safe, full_safe, filt_safe),
        "fastnn": _agg(ious_fastnn, recalls_fastnn, full_fastnn, filt_fastnn),
    }


# ============================================================
# 8. Random-point-cloud demo
# ============================================================
def run_vanilla_vs_precomp_evaluation(
    n_ref=2000,
    n_queries=100,
    k=16,
    dim=3,
    coord_scale=1000.0,
    seed=42,
    mode="safe",
):
    """
    在随机点云上评估：
      Vanilla KNN 为 GT
      对比 safe / fastnn 预比较器版
    """
    rng = np.random.default_rng(seed)
    ref_xyz = rng.uniform(0, coord_scale, (n_ref, dim)).astype(np.float64)
    queries_xyz = rng.uniform(0, coord_scale, (n_queries, dim)).astype(np.float64)

    ref_xyz_int = ref_xyz.astype(np.int64)
    queries_int = queries_xyz.astype(np.int64)

    precomp_res = evaluate_knn_module(
        queries_xyz, ref_xyz, queries_int, ref_xyz_int, k, mode=mode, scale_sq=1.0
    )

    vanilla_self_ious = []
    for qi in range(queries_xyz.shape[0]):
        idx1, _ = vanilla_knn_single_query(queries_xyz[qi], ref_xyz, k)
        idx2, _ = vanilla_knn_single_query(queries_xyz[qi], ref_xyz, k)
        vanilla_self_ious.append(knn_set_iou(idx1, idx2))
    vanilla_self_iou = float(np.mean(vanilla_self_ious))

    return {
        "mode": mode,
        "n_ref": n_ref,
        "n_queries": n_queries,
        "k": k,
        "vanilla_self_iou": vanilla_self_iou,
        "precomp_mean_iou": precomp_res["mean_iou"],
        "precomp_std_iou": precomp_res["std_iou"],
        "precomp_mean_recall": precomp_res["mean_recall"],
        "precomp_std_recall": precomp_res["std_recall"],
        "avg_full_compute": precomp_res["avg_full_compute"],
        "avg_filtered": precomp_res["avg_filtered"],
        "avg_filter_ratio": precomp_res["avg_filter_ratio"],
        "precomp_ious": precomp_res["ious"],
        "precomp_recalls": precomp_res["recalls"],
    }


def print_evaluation_report(stats: dict):
    mode = stats["mode"]
    print("\n" + "=" * 64)
    print(f"  Vanilla KNN vs 预比较器 KNN — IoU / Recall 评估 ({mode})")
    print("=" * 64)
    print(f"  设置: N_ref={stats['n_ref']}, N_queries={stats['n_queries']}, k={stats['k']}")
    print("-" * 64)
    print("  Ground truth: Vanilla KNN (精确距离，全量计算)")
    print(f"  Vanilla 自洽 IoU (sanity check): {stats['vanilla_self_iou']:.6f}")
    print("-" * 64)

    if mode == "safe":
        desc = "SAFE 下界过滤"
    elif mode == "fastnn":
        desc = "FastNN-style MSS/MSB 预比较器"
    else:
        desc = mode

    print(f"  预比较器 KNN ({desc}):")
    print(f"    Mean IoU (vs Vanilla):  {stats['precomp_mean_iou']:.6f}  (std: {stats['precomp_std_iou']:.6f})")
    print(f"    Mean Recall@k:          {stats['precomp_mean_recall']:.6f}  (std: {stats['precomp_std_recall']:.6f})")
    print(f"    平均每 query 精确距离计算次数: {stats['avg_full_compute']:.1f} / {stats['n_ref']}")
    print(f"    平均每 query 过滤点数:       {stats['avg_filtered']:.1f}")
    print(f"    平均过滤比例:               {stats['avg_filter_ratio'] * 100:.2f}%")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    # 你可以改成 mode="fastnn" 看近似版损失
    stats = run_vanilla_vs_precomp_evaluation(
        n_ref=2000,
        n_queries=100,
        k=16,
        seed=42,
        mode="fastnn",
    )
    print_evaluation_report(stats)