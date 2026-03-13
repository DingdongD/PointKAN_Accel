import heapq
import numpy as np
from collections import defaultdict, Counter


# ============================================================
# 0) Quantization
# ============================================================
def quantize_points_fixed_scale(xyz: np.ndarray, coord_scale: float) -> np.ndarray:
    """
    固定 scale 量化:
        x_int = round(x * coord_scale)
    xyz: [N, 3] float
    return: [N, 3] int64
    """
    return np.round(xyz * coord_scale).astype(np.int64)


def quantize_points_per_cloud_16bit(xyz: np.ndarray, eps: float = 1e-12):
    """
    每帧归一化到 [0, 65535]，返回:
      xyz_int: [N,3] int64
      meta: 量化参数，便于 query / ref 一致量化
    """
    xyz_min = xyz.min(axis=0, keepdims=True)
    xyz_max = xyz.max(axis=0, keepdims=True)
    denom = np.maximum(xyz_max - xyz_min, eps)
    xyz_norm = (xyz - xyz_min) / denom
    xyz_int = np.round(xyz_norm * 65535.0).astype(np.int64)
    meta = {
        "xyz_min": xyz_min,
        "denom": denom,
        "levels": 65535.0,
    }
    return xyz_int, meta


def quantize_with_meta(xyz: np.ndarray, meta: dict) -> np.ndarray:
    xyz_norm = (xyz - meta["xyz_min"]) / meta["denom"]
    xyz_int = np.round(xyz_norm * meta["levels"]).astype(np.int64)
    return xyz_int


def quantize_points_fixed_range(
    xyz: np.ndarray,
    coord_min: float,
    coord_max: float,
    bits: int,
    clip: bool = True,
) -> np.ndarray:
    """
    按数据集空间范围 [coord_min, coord_max] 和量化比特数 bits 做均匀量化。
    适用于已知数据范围的 dataset（如 ModelNet40 一般为 [-1, 1]）。

    xyz: [N, 3] float，坐标应在 [coord_min, coord_max] 内（若超出且 clip=True 则截断）
    coord_min, coord_max: 数据空间每维的最小/最大值（标量，各维共用；若需每维不同可后续扩展）
    bits: 量化比特数，每维映射到 [0, 2^bits - 1]
    clip: 若 True，将 (x - coord_min) / span 截断到 [0, 1] 再量化，避免越界

    return: [N, 3] int64
    """
    span = float(coord_max - coord_min)
    if span <= 0:
        raise ValueError("coord_max must be > coord_min")
    levels = (1 << bits) - 1  # 2^bits - 1，例如 8bit -> 255, 16bit -> 65535
    xyz_norm = (xyz - coord_min) / span
    if clip:
        xyz_norm = np.clip(xyz_norm, 0.0, 1.0)
    xyz_int = np.round(xyz_norm * levels).astype(np.int64)
    return xyz_int


# ============================================================
# 1) Vanilla KNN
# ============================================================
def vanilla_knn_single_query_float(query_xyz, ref_xyz, k):
    """
    原始浮点 KNN
    query_xyz: [3]
    ref_xyz:   [N, 3]
    """
    diff = ref_xyz - query_xyz[None, :]
    dist2 = np.sum(diff * diff, axis=1)
    k = min(k, len(dist2))
    idx = np.argpartition(dist2, k - 1)[:k]
    idx = idx[np.argsort(dist2[idx])]
    return idx, dist2[idx]


def vanilla_knn_single_query_int(query_int, ref_int, k):
    """
    整数域 KNN
    query_int: [3]
    ref_int:   [N, 3]
    """
    diff = ref_int - query_int[None, :]
    dist2 = np.sum(diff * diff, axis=1, dtype=np.int64)
    k = min(k, len(dist2))
    idx = np.argpartition(dist2, k - 1)[:k]
    idx = idx[np.argsort(dist2[idx])]
    return idx, dist2[idx]


# ============================================================
# 2) Bit helpers
# ============================================================
def leading_bit_pos(v: int):
    """
    require v > 0
    return floor(log2(v))
    """
    return v.bit_length() - 1


def msb_pos_of_int(v: int):
    """
    正整数 v 的 MSB 位置；若 v <= 0 返回 -1
    """
    if v <= 0:
        return -1
    return v.bit_length() - 1


# ============================================================
# 3) Strict SAFE lower-bound version
#    严格 safe 的前提：
#      - lower bound 在整数域构造
#      - 真实距离阈值也在整数域维护
#      - discard 判定在同一整数域完成
# ============================================================
def mss_lower_bound_sq(d: int):
    """
    给定一维整数差值 d = |q-p| >= 0，
    返回由 MSS 构造的 d^2 严格下界。

    若 d = 0，则下界为 0
    若 MSS = 10...，则 d >= 2^p  -> d^2 >= 2^(2p)
    若 MSS = 11...，则 d >= 3*2^(p-1) -> d^2 >= (3*2^(p-1))^2
    """
    if d == 0:
        return 0

    p = leading_bit_pos(d)
    second = (d >> (p - 1)) & 1 if p >= 1 else 0

    if second == 0:
        # MSS = 10...
        return 1 << (2 * p)
    else:
        # MSS = 11...
        base = 3 << (p - 1)
        return base * base


def precompare_lower_bound_sq_strict(q_int, p_int):
    """
    严格 safe 的 3D 整数平方距离下界。
    因为
        D_int^2 = dx^2 + dy^2 + dz^2 >= max(dx^2, dy^2, dz^2)
    所以 max(各维平方下界) 是整个 D_int^2 的合法下界。
    """
    dx = abs(int(q_int[0]) - int(p_int[0]))
    dy = abs(int(q_int[1]) - int(p_int[1]))
    dz = abs(int(q_int[2]) - int(p_int[2]))

    lbx = mss_lower_bound_sq(dx)
    lby = mss_lower_bound_sq(dy)
    lbz = mss_lower_bound_sq(dz)

    return max(lbx, lby, lbz)


def exact_int_dist2(q_int, p_int):
    dx = int(q_int[0]) - int(p_int[0])
    dy = int(q_int[1]) - int(p_int[1])
    dz = int(q_int[2]) - int(p_int[2])
    return dx * dx + dy * dy + dz * dz


def precomp_knn_single_query_safe_strict(q_int, ref_int, k):
    """
    对“整数域 KNN”严格无损的 safe 版本。

    返回:
      idx_topk, dist2_topk_int, full_compute_cnt, filtered_cnt
    """
    N = ref_int.shape[0]
    k = min(k, N)

    # 最大堆: (-dist2_int, idx)
    heap = []
    full_compute_cnt = 0
    filtered_cnt = 0

    # warmup：前 k 个点无条件精算
    for i in range(k):
        dist2_int = exact_int_dist2(q_int, ref_int[i])
        heapq.heappush(heap, (-dist2_int, i))
        full_compute_cnt += 1

    if N <= k:
        items = sorted([(-d, i) for d, i in heap])
        idx = np.array([i for _, i in items], dtype=np.int64)
        dist2 = np.array([d for d, _ in items], dtype=np.int64)
        return idx, dist2, full_compute_cnt, filtered_cnt

    tau_k_int = -heap[0][0]  # 当前第 k 近邻的整数真实平方距离

    for i in range(k, N):
        lb_int = precompare_lower_bound_sq_strict(q_int, ref_int[i])

        # 严格 safe discard：同一整数域比较
        if lb_int >= tau_k_int:
            filtered_cnt += 1
            continue

        dist2_int = exact_int_dist2(q_int, ref_int[i])
        full_compute_cnt += 1

        if dist2_int < tau_k_int:
            heapq.heapreplace(heap, (-dist2_int, i))
            tau_k_int = -heap[0][0]

    items = sorted([(-d, i) for d, i in heap])
    idx = np.array([i for _, i in items], dtype=np.int64)
    dist2 = np.array([d for d, _ in items], dtype=np.int64)
    return idx, dist2, full_compute_cnt, filtered_cnt


# ============================================================
# 4) FastNN-style approximate pre-comparator
#    尽量贴近论文思路：
#      - 由 MSS 生成 P(MSS)
#      - b = 当前第k近真实距离平方的 MSB
#      - if P(MSS) < b: full distance
#        else discard
#
#    注意：论文短文未完全展开 xyz 聚合，这里用 max(Px,Py,Pz) 作为实现补全
# ============================================================
def fastnn_p_mss_1d(d: int):
    """
    论文风格 1D P(MSS):
      MSS = 10 -> 2p - 1
      MSS = 11 -> 2p
    """
    if d == 0:
        return -10**9  # 极小分数，保证更容易通过

    p = leading_bit_pos(d)
    second = (d >> (p - 1)) & 1 if p >= 1 else 0

    if second == 0:
        return 2 * p - 1
    else:
        return 2 * p


def fastnn_precompare_score_xyz(q_int, p_int):
    dx = abs(int(q_int[0]) - int(p_int[0]))
    dy = abs(int(q_int[1]) - int(p_int[1]))
    dz = abs(int(q_int[2]) - int(p_int[2]))

    sx = fastnn_p_mss_1d(dx)
    sy = fastnn_p_mss_1d(dy)
    sz = fastnn_p_mss_1d(dz)

    return max(sx, sy, sz)


def precomp_knn_single_query_fastnn(q_int, ref_int, k):
    """
    FastNN 风格近似预比较器，完全在整数域里做真实距离与阈值维护。
    """
    N = ref_int.shape[0]
    k = min(k, N)

    heap = []
    full_compute_cnt = 0
    filtered_cnt = 0

    # warmup
    for i in range(k):
        dist2_int = exact_int_dist2(q_int, ref_int[i])
        heapq.heappush(heap, (-dist2_int, i))
        full_compute_cnt += 1

    if N <= k:
        items = sorted([(-d, i) for d, i in heap])
        idx = np.array([i for _, i in items], dtype=np.int64)
        dist2 = np.array([d for d, _ in items], dtype=np.int64)
        return idx, dist2, full_compute_cnt, filtered_cnt

    tau_k_int = -heap[0][0]
    b = msb_pos_of_int(tau_k_int)

    for i in range(k, N):
        p_score = fastnn_precompare_score_xyz(q_int, ref_int[i])

        # 论文风格比较：若 P(MSS) >= b，则直接丢弃
        if p_score >= b:
            filtered_cnt += 1
            continue

        dist2_int = exact_int_dist2(q_int, ref_int[i])
        full_compute_cnt += 1

        if dist2_int < tau_k_int:
            heapq.heapreplace(heap, (-dist2_int, i))
            tau_k_int = -heap[0][0]
            b = msb_pos_of_int(tau_k_int)

    items = sorted([(-d, i) for d, i in heap])
    idx = np.array([i for _, i in items], dtype=np.int64)
    dist2 = np.array([d for d, _ in items], dtype=np.int64)
    return idx, dist2, full_compute_cnt, filtered_cnt


# ============================================================
# 5) Metrics
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
# 5b) E2LSH index: FastNN-faithful two-level E2LSH
# ============================================================
class E2LSHTwoLevelIndex:
    """
    FastNN-faithful-ish two-level E2LSH index.

    Per table:
      1) compute a K-dimensional primitive E2LSH code h(v)
      2) compress h(v) into (H1, H2)

    Storage:
      tables[l][slot][fp] -> list(point_idx)
    """

    def __init__(
        self,
        dim=3,
        L=8,                  # number of hash tables
        K=4,                  # number of primitive hashes per table
        w=0.25,               # bucket width
        table_size=4096,      # top-level size
        C=(2**32 - 5),        # paper-suggested large prime on 32-bit
        seed=42,
    ):
        self.dim = dim
        self.L = L
        self.K = K
        self.w = float(w)
        self.table_size = int(table_size)
        self.C = int(C)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # per-table primitive E2LSH params
        self.A_list = []   # [L][K, dim]
        self.B_list = []   # [L][K]

        # per-table H1 / H2 random integers
        self.R1_list = []  # [L][K]
        self.R2_list = []  # [L][K]

        # tables[l][slot][fp] -> list(indices)
        self.tables = []

        self.ref_xyz = None

    def _primitive_hash_code(self, x: np.ndarray, A: np.ndarray, B: np.ndarray):
        """
        x: [dim]
        A: [K, dim]
        B: [K]
        return h: [K] int64
        """
        proj = A @ x
        h = np.floor((proj + B) / self.w).astype(np.int64)
        return h

    def _two_level_keys(self, h: np.ndarray, r1: np.ndarray, r2: np.ndarray):
        """
        H1(h) = [(r · h) mod C] mod tableSize
        H2(h) = (r' · h) mod C
        """
        dot1 = 0
        dot2 = 0
        for i in range(len(h)):
            dot1 += int(r1[i]) * int(h[i])
            dot2 += int(r2[i]) * int(h[i])

        h1 = (dot1 % self.C) % self.table_size
        h2 = dot2 % self.C
        return int(h1), int(h2)

    def build(self, ref_xyz: np.ndarray):
        """
        ref_xyz: [N, dim], float
        """
        self.ref_xyz = ref_xyz
        N, dim = ref_xyz.shape
        assert dim == self.dim

        self.A_list = []
        self.B_list = []
        self.R1_list = []
        self.R2_list = []
        self.tables = []

        for _ in range(self.L):
            # primitive E2LSH parameters for one table
            A = self.rng.normal(loc=0.0, scale=1.0, size=(self.K, self.dim))
            B = self.rng.uniform(0.0, self.w, size=(self.K,))

            # random integers for H1 / H2
            R1 = self.rng.integers(1, self.C, size=(self.K,), dtype=np.int64)
            R2 = self.rng.integers(1, self.C, size=(self.K,), dtype=np.int64)

            # two-level table
            table = defaultdict(lambda: defaultdict(list))

            for i in range(N):
                h = self._primitive_hash_code(ref_xyz[i], A, B)
                slot, fp = self._two_level_keys(h, R1, R2)
                table[slot][fp].append(i)

            self.A_list.append(A)
            self.B_list.append(B)
            self.R1_list.append(R1)
            self.R2_list.append(R2)
            self.tables.append(table)

        return self

    def _query_single_table(self, query_xyz: np.ndarray, l: int):
        """
        Return exact bucket match candidates from one table.
        """
        A = self.A_list[l]
        B = self.B_list[l]
        R1 = self.R1_list[l]
        R2 = self.R2_list[l]
        table = self.tables[l]

        hq = self._primitive_hash_code(query_xyz, A, B)
        slot, fp = self._two_level_keys(hq, R1, R2)

        if slot not in table:
            return np.empty((0,), dtype=np.int64)
        if fp not in table[slot]:
            return np.empty((0,), dtype=np.int64)

        return np.array(table[slot][fp], dtype=np.int64)

    def get_candidates_intersection(self, query_xyz: np.ndarray):
        """
        Faithful mode: intersection across L tables.
        This matches the paper text most closely.
        """
        sets = []
        for l in range(self.L):
            idx = self._query_single_table(query_xyz, l)
            sets.append(set(idx.tolist()))

        if len(sets) == 0:
            return np.empty((0,), dtype=np.int64)

        cand = sets[0]
        for s in sets[1:]:
            cand = cand.intersection(s)

        return np.array(sorted(cand), dtype=np.int64)

    def get_candidates_union(self, query_xyz: np.ndarray):
        """
        Engineering-friendly relaxed mode.
        """
        cand = set()
        for l in range(self.L):
            idx = self._query_single_table(query_xyz, l)
            cand.update(idx.tolist())
        return np.array(sorted(cand), dtype=np.int64)

    def get_candidates_min_hits(self, query_xyz: np.ndarray, min_hits=2):
        """
        Relaxed mode between union and intersection.
        """
        counter = Counter()
        for l in range(self.L):
            idx = self._query_single_table(query_xyz, l)
            for p in idx.tolist():
                counter[p] += 1
        cand = [p for p, c in counter.items() if c >= min_hits]
        return np.array(sorted(cand), dtype=np.int64)


# 兼容旧名（若有外部脚本仍引用 E2LSHIndex）
E2LSHIndex = E2LSHTwoLevelIndex


# ============================================================
# 5c) KNN on candidates only（包装已有 KNN，限定在候选子集）
# ============================================================
def knn_on_candidates_float(query_xyz, ref_xyz, candidate_idx, k):
    if len(candidate_idx) == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float64)
    sub_ref = ref_xyz[candidate_idx]
    local_idx, dist = vanilla_knn_single_query_float(
        query_xyz, sub_ref, min(k, len(sub_ref))
    )
    return candidate_idx[local_idx], dist


def knn_on_candidates_int(query_int, ref_int, candidate_idx, k):
    if len(candidate_idx) == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    sub_ref = ref_int[candidate_idx]
    local_idx, dist = vanilla_knn_single_query_int(
        query_int, sub_ref, min(k, len(sub_ref))
    )
    return candidate_idx[local_idx], dist


def knn_on_candidates_safe_strict(query_int, ref_int, candidate_idx, k):
    if len(candidate_idx) == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64), 0, 0
    sub_ref = ref_int[candidate_idx]
    local_idx, dist, full_cnt, filt_cnt = precomp_knn_single_query_safe_strict(
        query_int, sub_ref, min(k, len(sub_ref))
    )
    return candidate_idx[local_idx], dist, full_cnt, filt_cnt


def knn_on_candidates_fastnn(query_int, ref_int, candidate_idx, k):
    if len(candidate_idx) == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64), 0, 0
    sub_ref = ref_int[candidate_idx]
    local_idx, dist, full_cnt, filt_cnt = precomp_knn_single_query_fastnn(
        query_int, sub_ref, min(k, len(sub_ref))
    )
    return candidate_idx[local_idx], dist, full_cnt, filt_cnt


def pad_or_keep_topk(idx, k):
    """候选不足 k 时只返回实际个数；评估时可单独统计 shortage。"""
    return idx[: min(k, len(idx))]


# ============================================================
# 5d) E2LSH + KNN 统一评测
# ============================================================
def evaluate_knn_module_with_e2lsh(
    queries_xyz,
    refs_xyz,
    queries_int,
    refs_int,
    k,
    e2lsh_index: E2LSHIndex,
    candidate_mode="union",
    min_hits=2,
    search_mode="fastnn",
    gt_mode="float_vanilla",
):
    """
    candidate_mode: union / intersection / min_hits
    search_mode: int_vanilla / safe_strict / fastnn
    gt_mode: float_vanilla / int_vanilla
    返回含 shortage_query_ratio、avg_candidate_ratio 等统计。
    """
    ious = []
    recalls = []
    full_counts = []
    filtered_counts = []
    candidate_sizes = []
    shortage_queries = 0

    for qi in range(queries_xyz.shape[0]):
        q_xyz = queries_xyz[qi]
        q_int = queries_int[qi]

        if gt_mode == "float_vanilla":
            gt_idx, _ = vanilla_knn_single_query_float(q_xyz, refs_xyz, k)
        elif gt_mode == "int_vanilla":
            gt_idx, _ = vanilla_knn_single_query_int(q_int, refs_int, k)
        else:
            raise ValueError(f"Unsupported gt_mode={gt_mode}")

        if candidate_mode == "union":
            cand_idx = e2lsh_index.get_candidates_union(q_xyz)
        elif candidate_mode == "intersection":
            cand_idx = e2lsh_index.get_candidates_intersection(q_xyz)
        elif candidate_mode == "min_hits":
            cand_idx = e2lsh_index.get_candidates_min_hits(q_xyz, min_hits=min_hits)
        else:
            raise ValueError(f"Unsupported candidate_mode={candidate_mode}")

        candidate_sizes.append(len(cand_idx))
        if len(cand_idx) < k:
            shortage_queries += 1

        if search_mode == "int_vanilla":
            pd_idx, _ = knn_on_candidates_int(q_int, refs_int, cand_idx, k)
            full_cnt = len(cand_idx)
            filt_cnt = 0
        elif search_mode == "safe_strict":
            pd_idx, _, full_cnt, filt_cnt = knn_on_candidates_safe_strict(
                q_int, refs_int, cand_idx, k
            )
        elif search_mode == "fastnn":
            pd_idx, _, full_cnt, filt_cnt = knn_on_candidates_fastnn(
                q_int, refs_int, cand_idx, k
            )
        else:
            raise ValueError(f"Unsupported search_mode={search_mode}")

        ious.append(knn_set_iou(gt_idx, pd_idx))
        recalls.append(knn_recall_at_k(gt_idx, pd_idx, k))
        full_counts.append(full_cnt)
        filtered_counts.append(filt_cnt)

    n_ref = refs_xyz.shape[0]
    n_q = queries_xyz.shape[0]
    return {
        "mean_iou": float(np.mean(ious)),
        "mean_recall": float(np.mean(recalls)),
        "std_iou": float(np.std(ious)),
        "std_recall": float(np.std(recalls)),
        "avg_full_compute": float(np.mean(full_counts)),
        "avg_filtered": float(np.mean(filtered_counts)),
        "avg_candidate_size": float(np.mean(candidate_sizes)),
        "avg_candidate_ratio": float(np.mean(np.array(candidate_sizes, dtype=np.float64) / n_ref)),
        "shortage_query_ratio": float(shortage_queries / n_q) if n_q else 0.0,
        "ious": ious,
        "recalls": recalls,
    }


# ============================================================
# 6) Unified evaluator
# ============================================================
def evaluate_knn_module(
    queries_xyz,
    refs_xyz,
    queries_int,
    refs_int,
    k,
    mode="safe_strict",
    gt_mode="float_vanilla",
):
    """
    mode:
      - float_vanilla
      - int_vanilla
      - safe_strict
      - fastnn

    gt_mode:
      - float_vanilla
      - int_vanilla

    返回:
      iou/recall 是相对 gt_mode 而言
    """
    ious = []
    recalls = []
    full_counts = []
    filtered_counts = []

    for qi in range(queries_xyz.shape[0]):
        # GT
        if gt_mode == "float_vanilla":
            gt_idx, _ = vanilla_knn_single_query_float(queries_xyz[qi], refs_xyz, k)
        elif gt_mode == "int_vanilla":
            gt_idx, _ = vanilla_knn_single_query_int(queries_int[qi], refs_int, k)
        else:
            raise ValueError(f"Unsupported gt_mode: {gt_mode}")

        # Pred
        if mode == "float_vanilla":
            pd_idx, _ = vanilla_knn_single_query_float(queries_xyz[qi], refs_xyz, k)
            full_cnt = refs_xyz.shape[0]
            filt_cnt = 0
        elif mode == "int_vanilla":
            pd_idx, _ = vanilla_knn_single_query_int(queries_int[qi], refs_int, k)
            full_cnt = refs_int.shape[0]
            filt_cnt = 0
        elif mode == "safe_strict":
            pd_idx, _, full_cnt, filt_cnt = precomp_knn_single_query_safe_strict(
                queries_int[qi], refs_int, k
            )
        elif mode == "fastnn":
            pd_idx, _, full_cnt, filt_cnt = precomp_knn_single_query_fastnn(
                queries_int[qi], refs_int, k
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


# ============================================================
# 7) Random demo
# ============================================================
def run_random_demo(
    n_ref=2000,
    n_queries=100,
    k=16,
    coord_scale=1000.0,
    seed=42,
):
    rng = np.random.default_rng(seed)
    ref_xyz = rng.uniform(-1.0, 1.0, (n_ref, 3)).astype(np.float64)
    queries_xyz = rng.uniform(-1.0, 1.0, (n_queries, 3)).astype(np.float64)

    ref_int = quantize_points_fixed_scale(ref_xyz, coord_scale)
    query_int = quantize_points_fixed_scale(queries_xyz, coord_scale)

    res_float_vs_int = evaluate_knn_module(
        queries_xyz, ref_xyz, query_int, ref_int, k,
        mode="int_vanilla", gt_mode="float_vanilla"
    )

    res_int_vs_safe = evaluate_knn_module(
        queries_xyz, ref_xyz, query_int, ref_int, k,
        mode="safe_strict", gt_mode="int_vanilla"
    )

    res_float_vs_fastnn = evaluate_knn_module(
        queries_xyz, ref_xyz, query_int, ref_int, k,
        mode="fastnn", gt_mode="float_vanilla"
    )

    return {
        "float_vs_int": res_float_vs_int,
        "int_vs_safe": res_int_vs_safe,
        "float_vs_fastnn": res_float_vs_fastnn,
    }


if __name__ == "__main__":
    stats = run_random_demo()
    for name, res in stats.items():
        print(f"\n[{name}]")
        print(f"  Mean IoU      : {res['mean_iou']:.6f}")
        print(f"  Mean Recall@k : {res['mean_recall']:.6f}")
        print(f"  Avg full comp : {res['avg_full_compute']:.1f}")
        print(f"  Filter ratio  : {res['avg_filter_ratio']*100:.2f}%")
