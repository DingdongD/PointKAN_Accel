import numpy as np
import heapq
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ============================================================
# Utilities
# ============================================================

def l2_sq(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum((a - b) ** 2))


def pairwise_dist_sq(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # X: (N, D), y: (D,)
    diff = X - y[None, :]
    return np.sum(diff * diff, axis=1)


def bbox_lower_bound_sq(q: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> float:
    """
    Lower bound of squared L2 distance from q to an axis-aligned bounding box.
    """
    clamped = np.maximum(bbox_min, np.minimum(q, bbox_max))
    return l2_sq(q, clamped)


# ============================================================
# Tree Node
# ============================================================

@dataclass
class TreeNode:
    node_id: int
    point_indices: np.ndarray
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    is_leaf: bool = False

    centroid: Optional[np.ndarray] = None
    radius: float = 0.0
    bbox_min: Optional[np.ndarray] = None
    bbox_max: Optional[np.ndarray] = None

    max_md: float = -1.0
    argmax_idx: int = -1


# ============================================================
# Point Summary Tree
# ============================================================

class PointSummaryTree:
    def __init__(self, X: np.ndarray, leaf_size: int = 32):
        """
        X: (N, D)
        """
        assert X.ndim == 2
        self.X = X.astype(np.float32)
        self.N, self.D = X.shape
        self.leaf_size = leaf_size
        self._next_node_id = 0

        indices = np.arange(self.N, dtype=np.int64)
        self.root = self._build(indices)

    # ----------------------------
    # Tree Construction
    # ----------------------------
    def _new_node_id(self) -> int:
        nid = self._next_node_id
        self._next_node_id += 1
        return nid

    def _build(self, indices: np.ndarray) -> TreeNode:
        node = TreeNode(node_id=self._new_node_id(), point_indices=indices)

        pts = self.X[indices]
        node.centroid = np.mean(pts, axis=0)
        node.bbox_min = np.min(pts, axis=0)
        node.bbox_max = np.max(pts, axis=0)
        node.radius = float(np.sqrt(np.max(np.sum((pts - node.centroid[None, :]) ** 2, axis=1))))

        if len(indices) <= self.leaf_size:
            node.is_leaf = True
            return node

        # split by longest bbox axis
        spans = node.bbox_max - node.bbox_min
        axis = int(np.argmax(spans))
        sorted_idx = indices[np.argsort(self.X[indices, axis], kind="mergesort")]
        mid = len(sorted_idx) // 2

        left_idx = sorted_idx[:mid]
        right_idx = sorted_idx[mid:]

        node.left = self._build(left_idx)
        node.right = self._build(right_idx)
        node.is_leaf = False
        return node

    # ----------------------------
    # FPS Summary Refresh
    # ----------------------------
    def refresh_fps_summaries(self, min_dists: np.ndarray):
        """
        min_dists: (N,), current min distance (squared) to selected centers
        """
        self._refresh_fps_summaries(self.root, min_dists)

    def _refresh_fps_summaries(self, node: TreeNode, min_dists: np.ndarray):
        if node.is_leaf:
            local_md = min_dists[node.point_indices]
            local_pos = int(np.argmax(local_md))
            node.max_md = float(local_md[local_pos])
            node.argmax_idx = int(node.point_indices[local_pos])
            return

        self._refresh_fps_summaries(node.left, min_dists)
        self._refresh_fps_summaries(node.right, min_dists)

        if node.left.max_md >= node.right.max_md:
            node.max_md = node.left.max_md
            node.argmax_idx = node.left.argmax_idx
        else:
            node.max_md = node.right.max_md
            node.argmax_idx = node.right.argmax_idx

    # ----------------------------
    # FPS Winner Selection
    # ----------------------------
    def select_fps_winner(self) -> Tuple[int, float]:
        """
        Return global argmax point index and its max min-dist.
        """
        return self.root.argmax_idx, self.root.max_md

    # ----------------------------
    # KNN lower bound
    # ----------------------------
    def lower_bound_sq(self, node: TreeNode, q: np.ndarray) -> float:
        return bbox_lower_bound_sq(q, node.bbox_min, node.bbox_max)

    # ----------------------------
    # Incremental FPS update
    # ----------------------------
    def update_fps_with_center(
        self,
        center: np.ndarray,
        min_dists: np.ndarray,
        conservative: bool = True,
    ) -> dict:
        """
        Merkle-style incremental update:
        traverse tree, prune subtrees whose lower bound >= node.max_md
        update only touched leaves

        Returns stats dict.
        """
 
        stats = {
        "visited_nodes": 0,
        "pruned_nodes": 0,
        "touched_leaves": 0,
        "updated_points": 0,
        "distance_evals": 0,
        }
        self._update_fps_with_center(self.root, center, min_dists, stats, conservative)
        return stats

    def _update_fps_with_center(
        self,
        node: TreeNode,
        center: np.ndarray,
        min_dists: np.ndarray,
        stats: dict,
        conservative: bool,
    ):
        stats["visited_nodes"] += 1

        lb = self.lower_bound_sq(node, center)

        # prune rule:
        # if lower bound to entire subtree is already >= subtree max min-dist,
        # then new center cannot reduce the winner in this subtree; conservative but useful
        if lb >= node.max_md:
            stats["pruned_nodes"] += 1
            return

        if node.is_leaf:
            stats["touched_leaves"] += 1
            pts = self.X[node.point_indices]

            # count exact point-wise distance evaluations in this leaf
            stats["distance_evals"] += len(node.point_indices)

            d2 = np.sum((pts - center[None, :]) ** 2, axis=1)
            old_vals = min_dists[node.point_indices].copy()
            min_dists[node.point_indices] = np.minimum(min_dists[node.point_indices], d2)
            changed = np.sum(min_dists[node.point_indices] < old_vals)
            stats["updated_points"] += int(changed)

            local_md = min_dists[node.point_indices]
            local_pos = int(np.argmax(local_md))
            node.max_md = float(local_md[local_pos])
            node.argmax_idx = int(node.point_indices[local_pos])
            return

        self._update_fps_with_center(node.left, center, min_dists, stats, conservative)
        self._update_fps_with_center(node.right, center, min_dists, stats, conservative)

        if node.left.max_md >= node.right.max_md:
            node.max_md = node.left.max_md
            node.argmax_idx = node.left.argmax_idx
        else:
            node.max_md = node.right.max_md
            node.argmax_idx = node.right.argmax_idx

    # ----------------------------
    # Tree KNN
    # ----------------------------
    def tree_knn(self, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Exact-ish branch-and-bound KNN using bbox lower bound.
        Returns:
            idx: (k,)
            dist2: (k,)
            stats
        """
        stats = {
            "visited_nodes": 0,
            "pruned_nodes": 0,
            "touched_leaves": 0,
            "distance_evals": 0,
        }

        # max-heap for current top-k using negative distance not needed;
        # we'll store (-dist, idx) so heap[0] is worst among current best
        best_heap: List[Tuple[float, int]] = []

        def tau() -> float:
            if len(best_heap) < k:
                return np.inf
            return -best_heap[0][0]

        # priority queue by lower bound (best-first)
        pq = []
        root_lb = self.lower_bound_sq(self.root, q)
        heapq.heappush(pq, (root_lb, self.root.node_id, self.root))

        while pq:
            lb, _, node = heapq.heappop(pq)
            stats["visited_nodes"] += 1

            if lb >= tau():
                stats["pruned_nodes"] += 1
                continue

            if node.is_leaf:
                stats["touched_leaves"] += 1
                pts = self.X[node.point_indices]
                d2 = np.sum((pts - q[None, :]) ** 2, axis=1)
                stats["distance_evals"] += len(node.point_indices)

                for local_i, dist2 in enumerate(d2):
                    idx = int(node.point_indices[local_i])
                    if len(best_heap) < k:
                        heapq.heappush(best_heap, (-float(dist2), idx))
                    else:
                        if float(dist2) < -best_heap[0][0]:
                            heapq.heapreplace(best_heap, (-float(dist2), idx))
                continue

            lb_l = self.lower_bound_sq(node.left, q)
            lb_r = self.lower_bound_sq(node.right, q)

            if lb_l < tau():
                heapq.heappush(pq, (lb_l, node.left.node_id, node.left))
            else:
                stats["pruned_nodes"] += 1

            if lb_r < tau():
                heapq.heappush(pq, (lb_r, node.right.node_id, node.right))
            else:
                stats["pruned_nodes"] += 1

        best = sorted([(-d, idx) for d, idx in best_heap], key=lambda x: x[0])
        dist2 = np.array([x[0] for x in best], dtype=np.float32)
        idx = np.array([x[1] for x in best], dtype=np.int64)
        return idx, dist2, stats


# ============================================================
# Exact Baselines
# ============================================================

def exact_knn(X: np.ndarray, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    d2 = pairwise_dist_sq(X, q)
    idx = np.argsort(d2)[:k]
    return idx, d2[idx]


def exact_fps(X: np.ndarray, K: int, seed_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    N = X.shape[0]
    selected = np.zeros(K, dtype=np.int64)
    min_dists = np.full(N, np.inf, dtype=np.float32)

    cur = seed_idx
    for t in range(K):
        selected[t] = cur
        center = X[cur]
        d2 = pairwise_dist_sq(X, center)
        min_dists = np.minimum(min_dists, d2)
        cur = int(np.argmax(min_dists))
    return selected, min_dists


# ============================================================
# Tree FPS
# ============================================================

def tree_fps(
    X: np.ndarray,
    K: int,
    leaf_size: int = 32,
    seed_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[dict], PointSummaryTree]:
    """
    Merkle-style FPS:
      1) build summary tree
      2) maintain min_dists
      3) use tree max summary to select next center
      4) use bound-based subtree pruning to update only affected leaves
    """
    N = X.shape[0]
    tree = PointSummaryTree(X, leaf_size=leaf_size)
    selected = np.zeros(K, dtype=np.int64)
    min_dists = np.full(N, np.inf, dtype=np.float32)

    # initialize with seed
    first_center = X[seed_idx]
    d2 = pairwise_dist_sq(X, first_center)
    min_dists = np.minimum(min_dists, d2)
    selected[0] = seed_idx

    tree.refresh_fps_summaries(min_dists)

    stats_list = [{
        "iter": 0,
        "selected_idx": int(seed_idx),
        "visited_nodes": tree._next_node_id,  # rough init refresh cost not exact
        "pruned_nodes": 0,
        "touched_leaves": 0,
        "updated_points": int(N),
    }]

    for t in range(1, K):
        idx, score = tree.select_fps_winner()
        selected[t] = idx
        center = X[idx]
        stats = tree.update_fps_with_center(center, min_dists)
        stats["iter"] = t
        stats["selected_idx"] = int(idx)
        stats["selected_score"] = float(score)
        stats_list.append(stats)

    return selected, min_dists, stats_list, tree


# ============================================================
# Evaluation Helpers
# ============================================================

def compare_fps(selected_exact: np.ndarray, selected_tree: np.ndarray) -> dict:
    set_e = set(map(int, selected_exact.tolist()))
    set_t = set(map(int, selected_tree.tolist()))
    inter = len(set_e & set_t)
    return {
        "exact_unique": len(set_e),
        "tree_unique": len(set_t),
        "intersection": inter,
        "set_recall_vs_exact": inter / max(1, len(set_e)),
    }

    
def summarize_fps_stats(stats_list: List[dict]) -> dict:
    total_visited = sum(s.get("visited_nodes", 0) for s in stats_list[1:])
    total_pruned = sum(s.get("pruned_nodes", 0) for s in stats_list[1:])
    total_touched_leaves = sum(s.get("touched_leaves", 0) for s in stats_list[1:])
    total_updated_points = sum(s.get("updated_points", 0) for s in stats_list[1:])
    total_distance_evals = sum(s.get("distance_evals", 0) for s in stats_list[1:])

    return {
        "num_iters": len(stats_list),
        "total_visited_nodes": total_visited,
        "total_pruned_nodes": total_pruned,
        "total_touched_leaves": total_touched_leaves,
        "total_updated_points": total_updated_points,
        "total_distance_evals": total_distance_evals,
        "avg_visited_nodes_per_iter": total_visited / max(1, len(stats_list) - 1),
        "avg_touched_leaves_per_iter": total_touched_leaves / max(1, len(stats_list) - 1),
        "avg_distance_evals_per_iter": total_distance_evals / max(1, len(stats_list) - 1),
    }


# ============================================================
# Demo
# ============================================================

def demo():
    np.random.seed(0)

    # synthetic clustered point cloud
    N = 1024
    D = 3
    num_clusters = 8
    pts_per_cluster = N // num_clusters

    centers = np.random.randn(num_clusters, D).astype(np.float32) * 4.0
    X_list = []
    for i in range(num_clusters):
        pts = centers[i] + 0.35 * np.random.randn(pts_per_cluster, D).astype(np.float32)
        X_list.append(pts)
    X = np.concatenate(X_list, axis=0)

    # shuffle
    perm = np.random.permutation(len(X))
    X = X[perm]

    print("=== Build / FPS ===")
    K = 64
    sel_exact, md_exact = exact_fps(X, K=K, seed_idx=0)
    sel_tree, md_tree, stats_list, tree = tree_fps(X, K=K, leaf_size=32, seed_idx=0)

    fps_cmp = compare_fps(sel_exact, sel_tree)
    fps_stats = summarize_fps_stats(stats_list)

    print("FPS compare:", fps_cmp)
    print("FPS stats:", fps_stats)

    print("\n=== KNN ===")
    q = X[123] + 0.01
    k = 16
    idx_e, d2_e = exact_knn(X, q, k)
    idx_t, d2_t, knn_stats = tree.tree_knn(q, k)

    overlap = len(set(idx_e.tolist()) & set(idx_t.tolist()))
    print("KNN overlap:", overlap, "/", k)
    print("Tree-KNN stats:", knn_stats)

    print("\nDone.")
def benchmark_many_trials(
    num_trials=50,
    N=1024,
    D=3,
    K=32,
    k=16,
    leaf_size=32,
    num_clusters=8,
    cluster_std=0.35,
):
    fps_recalls = []
    fps_update_ratios = []
    knn_overlaps = []
    knn_eval_ratios = []

    for seed in range(num_trials):
        np.random.seed(seed)
        centers = np.random.randn(num_clusters, D).astype(np.float32) * 4.0
        pts_per_cluster = N // num_clusters
        X_list = []
        for i in range(num_clusters):
            pts = centers[i] + cluster_std * np.random.randn(pts_per_cluster, D).astype(np.float32)
            X_list.append(pts)
        X = np.concatenate(X_list, axis=0)
        X = X[np.random.permutation(len(X))]

        sel_e, _ = exact_fps(X, K=K, seed_idx=0)
        sel_t, _, fps_stats_list, tree = tree_fps(X, K=K, leaf_size=leaf_size, seed_idx=0)

        cmp = compare_fps(sel_e, sel_t)
        fps_recall = cmp["set_recall_vs_exact"]

        fps_stats = summarize_fps_stats(fps_stats_list)
        fps_update_ratio = fps_stats["total_updated_points"] / ((K - 1) * N)

        q = X[np.random.randint(0, N)]
        idx_e, _ = exact_knn(X, q, k)
        idx_t, _, knn_stats = tree.tree_knn(q, k)

        knn_overlap = len(set(idx_e.tolist()) & set(idx_t.tolist())) / k
        knn_eval_ratio = knn_stats["distance_evals"] / N

        fps_recalls.append(fps_recall)
        fps_update_ratios.append(fps_update_ratio)
        knn_overlaps.append(knn_overlap)
        knn_eval_ratios.append(knn_eval_ratio)

    print("FPS recall mean/std:", np.mean(fps_recalls), np.std(fps_recalls))
    print("FPS update ratio mean/std:", np.mean(fps_update_ratios), np.std(fps_update_ratios))
    print("KNN overlap mean/std:", np.mean(knn_overlaps), np.std(knn_overlaps))
    print("KNN eval ratio mean/std:", np.mean(knn_eval_ratios), np.std(knn_eval_ratios))

if __name__ == "__main__":
    # demo()
    benchmark_many_trials()