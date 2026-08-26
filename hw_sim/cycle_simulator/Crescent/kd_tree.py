from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np


@dataclass
class KDNode:
    idx: int
    point_ids: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    level: int
    split_dim: Optional[int] = None
    split_val: Optional[float] = None
    left: Optional[int] = None
    right: Optional[int] = None
    is_leaf: bool = False
    subtree_root: Optional[int] = None


class KDTree:
    def __init__(self, xyz: np.ndarray, leaf_size: int = 16):
        self.xyz = xyz
        self.leaf_size = leaf_size
        self.nodes: List[KDNode] = []
        ids = np.arange(len(xyz))
        self.root = self._build(ids, level=0)

    def _bbox(self, ids: np.ndarray):
        pts = self.xyz[ids]
        return pts.min(axis=0), pts.max(axis=0)

    def _build(self, ids: np.ndarray, level: int) -> int:
        idx = len(self.nodes)
        bbox_min, bbox_max = self._bbox(ids)
        node = KDNode(idx=idx, point_ids=ids, bbox_min=bbox_min, bbox_max=bbox_max, level=level)
        self.nodes.append(node)
        if len(ids) <= self.leaf_size:
            node.is_leaf = True
            return idx
        spreads = bbox_max - bbox_min
        dim = int(np.argmax(spreads))
        order = ids[np.argsort(self.xyz[ids, dim])]
        mid = len(order) // 2
        left_ids, right_ids = order[:mid], order[mid:]
        split_val = float(self.xyz[order[mid], dim])
        node.split_dim = dim
        node.split_val = split_val
        node.left = self._build(left_ids, level + 1)
        node.right = self._build(right_ids, level + 1)
        return idx

    def collect_subtree_roots(self, top_tree_height: int) -> List[int]:
        roots = []
        def dfs(nid: int):
            n = self.nodes[nid]
            if n.level == top_tree_height:
                roots.append(nid)
                return
            if n.is_leaf:
                roots.append(nid)
                return
            dfs(n.left)
            dfs(n.right)
        dfs(self.root)
        return roots

    def assign_subtree_ids(self, top_tree_height: int) -> Dict[int, int]:
        roots = self.collect_subtree_roots(top_tree_height)
        mapping = {r: i for i, r in enumerate(roots)}
        def mark(nid: int, root_id: int):
            n = self.nodes[nid]
            n.subtree_root = root_id
            if n.is_leaf:
                return
            mark(n.left, root_id)
            mark(n.right, root_id)
        for r in roots:
            mark(r, r)
        return mapping

    def linearize_subtree(self, root_id: int) -> List[int]:
        out = []
        def dfs(nid: int):
            out.append(nid)
            n = self.nodes[nid]
            if n.is_leaf:
                return
            dfs(n.left)
            dfs(n.right)
        dfs(root_id)
        return out


def dist2_point_aabb(q: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> float:
    delta = np.maximum(0.0, np.maximum(bmin - q, q - bmax))
    return float(np.dot(delta, delta))
