from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from kd_tree import KDTree, dist2_point_aabb
from memory import SRAMBuffer, DRAMModel, AccessReq


@dataclass
class QueryResult:
    query_id: int
    neighbors: List[int]
    aggregated_feature: np.ndarray


class ANSBallQuerySim:
    def __init__(self, xyz: np.ndarray, feat: np.ndarray, tree: KDTree,
                 top_tree_height: int, radius: float, max_neighbors: int,
                 tree_buf: SRAMBuffer, feat_buf: SRAMBuffer, index_buf: SRAMBuffer,
                 dram: DRAMModel, enable_bce_tree: bool, enable_bce_feat: bool,
                 elision_height_tree: int, elision_height_feat: int,
                 aggregate_mode: str = 'mean', fill_missing_with_zero: bool = True):
        self.xyz = xyz
        self.feat = feat
        self.tree = tree
        self.top_tree_height = top_tree_height
        self.radius2 = float(radius * radius)
        self.max_neighbors = max_neighbors
        self.tree_buf = tree_buf
        self.feat_buf = feat_buf
        self.index_buf = index_buf
        self.dram = dram
        self.enable_bce_tree = enable_bce_tree
        self.enable_bce_feat = enable_bce_feat
        self.elision_height_tree = elision_height_tree
        self.elision_height_feat = elision_height_feat
        self.aggregate_mode = aggregate_mode
        self.fill_missing_with_zero = fill_missing_with_zero

        self.subtree_root_to_sid = self.tree.assign_subtree_ids(top_tree_height)
        self.subtree_linear_nodes = {
            sid: self.tree.linearize_subtree(root_id)
            for root_id, sid in self.subtree_root_to_sid.items()
        }
        self.node_addr = {nid: nid * 64 for nid in range(len(tree.nodes))}
        self.feat_base_addr = 1 << 20
        self.feat_stride = feat.shape[1] * 4
        self.index_base_addr = 1 << 24
        self.cycles = 0
        self.tree_node_visits = 0
        self.feature_reads_served = 0

    def _load_subtree_streaming(self, sid: int):
        node_count = len(self.subtree_linear_nodes[sid])
        self.dram.stream_read(node_count * 64)
        self.cycles += max(1, node_count // 4)

    def _descend_top_tree(self, q: np.ndarray) -> int:
        nid = self.tree.root
        while True:
            node = self.tree.nodes[nid]
            req = AccessReq(cycle=self.cycles, requester='PE0', addr=self.node_addr[nid], size=64,
                            stage='FN_TOP', tree_level=node.level, tag=f'node{nid}')
            served, stalled, elided = self.tree_buf.process_cycle([req], self.enable_bce_tree, self.elision_height_tree)
            if served:
                self.tree_buf.account_read(1)
                self.tree_node_visits += 1
            elif elided:
                # top-tree elision means query cannot be reliably routed; fallback to current subtree/root approximation
                return self.tree.subtree_root_to_sid.get(node.subtree_root if node.subtree_root is not None else nid, 0)
            self.cycles += 1 + len(stalled)
            if node.level == self.top_tree_height or node.is_leaf:
                root_id = node.subtree_root if node.subtree_root is not None else nid
                return self.subtree_root_to_sid[root_id]
            if q[node.split_dim] <= node.split_val:
                nid = node.left
            else:
                nid = node.right

    def stage1_assign_queries(self, query_ids: np.ndarray, queries: np.ndarray):
        queues: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
        for qid, q in zip(query_ids, queries):
            sid = self._descend_top_tree(q)
            queues[sid].append((int(qid), q.copy()))
        return queues

    def _ball_query_subtree(self, qid: int, q: np.ndarray, sid: int) -> List[int]:
        root_id = None
        for rid, s in self.subtree_root_to_sid.items():
            if s == sid:
                root_id = rid
                break
        assert root_id is not None
        out = []
        stack = [root_id]
        while stack:
            nid = stack.pop()
            node = self.tree.nodes[nid]
            req = AccessReq(cycle=self.cycles, requester=f'Q{qid}', addr=self.node_addr[nid], size=64,
                            stage='FN_SUB', tree_level=node.level, tag=f'node{nid}')
            served, stalled, elided = self.tree_buf.process_cycle([req], self.enable_bce_tree, self.elision_height_tree)
            if served:
                self.tree_buf.account_read(1)
                self.tree_node_visits += 1
            elif elided:
                self.cycles += 1
                continue
            self.cycles += 1 + len(stalled)

            if dist2_point_aabb(q, node.bbox_min, node.bbox_max) > self.radius2:
                continue
            if node.is_leaf:
                pts = self.xyz[node.point_ids]
                d2 = np.sum((pts - q[None, :]) ** 2, axis=1)
                inside = node.point_ids[d2 <= self.radius2]
                for pid in inside.tolist():
                    if pid != qid:
                        out.append(int(pid))
                        if len(out) >= self.max_neighbors:
                            return out
            else:
                # heuristic order near child first
                left_d = dist2_point_aabb(q, self.tree.nodes[node.left].bbox_min, self.tree.nodes[node.left].bbox_max)
                right_d = dist2_point_aabb(q, self.tree.nodes[node.right].bbox_min, self.tree.nodes[node.right].bbox_max)
                if left_d < right_d:
                    stack.append(node.right)
                    stack.append(node.left)
                else:
                    stack.append(node.left)
                    stack.append(node.right)
        return out[:self.max_neighbors]

    def _gather_and_aggregate(self, qid: int, neighbors: List[int]) -> np.ndarray:
        # neighbor index writeback
        if neighbors:
            self.index_buf.account_write(len(neighbors))
        gathered = []
        pending = deque(neighbors)
        while pending:
            # issue one cycle of feature reads; multiple addresses may conflict
            reqs = []
            batch = []
            while pending:
                pid = pending.popleft()
                batch.append(pid)
                addr = self.feat_base_addr + pid * self.feat_stride
                reqs.append(AccessReq(cycle=self.cycles, requester=f'Q{qid}', addr=addr, size=self.feat_stride,
                                      stage='GATHER', tree_level=9999, tag=f'pt{pid}'))
                # keep same-cycle issue reasonable
                if len(reqs) >= 8:
                    break
            served, stalled, elided = self.feat_buf.process_cycle(reqs, self.enable_bce_feat, self.elision_height_feat)
            served_tags = {r.tag for r in served}
            stalled_tags = {r.tag for r in stalled}
            elided_tags = {r.tag for r in elided}
            for pid in batch:
                tag = f'pt{pid}'
                if tag in served_tags:
                    self.feat_buf.account_read(1)
                    self.feature_reads_served += 1
                    gathered.append(self.feat[pid])
                elif tag in stalled_tags:
                    pending.appendleft(pid)
                elif tag in elided_tags:
                    if self.fill_missing_with_zero:
                        gathered.append(np.zeros((self.feat.shape[1],), dtype=np.float32))
            self.cycles += 1
        if len(gathered) == 0:
            return np.zeros((self.feat.shape[1],), dtype=np.float32)
        G = np.stack(gathered, axis=0)
        if self.aggregate_mode == 'max':
            return G.max(axis=0)
        return G.mean(axis=0)

    def run(self, query_ids: np.ndarray, queries: np.ndarray) -> List[QueryResult]:
        queues = self.stage1_assign_queries(query_ids, queries)
        results = []
        for sid in sorted(queues.keys()):
            self._load_subtree_streaming(sid)
            for qid, q in queues[sid]:
                neigh = self._ball_query_subtree(qid, q, sid)
                aggr = self._gather_and_aggregate(qid, neigh)
                results.append(QueryResult(query_id=qid, neighbors=neigh, aggregated_feature=aggr))
        return results


def exact_ball_query_and_aggregate(xyz: np.ndarray, feat: np.ndarray, query_ids: np.ndarray, queries: np.ndarray,
                                   radius: float, max_neighbors: int, aggregate_mode: str = 'mean'):
    r2 = radius * radius
    out = []
    for qid, q in zip(query_ids, queries):
        d2 = np.sum((xyz - q[None, :]) ** 2, axis=1)
        neigh = [int(i) for i in np.where(d2 <= r2)[0].tolist() if int(i) != int(qid)]
        neigh = neigh[:max_neighbors]
        if len(neigh) == 0:
            aggr = np.zeros((feat.shape[1],), dtype=np.float32)
        else:
            G = feat[neigh]
            aggr = G.max(axis=0) if aggregate_mode == 'max' else G.mean(axis=0)
        out.append(QueryResult(query_id=int(qid), neighbors=neigh, aggregated_feature=aggr))
    return out
