from __future__ import annotations
from typing import Dict, Any
import numpy as np

from config import SimConfig
from data import generate_point_cloud, sample_queries
from kd_tree import KDTree
from memory import SRAMBuffer, DRAMModel
from ans import ANSBallQuerySim, exact_ball_query_and_aggregate
from metrics import recall_at_radius, l2_feature_error


def build_buffers(cfg: SimConfig):
    tree_buf = SRAMBuffer(
        'tree', cfg.tree_buf_capacity_bytes, cfg.tree_buf_num_banks,
        cfg.tree_buf_bank_width_bytes, cfg.tree_buf_ports_per_bank,
        cfg.tree_buf_access_latency_cycles, cfg.tree_buf_read_energy_pj,
        cfg.tree_buf_write_energy_pj,
    )
    feat_buf = SRAMBuffer(
        'feat', cfg.feat_buf_capacity_bytes, cfg.feat_buf_num_banks,
        cfg.feat_buf_bank_width_bytes, cfg.feat_buf_ports_per_bank,
        cfg.feat_buf_access_latency_cycles, cfg.feat_buf_read_energy_pj,
        cfg.feat_buf_write_energy_pj,
    )
    index_buf = SRAMBuffer(
        'index', cfg.index_buf_capacity_bytes, cfg.index_buf_num_banks,
        cfg.index_buf_bank_width_bytes, cfg.index_buf_ports_per_bank,
        cfg.index_buf_access_latency_cycles, cfg.index_buf_read_energy_pj,
        cfg.index_buf_write_energy_pj,
    )
    dram = DRAMModel(cfg.dram_burst_bytes, cfg.dram_read_energy_pj_per_burst, cfg.dram_write_energy_pj_per_burst)
    return tree_buf, feat_buf, index_buf, dram


def run_once(cfg: SimConfig) -> Dict[str, Any]:
    xyz, feat = generate_point_cloud(cfg.num_points, cfg.feat_dim, seed=cfg.seed)
    qids, queries = sample_queries(xyz, cfg.num_queries, seed=cfg.seed + 1)
    tree = KDTree(xyz, leaf_size=cfg.leaf_size)

    gt = exact_ball_query_and_aggregate(xyz, feat, qids, queries, cfg.radius, cfg.max_neighbors, cfg.aggregate_mode)
    gt_neighbors = [r.neighbors for r in gt]
    gt_aggr = np.stack([r.aggregated_feature for r in gt], axis=0)

    tree_buf, feat_buf, index_buf, dram = build_buffers(cfg)
    sim = ANSBallQuerySim(
        xyz, feat, tree, cfg.top_tree_height, cfg.radius, cfg.max_neighbors,
        tree_buf, feat_buf, index_buf, dram,
        cfg.enable_bce_tree, cfg.enable_bce_feat,
        cfg.elision_height_tree, cfg.elision_height_feat,
        cfg.aggregate_mode, cfg.fill_missing_with_zero,
    )
    pred = sim.run(qids, queries)
    pred_neighbors = [r.neighbors for r in pred]
    pred_aggr = np.stack([r.aggregated_feature for r in pred], axis=0)

    return {
        'cycles': sim.cycles,
        'tree_node_visits': sim.tree_node_visits,
        'feature_reads_served': sim.feature_reads_served,
        'recall_ball_query': recall_at_radius(gt_neighbors, pred_neighbors),
        'aggregation_l2_error': l2_feature_error(gt_aggr, pred_aggr),
        'tree_buf': tree_buf.stats.__dict__,
        'feat_buf': feat_buf.stats.__dict__,
        'index_buf': index_buf.stats.__dict__,
        'dram': dram.stats.__dict__,
        'num_queries': cfg.num_queries,
        'num_points': cfg.num_points,
    }
