from __future__ import annotations
import numpy as np


def recall_at_radius(gt_neighbors, pred_neighbors):
    recalls = []
    for g, p in zip(gt_neighbors, pred_neighbors):
        gs = set(g)
        ps = set(p)
        if len(gs) == 0:
            recalls.append(1.0)
        else:
            recalls.append(len(gs & ps) / max(1, len(gs)))
    return float(np.mean(recalls))


def l2_feature_error(gt_aggr: np.ndarray, pred_aggr: np.ndarray):
    diff = gt_aggr - pred_aggr
    return float(np.mean(np.linalg.norm(diff, axis=1)))
