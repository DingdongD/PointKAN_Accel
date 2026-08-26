from __future__ import annotations
import json
from copy import deepcopy

from config import SimConfig
from simulator import run_once


def main():
    base = SimConfig()

    no_bce = deepcopy(base)
    no_bce.enable_bce_tree = False
    no_bce.enable_bce_feat = False

    tree_only = deepcopy(base)
    tree_only.enable_bce_tree = True
    tree_only.enable_bce_feat = False

    both = deepcopy(base)
    both.enable_bce_tree = True
    both.enable_bce_feat = True

    results = {
        'no_bce': run_once(no_bce),
        'tree_bce_only': run_once(tree_only),
        'tree_and_feat_bce': run_once(both),
    }

    summary = {
        'cycle_reduction_tree_only_vs_no_bce': (results['no_bce']['cycles'] - results['tree_bce_only']['cycles']) / results['no_bce']['cycles'],
        'cycle_reduction_both_vs_no_bce': (results['no_bce']['cycles'] - results['tree_and_feat_bce']['cycles']) / results['no_bce']['cycles'],
        'recall_drop_tree_only_vs_no_bce': results['no_bce']['recall_ball_query'] - results['tree_bce_only']['recall_ball_query'],
        'recall_drop_both_vs_no_bce': results['no_bce']['recall_ball_query'] - results['tree_and_feat_bce']['recall_ball_query'],
        'aggr_error_increase_both_vs_no_bce': results['tree_and_feat_bce']['aggregation_l2_error'] - results['no_bce']['aggregation_l2_error'],
    }

    print(json.dumps({'results': results, 'summary': summary}, indent=2))


if __name__ == '__main__':
    main()
