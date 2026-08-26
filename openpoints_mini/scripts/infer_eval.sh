#!/usr/bin/env bash
# 评估检查点。示例:
#   bash openpoints_mini/scripts/infer_eval.sh --task cls --backbone pointnext --weights checkpoints/.../checkpoint_best.pth
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
exec python -m openpoints_mini.scripts.infer "$@"
