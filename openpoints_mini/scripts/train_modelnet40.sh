#!/usr/bin/env bash
# ModelNet40 分类。示例:
#   bash openpoints_mini/scripts/train_modelnet40.sh --backbone pointnext --epochs 50
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
exec python -m openpoints_mini.scripts.train --task cls "$@"
