#!/usr/bin/env bash
# ShapeNet-Part 部件分割。示例:
#   bash openpoints_mini/scripts/train_shapenet_part.sh --backbone pointvector --part_in_channels 6
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
exec python -m openpoints_mini.scripts.train --task partseg "$@"
