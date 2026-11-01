#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HERE/scripts/common.sh"
# fixed protocol: no official val -> N8, select=last
run seedvii --stage fixed --protocol pv2 --epochs 8 --schedule-epochs 25 --select last \
  --tuning full --head linear --norm running --accumulate 1 \
  --encoder-lr 1e-5 --adapter-lr 1e-4 --renderer-lr 1e-4 --lr 3e-4
