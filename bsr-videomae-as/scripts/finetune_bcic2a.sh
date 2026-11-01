#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HERE/scripts/common.sh"
# fixed protocol: no official val -> N19, select=last; EA inductive lambda=1e-5 tr/C^2
run bcic2a --stage fixed --protocol pv2 --epochs 19 --schedule-epochs 25 --select last \
  --tuning full --head linear --norm running --accumulate 1 \
  --encoder-lr 1e-5 --adapter-lr 1e-4 --renderer-lr 1e-4 --lr 3e-4 \
  --ea --ea-mode inductive --ea-lambda 1e-5 --ea-lambda-mode tr_over_C2
