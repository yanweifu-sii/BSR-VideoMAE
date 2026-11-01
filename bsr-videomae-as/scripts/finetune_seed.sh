#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HERE/scripts/common.sh"
# strict protocol: official val early-stop/selection; EA transductive lambda=1e-3 tr/C
run seed --stage strict --protocol pv2 --epochs 16 --schedule-epochs 25 --select best --patience 5 \
  --tuning full --head linear --norm running --accumulate 1 \
  --encoder-lr 1e-5 --adapter-lr 1e-4 --renderer-lr 1e-4 --lr 3e-4 \
  --ea --ea-mode transductive --ea-lambda 1e-3 --ea-lambda-mode tr_over_C
