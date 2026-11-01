#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HERE/scripts/common.sh"
# strict protocol: official val early-stop/selection (p15 = patience 15)
run shumi --stage strict --protocol pv1 --epochs 25 --schedule-epochs 25 --select best --patience 15 \
  --tuning full --head linear --norm batch --accumulate 6 \
  --encoder-lr 1e-5 --adapter-lr 1e-5 --renderer-lr 1e-5 --lr 1e-5
