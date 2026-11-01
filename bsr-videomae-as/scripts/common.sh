#!/usr/bin/env bash
# Shared launcher for the four public benchmarks.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
CKPT="${CKPT:-$HERE/ckpts/S2B2A_c70ceb7e.pt}"
SPLITS="${SPLITS:-$HERE/manifests/splits_dirs}"
LOCK="${LOCK:-$HERE/manifests/final_v3_lock.json}"
run() {
  local corpus="$1"; shift
  "$PY" "$HERE/finetune.py" \
    --ckpt "$CKPT" \
    --splits-root "$SPLITS" \
    --lock "$LOCK" \
    --corpus "$corpus" --subjects all \
    --seed 4523 --batch 24 \
    --output "$HERE/outputs/$corpus" \
    "$@"
}
