#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PLAYGROUND_PRESET="${PLAYGROUND_PRESET:-tiny_single}"
export PLAYGROUND_MODEL_KEY="${PLAYGROUND_MODEL_KEY:-tinyllama}"

OUT="${NSYS_REPORT_BASENAME:-profile_single_gpu}"

exec nsys profile \
  --trace=cuda,nvtx,osrt,cublas,cudnn,nvlink \
  --cuda-graph-trace=node \
  --trace-fork-before-exec=true \
  --capture-range=none \
  --output "$OUT" \
  --force-overwrite true \
  -- \
  serve run serve_config.yaml
