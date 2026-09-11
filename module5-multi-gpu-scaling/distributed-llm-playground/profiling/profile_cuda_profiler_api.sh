
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PLAYGROUND_PRESET="${PLAYGROUND_PRESET:-tiny_single}"
export PLAYGROUND_MODEL_KEY="${PLAYGROUND_MODEL_KEY:-tinyllama}"

OUT="${NSYS_REPORT_BASENAME:-profile_cuda_api}"

exec nsys profile \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --cuda-graph-trace=node \
  --trace-fork-before-exec=true \
  --capture-range=cudaProfilerApi \
  --output "$OUT" \
  --force-overwrite true \
  -- \
  serve run serve_config.yaml
