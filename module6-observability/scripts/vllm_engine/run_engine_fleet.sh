#!/usr/bin/env bash
# One vLLM process per engine profile on fixed ports (GPU host only — e.g. Lambda).
# Port layout must match gateway.py when VLLM_AUTO_ENGINE_ROUTING=1 (offsets from VLLM_BASE_URL's port).
#
# On your laptop: set VLLM_AUTO_ENGINE_ROUTING=1, keep VLLM_BASE_URL=http://127.0.0.1:8000, and
# forward every port in the SSH tunnel (README). Crew --technique selects the upstream; engine flags
# are not sent over HTTP — each process is started with the flags below.
#
# VRAM: six TinyLlama-sized servers may not fit one GPU; stop arms you are not testing, or run sequentially.
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=/dev/null
source ./_common.sh

declare -a PIDS=()

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

launch() {
  local port=$1
  shift
  echo "Starting vLLM on :$port $*" >&2
  VLLM_SERVE_PORT=$port vllm serve "$VLLM_MODEL" \
    --served-model-name "$VLLM_SERVED_NAME" \
    --host "$VLLM_BIND_HOST" \
    --port "$port" \
    "$@" &
  PIDS+=($!)
}

launch 8000
launch 8001 --enable-chunked-prefill
launch 8002 --enable-prefix-caching
launch 8003 --enable-chunked-prefill --enable-prefix-caching
launch 8004 --no-enable-chunked-prefill --no-enable-prefix-caching

if [[ -n "${VLLM_SPECULATIVE_CONFIG_JSON:-}" ]]; then
  launch 8005 --speculative-config "$VLLM_SPECULATIVE_CONFIG_JSON"
else
  echo "Skipping :8005 (speculative_decoding): export VLLM_SPECULATIVE_CONFIG_JSON to enable." >&2
fi

echo "Fleet PIDs: ${PIDS[*]} — laptop: VLLM_AUTO_ENGINE_ROUTING=1 + multi-port tunnel (see README)." >&2
wait
