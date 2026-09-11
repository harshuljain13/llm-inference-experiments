#!/usr/bin/env bash
# Enables chunked prefill (scheduler / engine). See --enable-chunked-prefill in:
# https://docs.vllm.ai/en/stable/configuration/engine_args/#schedulerconfig
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=/dev/null
source ./_common.sh
exec vllm serve "$VLLM_MODEL" \
  --served-model-name "$VLLM_SERVED_NAME" \
  --host "$VLLM_BIND_HOST" \
  --port "$VLLM_SERVE_PORT" \
  --enable-chunked-prefill
