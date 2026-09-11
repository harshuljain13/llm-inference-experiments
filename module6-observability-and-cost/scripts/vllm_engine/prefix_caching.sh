#!/usr/bin/env bash
# Enables automatic prefix caching (KV). See CacheConfig:
# https://docs.vllm.ai/en/stable/configuration/engine_args/#cacheconfig
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=/dev/null
source ./_common.sh
exec vllm serve "$VLLM_MODEL" \
  --served-model-name "$VLLM_SERVED_NAME" \
  --host "$VLLM_BIND_HOST" \
  --port "$VLLM_SERVE_PORT" \
  --enable-prefix-caching
