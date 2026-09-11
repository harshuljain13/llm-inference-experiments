#!/usr/bin/env bash
# Explicitly turn off chunked prefill + prefix caching (use if your vLLM build defaults them on
# and you want a strict control arm). If flags are unknown on your version, use baseline.sh only.
# https://docs.vllm.ai/en/stable/configuration/engine_args/ — CacheConfig / SchedulerConfig
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=/dev/null
source ./_common.sh
exec vllm serve "$VLLM_MODEL" \
  --served-model-name "$VLLM_SERVED_NAME" \
  --host "$VLLM_BIND_HOST" \
  --port "$VLLM_SERVE_PORT" \
  --no-enable-chunked-prefill \
  --no-enable-prefix-caching
