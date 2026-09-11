#!/usr/bin/env bash
# Baseline vLLM: same minimal flags as README Step 6 — rely on vLLM defaults for the rest.
# Engine args catalog: https://docs.vllm.ai/en/stable/configuration/engine_args/
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=/dev/null
source ./_common.sh
exec vllm serve "$VLLM_MODEL" \
  --served-model-name "$VLLM_SERVED_NAME" \
  --host "$VLLM_BIND_HOST" \
  --port "$VLLM_SERVE_PORT"
