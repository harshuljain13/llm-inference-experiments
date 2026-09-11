# shellcheck shell=bash
# Shared defaults for vllm serve wrappers. Source-only (do not execute directly).
#
# Engine / scheduler flags reference:
#   https://docs.vllm.ai/en/stable/configuration/engine_args/
#
# Override any variable when invoking, e.g.:
#   VLLM_SERVE_PORT=8001 bash scripts/vllm_engine/chunked_prefill.sh
#
# Beam search: there is no global "beam" engine flag in typical OpenAI-mode serving.
# It is applied per request (this repo: crew.py --technique beam_search → gateway injects
# use_beam_search / best_of). A/B beam vs greedy using the same vLLM process: run Crew twice.
#
# Run these on the GPU host (e.g. Lambda), not the laptop. Per-technique engine selection from
# Crew/gateway uses multiple vLLM ports (engine flags are not HTTP fields): run_engine_fleet.sh
# + VLLM_AUTO_ENGINE_ROUTING=1 — see README Step 6 / Step 17.

: "${VLLM_MODEL:=TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
: "${VLLM_SERVED_NAME:=texttinyllama}"
: "${VLLM_BIND_HOST:=0.0.0.0}"
: "${VLLM_SERVE_PORT:=8000}"
