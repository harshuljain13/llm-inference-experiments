#!/usr/bin/env bash
# Same CrewAI prompts/tasks every time; only the --technique label (and optional server profile) changes.
#
# This does NOT turn on chunked_prefill / speculative decoding / etc. on the server — it only sets
# X-Technique. To A/B real vLLM flags (Eagle on/off, two draft methods, …), change vllm serve
# (and VLLM_SERVER_PROFILE) between arms or use VLLM_BACKEND_MAP_JSON to two ports — see README Step 17.
#
# Lambda (single vLLM process): mostly labels gateway metrics unless you restart vLLM with different
#   flags per arm, use another port/instance, or use VLLM_BACKEND_MAP_JSON for multiple URLs.
#
# Client-only: beam_search — gateway injects use_beam_search on the body.
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

TECHS=(
  baseline
  chunked_prefill
  prefix_caching
  speculative_decoding
  beam_search
  combined
)

echo "=== Comparison run: same crew for each --technique label ==="
echo "For fair server-side compares, vLLM flags must match each arm (see README Step 17)."
echo ""

for t in "${TECHS[@]}"; do
  echo "========== crew --technique $t =========="
  echo "Metrics label technique=${t} (X-Technique header). server_profile comes from VLLM_SERVER_PROFILE in .env only."
  python crew.py --technique "$t" || echo "crew failed for $t (check gateway, .env, tunnel, vLLM)"
  sleep 2
done

echo ""
echo "Done. Grafana/Prometheus hints:"
echo "  - Chunked prefill: lower prefill/TTFT vs baseline on long contexts."
echo "  - Speculative: higher decode tok/s + acceptance metrics; often lower cost per token."
echo "  - Prefix caching: repeat identical-prefix runs → prefill drops."
echo "  - Beam: higher decode latency; compare output quality manually."
echo "  - Combined: label only (same as other techniques) unless you map it to another vLLM URL; vLLM spec_decode Grafana panel needs spec enabled in vllm serve, not this header."
