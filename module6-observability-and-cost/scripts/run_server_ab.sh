#!/usr/bin/env bash
# Real server-side A/B: each arm uses different vllm serve flags (you run those on the GPU).
#
# sequential — one port: restart vLLM + gateway between arms; compare in Grafana by server_profile.
# parallel   — two vLLM processes (two ports + tunnels); set VLLM_BACKEND_MAP_JSON; compare by technique.
#
# Setup: cp scripts/ab_arms.example.sh scripts/ab_arms.sh && edit hints/ports/profiles
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

CONF="$ROOT/scripts/ab_arms.sh"
if [[ ! -f "$CONF" ]]; then
  echo "Missing $CONF — copy the template:"
  echo "  cp scripts/ab_arms.example.sh scripts/ab_arms.sh"
  exit 1
fi
# shellcheck disable=SC1090
source "$CONF"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

MODE="${1:-}"
if [[ "$MODE" != "sequential" && "$MODE" != "seq" && "$MODE" != "parallel" && "$MODE" != "dual" ]]; then
  echo "Usage: $0 sequential|parallel"
  echo ""
  echo "  sequential  One vLLM port. For each arm: start vLLM with the printed flags, set"
  echo "              VLLM_SERVER_PROFILE in .env, restart python gateway.py, press Enter, then Crew runs."
  echo "  parallel    Two vLLM ports + VLLM_BACKEND_MAP_JSON (see below). Gateway restarted once;"
  echo "              script runs crew once per arm technique (hits different URLs)."
  exit 1
fi

if ! [[ "${AB_ARMS_COUNT:-0}" =~ ^[1-9][0-9]*$ ]]; then
  echo "AB_ARMS_COUNT must be a positive integer in scripts/ab_arms.sh"
  exit 1
fi

run_crew() {
  local tech="$1"
  echo "========== crew --technique $tech =========="
  python crew.py --technique "$tech" || echo "crew failed (gateway / vLLM / tunnel?)"
  sleep 2
}

if [[ "$MODE" == "parallel" || "$MODE" == "dual" ]]; then
  raw="${VLLM_BACKEND_MAP_JSON:-}"
  if [[ -z "$raw" ]]; then
    echo "parallel mode needs VLLM_BACKEND_MAP_JSON in .env, e.g.:"
    map_s=""
    for i in $(seq 1 "$AB_ARMS_COUNT"); do
      eval "k=\$AB_ARM_${i}_TECHNIQUE"
      p=$((7999 + i))
      [[ -n "$map_s" ]] && map_s+=","
      map_s+="\"$k\":\"http://127.0.0.1:$p\""
    done
    echo "  {$map_s}"
    echo "Match tunnel forwards to each vLLM process on the GPU. Restart gateway after editing .env."
    exit 1
  fi
  echo "=== Server A/B (parallel) ==="
  echo "VLLM_BACKEND_MAP_JSON is set. Restart gateway if you just changed it."
  echo "Ensure each arm's vLLM is running on the port implied by your map."
  read -r -p "Press Enter when both servers are up and gateway restarted..."
  for i in $(seq 1 "$AB_ARMS_COUNT"); do
    eval "tech=\$AB_ARM_${i}_TECHNIQUE"
    run_crew "$tech"
  done
  echo ""
  echo "Done. In Grafana, split by technique (each maps to a different vLLM / flag set)."
  exit 0
fi

# --- sequential ---
echo "=== Server A/B (sequential) ==="
echo "For each arm you will:"
echo "  1) Stop previous vLLM on Lambda (if any) and start it with the shown command."
echo "  2) Confirm SSH tunnel points at the right port."
echo "  3) Set VLLM_SERVER_PROFILE=<arm profile> in .env, restart python gateway.py (labels metrics)."
echo "  4) Press Enter here to run the same Crew workload."
echo ""

for i in $(seq 1 "$AB_ARMS_COUNT"); do
  eval "prof=\$AB_ARM_${i}_SERVER_PROFILE"
  eval "hint=\$AB_ARM_${i}_HINT"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "ARM $i / $AB_ARMS_COUNT  —  use VLLM_SERVER_PROFILE=$prof on the gateway"
  echo ""
  echo "Example vLLM command (edit in ab_arms.sh for your setup):"
  echo "$hint"
  echo ""
  read -r -p "When vLLM + tunnel are up and gateway restarted with VLLM_SERVER_PROFILE=$prof, press Enter..."
  run_crew "${AB_CREW_TECHNIQUE}"
done

echo ""
echo "Done. In Grafana, compare series by server_profile ($AB_ARMS_COUNT arms)."
