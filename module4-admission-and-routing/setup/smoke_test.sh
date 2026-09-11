#!/usr/bin/env bash
set -u

P1=${P1:-8001}
P2=${P2:-8002}
fail=0

green() { printf '\033[32m%s\033[0m\n' "$1"; }
red()   { printf '\033[31m%s\033[0m\n' "$1"; }

check_models() {
  local port=$1
  if curl -sf "http://127.0.0.1:${port}/v1/models" | grep -q '"id"'; then
    green "PASS  :${port} /v1/models"
  else
    red   "FAIL  :${port} /v1/models"
    fail=1
  fi
}

check_metrics() {
  local port=$1
  if curl -sf "http://127.0.0.1:${port}/metrics" | grep -q 'vllm:num_requests_waiting'; then
    green "PASS  :${port} /metrics has vllm:num_requests_waiting"
  else
    red   "FAIL  :${port} /metrics missing vllm:num_requests_waiting"
    fail=1
  fi
}

check_completion() {
  local port=$1
  local body
  body=$(curl -sf "http://127.0.0.1:${port}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"lab","messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":8}')
  if echo "$body" | grep -q 'choices'; then
    green "PASS  :${port} /v1/chat/completions"
  else
    red   "FAIL  :${port} /v1/chat/completions"
    fail=1
  fi
}

for p in "$P1" "$P2"; do
  check_models "$p"
  check_metrics "$p"
  check_completion "$p"
done

if [[ "$fail" -eq 0 ]]; then
  green "SMOKE PASS"
  exit 0
fi
red "SMOKE FAIL"
exit 1
