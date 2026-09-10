#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  set -a
  source "$ROOT/.env"
  set +a
fi
if [[ -x "$ROOT/.venv/bin/vllm" ]]; then
  export PATH="$ROOT/.venv/bin:$PATH"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

MODEL=${MODEL:-Qwen/Qwen3-0.6B}

if ! command -v nvidia-smi &>/dev/null; then
  echo "launch_replicas.sh runs on the Lambda GPU, not the Mac." >&2
  echo "On your Mac:  cd class-code/class7 && bash setup/sync_to_lambda.sh && bash setup/ssh.sh" >&2
  exit 1
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm not on PATH. On Lambda:  bash setup/lambda_setup.sh && source .venv/bin/activate"
  exit 1
fi

echo "starting two replicas of $MODEL  (max-num-seqs=8, gpu-memory-utilization=0.35)"

# Start one replica in the background, log to its own file, wait for /v1/models.
# Sequential, not concurrent: two engines profiling GPU memory at the same time
# race each other and one dies at engine-core init.
start_replica() {
  local port=$1
  local log=/tmp/llm-gateway-lab-$port.log

  vllm serve "$MODEL" --port "$port" \
    --gpu-memory-utilization 0.35 \
    --max-num-seqs 8 \
    --max-model-len 16384 \
    --scheduling-policy priority \
    --served-model-name lab >"$log" 2>&1 &
  local pid=$!
  echo "$pid" > /tmp/llm-gateway-lab-$port.pid
  echo "  :$port  pid $pid  log $log  — waiting (model download + compile can take minutes)"

  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      echo "  :$port  ready"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "  :$port  DIED during startup — last 40 lines of $log:" >&2
      tail -40 "$log" >&2
      return 1
    fi
    sleep 2
  done

  echo "  :$port  not ready after 6 minutes — see $log" >&2
  return 1
}

start_replica 8001
start_replica 8002

echo "replicas ready on :8001 and :8002"
echo "next:  bash setup/smoke_test.sh"
