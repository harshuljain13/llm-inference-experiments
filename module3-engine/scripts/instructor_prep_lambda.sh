#!/usr/bin/env bash
# Pre-run §7 on Lambda before class — keeps TinyLlama warm, validates GPU path.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  source .env
fi

if [[ -z "${LAMBDA:-}" || -z "${LAMBDA_SSH_KEY:-}" ]]; then
  echo "Set LAMBDA and LAMBDA_SSH_KEY in .env (see .env.example)"
  exit 1
fi

echo "== Sync module3-engine to Lambda =="
bash scripts/sync_to_lambda.sh

echo "== GPU smoke + §7 prefill/decode (remote) =="
ssh -i "$LAMBDA_SSH_KEY" -o BatchMode=yes "$LAMBDA" bash -s <<'REMOTE'
set -euo pipefail
cd ~/module3-engine
source .venv/bin/activate
export PYTHONPATH="$PWD"
python -c "
from smol_vllm import LLMEngine
from smol_vllm.demo import run_exp5_fake_vs_real, print_class3_gpu_summary, measure_kv_bytes_per_token
import torch
assert torch.cuda.is_available(), 'CUDA not available'
e = run_exp5_fake_vs_real(seed=0)
print_class3_gpu_summary(e)
measure_kv_bytes_per_token(seed=0)
print('§7 pre-run OK')
"
REMOTE

echo ""
echo "Next: T2 Lambda → bash scripts/lambda_jupyter.sh"
echo "      T3 Mac     → ssh -i \$LAMBDA_SSH_KEY -L 8889:127.0.0.1:8888 \$LAMBDA"
echo "      Browser    → http://127.0.0.1:8889/  (keep PDF speaker notes open)"
