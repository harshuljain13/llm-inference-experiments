#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -f "$ROOT/.env" ]]; then
  set -a
  source "$ROOT/.env"
  set +a
fi

echo "== GPU check =="
if ! command -v nvidia-smi &>/dev/null; then
  echo "nvidia-smi not found. This lab needs a Lambda GPU instance." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "== Python venv =="
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt

echo "== vLLM =="
pip install "vllm>=0.28"

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true
fi

echo "== Verify =="
python -c "
import torch
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
"
.venv/bin/python -c "import vllm; print('vllm', vllm.__version__)"

echo ""
echo "Setup done. Next (still on Lambda):"
echo "  source .venv/bin/activate"
echo "  bash setup/launch_replicas.sh"
echo "  make smoke"
echo "  python -m gateway.main --replicas http://127.0.0.1:8001,http://127.0.0.1:8002"
echo "  python app.py \"What is KV cache?\""
