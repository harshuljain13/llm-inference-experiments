#!/usr/bin/env bash
# One-time setup on a Lambda (or any CUDA) GPU instance.
set -euo pipefail

CLASS5_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CLASS5_ROOT"
if [[ -f "$CLASS5_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$CLASS5_ROOT/.env"
  set +a
fi

echo "== GPU check =="
if ! command -v nvidia-smi &>/dev/null; then
  echo "WARNING: nvidia-smi not found — Exp 5 (real model) needs a GPU."
else
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

echo "== Python venv =="
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel

echo "== Base deps (smol-vllm + jupyter + crewai) =="
pip install -r requirements.txt

echo "== GPU extras (torch + transformers for TinyLlama / Exp 5) =="
pip install -e "./smol-vllm[tinyllama-1.1b]"

echo "== PDF export (nbconvert + playwright) =="
pip install nbconvert playwright
playwright install chromium || echo "  Run: bash scripts/lambda_pdf_deps.sh if PDF export fails"

echo "== Verify =="
export PYTHONPATH="$CLASS5_ROOT"
python -c "
import torch
from smol_vllm import LLMEngine
print('smol_vllm OK')
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
"

echo ""
echo "Setup done. Next:"
echo "  source .venv/bin/activate && export PYTHONPATH=$CLASS5_ROOT"
echo "  bash scripts/lambda_jupyter.sh"
