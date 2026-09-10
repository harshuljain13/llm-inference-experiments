#!/usr/bin/env bash
# Start Jupyter on the GPU instance (bind localhost — use SSH -L from your Mac).
set -euo pipefail

CLASS5_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CLASS5_ROOT"
if [[ -f "$CLASS5_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$CLASS5_ROOT/.env"
  set +a
fi

if [[ ! -f "$CLASS5_ROOT/.venv/bin/activate" ]]; then
  echo "No .venv found — run one-time setup first:" >&2
  echo "  cd $CLASS5_ROOT && bash scripts/lambda_setup.sh" >&2
  exit 1
fi

source .venv/bin/activate
export PYTHONPATH="$CLASS5_ROOT"

PORT="${JUPYTER_PORT:-8888}"

# Stop stray notebook servers (avoids wrong port / wrong token confusion)
jupyter notebook stop "$PORT" 2>/dev/null || true
pkill -f "jupyter-notebook" 2>/dev/null || true
pkill -f "jupyter notebook" 2>/dev/null || true
pkill -f "jupyter-lab" 2>/dev/null || true
pkill -f "jupyter lab" 2>/dev/null || true
sleep 1

echo "Starting Jupyter on 127.0.0.1:$PORT (no login — safe via SSH tunnel only)"
echo ""
echo "On your Mac (new terminal), tunnel with:"
if [[ -n "${LAMBDA:-}" ]]; then
  key_hint=""
  if [[ -n "${LAMBDA_SSH_KEY:-}" ]]; then
    key_hint=" -i $LAMBDA_SSH_KEY"
  fi
  echo "  ssh${key_hint} -L ${PORT}:127.0.0.1:${PORT} ${LAMBDA}"
else
  echo "  ssh -L ${PORT}:127.0.0.1:${PORT} YOUR_USER@YOUR_LAMBDA_IP"
fi
echo ""
echo "Then open (no token needed): http://127.0.0.1:${PORT}/"
echo "If Mac already runs Jupyter on ${PORT}, use: ssh ... -L 8889:127.0.0.1:${PORT}  →  http://127.0.0.1:8889/"
echo ""

exec jupyter notebook \
  --no-browser \
  --ip=127.0.0.1 \
  --port="$PORT" \
  --ServerApp.token='' \
  --ServerApp.password='' \
  --NotebookApp.token='' \
  --NotebookApp.password='' \
  --ServerApp.allow_origin='*' \
  --NotebookApp.allow_origin='*' \
  class5.ipynb
