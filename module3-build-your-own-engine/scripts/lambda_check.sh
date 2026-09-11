#!/usr/bin/env bash
# Quick GPU/Lambda readiness check — run on Mac (deps only) or on the Lambda instance.
set -euo pipefail

CLASS5_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CLASS5_ROOT"

if [[ -f "$CLASS5_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$CLASS5_ROOT/.env"
  set +a
fi

if [[ -d "$CLASS5_ROOT/.venv" ]]; then
  # shellcheck disable=SC1091
  source "$CLASS5_ROOT/.venv/bin/activate"
fi

export PYTHONPATH="$CLASS5_ROOT"

echo "Host: $(hostname)"
echo "PWD:  $CLASS5_ROOT"
echo ""

python -c "from lib.gpu_check import print_gpu_report; print_gpu_report()"
