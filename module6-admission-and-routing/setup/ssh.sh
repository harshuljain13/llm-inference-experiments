#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  source "$ROOT/.env"
  set +a
fi

if [[ -z "${LAMBDA:-}" ]]; then
  echo "Set LAMBDA in $ROOT/.env" >&2
  exit 1
fi
if [[ -z "${LAMBDA_SSH_KEY:-}" || ! -f "$LAMBDA_SSH_KEY" ]]; then
  echo "Set LAMBDA_SSH_KEY in $ROOT/.env to a real private key file" >&2
  exit 1
fi

exec ssh -i "$LAMBDA_SSH_KEY" -o StrictHostKeyChecking=accept-new -t "$LAMBDA" 'mkdir -p ~/module6-admission-and-routing; cd ~/module6-admission-and-routing; exec bash -l'
