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
  echo "Set LAMBDA in .env (cp .env.example .env)" >&2
  exit 1
fi
if [[ -z "${LAMBDA_SSH_KEY:-}" || ! -f "$LAMBDA_SSH_KEY" ]]; then
  echo "Set LAMBDA_SSH_KEY in .env to your Lambda SSH private key" >&2
  exit 1
fi

echo "Sync Mac → $LAMBDA:~/module6-admission-and-routing/"
ssh -i "$LAMBDA_SSH_KEY" -o StrictHostKeyChecking=accept-new "$LAMBDA" echo "SSH OK"

rsync -avz --progress -e "ssh -i $LAMBDA_SSH_KEY -o StrictHostKeyChecking=accept-new" \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude 'results.json' \
  --exclude 'results.html' \
  --exclude '.env' \
  ./ \
  "$LAMBDA:~/module6-admission-and-routing/"

echo ""
echo "Synced. On your Mac:"
echo "  bash setup/ssh.sh"
