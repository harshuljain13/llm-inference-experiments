#!/usr/bin/env bash
# One-time on Lambda: Chromium + system libs for PDF export (webpdf).
set -euo pipefail

CLASS5_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CLASS5_ROOT"
source .venv/bin/activate

pip install -q nbconvert playwright
playwright install chromium

if command -v sudo &>/dev/null; then
  echo "Installing Chromium system dependencies (needs sudo)..."
  sudo "$(which python)" -m playwright install-deps chromium || true
fi

echo "PDF export ready — run last notebook cell or: bash scripts/export_notebook.sh"
