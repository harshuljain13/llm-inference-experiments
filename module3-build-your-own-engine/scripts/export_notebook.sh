#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -d .venv ]] && source .venv/bin/activate
export PYTHONPATH=$PWD
python -c "
from lib.export_notebook import save_pdf
from pathlib import Path
p = save_pdf()
print('Saved', Path(p).resolve())
"