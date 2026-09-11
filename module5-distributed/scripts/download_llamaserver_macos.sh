#!/usr/bin/env bash

set -euo pipefail
PLAYGROUND_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${PLAYGROUND_ROOT}/third_party/llama.cpp-bin"
mkdir -p "$DEST"
cd "$DEST"

MACHINE="$(uname -m)"
if [[ -n "${LLAMACPP_MACOS_SUFFIX:-}" ]]; then
  SUFFIX="$LLAMACPP_MACOS_SUFFIX"
elif [[ "$MACHINE" == "arm64" ]]; then
  SUFFIX="macos-arm64"
else
  SUFFIX="macos-x64"
fi

echo "Querying GitHub API for latest release (suffix=$SUFFIX) ..." >&2

JSON="$(
  curl -fsSL \
    -H 'Accept: application/vnd.github+json' \
    -H 'User-Agent: ray-project-llm-fetch' \
    'https://api.github.com/repos/ggml-org/llama.cpp/releases/latest'
)" || {
  echo "curl failed (network or GitHub). Try again or download manually from:" >&2
  echo "  https://github.com/ggml-org/llama.cpp/releases" >&2
  exit 1
}

META="$(printf '%s' "$JSON" | python3 -c "
import json, sys
suffix = sys.argv[1]
try:
    j = json.load(sys.stdin)
except json.JSONDecodeError as e:
    print('Invalid JSON from GitHub API:', e, file=sys.stderr)
    sys.exit(1)
assets = j.get('assets') or []
cands = [
    a for a in assets
    if suffix in a.get('name', '')
    and 'bin' in a.get('name', '')
    and a['name'].endswith('.tar.gz')
]
if not cands:
    print('No matching .tar.gz for', repr(suffix), file=sys.stderr)
    print('Sample asset names:', [a.get('name') for a in assets[:15]], file=sys.stderr)
    sys.exit(1)
a = cands[0]
sys.stdout.write(a['browser_download_url'] + chr(9) + a['name'])
" "$SUFFIX")" || {
  echo "Could not select a release asset." >&2
  exit 1
}

IFS=$'\t' read -r URL NAME <<<"$META" || true
if [[ -z "${URL:-}" || -z "${NAME:-}" ]]; then
  echo "Internal error: empty URL or filename after parsing." >&2
  exit 1
fi

echo "Downloading $NAME ..." >&2
curl -fL -o "$NAME" "$URL"
tar xzf "$NAME"
SERVER="$(find "$DEST" -type f -name llama-server 2>/dev/null | head -1)"
if [[ -z "$SERVER" ]]; then
  echo "Extracted archive but could not find llama-server under $DEST" >&2
  exit 1
fi
chmod +x "$SERVER"
echo ""
echo "llama-server is at:"
echo "  $SERVER"
echo ""
echo "Run (replace GGUF path):"
echo "  \"$SERVER\" -m /path/to/your/model.gguf --host 127.0.0.1 --port 8080"
