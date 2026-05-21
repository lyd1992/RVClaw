#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

MODEL_PATH="${1:-$RVCLAW_LLAMA_MODEL_PATH}"
SERVER_BIN="$LLAMA_CPP_HOME/bin/llama-server"

if [ ! -x "$SERVER_BIN" ]; then
  echo "llama-server not found or not executable: $SERVER_BIN" >&2
  echo "Install spacemit-llama.cpp first; see deploy/k3/install.md." >&2
  exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
  echo "model file not found: $MODEL_PATH" >&2
  echo "Download a GGUF planner model first; see deploy/k3/install.md." >&2
  exit 1
fi

exec "$SERVER_BIN" \
  --host "$RVCLAW_LLAMA_HOST" \
  --port "$RVCLAW_LLAMA_PORT" \
  -m "$MODEL_PATH" \
  --threads "$RVCLAW_LLAMA_THREADS" \
  --ctx-size "$RVCLAW_LLAMA_CTX_SIZE" \
  --batch-size "$RVCLAW_LLAMA_BATCH_SIZE" \
  --metrics \
  --no-mmap
