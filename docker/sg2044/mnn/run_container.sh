#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RVCLAW_ROOT="${RVCLAW_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
IMAGE="${RVCLAW_GCC15_IMAGE:-rvclaw/sg2044-gcc15.1:openeuler-riscv64}"

mkdir -p \
  "$RVCLAW_ROOT/third_party" \
  "$RVCLAW_ROOT/build" \
  "$RVCLAW_ROOT/runs" \
  "$RVCLAW_ROOT/.ccache"

TTY_ARGS=()
if [ -t 0 ] && [ -t 1 ]; then
  TTY_ARGS=(-it)
fi

if [ "$#" -eq 0 ]; then
  set -- bash
fi

docker run --rm "${TTY_ARGS[@]}" \
  -e RVCLAW_MNN_REF="${RVCLAW_MNN_REF:-master}" \
  -e RVCLAW_MNN_BUILD_DIR="${RVCLAW_MNN_BUILD_DIR:-build/mnn-sg2044-gcc15}" \
  -e RVCLAW_ENABLE_RVV_FLAGS="${RVCLAW_ENABLE_RVV_FLAGS:-auto}" \
  -e MAX_JOBS="${MAX_JOBS:-}" \
  -v "$RVCLAW_ROOT:/workspace/RVClaw" \
  -v "$RVCLAW_ROOT/third_party:/workspace/RVClaw/third_party" \
  -v "$RVCLAW_ROOT/build:/workspace/RVClaw/build" \
  -v "$RVCLAW_ROOT/runs:/workspace/RVClaw/runs" \
  -v "$RVCLAW_ROOT/.ccache:/workspace/.ccache" \
  -w /workspace/RVClaw \
  "$IMAGE" \
  "$@"
