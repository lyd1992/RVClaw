#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RVCLAW_ROOT="${RVCLAW_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "$RVCLAW_ROOT"

export RVCLAW_GCC15_IMAGE="${RVCLAW_GCC15_IMAGE:-rvclaw/sg2044-gcc15.1:openeuler-riscv64}"
export RVCLAW_MNN_REF="${RVCLAW_MNN_REF:-master}"
export RVCLAW_MNN_BUILD_DIR="${RVCLAW_MNN_BUILD_DIR:-build/mnn-sg2044-gcc15}"

echo "[RVClaw] image: $RVCLAW_GCC15_IMAGE"
echo "[RVClaw] MNN ref: $RVCLAW_MNN_REF"
echo "[RVClaw] MNN build dir: $RVCLAW_MNN_BUILD_DIR"

bash docker/sg2044/mnn/run_container.sh bash docker/sg2044/mnn/build_mnn.sh
