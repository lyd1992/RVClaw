#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE="${RVCLAW_GCC15_IMAGE:-rvclaw/sg2044-gcc15.1:openeuler-riscv64}"
BASE_IMAGE="${RVCLAW_OPENEULER_BASE_IMAGE:-openeuler/openeuler:25.03}"
GCC_VERSION="${RVCLAW_GCC_VERSION:-15.1.0}"
MAX_JOBS="${MAX_JOBS:-32}"

echo "[RVClaw] Building $IMAGE"
echo "[RVClaw] Base image: $BASE_IMAGE"
echo "[RVClaw] GCC version: $GCC_VERSION"
echo "[RVClaw] MAX_JOBS: $MAX_JOBS"

docker build \
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t "$IMAGE" \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg GCC_VERSION="$GCC_VERSION" \
  --build-arg MAX_JOBS="$MAX_JOBS" \
  "$SCRIPT_DIR"
