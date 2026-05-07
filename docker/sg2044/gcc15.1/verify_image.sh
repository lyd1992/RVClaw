#!/usr/bin/env bash
set -euo pipefail

IMAGE="${RVCLAW_GCC15_IMAGE:-rvclaw/sg2044-gcc15.1:openeuler-riscv64}"

echo "[RVClaw] Verifying $IMAGE"

docker run --rm "$IMAGE" gcc --version
docker run --rm "$IMAGE" g++ --version
docker run --rm "$IMAGE" cmake --version

docker run --rm "$IMAGE" bash -lc '
set -euo pipefail
echo "int main(){return 0;}" > /tmp/rvv_check.cc
g++ -march=rv64gcv -mabi=lp64d /tmp/rvv_check.cc -o /tmp/rvv_check
/tmp/rvv_check
echo "[RVClaw] RVV flag check passed"
'
