#!/usr/bin/env bash
set -euo pipefail

RVCLAW_ROOT="${RVCLAW_ROOT:-/workspace/RVClaw}"
cd "$RVCLAW_ROOT"

RUN_ID="${RVCLAW_RUN_ID:-env-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$RVCLAW_ROOT/runs/env/$RUN_ID"
LOG_PATH="$RUN_DIR/mnn_build.log"
MNN_REF="${RVCLAW_MNN_REF:-master}"
BUILD_DIR="${RVCLAW_MNN_BUILD_DIR:-build/mnn-sg2044-gcc15}"
MAX_JOBS="${MAX_JOBS:-$(nproc)}"

mkdir -p "$RUN_DIR" third_party build
exec > >(tee -a "$LOG_PATH") 2>&1

echo "[RVClaw] run id: $RUN_ID"
echo "[RVClaw] root: $RVCLAW_ROOT"
echo "[RVClaw] MNN ref: $MNN_REF"
echo "[RVClaw] build dir: $BUILD_DIR"
echo "[RVClaw] CC: ${CC:-gcc}"
echo "[RVClaw] CXX: ${CXX:-g++}"

if [ ! -d third_party/MNN/.git ]; then
  git clone https://github.com/alibaba/MNN.git third_party/MNN
fi

cd third_party/MNN
git fetch --tags --all --prune
git checkout "$MNN_REF"
MNN_COMMIT="$(git rev-parse HEAD)"
cd "$RVCLAW_ROOT"

RVV_FLAGS=""
if [ "${RVCLAW_ENABLE_RVV_FLAGS:-auto}" != "off" ]; then
  echo "int main(){return 0;}" > /tmp/rvv_check.cc
  if "${CXX:-g++}" -march=rv64gcv -mabi=lp64d /tmp/rvv_check.cc -o /tmp/rvv_check; then
    RVV_FLAGS="-march=rv64gcv -mabi=lp64d"
    echo "[RVClaw] RVV flags enabled: $RVV_FLAGS"
  else
    echo "[RVClaw] RVV flags not accepted; continuing without explicit RVV flags"
  fi
fi

export CFLAGS="${CFLAGS:-} $RVV_FLAGS"
export CXXFLAGS="${CXXFLAGS:-} $RVV_FLAGS"

cmake -S third_party/MNN -B "$BUILD_DIR" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${CC:-gcc}" \
  -DCMAKE_CXX_COMPILER="${CXX:-g++}" \
  -DMNN_BUILD_SHARED_LIBS=ON \
  -DMNN_BUILD_TOOLS=ON \
  -DMNN_BUILD_BENCHMARK=ON \
  -DMNN_BUILD_TEST=OFF \
  -DMNN_BUILD_CONVERTER=OFF

cmake --build "$BUILD_DIR" -j "$MAX_JOBS"

if [ -f "$BUILD_DIR/CMakeCache.txt" ]; then
  cp "$BUILD_DIR/CMakeCache.txt" "$RUN_DIR/mnn_cmake_cache.txt"
fi

find "$BUILD_DIR" \( -name "libMNN.so" -o -name "*benchmark*" -o -name "*MNN*" \) -print | sort > "$RUN_DIR/mnn_artifacts.txt"

python3 - <<PY
import json
import os
import platform
import subprocess
from pathlib import Path

run_dir = Path("$RUN_DIR")
build_dir = Path("$BUILD_DIR")

def capture(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return str(exc)

artifacts_path = run_dir / "mnn_artifacts.txt"
artifacts = artifacts_path.read_text(encoding="utf-8").splitlines() if artifacts_path.exists() else []

manifest = {
    "run_id": "$RUN_ID",
    "host": {
        "system": platform.system(),
        "machine": platform.machine(),
        "release": platform.release(),
    },
    "toolchain": {
        "cc": os.environ.get("CC"),
        "cxx": os.environ.get("CXX"),
        "gcc_version": capture([os.environ.get("CC", "gcc"), "--version"]).splitlines()[0],
        "gxx_version": capture([os.environ.get("CXX", "g++"), "--version"]).splitlines()[0],
        "cmake_version": capture(["cmake", "--version"]).splitlines()[0],
        "ninja_version": capture(["ninja", "--version"]),
        "cflags": os.environ.get("CFLAGS", ""),
        "cxxflags": os.environ.get("CXXFLAGS", ""),
    },
    "mnn": {
        "repo": "https://github.com/alibaba/MNN.git",
        "ref": "$MNN_REF",
        "commit": "$MNN_COMMIT",
        "build_dir": str(build_dir),
        "cmake_flags": [
            "MNN_BUILD_SHARED_LIBS=ON",
            "MNN_BUILD_TOOLS=ON",
            "MNN_BUILD_BENCHMARK=ON",
            "MNN_BUILD_TEST=OFF",
            "MNN_BUILD_CONVERTER=OFF",
        ],
    },
    "artifacts": artifacts,
    "logs": {
        "build_log": "$LOG_PATH",
        "cmake_cache": str(run_dir / "mnn_cmake_cache.txt"),
        "artifacts": str(artifacts_path),
    },
    "result": "success",
}
(run_dir / "container_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
PY

echo "[RVClaw] MNN build complete"
echo "[RVClaw] manifest: $RUN_DIR/container_manifest.json"
