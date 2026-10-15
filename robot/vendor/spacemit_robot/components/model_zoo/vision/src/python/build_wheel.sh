#!/usr/bin/env bash
set -euo pipefail

WHEEL_ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$WHEEL_ROOT/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
AUTO_BUILD_EXT=0
if [[ "${1:-}" == "--build-ext" ]]; then
  AUTO_BUILD_EXT=1
  shift
fi
BUILD_DIR="${1:-$REPO_ROOT/build}"
PKG_DIR="$WHEEL_ROOT/spacemit_vision"

if [[ "${BUILD_DIR}" == -D* ]]; then
  echo "error: '$BUILD_DIR' looks like a CMake definition, not a build directory."
  echo "usage:"
  echo "  ./build_wheel.sh                      # use default build dir: $REPO_ROOT/build"
  echo "  ./build_wheel.sh /path/to/build       # custom build dir"
  echo ""
  echo "build C++ first (repo root):"
  echo "  cmake -S \"$REPO_ROOT\" -B \"$REPO_ROOT/build\" -DBUILD_PYTHON_BINDINGS=ON"
  echo "  cmake --build \"$REPO_ROOT/build\" -j"
  exit 2
fi

echo "[1/4] Using build dir: $BUILD_DIR"
echo "[1/4] Using python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
echo "[1/4] Python version: $("$PYTHON_BIN" -c 'import sys; print(sys.version.split()[0])')"

find_extension_so() {
  find "$1/python" -maxdepth 1 -type f -name '_vision_service_cpp*.so' -print -quit 2>/dev/null || true
}

EXT_SO="$(find_extension_so "$BUILD_DIR")"
if [[ -z "${EXT_SO}" ]]; then
  if [[ "$AUTO_BUILD_EXT" -eq 1 ]]; then
    echo "[1/4] Extension not found, building extension via root CMake (BUILD_PYTHON_BINDINGS=ON) ..."
    cmake -S "$REPO_ROOT" -B "$BUILD_DIR" \
      -DBUILD_PYTHON_BINDINGS=ON \
      -DPython3_EXECUTABLE="$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
    cmake --build "$BUILD_DIR" --target _vision_service_cpp -j
    EXT_SO="$(find_extension_so "$BUILD_DIR")"
  fi
  if [[ -z "${EXT_SO}" ]]; then
    echo "error: missing $BUILD_DIR/python/_vision_service_cpp*.so"
    echo "hint: build the extension via root CMake:"
    echo "    cmake -S \"$REPO_ROOT\" -B \"$BUILD_DIR\" -DBUILD_PYTHON_BINDINGS=ON"
    echo "    cmake --build \"$BUILD_DIR\" -j"
    echo "  or let this script do it:"
    echo "    ./build_wheel.sh --build-ext \"$BUILD_DIR\""
    exit 1
  fi
fi

LIBVISION_SO=""
if [[ -f "$BUILD_DIR/libvision.so" ]]; then
  LIBVISION_SO="$BUILD_DIR/libvision.so"
elif [[ -f "$BUILD_DIR/src/libvision.so" ]]; then
  LIBVISION_SO="$BUILD_DIR/src/libvision.so"
else
  echo "error: missing libvision.so in $BUILD_DIR or $BUILD_DIR/src"
  exit 1
fi

echo "[2/4] Copying binaries into package"
echo "      EXT_SO: $EXT_SO"
echo "      LIBVISION_SO: $LIBVISION_SO"
cp -f "$EXT_SO" "$PKG_DIR/"
cp -f "$LIBVISION_SO" "$PKG_DIR/libvision.so"
echo "      Package dir now contains:"
find "$PKG_DIR" -maxdepth 1 -ls 2>/dev/null | sed -n '1,200p'

echo "[3/4] Building wheel"
cd "$WHEEL_ROOT"
"$PYTHON_BIN" -m build --wheel --no-isolation

echo "[4/4] Done"
echo "wheel: $(find dist -maxdepth 1 -type f -name '*.whl' -print | tr '\n' ' ')"

