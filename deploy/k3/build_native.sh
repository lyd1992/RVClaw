#!/usr/bin/env bash
set -eo pipefail
ROBOT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
SDK_ROOT=${SROBOTIS_ROOT:-/data/spacemit_robot}
test -f "$SDK_ROOT/output/staging/lib/libvision.so"
cmake -S "$ROBOT_ROOT/services/rvclaw_api/native" -B "$ROBOT_ROOT/build/camera" \
  -DSPACEMIT_ROBOT_ROOT="$SDK_ROOT" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROBOT_ROOT/build/camera" --target yolov8_mjpeg --parallel 2
install -m 755 "$ROBOT_ROOT/build/camera/yolov8_mjpeg" "$ROBOT_ROOT/build/yolov8_mjpeg"
cmake -S "$ROBOT_ROOT/services/meter_gauge/native" -B "$ROBOT_ROOT/build/meter-paddleocr" \
  -DSPACEMIT_ROBOT_ROOT="$SDK_ROOT" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROBOT_ROOT/build/meter-paddleocr" --target meter_paddleocr --parallel 2
