#!/usr/bin/env bash
set -eo pipefail
ROBOT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
VISION_RUNTIME="$ROBOT_ROOT/runtime/k3/vision"
SDK_ROOT=${SROBOTIS_ROOT:-/data/spacemit_robot}
test -f "$VISION_RUNTIME/services/rvclaw_api/rvclaw_api/server.py"
test -x "$ROBOT_ROOT/build/yolov8_mjpeg"
export PYTHONPATH="$VISION_RUNTIME/services/rvclaw_api:$VISION_RUNTIME/robot/ros2_ws/src/rvclaw_algo:$ROBOT_ROOT/services/rvclaw_report${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$SDK_ROOT/output/staging/lib:/opt/opencv-spacemit/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SPACEMIT_VISION_ROOT="$SDK_ROOT/components/model_zoo/vision"
export RVCLAW_CAMERA_MODE=${RVCLAW_CAMERA_MODE:-astra_raw}
export RVCLAW_STUDIO_ROOT="${RVCLAW_STUDIO_ROOT:-$ROBOT_ROOT/studio/web}"
export RVCLAW_METER_CONFIG="$VISION_RUNTIME/configs/meter/meters.json"
export RVCLAW_YOLOV8_BINARY="$ROBOT_ROOT/build/yolov8_mjpeg"
export RVCLAW_PPOCR_UPLOAD_BINARY="$ROBOT_ROOT/build/meter-paddleocr/meter_paddleocr"
export RVCLAW_PPOCR_UPLOAD_CONFIG="$VISION_RUNTIME/configs/vision/ppocr_meter_upload.yaml"
export RVCLAW_YOLOV8_CONFIG="$VISION_RUNTIME/configs/thresholds/general_yolov8.yaml"
export RVCLAW_PPE_CONFIG="$VISION_RUNTIME/configs/thresholds/helmet_yolov8.yaml"
export RVCLAW_HARDHAT_CONFIG="$VISION_RUNTIME/configs/thresholds/hardhat_yolov8.yaml"
export RVCLAW_ASTRA_PERSON_CONFIG="$VISION_RUNTIME/configs/thresholds/astra_person_bytetrack.yaml"
export RVCLAW_INSPECTION_PHOTO_ROOT=${RVCLAW_INSPECTION_PHOTO_ROOT:-/data/inspection_photos}
exec /usr/bin/python3 -m rvclaw_api.server "$@"
