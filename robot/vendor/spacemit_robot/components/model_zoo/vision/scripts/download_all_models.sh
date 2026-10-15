#!/bin/sh
# 一键执行所有 example 和 application 的 download_models.sh，将模型下载到 ~/.cache/models/vision/
# 在 cv 组件根目录执行: bash scripts/download_all_models.sh

set -e
CV_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CV_ROOT"

echo "========== Examples =========="
for dir in examples/resnet examples/efficientnet examples/efficientnet_v2s examples/mobilenet examples/mobilenetv1 examples/vit \
           examples/yolov8 examples/yolov8_pose examples/yolov8_seg examples/yolov11 \
           examples/yolo12 examples/yolo26 examples/yolo_world examples/yoloe \
           examples/yolov5 examples/yolov5_gesture examples/yolov5-face examples/arcface examples/ocsort \
           examples/bytetrack examples/emotion examples/pp_liteseg \
           examples/adaface examples/siglip2 examples/mobileclip2; do
  if [ -f "$dir/scripts/download_models.sh" ]; then
    echo ">>> $dir"
    bash "$dir/scripts/download_models.sh" || true
  fi
done

echo ""
echo "========== Applications =========="
for dir in applications/emotion_detection applications/fall_detection applications/fire_detection applications/intrusion_detection applications/face_recognition; do
  if [ -f "$dir/scripts/download_models.sh" ]; then
    echo ">>> $dir"
    bash "$dir/scripts/download_models.sh" || true
  fi
done

echo ""
echo "========== Done =========="
echo "Models are in: ${HOME:-~}/.cache/models/vision/"
ls -la "${HOME:-$HOME}/.cache/models/vision/" 2>/dev/null || true
