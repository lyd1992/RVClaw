# K3 DemoZoo Vision Bridge

This document records the v0.1.2 optional multi-vision path. RVClaw remains the
Web/API command center and evidence system; DemoZoo runs as a local K3 sidecar
for model inference.

## Goal

```text
RVClaw Web
  -> analyze_image
  -> DemoZoo HTTP sidecar
  -> classification / object_detection / segmentation / face_detection
  -> annotated image + vision_result.json + metrics + report
```

The default Web demo still works without DemoZoo. When DemoZoo is unavailable,
RVClaw falls back to `cv_sample` / `mock_fallback` and records that in
`metrics.json`.

## Enable DemoZoo Backend

Start DemoZoo separately according to the SpacemiT/Bianbu DemoZoo instructions.
Then configure RVClaw:

```bash
cd /opt/rvclaw/RVClaw
source deploy/k3/env.sh

export RVCLAW_DEVICE_BACKEND=cv_sample
export RVCLAW_VISION_BACKEND=demozoo
export RVCLAW_DEMOZOO_BASE_URL=http://127.0.0.1:8000
export RVCLAW_DEMOZOO_ENDPOINT_TEMPLATE='/predict/{model}'
export RVCLAW_VISION_TIMEOUT_S=30

bash deploy/k3/run_web_demo.sh
```

If your DemoZoo service uses a different route, only change
`RVCLAW_DEMOZOO_ENDPOINT_TEMPLATE`. The template can use `{task}` and `{model}`.

## Supported Tasks

| RVClaw task | Default model | Web preset |
|---|---|---|
| `classification` | `resnet` | 分类这张图片并说明结果 |
| `object_detection` | `yolov8` | 检测图片中的目标并生成结论 |
| `segmentation` | `yolov8_seg` | 分割画面中的主要区域 |
| `face_detection` | `yolov5_face` | 检测画面中是否有人脸 |

Face support is detection-only. RVClaw does not perform identity recognition,
face matching, or face-library management in v0.1.2.

## Expected Artifacts

Each successful vision run should produce:

```text
artifacts/
  a03_capture.png
  <task>_annotated.png
  vision_result.json
metrics.json
trace.jsonl
report.md
raw.log
```

`vision_result.json` uses the RVClaw normalized schema:

```json
{
  "task": "object_detection",
  "model": "yolov8",
  "backend": "demozoo",
  "summary": "检测到 3 个目标。",
  "labels": [],
  "objects": [{"label": "person", "confidence": 0.92, "bbox": [0.1, 0.2, 0.4, 0.8]}],
  "segments": [],
  "faces": [],
  "annotated_image_ref": ".../object_detection_annotated.png"
}
```

## Acceptance

Run these on K3:

```bash
source deploy/k3/env.sh
python3 -m unittest discover -s tests
python3 -m compileall -q src tests benchmarks
bash deploy/k3/run_web_demo.sh
```

In the Web UI, run the four presets. Confirm:

- the CV panel shows capture and annotated images;
- the result cards show top labels, boxes, masks, or face boxes;
- `metrics.json` records `vision_task`, `vision_backend`, `vision_model`, and count fields;
- `vision_result.json` exists for every visual run.
