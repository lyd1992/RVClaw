# K3 Real Vision DemoZoo Bridge

This document is the entry point for the real K3 vision demo. `cv_sample` is
only an offline smoke backend. For classification, detection, segmentation, or
face detection results that should be presented as model inference, RVClaw must
call a real local vision backend such as Bianbu DemoZoo.

## What Runs Where

```text
RVClaw Web / CLI
  -> analyze_image
  -> DemoZoo HTTP sidecar on K3
  -> ResNet / YOLO / YOLOv8-Seg / YOLOv5-Face
  -> normalized vision_result.json
  -> trace / metrics / report
```

Bianbu DemoZoo provides CV demos for classification, detection, segmentation,
and face-related models. The documented HTTP route is:

```bash
curl -X POST "http://localhost:8000/predict/yolov8" \
  -F "image=@test_image.jpg"
```

RVClaw also reads `GET /models` and prefers the endpoint advertised for the
selected model, for example `/predict/yolov8`. If a model route returns 404,
RVClaw retries the trailing-slash variant before reporting a failed run. The
multipart request includes both `image` and `file` aliases so minor DemoZoo API
field-name differences do not break the demo.

For model-script failures inside DemoZoo, RVClaw may retry another real model in
the same task family. For example, object detection tries `yolov8`, then
`yolov11`, `yolov8_seg`, `yolov8_pose`, `yolov5`, and `yolov6`. This is still a
real DemoZoo backend path, not `cv_sample`; `metrics.json` records the actual
`vision_model`, and `vision_result.json` records `requested_model` plus
`model_fallback_reason`.
If every real model in the task family returns no usable structured result and
no annotated image, `RVCLAW_REQUIRE_REAL_VISION=1` makes the run fail instead of
showing a misleading completed run with `0` targets.

## Start DemoZoo

Start DemoZoo according to the Bianbu/SpacemiT container guide. The common
image name in the official guide is:

```bash
sudo docker pull harbor.spacemit.com/bianbu-robot/spacemit-demo:latest
```

On some K3/Bianbu images, `dockerd` may fail with:

```text
iptables: Failed to initialize nft: Protocol not supported
failed to create NAT chain DOCKER
```

For this local sidecar demo, bypass Docker's bridge/NAT path and run DemoZoo on
the host network:

```bash
mkdir -p /etc/docker
cat >/etc/docker/daemon.json <<'EOF'
{
  "iptables": false,
  "ip6tables": false,
  "bridge": "none"
}
EOF

systemctl reset-failed docker
systemctl restart containerd
systemctl restart docker
systemctl status docker --no-pager -l
```

Then start DemoZoo with host networking:

```bash
docker run -itd \
  --network host \
  --name spacemit-demo-container \
  --privileged \
  harbor.spacemit.com/bianbu-robot/spacemit-demo:latest
```

If the container name already exists, use `docker start spacemit-demo-container`.

After the container is running, verify that the model service is reachable from
the K3 shell:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/models | jq
```

Then test one real model directly:

```bash
curl -X POST "http://127.0.0.1:8000/predict/resnet" \
  -F "image=@/data/rvclaw/uploads/<your-upload>.png" \
  -F "file=@/data/rvclaw/uploads/<your-upload>.png" | jq
```

Expected: JSON with a prediction field such as `predicted_class`, `labels`,
`objects`, `detections`, `boxes`, `segments`, or `faces`.

If direct detection still returns 404, test the trailing slash and inspect the
container route logs:

```bash
curl -X POST "http://127.0.0.1:8000/predict/yolov8/" \
  -F "image=@/data/rvclaw/uploads/<your-upload>.png" \
  -F "file=@/data/rvclaw/uploads/<your-upload>.png" -v

docker logs --tail=80 spacemit-demo-container
```

## Run RVClaw With Real Vision Required

Use this mode for roadshow demos where wrong fallback results are worse than a
clear failure:

```bash
cd /opt/rvclaw/RVClaw
source deploy/k3/env.sh

export RVCLAW_VISION_BACKEND=demozoo
export RVCLAW_DEMOZOO_BASE_URL=http://127.0.0.1:8000
export RVCLAW_DEMOZOO_ENDPOINT_TEMPLATE='/predict/{model}'
export RVCLAW_REQUIRE_REAL_VISION=1

bash deploy/k3/run_web_demo.sh
```

Equivalent shortcut:

```bash
bash deploy/k3/run_real_vision_web_demo.sh
```

With `RVCLAW_REQUIRE_REAL_VISION=1`, RVClaw will not fall back to `cv_sample`.
If DemoZoo is down or a model endpoint fails, the run becomes `failed`, the
Vision node shows the error, and the evidence pack records the reason.

## Task Display Policy

The Web UI intentionally uses different image layouts by task:

| Task | Display | Reason |
|---|---|---|
| image classification | single input image | classification returns labels, not geometry; before/after images add noise |
| object detection | input + annotated image | boxes need visual verification |
| segmentation | input + overlay image | mask/region overlay needs comparison |
| face detection | input + annotated image | boxes/count are the visual evidence; no identity recognition |
| A-03 inspection / detect_status | input + annotated image | status-light/risk annotation is the evidence |
| return BASE / safety rejection / planner failure | no CV image | the task does not need image evidence |

## Supported RVClaw Tasks

| RVClaw task | Default model | DemoZoo family |
|---|---|---|
| `classification` | `resnet` | ResNet/MobileNet/EfficientNet/Swin |
| `object_detection` | `yolov8` | YOLOv8/YOLOv11/YOLOv8-Seg/YOLOv8-Pose/YOLOv5/YOLOv6 |
| `segmentation` | `yolov8_seg` | YOLOv8-Seg/FCN/UNet/SAM |
| `face_detection` | `yolov5_face` | YOLOv5-Face |

Face support in RVClaw v0.1.x is detection-only. It does not do identity
recognition, face matching, or face library management.

## Acceptance

```bash
source deploy/k3/env.sh
python3 -m unittest discover -s tests
python3 -m compileall -q src tests benchmarks
bash deploy/k3/run_real_vision_web_demo.sh
```

In the Web UI:

1. Upload a normal image.
2. Run `图片分类`; confirm only one image is shown and result cards come from
   `vision_backend=demozoo`.
3. Run `目标检测`; confirm original and annotated images are shown.
4. Stop DemoZoo and run again; confirm the run fails instead of silently using
   `cv_sample`.

Evidence to check:

```text
artifacts/vision_result.json
metrics.json
trace.jsonl
report.md
raw.log
```
