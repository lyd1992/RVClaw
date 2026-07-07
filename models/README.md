# models

Place offline model files used by K3 local inference here.

Expected first-launch file:

```text
models/yolov8n.pt
```

The API reads `RVCLAW_YOLO_MODEL` when a custom path is needed. For the target
K3 deployment under `/data/RVClaw`, either keep the default relative path above
or set:

```bash
export RVCLAW_YOLO_MODEL=/data/RVClaw/models/yolov8n.pt
```

The Studio can start without this file. In that case `/api/vision-state` reports
`model.available=false` and uses a deterministic fallback payload so F5/F7 can
still be demonstrated while the runtime is being prepared.
