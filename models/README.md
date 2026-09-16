# models

Place local model files here.

For the YOLOv8n Studio demo, the expected file is:

```text
models/yolov8n.pt
```

Simple setup:

1. Install the runtime package:

```bash
python3 -m pip install ultralytics opencv-python
```

2. If the device can access the internet, start the service once and
   `ultralytics` can download `yolov8n.pt` automatically.
3. If the device cannot access the internet, download `yolov8n.pt` on another
   machine and copy it into this directory.

For K3 deployment, the full default path is:

```text
/data/RVClaw/models/yolov8n.pt
```

To use a different path:

```bash
export RVCLAW_YOLO_MODEL=/data/RVClaw/models/yolov8n.pt
```

The Studio can start without this file, but it will not show real YOLOv8n
detection boxes until `/api/vision-state` reports `"available": true`.
