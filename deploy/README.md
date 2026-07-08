# K3 deployment

Target path:

```text
/data/RVClaw
```

## 1. Copy or clone

```bash
mkdir -p /data/RVClaw
cd /data/RVClaw
git clone https://github.com/lyd1992/RVClaw.git .
```

If GitHub is unavailable on K3, upload the repository contents from the
development machine to `/data/RVClaw`.

## 2. YOLOv8n model

F5/F7 use the YOLOv8n adapter through the API and Studio display.

Default model path:

```text
/data/RVClaw/models/yolov8n.pt
```

Optional override:

```bash
export RVCLAW_YOLO_MODEL=/data/RVClaw/models/yolov8n.pt
export RVCLAW_YOLO_CONF=0.35
```

The service starts even if `ultralytics` or `yolov8n.pt` is not ready. In that
case `/api/vision-state` reports `model.available=false` and Studio shows the
fallback status instead of failing to boot.

## 3. Start Studio/API

```bash
cd /data/RVClaw
python3 scripts/serve_studio.py --host 0.0.0.0 --port 8017
```

Open:

```text
http://K3_IP:8017/
```

Verify F5/F7:

```bash
curl http://127.0.0.1:8017/api/vision-state
curl http://127.0.0.1:8017/api/person-tracking
```

## 4. systemd service

```ini
[Unit]
Description=RVClaw EdgeOne Studio
After=network.target

[Service]
Type=simple
WorkingDirectory=/data/RVClaw
Environment=RVCLAW_YOLO_MODEL=/data/RVClaw/models/yolov8n.pt
Environment=RVCLAW_YOLO_CONF=0.35
ExecStart=/usr/bin/python3 /data/RVClaw/scripts/serve_studio.py --host 0.0.0.0 --port 8017
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Install:

```bash
cat >/etc/systemd/system/rvclaw-studio.service
systemctl daemon-reload
systemctl enable rvclaw-studio
systemctl start rvclaw-studio
journalctl -u rvclaw-studio -f
```
