# K3 Agent Command Center + CV Demo

This document records the v0.1.1-v0.1.3 K3 visual demo path. It keeps RVClaw code in
the GitHub repository and keeps K3 runtime state under `/data/rvclaw`.

If this is your first K3 setup, finish `docs/k3_start_here.md` first. This
document focuses on the visual demo flow and expected results.

## Goal

The demo turns the previous CLI smoke test into a visual command center:

```text
browser natural-language task / uploaded image
  -> FastAPI Agent Command Center
  -> llama.cpp Planner
  -> Safety Guard
  -> cv_sample / DemoZoo / mock device skills
  -> Agent graph + image artifacts + metrics + trace + report
```

The default visual input is a sample image, so the demo works before a USB
camera is available. A future camera backend can reuse the same
`capture_image` / `detect_status` skill contract.

For the v0.1.2 multi-vision DemoZoo bridge, see `docs/k3_demozoo_bridge.md`.

## Dependencies

Install the K3 base packages and the recommended OpenCV package:

```bash
sudo apt update
sudo apt install -y git curl wget rsync unzip tar tree htop tmux \
  build-essential cmake ninja-build pkg-config ccache \
  python3 python3-venv python3-pip python3-dev \
  sqlite3 jq v4l-utils ffmpeg python3-opencv
```

Verify:

```bash
python3 - <<'PY'
import cv2
print(cv2.__version__)
PY
```

OpenCV is recommended, not mandatory. If `cv2` cannot be imported, the
`cv_sample` backend still copies the sample image to `a03_annotated.png` and
records `detector=file_copy_fallback`.

Install Web dependencies. `python-multipart` is included in `.[api]` for image upload:

```bash
cd /opt/rvclaw/RVClaw
python3 -m pip install -e '.[api]'
```

If system Python blocks global installs, use a venv:

```bash
python3 -m venv /data/rvclaw/venv
source /data/rvclaw/venv/bin/activate
python3 -m pip install -e '.[api]'
```

## K3 Model

Use the K3 recommended Qwen3-30B-A3B GGUF model for the formal demo:

```bash
wget https://www.modelscope.cn/models/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/resolve/master/Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf -P ~/
mkdir -p /data/rvclaw/models
ln -sfn ~/Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf \
  /data/rvclaw/models/Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf
```

The old `planner-smoke.gguf` remains useful for quick smoke tests:

```bash
export RVCLAW_LLAMA_MODEL=Qwen3-0.6B
export RVCLAW_LLAMA_MODEL_PATH=/data/rvclaw/models/planner-smoke.gguf
```

## Start Services

Terminal 1:

```bash
cd /opt/rvclaw/RVClaw
source deploy/k3/env.sh
bash deploy/k3/run_llama_server.sh
```

Terminal 2:

```bash
cd /opt/rvclaw/RVClaw
source deploy/k3/env.sh
bash deploy/k3/run_web_demo.sh
```

Open:

```text
http://<K3-IP>:8088
```

## Demo Script

1. Open the Web console.
2. Choose a task template from the dropdown; the active chip and expected tool chain should update.
3. Submit `检查 A-03 区域设备状态并生成报告`.
4. Confirm the Agent graph shows Task Intake, Planner, Safety Guard, Memory, Capture, Vision, Speak, and Report nodes.
5. Upload a `png/jpg/jpeg/webp` image, choose `目标检测`, and run again.
6. Confirm the CV panel shows capture and annotated images and the result cards show structured vision output.
7. Open `metrics.json`, `trace.jsonl`, `report.md`, and `raw.log` in the Web UI.
8. Open `历史记录`; it should appear in a drawer rather than occupying the main screen.
9. Submit an unsupported zone such as `移动到 Z-99 区域并拍照`; the Safety Guard node should show rejection/failure and still keep artifacts.

## Environment Overrides

```bash
export RVCLAW_WEB_HOST=0.0.0.0
export RVCLAW_WEB_PORT=8088
export RVCLAW_DEVICE_BACKEND=cv_sample
export RVCLAW_VISION_SOURCE=/data/rvclaw/cache/a03_normal.png
export RVCLAW_UPLOADS_DIR=/data/rvclaw/uploads
export RVCLAW_WEB_TOKEN=<optional-token>
```

`run_web_demo.sh` creates a sample image in `/data/rvclaw/cache` if the source
does not exist. If `cv_sample` cannot find the configured source, RVClaw falls
back to the mock device and records `device_backend=mock_fallback` in
`metrics.json`.

## Acceptance

- `rvclaw serve` starts on K3.
- `/api/health`, `/api/runs`, and `/api/benchmarks` respond.
- The Web UI can create a run, upload a local image, and view all evidence files.
- The main screen focuses on the current run; historical runs open from the `历史记录` drawer.
- The Agent graph shows Planner and Safety Guard nodes, including rejected unsafe runs.
- `metrics.json` records `planner_mode` and `device_backend`.
- CV sample runs produce `artifacts/a03_capture.png` and `artifacts/a03_annotated.png`.
- Unknown zones are rejected by Safety Guard and produce failed run artifacts.
