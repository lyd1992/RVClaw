# Operations Runbook

This runbook is the short operational reference. For first-time K3 deployment,
start with `docs/k3_start_here.md`; for the v0.1.2 multi-vision sidecar, use
`docs/k3_demozoo_bridge.md`. The current Web surface is the v0.1.3 Agent
Command Center.

## Local Development

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
python -m compileall -q src tests benchmarks
python -m rvclaw run "Check A-03 and generate report" --planner mock --json
```

## K3 CLI Smoke

```bash
cd /opt/rvclaw/RVClaw
git pull --ff-only
source deploy/k3/env.sh

python3 -m unittest discover -s tests
python3 -m compileall -q src tests benchmarks

python3 -m rvclaw run "检查 A-03 区域设备状态并生成报告" \
  --planner mock \
  --runs-dir /data/rvclaw/runs \
  --json
```

Expected baseline tool chain:

```text
memory_query
move_to
capture_image
detect_status
speak
upload_report
```

## K3 llama.cpp Planner

Keep llama-server in a dedicated SSH/tmux window:

```bash
cd /opt/rvclaw/RVClaw
source deploy/k3/env.sh
bash deploy/k3/run_llama_server.sh
```

Check the service from another SSH window:

```bash
curl http://127.0.0.1:9090/v1/models | jq
```

Run the CLI demo:

```bash
source deploy/k3/env.sh
bash deploy/k3/run_demo.sh
```

## K3 Agent Command Center + CV

Install OpenCV if possible:

```bash
sudo apt update
sudo apt install -y python3-opencv ffmpeg v4l-utils
python3 - <<'PY'
import cv2
print(cv2.__version__)
PY
```

Start the Web console:

```bash
cd /opt/rvclaw/RVClaw
source deploy/k3/env.sh
bash deploy/k3/run_web_demo.sh
```

Open the Web UI from Windows:

```text
http://<K3-LAN-IP>:8088
http://<K3-TAILSCALE-IP>:8088
```

The main page should show a task-template dropdown, optional image upload,
Agent execution graph, Runtime Stack Map, CV panel, and evidence file viewer.
Historical runs are opened from the `历史记录` drawer.

## K3 DemoZoo Multi-Vision

Start DemoZoo separately according to the SpacemiT/Bianbu instructions, then run
RVClaw with the sidecar enabled:

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

Use the four Web presets:

```text
分类这张图片并说明结果
检测图片中的目标并生成结论
分割画面中的主要区域
检测画面中是否有人脸
```

Expected evidence for each visual run:

```text
artifacts/a03_capture.png
artifacts/<task>_annotated.png
artifacts/vision_result.json
metrics.json
trace.jsonl
report.md
raw.log
```

## Safety Checks

Unknown zones, unknown skills, unknown vision tasks, unknown vision models, and
external image URLs should return `failed` while still preserving the run
evidence package.

Example:

```bash
python3 -m rvclaw run "移动到 Z-99 区域并拍照" \
  --planner mock \
  --runs-dir /data/rvclaw/runs \
  --json
```

## Benchmark

```bash
python3 benchmarks/run_agent_e2e.py --repeat 3 --planner llama_cpp --runs-dir /data/rvclaw/runs
```

The CSV is written under the selected runs directory. Record model name,
quantization, thread count, K3 image version, and whether DemoZoo was enabled.
