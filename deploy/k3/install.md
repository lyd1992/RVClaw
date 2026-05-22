# K3 Pico-ITX Install Notes

This path is for the K3 Pico-ITX 32GB RVClaw demo. The goal is to run the
existing RVClaw skeleton on K3 with a local `spacemit-llama.cpp` planner.

First-time users should start with `docs/k3_start_here.md`. This file is the
command reference for package installation, model paths, and environment
overrides.

For the SSH-oriented runbook, see `docs/k3_ssh_deployment.md`. Keep code
changes in the upstream GitHub repository; keep K3 machine state under
`/data/rvclaw` and environment variables.

## 1. Base Packages

Use the vendor-recommended Bianbu/Ubuntu riscv64 image, then install the
minimal tooling:

```bash
sudo apt update
sudo apt install -y git curl wget rsync unzip tar tree htop tmux \
  build-essential cmake ninja-build pkg-config ccache \
  python3 python3-venv python3-pip python3-dev \
  sqlite3 jq v4l-utils ffmpeg python3-opencv
```

`python3-opencv` is recommended for the v0.1.1 Web + CV demo. If it is not
installed, RVClaw still runs with a file-copy detector fallback, but the
annotated image will not contain OpenCV overlays.

Verify OpenCV:

```bash
python3 - <<'PY'
import cv2
print(cv2.__version__)
PY
```

If the distro package is unavailable, keep the demo running with the fallback
path first and install OpenCV later. Do not block the Web/API or llama.cpp demo
on OpenCV.

## 2. Data Directories

```bash
sudo mkdir -p /opt/rvclaw /data/rvclaw/{models,runs,logs,cache,src}
sudo chown -R "$USER":"$USER" /opt/rvclaw /data/rvclaw
```

## 3. Official SpacemiT llama.cpp Package

Start with the official prebuilt package before attempting source builds.
The Bianbu llama.cpp guide uses earlier packages, while the SpacemiT archive
currently includes `0.0.8`.

```bash
cd /data/rvclaw/src
wget https://archive.spacemit.com/spacemit-ai/llama.cpp/spacemit-llama.cpp.riscv64.0.0.8.tar.gz
tar -xzvf spacemit-llama.cpp.riscv64.0.0.8.tar.gz
ln -sfn spacemit-llama.cpp.riscv64.0.0.8 spacemit-llama.cpp
```

## 4. K3 Demo Model

For the formal K3 demo, use the K3 recommended Qwen3-30B-A3B GGUF model:

```bash
wget https://www.modelscope.cn/models/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/resolve/master/Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf -P ~/
mkdir -p /data/rvclaw/models
ln -sfn ~/Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf \
  /data/rvclaw/models/Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf
```

`deploy/k3/env.sh` defaults to this model:

```bash
export RVCLAW_LLAMA_MODEL=Qwen3-30B-A3B-Instruct-2507-Q4_0
export RVCLAW_LLAMA_MODEL_PATH=/data/rvclaw/models/Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf
```

## 4.1 Smoke Model

```bash
wget https://modelscope.cn/models/unsloth/Qwen3-0.6B-GGUF/resolve/master/Qwen3-0.6B-Q4_0.gguf \
  -O /data/rvclaw/models/planner-smoke.gguf
```

Use the smoke model only when you need a faster sanity check:

```bash
export RVCLAW_LLAMA_MODEL=Qwen3-0.6B
export RVCLAW_LLAMA_MODEL_PATH=/data/rvclaw/models/planner-smoke.gguf
```

## 5. Start llama-server

Install Web dependencies before starting the Web console:

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

```bash
cd /opt/rvclaw/RVClaw
source deploy/k3/env.sh
bash deploy/k3/run_llama_server.sh
```

In another terminal:

```bash
curl http://127.0.0.1:9090/v1/models | jq
```

## 6. Run RVClaw

First verify the framework with the mock planner:

```bash
cd /opt/rvclaw/RVClaw
source deploy/k3/env.sh
python3 -m rvclaw run "检查 A-03 区域设备状态并生成报告" --planner mock --runs-dir /data/rvclaw/runs
```

Then run the local llama.cpp planner:

```bash
bash deploy/k3/run_demo.sh
```

To run the Web + CV visual demo:

```bash
bash deploy/k3/run_web_demo.sh
```

Then open:

```text
http://<K3-IP>:8088
```

Expected `tool_calls` for the default inspection task:

```text
memory_query
move_to
capture_image
detect_status
speak
upload_report
```

If the small local model returns an incomplete plan such as only `speak`,
the llama.cpp planner adapter repairs inspection tasks to this deterministic
six-step workflow.
If it returns malformed JSON for the default inspection task, the adapter also
falls back to the same workflow so K3 smoke validation can continue.

The run should write:

```text
/data/rvclaw/runs/<run_id>/
  task.yaml
  metrics.json
  trace.jsonl
  report.md
  raw.log
```

## 7. Useful Environment Overrides

```bash
export RVCLAW_LLAMA_THREADS=4
export RVCLAW_LLAMA_CTX_SIZE=4096
export RVCLAW_LLAMA_BATCH_SIZE=256
export RVCLAW_LLAMA_MODEL_PATH=/data/rvclaw/models/Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf
export RVCLAW_DEVICE_BACKEND=cv_sample
export RVCLAW_VISION_SOURCE=/data/rvclaw/cache/a03_normal.png
export RVCLAW_WEB_HOST=0.0.0.0
export RVCLAW_WEB_PORT=8088
export RVCLAW_PLANNER=llama_cpp
```

Use `RVCLAW_PLANNER=mock` when the local model server is not running.

## 8. Full Verification

The SSH runbook has the complete verification flow and expected outputs:

```text
docs/k3_ssh_deployment.md
```

Minimum K3 acceptance commands:

```bash
source deploy/k3/env.sh
python3 -m unittest discover -s tests
bash deploy/k3/run_demo.sh | tee /tmp/rvclaw_llama_run.json
jq -r '.tool_calls[].name' /tmp/rvclaw_llama_run.json
python3 benchmarks/run_agent_e2e.py --repeat 3 --planner llama_cpp --runs-dir /data/rvclaw/runs
```

Do not type checklist labels such as `mock CLI / mock benchmark` as shell
commands; use the concrete `python3 -m rvclaw ...` and `python3 benchmarks/...`
commands above.
