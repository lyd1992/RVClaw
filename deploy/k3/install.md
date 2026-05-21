# K3 Pico-ITX Install Notes

This path is for the K3 Pico-ITX 32GB RVClaw demo. The goal is to run the
existing RVClaw skeleton on K3 with a local `spacemit-llama.cpp` planner.

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

## 4. Smoke Model

```bash
wget https://modelscope.cn/models/unsloth/Qwen3-0.6B-GGUF/resolve/master/Qwen3-0.6B-Q4_0.gguf \
  -O /data/rvclaw/models/planner-smoke.gguf
```

## 5. Start llama-server

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
export RVCLAW_LLAMA_MODEL_PATH=/data/rvclaw/models/planner-smoke.gguf
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
