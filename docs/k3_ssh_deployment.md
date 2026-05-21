# K3 SSH Deployment

本文只记录 K3 Pico-ITX 32GB 机器侧部署步骤。RVClaw 代码优化仍然在本地/上游 GitHub 仓库中完成，K3 只作为运行和验收环境。

## 1. 工作边界

- 代码仓：从上游 GitHub 拉取或更新 `RVClaw`，不要在 K3 上维护长期分叉。
- K3 配置：放在 `deploy/k3/` 脚本和本文档中，通过环境变量覆盖机器差异。
- 运行产物：写到 `/data/rvclaw/runs`，不要写回代码目录。
- 模型和官方 llama.cpp 包：写到 `/data/rvclaw`，不要提交到 Git。

## 2. SSH 登录 K3

```bash
ssh <user>@<k3-ip>
uname -m
cat /etc/os-release
free -h
```

期望：

```text
uname -m == riscv64
内存接近 K3 Pico-ITX 32GB 规格
```

## 3. 安装基础包

```bash
sudo apt update
sudo apt install -y git curl wget rsync unzip tar tree htop tmux \
  build-essential cmake ninja-build pkg-config ccache \
  python3 python3-venv python3-pip python3-dev \
  sqlite3 jq v4l-utils ffmpeg python3-opencv
```

## 4. 建立 K3 数据目录

```bash
sudo mkdir -p /opt/rvclaw /data/rvclaw/{models,runs,logs,cache,src}
sudo chown -R "$USER":"$USER" /opt/rvclaw /data/rvclaw
```

## 5. 获取 RVClaw 代码

首次部署：

```bash
cd /opt/rvclaw
git clone <your-rvclaw-github-url> RVClaw
cd RVClaw
```

后续更新：

```bash
cd /opt/rvclaw/RVClaw
git pull --ff-only
```

如果 K3 访问 GitHub 不稳定，在开发机推送到 GitHub 后，可用 `scp` 或内网 Git 镜像同步代码，但仍以 GitHub 仓库内容为准。

## 6. 安装官方 spacemit-llama.cpp

先使用进迭官方预编译包跑通。源码编译和自定义优化不作为 v0.1 的第一阻塞项。

```bash
cd /data/rvclaw/src
wget https://archive.spacemit.com/spacemit-ai/llama.cpp/spacemit-llama.cpp.riscv64.0.0.8.tar.gz
tar -xzvf spacemit-llama.cpp.riscv64.0.0.8.tar.gz
ln -sfn spacemit-llama.cpp.riscv64.0.0.8 spacemit-llama.cpp
```

## 7. 下载 smoke 模型

```bash
wget https://modelscope.cn/models/unsloth/Qwen3-0.6B-GGUF/resolve/master/Qwen3-0.6B-Q4_0.gguf \
  -O /data/rvclaw/models/planner-smoke.gguf
```

后续正式演示可以换成 1.5B、1.7B 或 4B 量化模型，只需要覆盖：

```bash
export RVCLAW_LLAMA_MODEL_PATH=/data/rvclaw/models/<your-model>.gguf
export RVCLAW_LLAMA_MODEL=<model-name>
```

## 8. 验证 mock 主链路

```bash
cd /opt/rvclaw/RVClaw
source deploy/k3/env.sh
python3 -m rvclaw run "检查 A-03 区域设备状态并生成报告" \
  --planner mock \
  --runs-dir /data/rvclaw/runs
```

确认输出目录包含：

```text
task.yaml
metrics.json
trace.jsonl
report.md
raw.log
```

## 9. 启动 llama-server

建议用 `tmux` 保持服务常驻：

```bash
tmux new -s rvclaw-llama
cd /opt/rvclaw/RVClaw
source deploy/k3/env.sh
bash deploy/k3/run_llama_server.sh
```

另开一个 SSH 终端验证：

```bash
curl http://127.0.0.1:9090/v1/models | jq
```

## 10. 运行 llama.cpp Planner Demo

```bash
cd /opt/rvclaw/RVClaw
source deploy/k3/env.sh
bash deploy/k3/run_demo.sh
```

等价手工命令：

```bash
python3 -m rvclaw run "检查 A-03 区域设备状态并生成报告" \
  --planner llama_cpp \
  --runs-dir /data/rvclaw/runs \
  --json
```

预期输出应包含 6 个工具调用，而不是只有 `speak`：

```text
memory_query
move_to
capture_image
detect_status
speak
upload_report
```

如果模型只返回了单个 `speak`，当前代码会把巡检任务修复为 deterministic 6 步计划，避免 demo 表面 completed 但没有真实执行巡检闭环。
如果模型返回了半截 JSON、缺逗号、错误参数名等不稳定输出，默认巡检任务同样会回退到 deterministic 6 步计划；非巡检任务仍会报错，方便继续定位真实模型输出问题。

## 11. 常用覆盖项

这些环境变量只影响 K3 运行，不需要写死进 RVClaw 代码：

```bash
export RVCLAW_LLAMA_THREADS=4
export RVCLAW_LLAMA_CTX_SIZE=4096
export RVCLAW_LLAMA_BATCH_SIZE=256
export RVCLAW_LLAMA_TIMEOUT_S=120
export RVCLAW_PLATFORM_SOC=K3-Pico-ITX-32GB
export RVCLAW_RVV_VLEN=unknown
```

如果本地模型服务未启动，可临时退回 mock：

```bash
export RVCLAW_PLANNER=mock
bash deploy/k3/run_demo.sh
```

## 12. 验收命令

```bash
cd /opt/rvclaw/RVClaw
source deploy/k3/env.sh
python3 -m unittest discover -s tests
python3 benchmarks/run_agent_e2e.py --repeat 3 --planner mock --runs-dir /data/rvclaw/runs
python3 benchmarks/run_agent_e2e.py --repeat 3 --planner llama_cpp --runs-dir /data/rvclaw/runs
```

`llama_cpp` benchmark 需要 `llama-server` 已经在另一个 SSH/tmux 会话中运行。

## 13. 完整验证步骤与预期结果

### 13.1 更新到最新代码

```bash
cd /opt/rvclaw/RVClaw
git pull --ff-only
git log -1 --oneline
```

预期：

```text
最新提交包含 K3 llama.cpp planner hardening / deployment 相关说明
```

### 13.2 确认环境变量

```bash
source deploy/k3/env.sh
echo "$RVCLAW_PLATFORM_SOC"
echo "$RVCLAW_LLAMA_BASE_URL"
echo "$RVCLAW_LLAMA_MODEL_PATH"
ls -lh "$RVCLAW_LLAMA_MODEL_PATH"
```

预期：

```text
K3-Pico-ITX-32GB
http://127.0.0.1:9090/v1
/data/rvclaw/models/planner-smoke.gguf
模型文件存在，大小约 364M
```

### 13.3 启动并验证 llama-server

一个 SSH/tmux 窗口：

```bash
cd /opt/rvclaw/RVClaw
source deploy/k3/env.sh
bash deploy/k3/run_llama_server.sh
```

另一个 SSH 窗口：

```bash
curl http://127.0.0.1:9090/v1/models | jq '.data[0].id'
```

预期：

```text
"planner-smoke.gguf"
```

### 13.4 跑 mock 基线

```bash
python3 -m rvclaw run "检查 A-03 区域设备状态并生成报告" \
  --planner mock \
  --runs-dir /data/rvclaw/runs \
  --json | tee /tmp/rvclaw_mock_run.json
```

预期：

```bash
jq -r '.status' /tmp/rvclaw_mock_run.json
jq -r '.tool_calls[].name' /tmp/rvclaw_mock_run.json
```

输出：

```text
completed
memory_query
move_to
capture_image
detect_status
speak
upload_report
```

### 13.5 跑 llama.cpp Planner

```bash
bash deploy/k3/run_demo.sh | tee /tmp/rvclaw_llama_run.json
```

预期：

```bash
jq -r '.status' /tmp/rvclaw_llama_run.json
jq -r '.tool_calls[].name' /tmp/rvclaw_llama_run.json
```

输出：

```text
completed
memory_query
move_to
capture_image
detect_status
speak
upload_report
```

### 13.6 检查运行产物

```bash
RUN_DIR="$(jq -r '.run_dir' /tmp/rvclaw_llama_run.json)"
ls "$RUN_DIR"
cat "$RUN_DIR/task.yaml"
jq . "$RUN_DIR/metrics.json"
tail -n 5 "$RUN_DIR/trace.jsonl"
cat "$RUN_DIR/report.md"
```

预期：

```text
目录包含 artifacts、metrics.json、raw.log、report.md、task.yaml、trace.jsonl
task.yaml 中 platform.soc 为 K3-Pico-ITX-32GB
metrics.json 中 status 为 completed，task_success 为 true，tool_call_count 为 6
trace.jsonl 中能看到 planner.completed、skill_call.started、skill_call.completed、run.finished
report.md 中列出 6 个 Tool Calls
```

### 13.7 跑测试和 benchmark

```bash
python3 -m unittest discover -s tests
python3 benchmarks/run_agent_e2e.py --repeat 3 --planner mock --runs-dir /data/rvclaw/runs
python3 benchmarks/run_agent_e2e.py --repeat 3 --planner llama_cpp --runs-dir /data/rvclaw/runs
tail -n 5 /data/rvclaw/runs/benchmark_agent_e2e.csv
```

不要把文档里的说明文字当 shell 命令输入；例如 `mock CLI / mock benchmark` 只是说明项，不是命令。

预期：

```text
11 个以上测试通过
benchmark_agent_e2e.csv 生成
CSV 中包含 planner_model、planner_model_path、planner_base_url、llama_threads、platform_soc
llama_cpp 三次 run 的 status 为 completed
```

### 13.8 抓取 llama-server 原始响应

如果 `bash deploy/k3/run_demo.sh` 报 `llama.cpp planner response missing message content` 或非巡检任务的 `malformed JSON content`，先抓一次原始响应：

```bash
curl "$RVCLAW_LLAMA_BASE_URL/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$RVCLAW_LLAMA_MODEL\",
    \"messages\": [
      {\"role\":\"system\",\"content\":\"Return strict JSON only.\"},
      {\"role\":\"user\",\"content\":\"返回 {\\\"tool_calls\\\":[{\\\"name\\\":\\\"speak\\\",\\\"arguments\\\":{\\\"text\\\":\\\"hello\\\"}}]}\"}
    ],
    \"temperature\": 0,
    \"max_tokens\": 128,
    \"stream\": false
  }" | tee /tmp/rvclaw_llama_raw_response.json | jq
```

常见兼容结构：

```text
choices[0].message.content
choices[0].message.reasoning_content
choices[0].content
choices[0].text
```

当前适配器会依次尝试这些字段。若仍失败，把 `/tmp/rvclaw_llama_raw_response.json` 保存下来，用于更新适配器。

### 13.9 验收结论口径

全部通过后，可以记录为：

```text
K3 Pico-ITX 32GB 上 RVClaw Demo Claw v0.1 初步闭环通过：
本地 spacemit-llama.cpp/Qwen3-0.6B Planner 接入成功，
Agent Core/Safety Guard/Mock Device/SQLite memory/RVBench 产物链路可复现。
```

不要写成：

```text
K3 上完整机器人控制或 ROS2/OpenClaw 实机闭环已完成
```

当前验证仍是 Mock Device + 本地 LLM Planner 的软件闭环。
