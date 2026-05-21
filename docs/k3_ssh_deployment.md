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
