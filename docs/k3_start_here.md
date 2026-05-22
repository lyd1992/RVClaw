# K3 First Run Guide

这是 K3 Pico-ITX 32GB 上第一次跑 RVClaw 的入口文档。先按本页走通一遍，再根据需要跳到更细的部署或排障文档。

## 读哪份文档

| 场景 | 先看 |
|---|---|
| 第一次在 K3 上部署 RVClaw | 本文 |
| 只想复制安装命令 | `deploy/k3/install.md` |
| 要做 Web + CV 可视化演示 | `docs/k3_web_cv_demo.md` |
| SSH/tmux/trace/benchmark 逐项排障 | `docs/k3_ssh_deployment.md` |
| 看当前工程状态和 tag | `docs/development_status.md` |

代码以 GitHub 上游仓库为准；K3 上的模型、运行产物、官方 llama.cpp 包统一放在 `/data/rvclaw`，不要提交进 Git。

## 0. 目标结果

完整演示链路是：

```text
浏览器自然语言任务
  -> FastAPI Web 控制台
  -> llama.cpp Planner
  -> Safety Guard
  -> cv_sample / mock device
  -> timeline + 图片 artifact + metrics + trace + report
```

最低验收要看到：

```text
memory_query -> move_to -> capture_image -> detect_status -> speak -> upload_report
```

以及一个 run 目录：

```text
/data/rvclaw/runs/<run_id>/
  task.yaml
  metrics.json
  trace.jsonl
  report.md
  raw.log
  artifacts/
```

## 1. 安装基础环境

```bash
sudo apt update
sudo apt install -y git curl wget rsync unzip tar tree htop tmux \
  build-essential cmake ninja-build pkg-config ccache \
  python3 python3-venv python3-pip python3-dev \
  sqlite3 jq v4l-utils ffmpeg python3-opencv
```

OpenCV 用于 v0.1.1 的样例图识别和标注，建议安装：

```bash
python3 - <<'PY'
import cv2
print(cv2.__version__)
PY
```

如果 `cv2` 暂时不可用，Web demo 仍会用 file-copy fallback 跑完，只是标注图不会有 OpenCV 绘制框。

## 2. 建立目录并拉代码

```bash
sudo mkdir -p /opt/rvclaw /data/rvclaw/{models,runs,logs,cache,src}
sudo chown -R "$USER":"$USER" /opt/rvclaw /data/rvclaw

cd /opt/rvclaw
git clone https://github.com/lyd1992/RVClaw.git
cd RVClaw
```

后续更新：

```bash
cd /opt/rvclaw/RVClaw
git pull --ff-only
```

## 3. 安装 Web 依赖

优先用 venv，避免系统 Python 限制全局安装：

```bash
cd /opt/rvclaw/RVClaw
python3 -m venv /data/rvclaw/venv
source /data/rvclaw/venv/bin/activate
python3 -m pip install -e '.[api]'
```

每次新 SSH 窗口如果使用 venv，需要先激活：

```bash
source /data/rvclaw/venv/bin/activate
```

## 4. 安装 spacemit-llama.cpp

```bash
cd /data/rvclaw/src
wget https://archive.spacemit.com/spacemit-ai/llama.cpp/spacemit-llama.cpp.riscv64.0.0.8.tar.gz
tar -xzvf spacemit-llama.cpp.riscv64.0.0.8.tar.gz
ln -sfn spacemit-llama.cpp.riscv64.0.0.8 spacemit-llama.cpp
```

## 5. 下载模型

正式 K3 演示默认使用厂商文档推荐的 Qwen3-30B-A3B：

```bash
wget https://www.modelscope.cn/models/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/resolve/master/Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf -P ~/
mkdir -p /data/rvclaw/models
ln -sfn ~/Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf \
  /data/rvclaw/models/Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf
```

如果只是快速确认链路，也可以下载小模型 smoke 包，并临时覆盖环境变量：

```bash
wget https://modelscope.cn/models/unsloth/Qwen3-0.6B-GGUF/resolve/master/Qwen3-0.6B-Q4_0.gguf \
  -O /data/rvclaw/models/planner-smoke.gguf

export RVCLAW_LLAMA_MODEL=Qwen3-0.6B
export RVCLAW_LLAMA_MODEL_PATH=/data/rvclaw/models/planner-smoke.gguf
```

## 6. 先跑 mock 基线

```bash
cd /opt/rvclaw/RVClaw
source deploy/k3/env.sh
python3 -m rvclaw run "检查 A-03 区域设备状态并生成报告" \
  --planner mock \
  --runs-dir /data/rvclaw/runs \
  --json
```

预期：`status=completed`，并输出 6 个 tool calls。

## 7. 启动 llama-server

用一个 SSH/tmux 窗口保持服务：

```bash
cd /opt/rvclaw/RVClaw
source deploy/k3/env.sh
bash deploy/k3/run_llama_server.sh
```

另开一个窗口确认模型服务：

```bash
curl http://127.0.0.1:9090/v1/models | jq
```

## 8. 启动 Web + CV 演示

```bash
cd /opt/rvclaw/RVClaw
source deploy/k3/env.sh
bash deploy/k3/run_web_demo.sh
```

浏览器打开：

```text
http://<K3-IP>:8088
```

输入：

```text
检查 A-03 区域设备状态并生成报告
```

页面应展示 skill timeline、原图、标注图、运行状态、`metrics.json`、`trace.jsonl`、`report.md` 和 `raw.log`。

## 9. 常见分支

想验证 Safety Guard：

```bash
python3 -m rvclaw run "移动到 Z-99 区域并拍照" \
  --planner mock \
  --runs-dir /data/rvclaw/runs \
  --json
```

预期是 `failed`，但 run 目录和证据包仍会生成。

想只跑 CLI llama.cpp demo：

```bash
bash deploy/k3/run_demo.sh
```

想跑 benchmark：

```bash
python3 benchmarks/run_agent_e2e.py --repeat 3 --planner llama_cpp --runs-dir /data/rvclaw/runs
```

## 10. 继续阅读

- `docs/k3_web_cv_demo.md`：Web 页面、CV sample、演示脚本和验收项。
- `deploy/k3/install.md`：完整安装命令和环境变量参考。
- `docs/k3_ssh_deployment.md`：详细 SSH 操作、排障和历史 smoke 验收记录。
- `docs/openclaw_adapter_contract.md`：v0.2 后续 OpenClaw/ROS2/飞书 sidecar 合同。
