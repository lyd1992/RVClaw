# RVClaw

RVClaw 是 Demo Claw v0.1 的最小可运行框架，面向 RISC-V Linux 环境，优先在 SG2044 / openEuler 等 RISC-V Linux 系统上完成安装、部署、运行、测试和验收。

当前阶段的目标不是追求单点 kernel 性能，而是先把具身智能 Agent 的任务闭环跑通：

```text
自然语言任务
  -> PlannerBackend
  -> Agent Core
  -> Safety Guard
  -> Skill Router
  -> Memory + Mock Device
  -> trace / metrics / report 运行产物
```

一句话定位：

> RVClaw 不是机器人整机，也不是大模型训练项目，而是 RISC-V 具身智能设备上可交付、可复现、可运维的软件底座。

## 目标运行环境

第一阶段主验收环境：

| 项目 | 当前口径 |
|---|---|
| SoC | SG2044 |
| ISA | RV64GCV / RVV 1.0 |
| RVV VLEN | 128 |
| OS | openEuler RISC-V 或兼容 RISC-V Linux |
| Python | Python 3.10+ |
| 外部依赖 | mock Demo Claw 主链路零第三方依赖 |

`K3 / VLEN=256` 只作为后续兼容验证目标，不作为 v0.1 首要验收平台。

## 在 openEuler / RISC-V Linux 上安装

```bash
sudo dnf install -y git python3
git clone https://github.com/lyd1992/RVClaw.git
cd RVClaw
python3 --version
```

不安装包，直接运行：

```bash
export PYTHONPATH="$PWD/src"
python3 -m rvclaw run --planner mock
```

可选：以 editable 模式安装：

```bash
sudo dnf install -y python3-pip
python3 -m pip install -e .
rvclaw run --planner mock
```

## 运行 Demo Claw

默认巡检任务：

```bash
export PYTHONPATH="$PWD/src"
python3 -m rvclaw run --planner mock
```

自定义任务：

```bash
python3 -m rvclaw run "检查 A-03 区域设备状态并生成报告" --planner mock
```

输出 JSON 摘要：

```bash
python3 -m rvclaw run "检查 A-03 区域设备状态并生成报告" --planner mock --json
```

每次运行都会在 `runs/` 下生成一个运行目录：

```text
runs/run-YYYYMMDDTHHMMSSZ/
  task.yaml
  metrics.json
  trace.jsonl
  report.md
  raw.log
  artifacts/
```

这些文件是 Demo Claw v0.1 的基本验收产物。

## 在 RISC-V Linux 上测试

```bash
export PYTHONPATH="$PWD/src"
python3 -m unittest discover -s tests
python3 -m compileall -q src tests benchmarks
```

运行端到端 benchmark：

```bash
python3 benchmarks/run_agent_e2e.py --repeat 3 --planner mock
```

benchmark 结果会写入：

```text
runs/benchmark_agent_e2e.csv
```

指标字段参考：

```text
benchmarks/benchmark_schema.yaml
```

## SG2044 辅助脚本

```bash
source deploy/sg2044/env.sh
bash deploy/sg2044/run_demo.sh
```

部署说明：

- `deploy/sg2044/install.md`
- `docs/operations.md`
- `docs/mnn_container_build.md`

## K3 Pico-ITX / llama.cpp 辅助脚本

K3 作为边缘盒子演示环境时，代码仍以本仓库为准，机器侧配置独立放在 `deploy/k3/` 和文档中：

```bash
source deploy/k3/env.sh
bash deploy/k3/run_llama_server.sh
bash deploy/k3/run_demo.sh
```

SSH 部署步骤见：

- `docs/k3_ssh_deployment.md`
- `deploy/k3/install.md`

默认巡检任务在 K3 上的 `llama_cpp` planner 预期输出 6 个 tool calls：

```text
memory_query -> move_to -> capture_image -> detect_status -> speak -> upload_report
```

小模型若只返回单个 `speak`，适配器会把巡检任务修复为上述 deterministic workflow，保证 v0.1 demo 验收看到完整闭环。

当前 mock planner 还支持 `返回 BASE` 这类基础动作，预期输出：

```text
memory_query -> move_to(BASE) -> speak
```

超出白名单或模型输出异常的非巡检任务应以 `failed` 状态写出运行产物，而不是让 CLI 打印 Python traceback。

## MNN Docker 构建流程

第一阶段以 MNN 为例打通容器化构建。需要先在 SG2044 上构建一次 GCC 15.1 工具链镜像：

```bash
bash docker/sg2044/gcc15.1/build_image.sh
bash docker/sg2044/gcc15.1/verify_image.sh
```

该镜像只内置 openEuler、GCC/G++ 15.1、CMake、Ninja、Python、git、make、ccache 等构建环境；MNN 源码和 build 产物不写进镜像，而是挂载到宿主机目录：

```text
third_party/MNN/
build/mnn-sg2044-gcc15/
runs/env/<run_id>/
```

查看 MNN 容器构建计划：

```bash
export PYTHONPATH="$PWD/src"
python3 -m rvclaw container doctor
python3 -m rvclaw container mnn plan
```

生成或执行 MNN 容器构建脚本：

```bash
python3 -m rvclaw container mnn script --output deploy/sg2044/mnn_container_build.generated.sh
bash deploy/sg2044/mnn_container_build.sh
```

构建完成后检查：

```bash
find build/mnn-sg2044-gcc15 -name "libMNN.so" -o -name "*benchmark*" -o -name "*MNN*"
cat runs/env/*/container_manifest.json
```

## 后端框架安装管理

RVClaw 可以统一管理 llama.cpp、MNN、vLLM 等可选后端的源码安装。v0.1 的 mock 主链路仍然保持零第三方依赖；这些后端通过 `rvclaw install` 作为可插拔能力接入。

检查 SG2044 / openEuler 环境：

```bash
export PYTHONPATH="$PWD/src"
python3 -m rvclaw doctor
```

列出支持的后端：

```bash
python3 -m rvclaw install list
```

查看安装计划，不执行：

```bash
python3 -m rvclaw install plan llama_cpp mnn
python3 -m rvclaw install plan vllm
```

生成可审计 shell 脚本：

```bash
python3 -m rvclaw install script llama_cpp mnn --output deploy/sg2044/install_backends.generated.sh
bash deploy/sg2044/install_backends.generated.sh
```

直接执行安装：

```bash
python3 -m rvclaw install run llama_cpp --yes
python3 -m rvclaw install run mnn --yes
```

SG2044 辅助入口：

```bash
bash deploy/sg2044/install_backends.sh llama_cpp
bash deploy/sg2044/install_backends.sh mnn
```

当前安装边界：

| 后端 | 当前定位 | 安装动作 |
|---|---|---|
| llama.cpp | P0 本地 LLM/GGUF baseline | clone 源码，CMake Release 构建，验证 `llama-cli` |
| MNN | P1 端侧视觉/小模型插件 | clone 源码，CMake Release 构建 CPU/tools/benchmark |
| vLLM | P1/P2 服务化 LLM 实验后端 | clone 源码，Python development install；不承诺 RISC-V 高性能 backend |

详细说明见 `docs/backend_installation.md`。

## 目前已经实现的功能

| 模块 | 当前状态 | 主要文件 |
|---|---|---|
| CLI 任务入口 | 已实现，支持自然语言任务提交 | `src/rvclaw/cli.py` |
| Python API | 已实现，提供 `run_demo()` | `src/rvclaw/api.py` |
| `task.yaml` 生成 | 已实现，每次 run 生成结构化任务文件 | `src/rvclaw/observability.py` |
| Mock PlannerBackend | 已实现，能生成固定巡检 tool_calls | `src/rvclaw/agent/planner.py` |
| Claude CLI PlannerBackend | 已实现适配器，需要本机存在 `claude` 命令 | `src/rvclaw/agent/planner.py` |
| Auto Planner | 已实现，优先探测 Claude CLI，不存在时回退 mock planner | `src/rvclaw/agent/planner.py` |
| Agent Core | 已实现基础执行循环、状态收敛和错误中止 | `src/rvclaw/agent/core.py` |
| Skill Registry | 已实现白名单 registry | `src/rvclaw/skills/registry.yaml` |
| Safety Guard | 已实现白名单、必填参数、类型、枚举、范围、timeout 默认值校验 | `src/rvclaw/agent/safety_guard.py` |
| Tool Router | 已实现 skill 调用分发和结果记录 | `src/rvclaw/agent/tool_router.py` |
| Mock Skills | 已实现 `memory_query`、`move_to`、`capture_image`、`detect_status`、`speak`、`upload_report`、`stop` | `src/rvclaw/skills/builtin.py` |
| SQLite 事件记忆 | 已实现，内置 A-03 设备画像和历史巡检 seed | `src/rvclaw/memory/sqlite_event_store.py` |
| Flat Vector baseline | 已实现轻量词法检索 baseline | `src/rvclaw/memory/flat_vector_store.py` |
| Mock Device | 已实现移动、拍照、状态检测、播报、上报、停止 mock 行为 | `src/rvclaw/adapters/mock_device.py` |
| 运行产物 | 已实现 `task.yaml`、`metrics.json`、`trace.jsonl`、`report.md`、`raw.log` | `src/rvclaw/observability.py` |
| E2E Benchmark | 已实现 mock 端到端 benchmark CSV 输出 | `benchmarks/run_agent_e2e.py` |
| 基础单元测试 | 已实现 demo run 产物检查 | `tests/test_demo_run.py` |
| SG2044 部署脚本 | 已实现环境变量和 demo 启动脚本 | `deploy/sg2044/` |
| 后端安装管理 | 已实现 `doctor`、`install list/plan/script/run` | `src/rvclaw/install/` |
| 后端安装文档 | 已实现 llama.cpp、MNN、vLLM 安装边界说明 | `docs/backend_installation.md` |

## 尚未实现或仅预留接口的功能

| 模块 | 当前状态 |
|---|---|
| FastAPI / Web API | 尚未实现；当前先提供 CLI 和 Python API |
| ROS 2 Adapter | 仅占位 |
| OpenClaw Adapter | 仅占位 |
| llama.cpp / GGUF RuntimeBackend | 推理适配器仍为占位；源码安装管理已实现 |
| MNN RuntimeBackend | 推理适配器仍为占位；源码安装管理已实现 |
| vLLM RuntimeBackend | 服务适配器仍为占位；源码/Python development 安装管理已实现 |
| ONNX Runtime 后端 | 仅占位 |
| Knowhere / Milvus MemoryBackend | 尚未实现 |
| 真实相机 / IMU / 底盘控制 | 尚未实现，当前为 Mock Device |
| 人工确认 UI | Safety Guard 预留边界，尚未实现交互式确认界面 |

## 仓库结构

```text
src/rvclaw/
  agent/              # Planner、Agent Core、Safety Guard、Tool Router
  adapters/           # Mock Device，后续 ROS2/OpenClaw
  memory/             # SQLite event store 与 flat retrieval baseline
  runtime/            # RuntimeBackend API 与后端占位
  skills/             # Skill Registry 与内置 mock skills
  api.py              # run_demo() 编程接口
  cli.py              # 命令行入口
benchmarks/           # 端到端 benchmark 入口
deploy/sg2044/        # SG2044 / openEuler 辅助脚本
docs/                 # 架构、SRS/SDS、运维、路线图
examples/             # 示例任务和 memory seed
tests/                # 基础回归测试
```

## Demo Claw v0.1 验收清单

在 SG2044 / openEuler 上，一次有效 demo run 应证明：

- CLI 能接收自然语言巡检任务。
- Planner 能生成受控 skill-call 序列。
- Skill Registry 和 Safety Guard 会校验每个 skill 调用。
- Mock Device 和 SQLite memory 能完成一次巡检闭环。
- 运行目录包含 `task.yaml`、`metrics.json`、`trace.jsonl`、`report.md`、`raw.log`。
- `python3 -m unittest discover -s tests` 通过。
- `benchmarks/run_agent_e2e.py` 能生成 benchmark CSV。

## 后续路线

近期路线：

```text
Demo Claw -> Robot Agent -> Embodied Claw -> Fleet Claw
```

MNN、vLLM、llama.cpp、Knowhere、Milvus 等后端应在 v0.1 主链路稳定后，通过统一接口作为可插拔优化后端接入。
