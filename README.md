# RVClaw

RVClaw 是 Demo Claw v0.1 的最小可运行框架，定位为面向具身智能设备的 RISC-V 端侧 Agent Runtime。当前仓库先把任务闭环、接口边界、mock device、SQLite 记忆、白名单 skill、安全校验、trace/metrics/report 产物跑通，再逐步接入 llama.cpp、MNN、vLLM、Knowhere/Milvus 等优化后端。

一句话口径：

> RVClaw 不是机器人整机，也不是大模型训练项目，而是 RISC-V 具身智能设备上可交付、可复现、可运维的软件底座。

## 当前目标

- 主验收平台：SG2044 / RVV VLEN=128。
- Demo v0.1：设备巡检助手，默认任务为“检查 A-03 区域设备状态并生成报告”。
- 默认闭环：Task Intake -> PlannerBackend -> Agent Core -> Safety Guard -> Skill Router -> Mock Device/Memory -> Observability。
- 运行产物：`task.yaml`、`trace.jsonl`、`metrics.json`、`report.md`、`raw.log`。
- 后端策略：v0.1 不强依赖 vLLM/MNN/Knowhere；它们在接口稳定后作为插件接入。

## 快速开始

```powershell
cd RVClaw
$env:PYTHONPATH = "src"
python -m rvclaw run --planner mock
```

也可以提交自定义任务：

```powershell
$env:PYTHONPATH = "src"
python -m rvclaw run "检查 A-03 区域设备状态并生成报告" --planner mock
```

运行完成后会生成类似目录：

```text
runs/run-20260507T023000Z/
  task.yaml
  metrics.json
  trace.jsonl
  report.md
  raw.log
  artifacts/
```

如已安装 Claude CLI，可使用：

```powershell
$env:PYTHONPATH = "src"
python -m rvclaw run "检查 A-03 区域设备状态并生成报告" --planner claude
```

`--planner auto` 会优先探测 Claude CLI，不存在时回退到 mock planner。

## 代码结构

```text
src/rvclaw/
  agent/              # Agent Core、Planner、Safety Guard、Tool Router
  adapters/           # Mock Device，后续 ROS2/OpenClaw
  memory/             # SQLite event store 与 flat vector baseline
  runtime/            # RuntimeBackend 接口与后端占位
  skills/             # Skill Registry 与内置 mock skills
  api.py              # run_demo 编程接口
  cli.py              # CLI
benchmarks/           # Demo e2e 与后端 benchmark 入口
deploy/sg2044/        # SG2044 部署说明和启动脚本
docs/                 # 架构、SRS/SDS、路线图、运维说明
examples/             # 示例任务和记忆种子
tests/                # 基础回归测试
```

## 测试

```powershell
cd RVClaw
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
```

## Benchmark

```powershell
cd RVClaw
$env:PYTHONPATH = "src"
python benchmarks/run_agent_e2e.py --repeat 3 --planner mock
```

结果会写入 `runs/benchmark_agent_e2e.csv`，字段遵循 `benchmarks/benchmark_schema.yaml`。

## 路线图

近期路线是：

1. Demo Claw：SG2044 上可运行、可复现、可验收的 mock/baseline 样机。
2. Robot Agent：单机任务型智能体，接入真实设备 adapter 与本地模型 baseline。
3. Embodied Claw：单机长期值守智能体，增加事件循环、长期记忆、异常处理。
4. Fleet Claw：多机协同、OTA、日志回放、远程诊断与运维。

更多细节见 [docs/architecture.md](docs/architecture.md)、[docs/srs_sds.md](docs/srs_sds.md) 和 [docs/roadmap.md](docs/roadmap.md)。

## 状态

当前为 v0.1 skeleton。接口优先于单点性能，mock/baseline 优先于优化后端。所有性能数字必须记录平台、模型、后端、commit/build flags 和测试环境。
