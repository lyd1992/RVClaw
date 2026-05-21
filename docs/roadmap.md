# Roadmap

## Current Checkpoint: K3 llama.cpp Smoke

Status as of 2026-05-21:

- K3 Pico-ITX 32GB + Bianbu/RISC-V Linux validated.
- Local `spacemit-llama.cpp` server reachable through OpenAI-compatible API.
- Qwen3-0.6B GGUF smoke model can drive the `llama_cpp` PlannerBackend.
- Default A-03 inspection task completes with the 6-step mock workflow.
- `返回 BASE` completes with `move_to(BASE)`.
- Unsupported targets such as `B-01` fail safely with run artifacts instead of Python tracebacks.

Checkpoint tag:

```text
v0.1.0-k3-llama-smoke
```

## Week 1: Demo Claw Skeleton

- [x] 固化代码结构和接口边界。
- [x] 跑通 mock planner、mock device、SQLite memory、skill registry。
- [x] 生成标准 run artifacts。
- [x] 建立基础 e2e benchmark。

## Week 2: K3 llama.cpp Baseline

- [x] 在 K3 上固定 OS、Python、模型、运行目录记录口径。
- [x] 接入 `spacemit-llama.cpp` / GGUF smoke baseline。
- [x] 建立 mock/llama.cpp E2E benchmark CSV。
- [ ] 补充 tokens/s、TTFT、内存峰值等更细指标进入 `metrics.json`。

## Week 3: Device and Vision Baseline

- 固化 ROS 2/OpenClaw schema。
- 增加 mock camera/IMU/controller 数据回放。
- 预留 MNN/ONNX 视觉后端插件接口。

## Week 4: Reportable Demo

- 打包 Demo Claw v0.1 runbook。
- 输出可复现 benchmark CSV。
- 完成 BP/路演可演示脚本。
- 梳理 Robot Agent 与 Embodied Claw 的下一阶段需求。
