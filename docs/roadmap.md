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

## Current Checkpoint: v0.1.3 K3 Agent Command Center

- [x] Add image upload with local-only `upload:<id>` references.
- [x] Move history into a drawer so the main screen focuses on the active run.
- [x] Upgrade the Web UI from a log timeline to an Agent execution graph.
- [x] Add explicit Safety Guard trace events for approved/rejected tool calls.
- [x] Add Runtime Stack Map for active, fallback, and reserved backends.
- [ ] Validate upload + Agent graph on the physical K3.

## Previous Checkpoint: v0.1.2 K3 Multi-Vision DemoZoo Bridge

- [x] Add `analyze_image` as the unified vision skill.
- [x] Support classification, object detection, segmentation, and face detection task mapping.
- [x] Add DemoZoo HTTP sidecar bridge with `cv_sample` fallback.
- [x] Upgrade Web UI with visual task presets and result cards.
- [ ] Validate DemoZoo sidecar on the physical K3.

## Earlier Checkpoint: v0.1.1 K3 Web + CV Demo

- [x] Add FastAPI-compatible Web/API surface for runs, history, artifacts, and benchmark CSV.
- [x] Add configurable zones with `A-03`, `B-01`, and `BASE`.
- [x] Add `planner_mode` metrics for direct, repaired, fallback, failed, and mock planner paths.
- [x] Add sample-image CV device backend with mock fallback.
- [x] Default K3 formal demo model to Qwen3-30B-A3B GGUF while preserving the 0.6B smoke model.
- [ ] Validate the Web demo on the physical K3.

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
