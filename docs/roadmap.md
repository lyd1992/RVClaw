# Roadmap

## Week 1: Demo Claw Skeleton

- 固化代码结构和接口边界。
- 跑通 mock planner、mock device、SQLite memory、skill registry。
- 生成标准 run artifacts。
- 建立基础 e2e benchmark。

## Week 2: SG2044 Baseline

- 在 SG2044 上固定 Python、OS、compiler、RVV VLEN 记录口径。
- 接入 llama.cpp/GGUF baseline。
- 建立模型、量化、tokens/s、TTFT 的最小指标集。

## Week 3: Device and Vision Baseline

- 固化 ROS 2/OpenClaw schema。
- 增加 mock camera/IMU/controller 数据回放。
- 预留 MNN/ONNX 视觉后端插件接口。

## Week 4: Reportable Demo

- 打包 Demo Claw v0.1 runbook。
- 输出可复现 benchmark CSV。
- 完成 BP/路演可演示脚本。
- 梳理 Robot Agent 与 Embodied Claw 的下一阶段需求。
