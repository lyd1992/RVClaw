# RVClaw Architecture

RVClaw 采用“先闭环、后优化”的 Demo Claw v0.1 架构。

## 八层架构

1. RISC-V Hardware + OS：SG2044、K3、openEuler/EulixOS、GCC/Clang。
2. RVV Kernel Layer：Attention、GEMM、Pack/Reorder、Dynamic Quant、SQ、Distance、MPMI、ROrder。
3. AI Runtime：llama.cpp baseline，后续接入 MNN、vLLM、ONNX Runtime。
4. RAG + Memory：SQLite/flat baseline，后续接入 Knowhere/Milvus/RVFlow。
5. Device Adapter：Mock Device、ROS 2、OpenClaw、Camera、IMU、Mic、底盘、控制器。
6. Agent Orchestrator：Planner、State Machine、Tool Router、Task Context。
7. Safety + Observability：Skill 白名单、人工接管、日志、trace、replay、diagnostics。
8. Robot / Fleet Apps：巡检、实验室运维、园区安防、仓储 AMR、低空节点等。

## v0.1 执行链路

```text
CLI/API task
  -> Task Intake
  -> PlannerBackend(auto/mock/claude)
  -> Agent Core
  -> Safety Guard
  -> Tool Router
  -> Skills(memory_query/move_to/capture_image/detect_status/speak/upload_report/stop)
  -> Mock Device + SQLite Memory
  -> Observability(task/trace/metrics/report/raw log)
```

## 插件边界

- `RuntimeBackend`：LLM、embedding、视觉、小模型、量化和后端指标。
- `MemoryManager`：事件库、设备画像、小规模检索，后续接 Knowhere/Milvus。
- `Skill Registry`：skill 白名单、参数 schema、安全级别。
- `Device Adapter`：隔离 Mock Device、ROS 2、OpenClaw 和真实传感器。
- `Observability`：统一 run artifacts，支撑 replay、benchmark 和验收。
