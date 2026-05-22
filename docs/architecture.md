# RVClaw Architecture

RVClaw follows a "close the loop first, optimize later" architecture. The
current K3 path is a demo-grade edge Agent Runtime: RVClaw owns task intake,
planning, safety, skills, evidence, and Web visualization; local model services
such as llama.cpp or DemoZoo are sidecars behind stable interfaces.

## Eight-Layer View

1. RISC-V Hardware + OS: K3 Pico-ITX, SG2044, Bianbu/Ubuntu/openEuler.
2. RVV Kernel Layer: attention, GEMM, quantization, distance, reorder.
3. AI Runtime: llama.cpp baseline, later MNN, ONNX Runtime, vLLM.
4. RAG + Memory: SQLite/JSONL/flat baseline, later Knowhere/Milvus.
5. Device Adapter: mock, cv_sample, DemoZoo sidecar, future ROS2/OpenClaw.
6. Agent Orchestrator: Planner, Agent Core, Tool Router, Task Context.
7. Safety + Observability: Skill whitelist, guard, trace, replay, metrics.
8. Robot / Fleet Apps: inspection, lab operations, AMR, security, diagnostics.

## v0.1 Core Loop

```text
CLI/API task
  -> Task Intake
  -> PlannerBackend(auto/mock/claude/llama_cpp)
  -> Agent Core
  -> Safety Guard
  -> Tool Router
  -> Skills(memory_query/move_to/capture_image/detect_status/speak/upload_report/stop)
  -> Mock/CV sample Device + SQLite Memory
  -> Observability(task/trace/metrics/report/raw log)
```

## v0.1.1 K3 Web + CV Loop

```text
Web console
  -> POST /api/runs
  -> run_demo()
  -> llama.cpp Planner or deterministic repair
  -> configurable zone Safety Guard
  -> cv_sample capture/detect
  -> artifacts/a03_capture.png + artifacts/a03_annotated.png
  -> Web timeline + artifact viewer
```

## v0.1.2 Multi-Vision Loop

```text
Web preset or natural-language vision task
  -> Planner / deterministic vision workflow
  -> analyze_image
  -> DemoZoo HTTP sidecar
  -> classification / object_detection / segmentation / face_detection
  -> normalized vision_result.json
  -> annotated image + result cards + metrics + report
```

If DemoZoo is unavailable, the same `analyze_image` skill falls back to
`cv_sample` and records `vision_backend=mock_fallback` or `cv_sample` in the run
evidence. Face support is detection-only in v0.1.2; RVClaw does not perform
identity recognition, face matching, or face-library management.

## v0.1.3 Agent Command Center Loop

```text
Web task template / uploaded image
  -> local upload:image_ref validation
  -> async POST /api/runs
  -> Agent graph polling trace.jsonl
  -> Planner + Safety Guard nodes
  -> Skill execution nodes
  -> Runtime Stack Map + evidence file viewer
```

The main Web screen focuses on the current run. Historical runs are loaded from
a drawer so roadshow demos can keep the story on the active Agent workflow.

## Stable Interfaces

- `PlannerBackend`: converts task context into JSON `tool_calls`. Current local
  K3 path uses OpenAI-compatible `llama-server` plus deterministic repair.
- `Skill Registry`: defines allowed skill names, schemas, enums, timeouts, and
  safety levels.
- `Device Adapter`: hides mock, CV sample, DemoZoo, and future ROS2/OpenClaw
  implementations behind the same skill functions.
- `Memory Manager`: handles SQLite/JSONL memory and is the future expansion
  point for Knowhere/Milvus.
- `Observability`: every run must leave `task.yaml`, `trace.jsonl`,
  `metrics.json`, `report.md`, `raw.log`, and task-specific artifacts.

## Main Verification Documents

- First K3 setup and verification: `docs/k3_start_here.md`.
- Agent Command Center + CV demo: `docs/k3_web_cv_demo.md`.
- Optional v0.1.2 DemoZoo sidecar: `docs/k3_demozoo_bridge.md`.
- Current checkpoint and known gaps: `docs/development_status.md`.
