# RVClaw Development Status

Updated: 2026-05-22

Checkpoint tag: `v0.1.0-k3-llama-smoke`

Current code checkpoint target: `v0.1.3-k3-agent-command-center`

## Summary

RVClaw has reached the first K3 smoke checkpoint and now has a v0.1.3 Agent Command Center implementation ready for K3 validation. The v0.1.0 checkpoint proved the Demo Claw software loop on K3 Pico-ITX 32GB with local `spacemit-llama.cpp`; v0.1.1 added the Web/CV surface; v0.1.2 added `analyze_image` and a DemoZoo sidecar bridge; v0.1.3 makes the Agent workflow visible with image upload, an execution graph, Safety Guard nodes, a Runtime Stack Map, and a history drawer.

This checkpoint proves the RVClaw runtime path is executable and auditable on the target RISC-V edge box:

```text
browser or CLI natural-language task / uploaded image
  -> task.yaml
  -> llama.cpp / mock PlannerBackend
  -> Agent Core
  -> Safety Guard
  -> Tool Router
  -> Mock / CV sample / DemoZoo Device + SQLite memory
  -> trace.jsonl / metrics.json / report.md / raw.log
```

This is still not ROS2/OpenClaw real-device control, real camera capture, or MNN/ONNX vision inference. Those remain v0.2+ adapter work.

## Validated On K3

Environment observed during validation:

| Item | Value |
|---|---|
| Board | K3 Pico-ITX 32GB |
| OS | Bianbu / RISC-V Linux |
| Python | 3.14.3 |
| Local LLM server | `spacemit-llama.cpp` |
| Formal demo model | Qwen3-30B-A3B-Instruct-2507-Q4_0 GGUF |
| Smoke model | `planner-smoke.gguf` / Qwen3-0.6B GGUF |
| RVClaw data root | `/data/rvclaw` |
| RVClaw source root | `/opt/rvclaw/RVClaw` |

Validated commands:

```bash
source deploy/k3/env.sh
bash deploy/k3/run_demo.sh
python3 -m rvclaw run "返回 BASE" --planner mock --runs-dir /data/rvclaw/runs --json
python3 -m rvclaw run "移动到 B-01 区域并拍照" --planner llama_cpp --runs-dir /data/rvclaw/runs --json
python3 -m unittest discover -s tests
python3 benchmarks/run_agent_e2e.py --repeat 3 --planner mock --runs-dir /data/rvclaw/runs
```

For v0.1.1 Web + CV validation, start from `docs/k3_start_here.md`, then run:

```bash
source deploy/k3/env.sh
bash deploy/k3/run_llama_server.sh
bash deploy/k3/run_web_demo.sh
```

For v0.1.2 DemoZoo validation, also see `docs/k3_demozoo_bridge.md`. For the current Web demo, use `docs/k3_web_cv_demo.md`.

Known successful K3 runs:

| Run ID | Goal | Planner | Expected result |
|---|---|---|---|
| `run-20260521T083749Z` | 检查 A-03 区域设备状态并生成报告 | `llama_cpp` | `completed`, 6-step inspection workflow |
| `run-20260521T091220Z` | 返回 BASE | `mock` | `completed`, moves to `BASE` |
| `run-20260521T091245Z` | 移动到 B-01 区域并拍照 | `llama_cpp` | historical pre-v0.1.1 result: `failed`, unsupported target handled with artifacts |

## Implemented

- CLI and Python API entrypoints for natural-language tasks.
- Deterministic `task.yaml` generation from CLI task intake.
- `mock`, `claude_cli`, `auto`, and `llama_cpp` PlannerBackend selection.
- OpenAI-compatible `llama.cpp` planner adapter for local `llama-server`.
- K3-specific environment and run scripts under `deploy/k3/`.
- K3 first-run guide, SSH deployment runbook, and Web + CV demo runbook.
- Skill whitelist, argument schema checks, timeout defaults, and failed-run artifact capture.
- Configurable zone whitelist loaded from `configs/zones.yaml` with `A-03`, `B-01`, and `BASE`.
- Built-in mock skills:
  - `memory_query`
  - `move_to`
  - `capture_image`
  - `detect_status`
  - `speak`
  - `upload_report`
  - `stop`
- SQLite event memory and flat retrieval baseline.
- Mock Device movement, image placeholder capture, status detection, speak, upload, and stop behavior.
- CV sample device backend that writes real image artifacts:
  - `artifacts/a03_capture.png`
  - `artifacts/a03_annotated.png`
- Multi-vision `analyze_image` skill with normalized `vision_result.json`.
- DemoZoo sidecar bridge for real classification, object detection, segmentation, and face detection.
- `cv_sample` remains a smoke/fallback backend only; `RVCLAW_REQUIRE_REAL_VISION=1` disables fallback and fails clearly if DemoZoo is unavailable.
- Web CV display policy now follows task semantics: classification uses one image, detection/segmentation/face/inspection use input plus processed image, and non-vision tasks show no image.
- Agent Command Center Web UI with task templates, active preset state, local image upload, Agent execution graph, Runtime Stack Map, and history drawer.
- Explicit `safety_guard.approved` / `safety_guard.rejected` trace events for Web graph visualization.
- Web console and API:
  - `GET /api/health`
  - `POST /api/uploads`
  - `GET /api/uploads/{upload_id}`
  - `POST /api/runs`
  - `GET /api/runs`
  - `GET /api/runs/{run_id}`
  - `GET /api/runs/{run_id}/files`
  - `GET /api/benchmarks`
- `planner_mode`, `device_backend`, `vision_backend`, and `vision_task` metrics for demo explainability.
- E2E benchmark CSV with K3/llama.cpp environment metadata.
- Planner hardening for small-model instability:
  - repairs incomplete inspection plans to the deterministic 6-step workflow;
  - repairs complete-but-invalid inspection or vision plans to schema-safe deterministic workflows;
  - falls back to deterministic inspection workflow on malformed JSON for default inspection tasks;
  - records unsupported/non-inspection planner failures as `failed` artifacts instead of Python tracebacks.

## Current Supported Behaviors

Default inspection:

```text
检查 A-03 区域设备状态并生成报告
```

Expected tool calls:

```text
memory_query
move_to(A-03)
capture_image(A-03)
detect_status(A-03)
speak
upload_report
```

Return to base:

```text
返回 BASE
```

Expected tool calls:

```text
memory_query
move_to(BASE)
speak
```

`B-01` is now in the default zone whitelist and can be used by mock/CV sample workflows. Unknown targets such as `Z-99` are expected to fail safely while still writing artifacts.

## Not Yet Implemented

- Real USB camera capture in `capture_image`.
- ROS2/OpenClaw real-device adapter execution.
- Real robot chassis or actuator control.
- Real speaker output.
- Real report upload service.
- MNN/ONNX/CV model-backed anomaly detection.
- Milvus/Knowhere high-performance memory backend.
- Interactive human confirmation UI.
- Face identity recognition or face-library matching.

## Acceptance Criteria For This Checkpoint

The checkpoint is considered valid when all of the following hold on K3:

```bash
git log -1 --oneline
source deploy/k3/env.sh
curl http://127.0.0.1:9090/v1/models | jq
bash deploy/k3/run_demo.sh | tee /tmp/rvclaw_llama_run.json
jq -r '.status' /tmp/rvclaw_llama_run.json
jq -r '.tool_calls[].name' /tmp/rvclaw_llama_run.json
python3 -m unittest discover -s tests
bash deploy/k3/run_web_demo.sh
```

Expected `run_demo.sh` summary:

```text
completed
memory_query
move_to
capture_image
detect_status
speak
upload_report
```

Every run must leave:

```text
task.yaml
metrics.json
trace.jsonl
report.md
raw.log
artifacts/
```

## Next Steps

1. Add a real camera-backed `capture_image` path while preserving mock fallback.
2. Improve the OpenCV detector from sample-image annotation to a real status-light or anomaly rule set.
3. Validate v0.1.3 image upload and Agent graph on physical K3.
4. Add a real camera backend behind the same `capture_image` contract.
5. Add RuntimeBackend and MemoryBackend capability probes before wiring MNN/vLLM/Milvus into the main chain.
6. Prepare ROS2/OpenClaw adapter contracts for v0.2, but keep real-device control gated by Safety Guard and explicit confirmation.
