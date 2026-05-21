# RVClaw Development Status

Updated: 2026-05-21

Checkpoint tag: `v0.1.0-k3-llama-smoke`

## Summary

RVClaw has reached the first K3 smoke checkpoint: the Demo Claw v0.1 software loop runs on K3 Pico-ITX 32GB with a local `spacemit-llama.cpp` server and a Qwen3-0.6B GGUF smoke model.

This checkpoint proves the RVClaw runtime path is executable and auditable on the target RISC-V edge box:

```text
natural-language task
  -> task.yaml
  -> llama.cpp / mock PlannerBackend
  -> Agent Core
  -> Safety Guard
  -> Tool Router
  -> Mock Skills + SQLite memory
  -> trace.jsonl / metrics.json / report.md / raw.log
```

This is still a Mock Device software loop. It is not yet ROS2/OpenClaw real-device control, real camera capture, or MNN/ONNX vision inference.

## Validated On K3

Environment observed during validation:

| Item | Value |
|---|---|
| Board | K3 Pico-ITX 32GB |
| OS | Bianbu / RISC-V Linux |
| Python | 3.14.3 |
| Local LLM server | `spacemit-llama.cpp` |
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

Known successful K3 runs:

| Run ID | Goal | Planner | Expected result |
|---|---|---|---|
| `run-20260521T083749Z` | 检查 A-03 区域设备状态并生成报告 | `llama_cpp` | `completed`, 6-step inspection workflow |
| `run-20260521T091220Z` | 返回 BASE | `mock` | `completed`, moves to `BASE` |
| `run-20260521T091245Z` | 移动到 B-01 区域并拍照 | `llama_cpp` | `failed`, unsupported target handled with artifacts |

## Implemented

- CLI and Python API entrypoints for natural-language tasks.
- Deterministic `task.yaml` generation from CLI task intake.
- `mock`, `claude_cli`, `auto`, and `llama_cpp` PlannerBackend selection.
- OpenAI-compatible `llama.cpp` planner adapter for local `llama-server`.
- K3-specific environment and run scripts under `deploy/k3/`.
- K3 SSH deployment and verification runbook.
- Skill whitelist, argument schema checks, timeout defaults, and failed-run artifact capture.
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
- E2E benchmark CSV with K3/llama.cpp environment metadata.
- Planner hardening for small-model instability:
  - repairs incomplete inspection plans to the deterministic 6-step workflow;
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

Unsupported targets such as `B-01` are expected to fail safely unless the registry is expanded.

## Not Yet Implemented

- Real USB camera capture in `capture_image`.
- ROS2/OpenClaw real-device adapter execution.
- Real robot chassis or actuator control.
- Real speaker output.
- Real report upload service.
- MNN/ONNX/CV model-backed anomaly detection.
- Milvus/Knowhere high-performance memory backend.
- Web/FastAPI control console.
- Interactive human confirmation UI.

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
2. Add a small rule-based or OpenCV status detector before introducing MNN/ONNX.
3. Expand the skill registry from `A-03`/`BASE` to a configurable zone list.
4. Add a `planner_mode` metric that records whether `llama_cpp` output was used directly or repaired/fallback.
5. Add a simple FastAPI endpoint after the CLI path remains stable on K3.
6. Prepare ROS2/OpenClaw adapter contracts for v0.2, but keep real-device control gated by Safety Guard and explicit confirmation.
