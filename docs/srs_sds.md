# Demo Claw v0.1.1 SRS/SDS Mapping

## Functional Requirements

| ID | Requirement | Current implementation |
|---|---|---|
| FR-001 | Submit natural-language tasks through CLI or Web/API | `src/rvclaw/cli.py`, `src/rvclaw/api.py`, `src/rvclaw/web/` |
| FR-002 | Generate structured `task.yaml` | `RunRecorder.write_task()` |
| FR-003 | Claude CLI / llama.cpp PlannerBackend outputs JSON `tool_calls` | `ClaudeCliPlannerBackend`, `LlamaCppPlannerBackend` |
| FR-004 | Skill Registry whitelist validation | `skills/registry.yaml`, `SkillRegistry` |
| FR-005 | Safety Guard validates skill name, arguments, timeout, confirmation, and stop policy | `SafetyGuard.validate()` |
| FR-006 | Mock Skills | `skills/builtin.py` |
| FR-007 | SQLite/JSONL event memory and device profile query | `SQLiteEventStore`; JSONL remains an import/export format |
| FR-008 | Mock Device / CV sample Device, with future ROS2/OpenClaw adapters | `MockDevice`, `CVSampleDevice`, `docs/openclaw_adapter_contract.md` |
| FR-009 | Generate `run_id` and run directory for each run | `run_demo()`, `RunRecorder` |
| FR-010 | Output metrics/trace/report/raw log | `RunRecorder` |
| FR-011 | Benchmark Schema | `benchmarks/benchmark_schema.yaml` |
| FR-012 | Replaceable PlannerBackend | `planner_from_name()` supports `auto`, `mock`, `claude_cli`, `llama_cpp` |
| FR-013 | Replaceable RuntimeBackend | `runtime/runtime_api.py` and backend placeholders |
| FR-014 | Replaceable MemoryBackend | `MemoryManager` and `SQLiteEventStore` boundaries |
| FR-015 | Web console can inspect run history, artifacts, and benchmark CSV | `src/rvclaw/web/` |
| FR-016 | Record planner output mode | `planner_mode` in `metrics.json` and `trace.jsonl` |
| FR-017 | Configurable zone whitelist | `configs/zones.yaml` |

## Design Modules

| Module | Responsibility | Files |
|---|---|---|
| Task Intake | Receive user tasks and generate `task.yaml` | `cli.py`, `api.py`, `observability.py`, `web/` |
| Agent Core | Execution loop, status convergence, error handling | `agent/core.py` |
| PlannerBackend | Generate JSON `tool_calls`; record direct/repaired/fallback modes | `agent/planner.py` |
| Memory Manager | SQLite/flat memory query and write | `memory/` |
| Skill Registry | Whitelist, parameter schema, safety level, zone constraints | `skills/registry.yaml`, `configs/zones.yaml` |
| Safety Guard | Validate legality, safety, and executability | `agent/safety_guard.py` |
| Device Adapter | Mock Device, CV sample Device, future ROS2/OpenClaw | `adapters/` |
| Web Console | Submit tasks, view run history, evidence files, and image artifacts | `web/` |
| Observability | trace, metrics, report, raw log | `observability.py` |
