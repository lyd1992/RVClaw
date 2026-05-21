# Demo Claw v0.1 SRS/SDS Mapping

## 功能需求映射

| 编号 | 需求 | 当前实现 |
|---|---|---|
| FR-001 | CLI 或 Web/API 提交自然语言任务 | `src/rvclaw/cli.py`、`src/rvclaw/api.py`；Web/API 预留 optional dependency |
| FR-002 | 生成结构化 `task.yaml` | `RunRecorder.write_task()` |
| FR-003 | Claude CLI / llama.cpp PlannerBackend 输出 JSON tool_calls | `ClaudeCliPlannerBackend`、`LlamaCppPlannerBackend` |
| FR-004 | Skill Registry 白名单校验 | `skills/registry.yaml`、`SkillRegistry` |
| FR-005 | Safety Guard 校验参数、权限、超时、确认和急停 | `SafetyGuard.validate()`；人工确认接口预留 |
| FR-006 | Mock Skills | `skills/builtin.py` |
| FR-007 | SQLite/JSONL 事件记忆和设备画像查询 | `SQLiteEventStore`；JSONL 后续作为导入导出格式 |
| FR-008 | Mock Device，后续 ROS2/OpenClaw | `MockDevice`，`ros2_adapter.py`、`openclaw_adapter.py` 占位 |
| FR-009 | 每次运行生成 `run_id` 和运行目录 | `run_demo()`、`RunRecorder` |
| FR-010 | 输出 metrics/trace/report/raw log | `RunRecorder` |
| FR-011 | Benchmark Schema | `benchmarks/benchmark_schema.yaml` |
| FR-012 | PlannerBackend 可替换 | `planner_from_name()`；支持 `auto`、`mock`、`claude_cli`、`llama_cpp` |
| FR-013 | RuntimeBackend 可替换 | `runtime/runtime_api.py` 与后端占位 |
| FR-014 | MemoryBackend 可替换 | `MemoryManager` 与 `SQLiteEventStore` 边界 |

## 详细设计模块

| 模块 | 责任 | 文件 |
|---|---|---|
| Task Intake | 接收用户任务，生成 `task.yaml` | `cli.py`、`api.py`、`observability.py` |
| Agent Core | 状态机、流程控制、错误处理 | `agent/core.py` |
| PlannerBackend | 生成 JSON tool_calls | `agent/planner.py` |
| Memory Manager | SQLite/flat memory 查询与写入 | `memory/` |
| Skill Registry | 白名单、参数 schema、安全级别 | `skills/registry.yaml` |
| Safety Guard | 合法性、安全性、可执行性校验 | `agent/safety_guard.py` |
| Device Adapter | Mock Device，后续 ROS2/OpenClaw | `adapters/` |
| Observability | trace、metrics、report、raw log | `observability.py` |
