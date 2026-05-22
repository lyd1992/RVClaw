from __future__ import annotations

from rvclaw.agent.planner import PlannerBackend
from rvclaw.agent.tool_router import ToolRouter
from rvclaw.memory.memory_api import MemoryManager
from rvclaw.models import RunSummary, Task
from rvclaw.observability import RunRecorder
from rvclaw.utils import platform_profile, utc_now_iso


class AgentCore:
    def __init__(
        self,
        planner: PlannerBackend,
        router: ToolRouter,
        memory: MemoryManager,
        recorder: RunRecorder,
        run_metadata: dict | None = None,
    ):
        self.planner = planner
        self.router = router
        self.memory = memory
        self.recorder = recorder
        self.run_metadata = run_metadata or {}

    def run(self, goal: str, source: str = "cli") -> RunSummary:
        task = Task(
            task_id=self.recorder.run_id,
            goal=goal,
            created_at=utc_now_iso(),
            source=source,
            platform=platform_profile(),
            metadata={"demo": "Demo Claw v0.1", "planner": self.planner.name},
        )
        self.recorder.write_task(task.to_dict())
        self.recorder.trace("task.received", task.to_dict())
        self.recorder.raw(f"Task received: {goal}")

        memory_context = self.memory.query(goal, limit=5)
        self.recorder.trace("memory.context", {"matches": memory_context})

        status = "completed"
        results = []
        planner_error = None
        planner_mode = getattr(self.planner, "last_mode", self.planner.name)
        try:
            calls = self.planner.plan(task, memory_context)
            planner_mode = getattr(self.planner, "last_mode", planner_mode)
            self.recorder.trace(
                "planner.completed",
                {
                    "planner": self.planner.name,
                    "planner_mode": planner_mode,
                    "tool_calls": [c.to_dict() for c in calls],
                },
            )

            for call in calls:
                result = self.router.execute(call)
                results.append({"call": call.to_dict(), "result": result.to_dict()})
                if not result.ok:
                    status = "failed"
                    break
                if call.name == "stop":
                    status = "stopped"
                    break
        except Exception as exc:
            status = "failed"
            calls = []
            planner_error = str(exc)
            planner_mode = getattr(self.planner, "last_mode", "failed")
            self.recorder.trace("planner.failed", {"planner": self.planner.name, "planner_mode": planner_mode, "error": planner_error})
            self.recorder.raw(f"Planner failed: {planner_error}")

        report = self._build_report(task, status, results, planner_error=planner_error)
        self.recorder.write_report(report)
        self.memory.remember(
            "inspection_report",
            {"run_id": self.recorder.run_id, "status": status, "report_path": str(self.recorder.report_path)},
            tags=["run", self.recorder.run_id],
        )

        metrics = {
            "run_id": self.recorder.run_id,
            "status": status,
            "planner": self.planner.name,
            "planner_mode": planner_mode,
            "platform": task.platform,
            "task_success": status == "completed",
            "tool_call_count": len(results),
            "artifacts": {
                "task": str(self.recorder.task_path),
                "trace": str(self.recorder.trace_path),
                "report": str(self.recorder.report_path),
                "raw_log": str(self.recorder.raw_log_path),
            },
        }
        metrics.update(self.run_metadata)
        metrics.update(_vision_metrics_from_results(results))
        if planner_error:
            metrics["planner_error"] = planner_error
        self.recorder.write_metrics(metrics)
        self.recorder.trace("run.finished", metrics)

        return RunSummary(
            run_id=self.recorder.run_id,
            status=status,
            run_dir=str(self.recorder.run_dir),
            report_path=str(self.recorder.report_path),
            metrics_path=str(self.recorder.metrics_path),
            trace_path=str(self.recorder.trace_path),
            task_path=str(self.recorder.task_path),
            tool_calls=[item["call"] for item in results],
        )

    @staticmethod
    def _build_report(task: Task, status: str, results: list[dict], planner_error: str | None = None) -> str:
        lines = [
            f"# RVClaw Inspection Report - {task.task_id}",
            "",
            f"- Goal: {task.goal}",
            f"- Status: {status}",
            f"- Planner: {task.metadata.get('planner')}",
            f"- Platform: {task.platform.get('soc')} / RVV VLEN={task.platform.get('rvv_vlen')}",
            "",
            "## Tool Calls",
            "",
        ]
        if planner_error:
            lines.extend(
                [
                    "## Planner Error",
                    "",
                    planner_error,
                    "",
                ]
            )
        for index, item in enumerate(results, start=1):
            call = item["call"]
            result = item["result"]
            lines.append(f"{index}. `{call['name']}` -> {'ok' if result['ok'] else 'failed'}")
            if result.get("error"):
                lines.append(f"   - Error: {result['error']}")
            output = result.get("output") or {}
            if "status" in output:
                lines.append(f"   - Status: {output['status']}")
            if "risk_level" in output:
                lines.append(f"   - Risk: {output['risk_level']}")
            if "remote_uri" in output:
                lines.append(f"   - Upload: {output['remote_uri']}")
            if "summary" in output:
                lines.append(f"   - Vision: {output['summary']}")
            if "objects" in output:
                lines.append(f"   - Objects: {len(output.get('objects') or [])}")
            if "faces" in output:
                lines.append(f"   - Faces: {len(output.get('faces') or [])}")
        lines.extend(
            [
                "",
                "## Acceptance Artifacts",
                "",
                "This run produced task.yaml, trace.jsonl, metrics.json, report.md, and raw.log.",
                "",
            ]
        )
        return "\n".join(lines)


def _vision_metrics_from_results(results: list[dict]) -> dict:
    for item in reversed(results):
        call = item.get("call", {})
        output = (item.get("result", {}) or {}).get("output", {}) or {}
        if call.get("name") == "analyze_image":
            metrics = {
                "vision_task": output.get("task"),
                "vision_backend": output.get("backend"),
                "vision_backend_detail": output.get("backend_detail"),
                "vision_model": output.get("model"),
                "vision_requested_model": output.get("requested_model"),
                "vision_model_fallback_reason": output.get("model_fallback_reason"),
                "vision_summary": output.get("summary"),
                "vision_latency_ms": output.get("latency_ms"),
                "objects_count": len(output.get("objects") or []),
                "labels_count": len(output.get("labels") or []),
                "segments_count": len(output.get("segments") or []),
                "faces_count": len(output.get("faces") or []),
            }
            return {key: value for key, value in metrics.items() if value is not None}
    return {}
