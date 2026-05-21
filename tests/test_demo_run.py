from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rvclaw.api import run_demo
from rvclaw.agent.core import AgentCore


class DemoRunTest(unittest.TestCase):
    def test_demo_run_outputs_acceptance_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            summary = run_demo(
                goal="检查 A-03 区域设备状态并生成报告",
                runs_dir=Path(scratch),
                planner_name="mock",
                run_id="test-run",
            )

            self.assertEqual(summary.status, "completed")
            for artifact in [
                summary.task_path,
                summary.metrics_path,
                summary.trace_path,
                summary.report_path,
                str(Path(summary.run_dir) / "raw.log"),
            ]:
                self.assertTrue(Path(artifact).exists(), artifact)

            metrics = json.loads(Path(summary.metrics_path).read_text(encoding="utf-8"))
            self.assertTrue(metrics["task_success"])
            self.assertEqual(metrics["tool_call_count"], 6)

    def test_mock_planner_return_to_base_uses_base_target(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            summary = run_demo(
                goal="返回 BASE",
                runs_dir=Path(scratch),
                planner_name="mock",
                run_id="test-return-base",
            )

            self.assertEqual(summary.status, "completed")
            self.assertEqual([call["name"] for call in summary.tool_calls], ["memory_query", "move_to", "speak"])
            self.assertEqual(summary.tool_calls[1]["arguments"]["target"], "BASE")

    def test_planner_exception_writes_failed_artifacts(self) -> None:
        class FailingPlanner:
            name = "failing"

            def plan(self, task, memory_context):
                raise RuntimeError("planner exploded")

        with tempfile.TemporaryDirectory() as scratch:
            from rvclaw.adapters.mock_device import MockDevice
            from rvclaw.agent.safety_guard import SafetyGuard, SkillRegistry
            from rvclaw.agent.tool_router import ToolRouter
            from rvclaw.memory.memory_api import MemoryManager
            from rvclaw.observability import RunRecorder
            from rvclaw.skills.builtin import build_builtin_skills

            runs_dir = Path(scratch)
            run_dir = runs_dir / "test-planner-failure"
            recorder = RunRecorder(run_id="test-planner-failure", run_dir=run_dir)
            memory = MemoryManager.from_path(runs_dir / "memory.sqlite3")
            device = MockDevice(artifact_dir=run_dir / "artifacts")
            guard = SafetyGuard(SkillRegistry.from_default())
            router = ToolRouter(build_builtin_skills(device, memory, "test-planner-failure"), guard, recorder)
            core = AgentCore(planner=FailingPlanner(), router=router, memory=memory, recorder=recorder)

            failed = core.run("移动到 B-01 区域并拍照")

            self.assertEqual(failed.status, "failed")
            self.assertEqual(failed.tool_calls, [])
            metrics = json.loads(Path(failed.metrics_path).read_text(encoding="utf-8"))
            self.assertFalse(metrics["task_success"])
            self.assertEqual(metrics["planner_error"], "planner exploded")
            self.assertTrue(Path(failed.report_path).exists())


if __name__ == "__main__":
    unittest.main()
