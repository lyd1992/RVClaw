from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rvclaw.api import run_demo
from rvclaw.agent.safety_guard import SafetyGuard, SkillRegistry
from rvclaw.config import load_zone_ids
from rvclaw.models import ToolCall


class ZoneConfigTest(unittest.TestCase):
    def test_load_zone_ids_accepts_simple_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "zones.yaml"
            path.write_text(
                'version: "0.1.1"\n'
                "zones:\n"
                "  - id: A-03\n"
                "  - id: B-01\n"
                "  - id: BASE\n",
                encoding="utf-8",
            )

            self.assertEqual(load_zone_ids(path), ["A-03", "B-01", "BASE"])

    def test_default_registry_allows_configured_b01_zone(self) -> None:
        guard = SafetyGuard(SkillRegistry.from_default())

        checked = guard.validate(ToolCall("move_to", {"target": "B-01"}))

        self.assertEqual(checked.arguments["target"], "B-01")

    def test_safety_guard_strips_unknown_arguments_before_skill_execution(self) -> None:
        guard = SafetyGuard(SkillRegistry.from_default())

        checked = guard.validate(ToolCall("memory_query", {"query": "A-03", "context": "extra"}))

        self.assertEqual(checked.arguments, {"query": "A-03"})

    def test_unknown_zone_is_rejected_by_safety_guard(self) -> None:
        guard = SafetyGuard(SkillRegistry.from_default())

        with self.assertRaisesRegex(ValueError, "outside whitelist"):
            guard.validate(ToolCall("move_to", {"target": "Z-99"}))

    def test_mock_inspection_uses_requested_b01_zone(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            summary = run_demo(
                goal="检查 B-01 区域设备状态并生成报告",
                runs_dir=Path(scratch),
                planner_name="mock",
                run_id="test-b01-zone",
            )

            self.assertEqual(summary.status, "completed")
            self.assertEqual(summary.tool_calls[1]["name"], "move_to")
            self.assertEqual(summary.tool_calls[1]["arguments"]["target"], "B-01")
            metrics = json.loads(Path(summary.metrics_path).read_text(encoding="utf-8"))
            self.assertTrue(metrics["task_success"])


if __name__ == "__main__":
    unittest.main()
