from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.run_agent_e2e import build_benchmark_row
from rvclaw.models import RunSummary


class BenchmarkMetadataTest(unittest.TestCase):
    def test_benchmark_row_records_k3_llama_environment(self) -> None:
        old_env = {
            "RVCLAW_LLAMA_MODEL": os.environ.get("RVCLAW_LLAMA_MODEL"),
            "RVCLAW_LLAMA_MODEL_PATH": os.environ.get("RVCLAW_LLAMA_MODEL_PATH"),
            "RVCLAW_LLAMA_BASE_URL": os.environ.get("RVCLAW_LLAMA_BASE_URL"),
            "RVCLAW_LLAMA_THREADS": os.environ.get("RVCLAW_LLAMA_THREADS"),
            "RVCLAW_PLATFORM_SOC": os.environ.get("RVCLAW_PLATFORM_SOC"),
        }
        try:
            os.environ["RVCLAW_LLAMA_MODEL"] = "Qwen3-0.6B"
            os.environ["RVCLAW_LLAMA_MODEL_PATH"] = "/data/rvclaw/models/planner-smoke.gguf"
            os.environ["RVCLAW_LLAMA_BASE_URL"] = "http://127.0.0.1:9090/v1"
            os.environ["RVCLAW_LLAMA_THREADS"] = "8"
            os.environ["RVCLAW_PLATFORM_SOC"] = "K3-Pico-ITX-32GB"
            summary = RunSummary(
                run_id="run-test",
                status="completed",
                run_dir="/data/rvclaw/runs/run-test",
                report_path="report.md",
                metrics_path="metrics.json",
                trace_path="trace.jsonl",
                task_path="task.yaml",
                tool_calls=[],
            )

            row = build_benchmark_row(summary, planner="llama_cpp", elapsed_ms=12.5)

            self.assertEqual(row["planner_model"], "Qwen3-0.6B")
            self.assertEqual(row["planner_model_path"], "/data/rvclaw/models/planner-smoke.gguf")
            self.assertEqual(row["planner_base_url"], "http://127.0.0.1:9090/v1")
            self.assertEqual(row["llama_threads"], "8")
            self.assertEqual(row["platform_soc"], "K3-Pico-ITX-32GB")
            self.assertTrue(row["task_success"])
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
