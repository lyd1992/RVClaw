from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from rvclaw.api import run_demo
from rvclaw.benchmark.mnn_rvv import resolve_test_spec, summarize_results, write_speedup_svg


class MnnRvvBenchmarkTest(unittest.TestCase):
    def test_resolve_function_aliases(self):
        spec = resolve_test_spec("softmax")
        self.assertEqual(spec.function_name, "MNNSoftmax")
        self.assertEqual(spec.test_binary, "test_softmax")

    def test_summarize_and_svg(self):
        spec = resolve_test_spec("MNNSoftmax")
        rows = [
            {"test": "test_softmax", "config": "axis=32", "passed": True, "speedup": 2.0, "scalar_s": 1.0, "rvv_s": 0.5},
            {"test": "test_softmax", "config": "axis=64", "passed": True, "speedup": 4.0, "scalar_s": 1.0, "rvv_s": 0.25},
        ]
        summary = summarize_results(spec, rows)
        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(summary["speedup_mean"], 3.0)
        with tempfile.TemporaryDirectory() as tmp:
            chart = write_speedup_svg(Path(tmp) / "chart.svg", spec, rows)
            self.assertIn("MNNSoftmax", chart.read_text(encoding="utf-8"))

    def test_run_demo_imports_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"
            output.mkdir()
            _write_jsonl(
                output / "all_results.jsonl",
                [
                    {"test": "test_softmax", "config": "axis=32", "passed": True, "speedup": 2.0, "scalar_s": 1.0, "rvv_s": 0.5, "vlen": 128, "isa": "rv64gcv"},
                    {"test": "test_softmax", "config": "axis=64", "passed": True, "speedup": 4.0, "scalar_s": 1.0, "rvv_s": 0.25, "vlen": 128, "isa": "rv64gcv"},
                ],
            )
            (output / "report.md").write_text("# source\n", encoding="utf-8")
            old_output = os.environ.get("RVCLAW_MNN_RVV_OUTPUT_DIR")
            os.environ["RVCLAW_MNN_RVV_OUTPUT_DIR"] = str(output)
            try:
                summary = run_demo(
                    "测试MNNSoftmax在RVV上的优化提升，使用已有output",
                    runs_dir=root / "runs",
                    planner_name="mock",
                )
            finally:
                if old_output is None:
                    os.environ.pop("RVCLAW_MNN_RVV_OUTPUT_DIR", None)
                else:
                    os.environ["RVCLAW_MNN_RVV_OUTPUT_DIR"] = old_output
            self.assertEqual(summary.status, "completed")
            metrics = json.loads(Path(summary.metrics_path).read_text(encoding="utf-8"))
            self.assertEqual(metrics["benchmark_function"], "MNNSoftmax")
            self.assertEqual(metrics["speedup_mean"], 3.0)
            self.assertTrue((Path(summary.run_dir) / "artifacts" / "mnn_speedup_bar.svg").is_file())
            self.assertTrue((Path(summary.run_dir) / "artifacts" / "mnn_rvv" / "all_results.jsonl").is_file())

    def test_unsupported_framework_keeps_failed_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_demo("测试PyTorchConv优化提升", runs_dir=Path(tmp) / "runs", planner_name="mock")
            self.assertEqual(summary.status, "failed")
            report = Path(summary.report_path).read_text(encoding="utf-8")
            self.assertIn("当前仅支持MNNRVV静态函数测试", report)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    unittest.main()
