from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rvclaw.api import run_demo
from rvclaw.benchmark.mnn_rvv import (
    parse_text_results,
    resolve_test_spec,
    run_benchmark,
    single_test_shell_command,
    summarize_results,
    write_speedup_svg,
)


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

    def test_refresh_with_local_output_reruns_before_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workdir = root / "mnn_rvv_tests"
            output = workdir / "output"
            output.mkdir(parents=True)
            bin_dir = workdir / "bin"
            bin_dir.mkdir()
            (bin_dir / "test_softmax").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            _write_jsonl(
                output / "all_results.jsonl",
                [
                    {"test": "test_softmax", "config": "axis=32", "passed": True, "speedup": 2.0, "scalar_s": 1.0, "rvv_s": 0.5},
                ],
            )
            old_output = os.environ.get("RVCLAW_MNN_RVV_OUTPUT_DIR")
            old_workdir = os.environ.get("RVCLAW_SG2044_WORKDIR")
            os.environ["RVCLAW_MNN_RVV_OUTPUT_DIR"] = str(output)
            os.environ["RVCLAW_SG2044_WORKDIR"] = str(workdir)
            try:
                with patch("rvclaw.benchmark.mnn_rvv.run_command") as run_command:
                    run_command.return_value.stdout = "reran\n"
                    run_command.return_value.stderr = ""
                    result = run_benchmark("MNN", "MNNSoftmax", "single", True, root / "artifacts")
            finally:
                if old_output is None:
                    os.environ.pop("RVCLAW_MNN_RVV_OUTPUT_DIR", None)
                else:
                    os.environ["RVCLAW_MNN_RVV_OUTPUT_DIR"] = old_output
                if old_workdir is None:
                    os.environ.pop("RVCLAW_SG2044_WORKDIR", None)
                else:
                    os.environ["RVCLAW_SG2044_WORKDIR"] = old_workdir
            self.assertEqual(result["benchmark_source"], "local_refresh")
            self.assertTrue(result["refresh"])
            run_command.assert_called_once()
            called_args = run_command.call_args.args[0]
            self.assertIn("bin/test_softmax", called_args[2])
            self.assertNotIn("run_all_and_report.sh", called_args[2])

    def test_single_test_shell_command_targets_only_one_binary(self):
        spec = resolve_test_spec("MNNSoftmax")
        command = single_test_shell_command("/data/zl/mnn_rvv_tests", spec)
        self.assertIn("bin/test_softmax", command)
        self.assertIn("output/all_results.jsonl", command)
        self.assertNotIn("run_all_and_report.sh", command)

    def test_parse_single_binary_text_output(self):
        spec = resolve_test_spec("MNNSoftmax")
        rows = parse_text_results(
            "\n".join(
                [
                    "size=100",
                    "Scalar time: 0.0001 sec",
                    "RVV time   : 0.0001 sec",
                    "Speedup    : 1.00x",
                    "Test size=100: PASSED",
                    "size=1024",
                    "Scalar time: 0.0002 sec",
                    "RVV time   : 0.0001 sec",
                    "Speedup    : 2.00x",
                    "Test size=1024: PASSED",
                ]
            ),
            spec,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["test"], "test_softmax")
        self.assertEqual(rows[0]["config"], "size=100")
        self.assertEqual(rows[1]["speedup"], 2.0)
        self.assertTrue(rows[1]["passed"])

    def test_parse_generic_config_text_output(self):
        spec = resolve_test_spec("MNNMatrixProd")
        rows = parse_text_results(
            "\n".join(
                [
                    "M=16,N=32,K=64",
                    "Scalar time: 0.0010 sec",
                    "RVV time   : 0.0005 sec",
                    "Speedup    : 2.00x",
                    "Test M=16,N=32,K=64: PASSED",
                ]
            ),
            spec,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["test"], "test_matrix_prod")
        self.assertEqual(rows[0]["config"], "M=16,N=32,K=64")
        self.assertEqual(rows[0]["speedup"], 2.0)
        self.assertTrue(rows[0]["passed"])

    def test_parse_inline_speedup_text_output(self):
        spec = resolve_test_spec("MNNConvRunForLineInt8")
        rows = parse_text_results(
            "\n".join(
                [
                    "w=4 sdq=1 Speedup: 2.62x Test: PASSED",
                    "w=4 sdq=2 Speedup: infx Test: PASSED",
                ]
            ),
            spec,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["config"], "w=4 sdq=1")
        self.assertEqual(rows[0]["speedup"], 2.62)
        self.assertTrue(rows[0]["passed"])
        self.assertEqual(rows[1]["speedup"], 0.0)
        self.assertTrue(rows[1]["passed"])

    def test_parse_pipe_speedup_text_output(self):
        spec = resolve_test_spec("MNNC3ToBGR555")
        rows = parse_text_results(
            "\n".join(
                [
                    "[RGB] count=1024 | Scalar: 0.000001 s | RVV: 0.000002 s | Speedup: 0.50x",
                    "[BGR] count=1024 | Scalar: 0.000001 s | RVV: 0.000001 s | Speedup: 1.00x",
                    "Test size=1024 PASSED",
                ]
            ),
            spec,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["config"], "[RGB] count=1024")
        self.assertEqual(rows[0]["scalar_s"], 0.000001)
        self.assertEqual(rows[0]["rvv_s"], 0.000002)
        self.assertEqual(rows[1]["speedup"], 1.0)
        self.assertTrue(rows[0]["passed"])
        self.assertTrue(rows[1]["passed"])

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
