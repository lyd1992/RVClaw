import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "robot" / "ros2_ws" / "src" / "rvclaw_algo"))
sys.path.insert(0, str(REPO_ROOT / "services" / "rvclaw_report"))
sys.path.insert(0, str(REPO_ROOT / "services" / "rvclaw_api"))

from rvclaw_api.sample_flow import run_sample_flow


SAMPLE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360" viewBox="0 0 640 360">
  <rect width="640" height="360" fill="#f3f4f6"/>
  <rect x="220" y="70" width="200" height="160" rx="8" fill="#111827"/>
  <circle cx="320" cy="150" r="62" fill="#f9fafb" stroke="#374151" stroke-width="6"/>
  <path d="M320 150 L356 118" stroke="#dc2626" stroke-width="7" stroke-linecap="round"/>
  <text x="320" y="265" text-anchor="middle" font-family="monospace" font-size="24" fill="#111827">A-03 PRESSURE</text>
</svg>
"""


class SampleInspectionFlowTest(unittest.TestCase):
    def test_inspection_result_contract_declares_runtime_and_detections(self):
        schema_path = REPO_ROOT / "contracts" / "inspection_result.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        properties = schema["properties"]
        self.assertIn("runtime", properties)
        self.assertIn("detections", properties)
        self.assertIn("bbox", properties["detections"]["items"]["properties"])

    def test_sample_image_flow_writes_json_annotation_and_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            image_path = work_dir / "sample_meter.svg"
            image_path.write_text(SAMPLE_SVG, encoding="utf-8")

            payload = run_sample_flow(
                image_path=image_path,
                output_root=work_dir / "runs",
                task_id="task-sample-001",
                point_id="A-03",
                category="meter_reading",
            )

            run_id = payload["run"]["run_id"]
            run_dir = work_dir / "runs" / run_id
            result_path = run_dir / "inspection_result.json"
            annotated_path = run_dir / "annotated_image.svg"
            report_path = run_dir / "report.md"

            self.assertTrue(result_path.exists())
            self.assertTrue(annotated_path.exists())
            self.assertTrue(report_path.exists())

            result = json.loads(result_path.read_text(encoding="utf-8"))
            for key in ["task_id", "point_id", "category", "is_anomaly", "timestamp"]:
                self.assertIn(key, result)
            self.assertEqual(result["task_id"], "task-sample-001")
            self.assertEqual(result["point_id"], "A-03")
            self.assertEqual(result["category"], "meter_reading")
            self.assertEqual(result["runtime"]["target"], "K3 CoM260 local")
            self.assertGreater(len(result["detections"]), 0)
            self.assertIn("bbox", result["detections"][0])

            annotated = annotated_path.read_text(encoding="utf-8")
            self.assertIn("<rect", annotated)
            self.assertIn("data:image/svg+xml;base64", annotated)
            self.assertIn("meter_reading", annotated)

            report = report_path.read_text(encoding="utf-8")
            self.assertIn("task-sample-001", report)
            self.assertIn("A-03", report)
            self.assertIn("meter_reading", report)

            self.assertEqual(payload["inspection"]["task_id"], "task-sample-001")
            self.assertEqual(payload["artifacts"]["annotated_image_url"], f"/artifacts/{run_id}/annotated_image.svg")
            self.assertEqual(payload["artifacts"]["report_url"], f"/artifacts/{run_id}/report.md")


if __name__ == "__main__":
    unittest.main()
