from __future__ import annotations

import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rvclaw.api import run_demo
from rvclaw.web.uploads import resolve_upload_image_ref, save_upload_bytes


ONE_PIXEL_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP4z8AAAAMBAQDJ/pLv"
    "AAAAAElFTkSuQmCC"
)


class UploadsAndImageRefTest(unittest.TestCase):
    def test_save_upload_bytes_returns_local_image_ref_and_rejects_unknown_ref(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            uploads_dir = Path(scratch) / "uploads"
            payload = save_upload_bytes(
                uploads_dir=uploads_dir,
                filename="Panel.PNG",
                content=base64.b64decode(ONE_PIXEL_PNG),
            )

            self.assertRegex(payload["upload_id"], r"^[a-f0-9]{32}\.png$")
            self.assertEqual(payload["image_ref"], f"upload:{payload['upload_id']}")
            resolved = resolve_upload_image_ref(uploads_dir, payload["image_ref"])
            self.assertTrue(resolved.is_file())

            with self.assertRaises(ValueError):
                resolve_upload_image_ref(uploads_dir, "https://example.com/panel.png")
            with self.assertRaises(ValueError):
                save_upload_bytes(uploads_dir=uploads_dir, filename="panel.txt", content=b"not image")

    def test_run_demo_can_use_image_ref_as_cv_sample_source(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch) / "uploaded.png"
            source.write_bytes(base64.b64decode(ONE_PIXEL_PNG))

            summary = run_demo(
                goal="检测图片中的目标并生成结论",
                runs_dir=Path(scratch) / "runs",
                planner_name="mock",
                run_id="uploaded-image-run",
                image_ref=source,
            )

            self.assertEqual(summary.status, "completed")
            metrics = json.loads(Path(summary.metrics_path).read_text(encoding="utf-8"))
            self.assertEqual(metrics["device_backend"], "cv_sample")
            self.assertEqual(metrics["vision_source"], str(source))
            self.assertEqual(metrics["vision_task"], "object_detection")
            self.assertTrue((Path(summary.run_dir) / "artifacts" / "vision_result.json").exists())


if __name__ == "__main__":
    unittest.main()
