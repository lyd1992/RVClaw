from __future__ import annotations

import base64
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - optional API dependency
    TestClient = None

from rvclaw.web.app import create_app


ONE_PIXEL_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP4z8AAAAMBAQDJ/pLv"
    "AAAAAElFTkSuQmCC"
)


@unittest.skipIf(TestClient is None, "FastAPI TestClient is not installed")
class WebCommandCenterTest(unittest.TestCase):
    def test_upload_image_ref_can_drive_vision_run(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            app = create_app(runs_dir=Path(scratch) / "runs", planner="mock")
            client = TestClient(app)

            upload = client.post(
                "/api/uploads",
                files={"file": ("panel.png", base64.b64decode(ONE_PIXEL_PNG), "image/png")},
            )
            self.assertEqual(upload.status_code, 200)
            upload_payload = upload.json()
            self.assertRegex(upload_payload["image_ref"], r"^upload:[a-f0-9]{32}\.png$")

            preview = client.get(f"/api/uploads/{upload_payload['upload_id']}")
            self.assertEqual(preview.status_code, 200)
            self.assertEqual(preview.content[:8], b"\x89PNG\r\n\x1a\n")

            created = client.post(
                "/api/runs",
                json={
                    "goal": "检测图片中的目标并生成结论",
                    "planner": "mock",
                    "image_ref": upload_payload["image_ref"],
                },
            )
            self.assertEqual(created.status_code, 200)
            self.assertEqual(created.json()["status"], "running")

            detail = self._wait_for_run(client, created.json()["run_id"])
            self.assertEqual(detail["summary"]["status"], "completed")
            self.assertEqual(detail["metrics"]["vision_task"], "object_detection")
            self.assertIn("artifacts/vision_result.json", detail["files"])
            self.assertTrue(any(name.endswith("_capture.png") for name in detail["files"]))

    def test_upload_rejects_unsupported_file_type_and_external_image_ref(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            app = create_app(runs_dir=Path(scratch) / "runs", planner="mock")
            client = TestClient(app)

            bad_upload = client.post(
                "/api/uploads",
                files={"file": ("note.txt", b"not an image", "text/plain")},
            )
            self.assertEqual(bad_upload.status_code, 400)

            bad_run = client.post(
                "/api/runs",
                json={
                    "goal": "检测图片中的目标并生成结论",
                    "planner": "mock",
                    "image_ref": "https://example.com/panel.png",
                },
            )
            self.assertEqual(bad_run.status_code, 400)

    def test_index_contains_command_center_controls_not_inline_history_list(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            app = create_app(runs_dir=Path(scratch) / "runs", planner="mock")
            client = TestClient(app)

            html = client.get("/").text
            self.assertIn("Agent Command Center", html)
            self.assertIn("taskTemplate", html)
            self.assertIn("historyDrawer", html)
            self.assertIn("Runtime Stack Map", html)
            self.assertIn("agentGraph", html)
            self.assertNotIn('id="runs" class="timeline"', html)

    def _wait_for_run(self, client: TestClient, run_id: str) -> dict:
        deadline = time.time() + 5
        while time.time() < deadline:
            response = client.get(f"/api/runs/{run_id}")
            self.assertEqual(response.status_code, 200)
            detail = response.json()
            if detail["summary"]["status"] != "running":
                return detail
            time.sleep(0.05)
        self.fail(f"run {run_id} did not finish")


class WebUploadAnnotationTest(unittest.TestCase):
    def test_upload_endpoint_uses_globally_resolvable_uploadfile_alias(self) -> None:
        source = (ROOT / "src" / "rvclaw" / "web" / "app.py").read_text(encoding="utf-8")
        self.assertIn("FastAPIUploadFile: Any = Any", source)
        self.assertIn("globals()[\"FastAPIUploadFile\"] = UploadFile", source)
        self.assertIn("file: FastAPIUploadFile = File(...)", source)
        self.assertNotIn("file: UploadFile = File(...)", source)


if __name__ == "__main__":
    unittest.main()
