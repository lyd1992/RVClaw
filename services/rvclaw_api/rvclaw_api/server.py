import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .person_tracking import get_person_tracking_payload
from .sample_flow import run_sample_flow
from .video_detection import get_video_detection_payload
from .vision_state import get_vision_state


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IMAGE = REPO_ROOT / "samples" / "sample_meter.svg"
DEFAULT_RUNS = REPO_ROOT / "runs"
STUDIO_ROOT = REPO_ROOT / "studio" / "web"


class RVClawRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/sample-flow":
            self._write_json(self._run_sample_flow())
            return
        if parsed.path == "/api/person-tracking":
            self._write_json(get_person_tracking_payload())
            return
        if parsed.path == "/api/vision-state":
            self._write_json(get_vision_state())
            return
        if parsed.path == "/api/video-detections":
            self._write_json(get_video_detection_payload())
            return
        if parsed.path.startswith("/artifacts/"):
            self._serve_artifact(parsed.path.removeprefix("/artifacts/"))
            return
        if parsed.path.startswith("/samples/"):
            self._serve_sample(parsed.path.removeprefix("/samples/"))
            return
        self._serve_studio(parsed.path)

    def _run_sample_flow(self):
        return run_sample_flow(
            image_path=DEFAULT_IMAGE,
            output_root=DEFAULT_RUNS,
            task_id="studio-sample-001",
            point_id="A-03",
            category="meter_reading",
        )

    def _serve_artifact(self, relative_path):
        target = (DEFAULT_RUNS / relative_path).resolve()
        runs_root = DEFAULT_RUNS.resolve()
        if not str(target).startswith(str(runs_root)) or not target.exists() or target.is_dir():
            self.send_error(404)
            return
        self._write_bytes(target.read_bytes(), _content_type(target))

    def _serve_sample(self, relative_path):
        samples_root = (REPO_ROOT / "samples").resolve()
        target = (samples_root / relative_path).resolve()
        if not str(target).startswith(str(samples_root)) or not target.exists() or target.is_dir():
            self.send_error(404)
            return
        self._write_bytes(target.read_bytes(), _content_type(target))

    def _serve_studio(self, request_path):
        relative = request_path.lstrip("/") or "index.html"
        target = (STUDIO_ROOT / relative).resolve()
        studio_root = STUDIO_ROOT.resolve()
        if not str(target).startswith(str(studio_root)) or not target.exists() or target.is_dir():
            self.send_error(404)
            return
        self._write_bytes(target.read_bytes(), _content_type(target))

    def _write_json(self, payload):
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self._write_bytes(body, "application/json; charset=utf-8")

    def _write_bytes(self, body, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _content_type(path):
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def main():
    parser = argparse.ArgumentParser(description="RVClaw local Studio API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8017, type=int)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), RVClawRequestHandler)
    print(f"RVClaw Studio API listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
