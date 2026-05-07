from __future__ import annotations

from pathlib import Path

from rvclaw.utils import ensure_dir, utc_now_iso


class MockDevice:
    def __init__(self, artifact_dir: str | Path):
        self.artifact_dir = ensure_dir(artifact_dir)
        self.current_zone = "BASE"
        self.stopped = False

    def move_to(self, target: str) -> dict:
        self.current_zone = target
        return {
            "target": target,
            "pose": {"zone": target, "x": 3.2, "y": 1.4, "theta": 0.0},
            "status": "arrived",
        }

    def capture_image(self, target: str, mode: str = "inspection") -> dict:
        image_ref = self.artifact_dir / f"{target.lower().replace('-', '_')}_{mode}.txt"
        image_ref.write_text(
            f"mock image placeholder\nzone={target}\nmode={mode}\nts={utc_now_iso()}\n",
            encoding="utf-8",
        )
        return {
            "target": target,
            "mode": mode,
            "image_ref": str(image_ref),
            "status": "captured",
        }

    def detect_status(self, target: str, image_ref: str | None = None) -> dict:
        return {
            "target": target,
            "image_ref": image_ref,
            "device": "edge-inference-node-a03",
            "status": "normal",
            "signals": {
                "temperature_c": 42.1,
                "vibration": "normal",
                "smoke": "none",
                "path_blocked": False,
            },
            "risk_level": "low",
        }

    def speak(self, text: str) -> dict:
        return {"text": text, "status": "spoken"}

    def upload_report(self, title: str, run_id: str) -> dict:
        return {
            "title": title,
            "run_id": run_id,
            "remote_uri": f"mock://reports/{run_id}",
            "status": "uploaded",
        }

    def stop(self, reason: str = "manual stop") -> dict:
        self.stopped = True
        return {"stopped": True, "reason": reason}
