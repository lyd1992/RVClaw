"""Installed entry point for the named waypoint client."""

import os
from pathlib import Path
import runpy


def main() -> None:
    lab_root = Path(os.environ.get("RVCLAW_LAB_ROOT", "/data/rvclaw-k3-lab"))
    script = lab_root / "scripts" / "rvclaw_waypoint_client.py"
    runpy.run_path(str(script), run_name="__main__")
