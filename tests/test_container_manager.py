from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rvclaw.container.manager import build_mnn_script, format_mnn_plan


class ContainerManagerTest(unittest.TestCase):
    def test_mnn_plan_mentions_required_scripts(self) -> None:
        plan = format_mnn_plan()
        self.assertIn("docker/sg2044/gcc15.1/build_image.sh", plan)
        self.assertIn("docker/sg2044/mnn/run_container.sh", plan)
        self.assertIn("runs/env/<run_id>/container_manifest.json", plan)

    def test_mnn_script_uses_gcc15_image_and_ref(self) -> None:
        script = build_mnn_script(mnn_ref="test-ref", build_dir="build/test-mnn")
        self.assertIn("rvclaw/sg2044-gcc15.1:openeuler-riscv64", script)
        self.assertIn("RVCLAW_MNN_REF=\"${RVCLAW_MNN_REF:-test-ref}\"", script)
        self.assertIn("RVCLAW_MNN_BUILD_DIR=\"${RVCLAW_MNN_BUILD_DIR:-build/test-mnn}\"", script)


if __name__ == "__main__":
    unittest.main()
