import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class StudioStaticAssetsTest(unittest.TestCase):
    def test_studio_page_contains_sample_flow_view(self):
        page = (REPO_ROOT / "studio" / "web" / "index.html").read_text(encoding="utf-8")

        self.assertIn("/api/sample-flow", page)
        self.assertIn("annotated-image", page)
        self.assertIn("json-output", page)
        self.assertIn("report-link", page)
        self.assertIn("RVClaw EdgeOne Studio", page)


if __name__ == "__main__":
    unittest.main()
