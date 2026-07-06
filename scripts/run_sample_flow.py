import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "robot" / "ros2_ws" / "src" / "rvclaw_algo"))
sys.path.insert(0, str(REPO_ROOT / "services" / "rvclaw_report"))
sys.path.insert(0, str(REPO_ROOT / "services" / "rvclaw_api"))

from rvclaw_api import run_sample_flow


def main():
    parser = argparse.ArgumentParser(description="Run RVClaw sample image inspection flow")
    parser.add_argument("--image", default=str(REPO_ROOT / "samples" / "sample_meter.svg"))
    parser.add_argument("--output-root", default=str(REPO_ROOT / "runs"))
    parser.add_argument("--task-id", default="sample-task-001")
    parser.add_argument("--point-id", default="A-03")
    parser.add_argument("--category", default="meter_reading")
    args = parser.parse_args()

    payload = run_sample_flow(
        image_path=Path(args.image),
        output_root=Path(args.output_root),
        task_id=args.task_id,
        point_id=args.point_id,
        category=args.category,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
