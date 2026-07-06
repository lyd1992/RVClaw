import json
from datetime import datetime, timezone
from pathlib import Path

from rvclaw_algo import LocalInspectionRecognizer
from rvclaw_report import write_markdown_report


def run_sample_flow(image_path, output_root, task_id, point_id, category):
    run_id = _build_run_id(task_id, point_id)
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    recognizer = LocalInspectionRecognizer()
    result = recognizer.inspect(
        image_path=Path(image_path),
        task_id=task_id,
        point_id=point_id,
        category=category,
    )

    result_path = run_dir / "inspection_result.json"
    annotated_path = run_dir / "annotated_image.svg"
    report_path = run_dir / "report.md"

    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    recognizer.save_annotated_svg(image_path, result, annotated_path)
    write_markdown_report(result, annotated_path.name, report_path)

    return {
        "run": {
            "run_id": run_id,
            "created_at": result["timestamp"],
            "status": "completed",
        },
        "inspection": result,
        "artifacts": {
            "inspection_result_url": f"/artifacts/{run_id}/inspection_result.json",
            "annotated_image_url": f"/artifacts/{run_id}/annotated_image.svg",
            "report_url": f"/artifacts/{run_id}/report.md",
        },
    }


def _build_run_id(task_id, point_id):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_task = _slug(task_id)
    safe_point = _slug(point_id)
    return f"{timestamp}-{safe_task}-{safe_point}"


def _slug(value):
    allowed = []
    for char in str(value).lower():
        if char.isalnum():
            allowed.append(char)
        elif char in {"-", "_"}:
            allowed.append(char)
        else:
            allowed.append("-")
    return "".join(allowed).strip("-") or "run"
