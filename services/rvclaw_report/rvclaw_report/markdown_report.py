from pathlib import Path


def write_markdown_report(result, annotated_image_name, output_path):
    target = Path(output_path)
    status = "ANOMALY" if result["is_anomaly"] else "OK"
    lines = [
        "# RVClaw Inspection Report",
        "",
        f"- Task: {result['task_id']}",
        f"- Point: {result['point_id']}",
        f"- Category: {result['category']}",
        f"- Value: {result['value']}",
        f"- Confidence: {result['confidence']}",
        f"- Status: {status}",
        f"- Timestamp: {result['timestamp']}",
        f"- Runtime: {result['runtime']['target']} / {result['runtime']['backend']}",
        "",
        "## Evidence",
        "",
        f"![Annotated evidence]({annotated_image_name})",
        "",
        "## Structured Result",
        "",
        "```json",
        _to_json_block(result),
        "```",
        "",
    ]
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def _to_json_block(value):
    import json

    return json.dumps(value, ensure_ascii=False, indent=2)
