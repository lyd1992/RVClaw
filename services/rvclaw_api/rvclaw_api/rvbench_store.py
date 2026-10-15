"""Small file-backed store for the manual RVBench workspace.

The store deliberately keeps the schema open: the operator defines projects,
cases, steps, metrics, thresholds, and evidence in the browser.  No test
suite is invented by the service.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from urllib.parse import parse_qs


_LOCK = RLock()
_SAFE = re.compile(r"^[A-Za-z0-9_-]{1,96}$")
_STATUS = {"pending", "running", "passed", "failed"}


class RvbenchError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _safe(value: object, label: str) -> str:
    text = str(value or "").strip()
    if not _SAFE.fullmatch(text):
        raise RvbenchError(f"{label} 格式无效")
    return text


def _state_paths(root: Path) -> tuple[Path, Path]:
    state_root = root / "state" / "rvbench"
    return state_root / "store.json", state_root / "evidence"


def _load(root: Path) -> dict:
    store_path, _ = _state_paths(root)
    try:
        value = json.loads(store_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        value = {}
    if not isinstance(value, dict):
        value = {}
    value.setdefault("projects", [])
    value.setdefault("runs", [])
    return value


def _save(root: Path, value: dict) -> None:
    store_path, _ = _state_paths(root)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    temp = store_path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(store_path)


def _find(items: list[dict], item_id: str, label: str) -> dict:
    item = next((item for item in items if item.get("id") == item_id), None)
    if not item:
        raise RvbenchError(f"{label} 不存在", 404)
    return item


def _parse_steps(payload: dict) -> list[dict]:
    raw = payload.get("steps")
    if raw is None:
        text = str(payload.get("steps_text") or "")
        raw = [line.strip() for line in text.splitlines() if line.strip()]
    if not isinstance(raw, list) or not raw:
        raise RvbenchError("至少需要一个测试步骤")
    steps = []
    for item in raw:
        if isinstance(item, str):
            title, _, description = item.partition("|")
            item = {"title": title.strip(), "description": description.strip()}
        if not isinstance(item, dict) or not str(item.get("title") or "").strip():
            raise RvbenchError("测试步骤必须有名称")
        steps.append({
            "id": _id("step"),
            "title": str(item.get("title")).strip()[:200],
            "description": str(item.get("description") or "").strip()[:1000],
            "default_threshold": item.get("default_threshold"),
        })
    return steps


def _run_status(run: dict) -> str:
    states = [str(step.get("status") or "pending") for step in run.get("steps", [])]
    if any(item == "failed" for item in states):
        return "failed"
    if states and all(item == "passed" for item in states):
        return "passed"
    if any(item == "running" for item in states):
        return "running"
    return "draft"


def _threshold_result(metric: object, threshold: object) -> bool | None:
    """Return a threshold decision when the operator supplied numeric values."""
    try:
        value = float(metric)
    except (TypeError, ValueError):
        return None
    low = high = None
    if isinstance(threshold, dict):
        low, high = threshold.get("min"), threshold.get("max")
    elif isinstance(threshold, (list, tuple)) and len(threshold) == 2:
        low, high = threshold
    elif isinstance(threshold, str):
        match = re.fullmatch(r"\s*(-?(?:\d+(?:\.\d+)?))?\s*(?:\.\.|-|至)\s*(-?(?:\d+(?:\.\d+)?))?\s*", threshold)
        if match:
            low, high = match.groups()
        else:
            try:
                low = high = float(threshold)
            except ValueError:
                return None
    try:
        low = float(low) if low not in (None, "") else None
        high = float(high) if high not in (None, "") else None
    except (TypeError, ValueError):
        return None
    if low is None and high is None:
        return None
    return (low is None or value >= low) and (high is None or value <= high)


def _summary(run: dict) -> dict:
    steps = run.get("steps", [])
    return {
        "total": len(steps),
        "pending": sum(step.get("status") == "pending" for step in steps),
        "running": sum(step.get("status") == "running" for step in steps),
        "passed": sum(step.get("status") == "passed" for step in steps),
        "failed": sum(step.get("status") == "failed" for step in steps),
    }


def _public_run(run: dict) -> dict:
    copy = json.loads(json.dumps(run))
    copy["summary"] = _summary(run)
    copy["status"] = _run_status(run)
    return copy


def _inspection_snapshot(studio_root: Path) -> dict:
    try:
        value = json.loads((studio_root / "inspection-progress.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"history": [], "state": "idle"}
    return value if isinstance(value, dict) else {"history": [], "state": "idle"}


def get(root: Path, studio_root: Path, path: str, query: str = "") -> dict | bytes | None:
    with _LOCK:
        store = _load(root)
        if path == "/api/rvbench/projects":
            return {"projects": store["projects"]}
        if path == "/api/rvbench/runs":
            params = parse_qs(query)
            project_id = (params.get("project_id") or [""])[0]
            runs = [item for item in store["runs"] if not project_id or item.get("project_id") == project_id]
            return {"runs": [_public_run(item) for item in runs]}
        match = re.fullmatch(r"/api/rvbench/projects/([A-Za-z0-9_-]+)", path)
        if match:
            return {"project": _find(store["projects"], match.group(1), "项目")}
        match = re.fullmatch(r"/api/rvbench/runs/([A-Za-z0-9_-]+)(?:/report)?", path)
        if match:
            run = _find(store["runs"], match.group(1), "测试运行")
            return {"run": _public_run(run), "report": _public_run(run)}
        if path == "/api/rvbench/inspection-candidates":
            snapshot = _inspection_snapshot(studio_root)
            meter_records = []
            try:
                value = json.loads((root / "runs" / "meter-uploads" / "index.json").read_text(encoding="utf-8"))
                meter_records = value if isinstance(value, list) else []
            except (OSError, json.JSONDecodeError):
                pass
            return {
                "current": {
                    "task_id": snapshot.get("task_id"),
                    "state": snapshot.get("state"),
                    "state_text": snapshot.get("state_text"),
                    "route": snapshot.get("route", []),
                    "steps": snapshot.get("steps", []),
                    "evidence": snapshot.get("evidence"),
                },
                "history": snapshot.get("history", []),
                "meter_records": meter_records[:20],
            }
        if path == "/api/inspection-runs":
            snapshot = _inspection_snapshot(studio_root)
            return {"runs": snapshot.get("history", []), "current": snapshot}
        match = re.fullmatch(r"/api/inspection-runs/([A-Za-z0-9_-]+)", path)
        if match:
            snapshot = _inspection_snapshot(studio_root)
            task_id = match.group(1)
            if snapshot.get("task_id") == task_id:
                item = dict(snapshot)
            else:
                item = next((row for row in snapshot.get("history", []) if row.get("task_id") == task_id), None)
                if item is None:
                    raise RvbenchError("巡检任务不存在", 404)
            timeline = [{
                "id": step.get("id"), "label": step.get("label"),
                "status": step.get("status"), "status_text": step.get("status_text"),
                "at": step.get("at"),
            } for step in item.get("steps", [])]
            if not timeline:
                completed = set(item.get("completed_waypoints") or [])
                timeline = [{
                    "id": waypoint,
                    "label": waypoint,
                    "status": "done" if waypoint in completed else "pending",
                    "status_text": "已完成" if waypoint in completed else "未记录时刻",
                    "at": None,
                } for waypoint in item.get("route", [])]
                if item.get("finished_at") is not None:
                    timeline.append({"id": "finish", "label": "巡检结束", "status": item.get("state"), "status_text": item.get("state_text"), "at": item.get("finished_at")})
            linked_runs = [_public_run(row) for row in store.get("runs", []) if row.get("inspection_task_id") == task_id]
            return {"run": item, "timeline": timeline, "evidence": item.get("evidence"), "linked_runs": linked_runs}
        match = re.fullmatch(r"/api/rvbench/runs/([A-Za-z0-9_-]+)/evidence/([A-Za-z0-9_.-]+)", path)
        if match:
            _, evidence_root = _state_paths(root)
            target = (evidence_root / match.group(1) / match.group(2)).resolve()
            if not str(target).startswith(str(evidence_root.resolve())) or not target.is_file():
                raise RvbenchError("证据不存在", 404)
            return target.read_bytes()
    return None


def post(root: Path, studio_root: Path, path: str, payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise RvbenchError("请求必须是 JSON 对象")
    with _LOCK:
        store = _load(root)
        if path == "/api/rvbench/projects":
            name = str(payload.get("name") or "").strip()
            if not name:
                raise RvbenchError("项目名称不能为空")
            project = {"id": _id("proj"), "name": name[:160], "description": str(payload.get("description") or "")[:1000], "use_cases": [], "created_at": _now(), "updated_at": _now()}
            store["projects"].insert(0, project)
            _save(root, store)
            return {"project": project}
        match = re.fullmatch(r"/api/rvbench/projects/([A-Za-z0-9_-]+)/use-cases", path)
        if match:
            project = _find(store["projects"], match.group(1), "项目")
            case = {"id": _id("case"), "name": str(payload.get("name") or "").strip()[:160], "description": str(payload.get("description") or "")[:1000], "steps": _parse_steps(payload), "created_at": _now()}
            if not case["name"]:
                raise RvbenchError("用例名称不能为空")
            project["use_cases"].append(case)
            project["updated_at"] = _now()
            _save(root, store)
            return {"project": project, "use_case": case}
        if path == "/api/rvbench/runs":
            project = _find(store["projects"], _safe(payload.get("project_id"), "project_id"), "项目")
            case = _find(project.get("use_cases", []), _safe(payload.get("use_case_id"), "use_case_id"), "用例")
            run = {"id": _id("run"), "project_id": project["id"], "use_case_id": case["id"], "inspection_task_id": payload.get("inspection_task_id"), "status": "draft", "created_at": _now(), "started_at": None, "finished_at": None, "steps": [{**step, "status": "pending", "notes": "", "metric": None, "threshold": step.get("default_threshold"), "evidence": []} for step in case["steps"]]}
            store["runs"].insert(0, run)
            _save(root, store)
            return {"run": _public_run(run)}
        match = re.fullmatch(r"/api/rvbench/runs/([A-Za-z0-9_-]+)/start", path)
        if match:
            run = _find(store["runs"], match.group(1), "测试运行")
            run["started_at"] = run.get("started_at") or _now()
            run["finished_at"] = None
            for step in run["steps"]:
                if step.get("status") == "pending":
                    step["status"] = "running"
                    break
            _save(root, store)
            return {"run": _public_run(run)}
        match = re.fullmatch(r"/api/rvbench/runs/([A-Za-z0-9_-]+)/finish", path)
        if match:
            run = _find(store["runs"], match.group(1), "测试运行")
            run["finished_at"] = _now()
            for step in run["steps"]:
                if step.get("status") in {"pending", "running"}:
                    step["status"] = "failed"
                    step.setdefault("notes", "未完成")
            _save(root, store)
            return {"run": _public_run(run)}
        match = re.fullmatch(r"/api/rvbench/runs/([A-Za-z0-9_-]+)/link-inspection", path)
        if match:
            run = _find(store["runs"], match.group(1), "测试运行")
            task_id = str(payload.get("task_id") or "").strip()
            if not task_id:
                raise RvbenchError("task_id 不能为空")
            run["inspection_task_id"] = task_id
            _save(root, store)
            return {"run": _public_run(run)}
        match = re.fullmatch(r"/api/rvbench/runs/([A-Za-z0-9_-]+)/import-candidate", path)
        if match:
            run = _find(store["runs"], match.group(1), "测试运行")
            task_id = str(payload.get("task_id") or "").strip()
            meter_id = str(payload.get("meter_evidence_id") or "").strip()
            source = None
            meter_source = None
            if task_id:
                snapshot = _inspection_snapshot(studio_root)
                source = snapshot if snapshot.get("task_id") == task_id else next((row for row in snapshot.get("history", []) if row.get("task_id") == task_id), None)
                if source is None:
                    raise RvbenchError("巡检任务不存在", 404)
                run["inspection_task_id"] = task_id
            elif meter_id:
                try:
                    meter_rows = json.loads((root / "runs" / "meter-uploads" / "index.json").read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    meter_rows = []
                meter_source = next((row for row in meter_rows if row.get("evidence_id") == meter_id), None)
                if meter_source is None:
                    raise RvbenchError("仪表证据不存在", 404)
            else:
                raise RvbenchError("task_id 或 meter_evidence_id 不能为空")
            step_id = str(payload.get("step_id") or "").strip()
            step = next((item for item in run.get("steps", []) if item.get("id") == step_id), None) or (run.get("steps") or [None])[0]
            if step is not None:
                if source is not None:
                    summary = f"巡检任务 {task_id}：状态={source.get('state_text') or source.get('state')}; 路线={' / '.join(source.get('route') or [])}; 完成点位={' / '.join(source.get('completed_waypoints') or []) or '无'}"
                else:
                    summary = f"仪表证据 {meter_id}：读数={meter_source.get('value') or meter_source.get('raw_text') or '未识别'}; 状态={meter_source.get('status') or 'unknown'}; 时间={meter_source.get('created_at') or '--'}"
                step["notes"] = (step.get("notes", "") + ("\n" if step.get("notes") else "") + summary)[:2000]
                if source is not None and source.get("evidence_url"):
                    step.setdefault("evidence", []).append({"name": "inspection-evidence", "url": source["evidence_url"], "created_at": _now(), "content_type": "text/html"})
                if meter_source is not None and meter_source.get("evidence_url"):
                    step.setdefault("evidence", []).append({"name": "meter-evidence", "url": meter_source["evidence_url"], "created_at": _now(), "content_type": "image/jpeg"})
            _save(root, store)
            return {"run": _public_run(run), "imported": {"task_id": task_id, "meter_evidence_id": meter_id, "step_id": step.get("id") if step else None}}
        match = re.fullmatch(r"/api/rvbench/runs/([A-Za-z0-9_-]+)/steps/([A-Za-z0-9_-]+)", path)
        if match:
            run = _find(store["runs"], match.group(1), "测试运行")
            step = _find(run.get("steps", []), match.group(2), "测试步骤")
            status = str(payload.get("status") or step.get("status") or "pending")
            if status not in _STATUS:
                raise RvbenchError("步骤状态必须是 pending、running、passed 或 failed")
            step["status"] = status
            step["notes"] = str(payload.get("notes") or "")[:2000]
            if "metric" in payload:
                step["metric"] = payload.get("metric")
            if "threshold" in payload:
                step["threshold"] = payload.get("threshold")
            decision = _threshold_result(step.get("metric"), step.get("threshold"))
            if decision is not None and status in {"pending", "running"}:
                step["status"] = "passed" if decision else "failed"
            evidence = payload.get("evidence")
            if isinstance(evidence, dict) and evidence.get("data"):
                try:
                    raw = str(evidence["data"]).split(",", 1)[-1]
                    data = base64.b64decode(raw, validate=True)
                except (ValueError, base64.binascii.Error) as error:
                    raise RvbenchError("证据文件编码无效") from error
                if len(data) > 10 * 1024 * 1024:
                    raise RvbenchError("证据文件不能超过 10 MB")
                _, evidence_root = _state_paths(root)
                folder = evidence_root / run["id"]
                folder.mkdir(parents=True, exist_ok=True)
                filename = re.sub(r"[^A-Za-z0-9_.-]", "_", str(evidence.get("name") or "evidence.bin"))[:120]
                if not filename or filename in {".", ".."}:
                    filename = "evidence.bin"
                target = folder / f"{uuid.uuid4().hex[:8]}-{filename}"
                target.write_bytes(data)
                step.setdefault("evidence", []).append({"name": filename, "url": f"/api/rvbench/runs/{run['id']}/evidence/{target.name}", "created_at": _now(), "content_type": evidence.get("mime") or mimetypes.guess_type(filename)[0] or "application/octet-stream"})
            if status in {"passed", "failed"}:
                next_step = next((item for item in run["steps"] if item.get("status") == "pending"), None)
                if next_step and run.get("started_at"):
                    next_step["status"] = "running"
            run["status"] = _run_status(run)
            if run["status"] in {"passed", "failed"}:
                run["finished_at"] = run.get("finished_at") or _now()
            _save(root, store)
            return {"run": _public_run(run), "step": step}
    raise RvbenchError("未知 RVBench 接口", 404)


def delete(root: Path, path: str) -> dict:
    with _LOCK:
        store = _load(root)
        match = re.fullmatch(r"/api/rvbench/projects/([A-Za-z0-9_-]+)", path)
        if match:
            before = len(store["projects"])
            store["projects"] = [item for item in store["projects"] if item.get("id") != match.group(1)]
            if len(store["projects"]) == before:
                raise RvbenchError("项目不存在", 404)
            _save(root, store)
            return {"deleted": True}
    raise RvbenchError("未知 RVBench 接口", 404)


def is_path(path: str) -> bool:
    return path.startswith("/api/rvbench/") or path.startswith("/api/inspection-runs")
