from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path
from typing import Any

from rvclaw.api import run_demo
from rvclaw.utils import make_run_id
from rvclaw.web.service import get_run_detail, list_run_files, list_runs, read_benchmark_rows, read_run_file
from rvclaw.web.uploads import read_upload_bytes, resolve_upload_image_ref, save_upload_bytes

FastAPIUploadFile: Any = Any


def create_app(runs_dir: str | Path, planner: str = "llama_cpp", web_token: str | None = None):
    try:
        from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
        from fastapi.responses import HTMLResponse, PlainTextResponse, Response
    except ImportError as exc:
        raise RuntimeError("FastAPI is not installed. Install RVClaw with: python3 -m pip install -e '.[api]'") from exc

    globals()["FastAPIUploadFile"] = UploadFile
    runs_root = Path(runs_dir)
    uploads_root = Path(os.environ.get("RVCLAW_UPLOADS_DIR") or runs_root.parent / "uploads")
    app = FastAPI(title="RVClaw K3 Agent Command Center", version="0.1.3")
    jobs: dict[str, dict[str, Any]] = {}
    jobs_lock = threading.Lock()

    def check_token(x_rvclaw_token: str | None = Header(default=None)) -> None:
        if web_token and x_rvclaw_token != web_token:
            raise HTTPException(status_code=401, detail="missing or invalid RVClaw token")

    def set_job(run_id: str, **updates: Any) -> None:
        with jobs_lock:
            jobs.setdefault(run_id, {}).update(updates)

    def get_job(run_id: str) -> dict[str, Any]:
        with jobs_lock:
            return dict(jobs.get(run_id, {}))

    def run_in_background(run_id: str, goal: str, selected_planner: str, image_path: Path | None) -> None:
        set_job(run_id, status="running", goal=goal, planner=selected_planner)
        try:
            summary = run_demo(
                goal=goal,
                runs_dir=runs_root,
                planner_name=selected_planner,
                run_id=run_id,
                image_ref=image_path,
            )
            set_job(run_id, status=summary.status, summary=summary.to_dict())
        except Exception as exc:  # pragma: no cover - defensive envelope for web worker
            set_job(run_id, status="failed", error=str(exc))

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _index_html()

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "runs_dir": str(runs_root),
            "uploads_dir": str(uploads_root),
            "default_planner": planner,
        }

    @app.post("/api/uploads")
    async def upload_image(file: FastAPIUploadFile = File(...), _: None = Depends(check_token)) -> dict[str, Any]:
        try:
            payload = save_upload_bytes(uploads_root, file.filename or "upload.png", await file.read())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        payload["url"] = f"/api/uploads/{payload['upload_id']}"
        return payload

    @app.get("/api/uploads/{upload_id}")
    def upload_file(upload_id: str, _: None = Depends(check_token)):
        try:
            payload = read_upload_bytes(uploads_root, upload_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(payload["bytes"], media_type=payload["content_type"])

    @app.post("/api/runs")
    def create_run(payload: dict[str, Any], _: None = Depends(check_token)) -> dict[str, Any]:
        goal = str(payload.get("goal") or "检查 A-03 区域设备状态并生成报告")
        selected_planner = str(payload.get("planner") or planner)
        image_ref = payload.get("image_ref")
        image_path: Path | None = None
        if image_ref:
            try:
                image_path = resolve_upload_image_ref(uploads_root, str(image_ref))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        run_id = f"{make_run_id()}-{uuid.uuid4().hex[:6]}"
        set_job(run_id, status="running", goal=goal, planner=selected_planner)
        worker = threading.Thread(
            target=run_in_background,
            args=(run_id, goal, selected_planner, image_path),
            daemon=True,
        )
        worker.start()
        return {"run_id": run_id, "status": "running", "goal": goal, "planner": selected_planner}

    @app.get("/api/runs")
    def runs() -> list[dict[str, Any]]:
        rows = {row["run_id"]: row for row in list_runs(runs_root)}
        with jobs_lock:
            for run_id, job in jobs.items():
                if run_id not in rows:
                    rows[run_id] = {
                        "run_id": run_id,
                        "status": job.get("status", "running"),
                        "planner": job.get("planner", planner),
                        "planner_mode": "running",
                        "task_success": False,
                        "tool_call_count": 0,
                        "latency_ms": None,
                        "run_dir": str(runs_root / run_id),
                    }
        return sorted(rows.values(), key=lambda row: row["run_id"], reverse=True)

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str) -> dict[str, Any]:
        try:
            detail = get_run_detail(runs_root, run_id)
            return _merge_job_detail(detail, get_job(run_id), planner)
        except FileNotFoundError:
            job = get_job(run_id)
            if job:
                return _job_placeholder_detail(run_id, runs_root / run_id, job, planner)
            raise HTTPException(status_code=404, detail=run_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}/files")
    def run_files(run_id: str) -> list[str]:
        try:
            return list_run_files(runs_root, run_id)
        except FileNotFoundError:
            if get_job(run_id):
                return []
            raise HTTPException(status_code=404, detail=run_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}/files/{name:path}")
    def run_file(run_id: str, name: str):
        try:
            file_payload = read_run_file(runs_root, run_id, name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if "bytes" in file_payload:
            return Response(file_payload["bytes"], media_type=file_payload["content_type"])
        media_type = file_payload["content_type"]
        if media_type.startswith("text/") or media_type in {"application/yaml", "application/jsonl"}:
            return PlainTextResponse(file_payload["content"], media_type=media_type)
        return Response(file_payload["content"], media_type=media_type)

    @app.get("/api/benchmarks")
    def benchmarks() -> list[dict[str, str]]:
        return read_benchmark_rows(runs_root)

    return app


def _job_placeholder_detail(run_id: str, run_dir: Path, job: dict[str, Any], default_planner: str) -> dict[str, Any]:
    status = str(job.get("status") or "running")
    metrics = {"status": status, "planner": job.get("planner", default_planner)}
    if job.get("error"):
        metrics["worker_error"] = job["error"]
    return {
        "summary": {
            "run_id": run_id,
            "status": status,
            "planner": job.get("planner", default_planner),
            "planner_mode": "running" if status == "running" else "failed",
            "task_success": False,
            "tool_call_count": 0,
            "latency_ms": None,
            "run_dir": str(run_dir),
        },
        "metrics": metrics,
        "trace": [],
        "files": [],
    }


def _merge_job_detail(detail: dict[str, Any], job: dict[str, Any], default_planner: str) -> dict[str, Any]:
    if not job:
        return detail
    summary = dict(detail.get("summary") or {})
    metrics = dict(detail.get("metrics") or {})
    if summary.get("status") == "unknown" or not metrics:
        status = str(job.get("status") or "running")
        summary["status"] = status
        summary["planner"] = job.get("planner", summary.get("planner", default_planner))
        summary["planner_mode"] = "running" if status == "running" else "failed"
        metrics.setdefault("status", status)
        metrics.setdefault("planner", summary["planner"])
        if job.get("error"):
            metrics["worker_error"] = job["error"]
    detail = dict(detail)
    detail["summary"] = summary
    detail["metrics"] = metrics
    return detail


def _index_html() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RVClaw Agent Command Center</title>
  <style>
    :root { color-scheme: dark; --bg:#0d1214; --panel:#141b1f; --panel2:#101619; --line:#2a363c; --text:#eef5f2; --muted:#91a29e; --accent:#35d49d; --accent2:#70a7ff; --warn:#f1b24a; --bad:#ff6b6b; --idle:#59666b; }
    * { box-sizing:border-box; }
    body { margin:0; background:linear-gradient(135deg,#0d1214 0%,#13201d 46%,#0e1518 100%); color:var(--text); font-family:"Segoe UI","Microsoft YaHei",sans-serif; }
    header { display:flex; justify-content:space-between; gap:16px; align-items:center; padding:18px 24px; border-bottom:1px solid var(--line); background:rgba(13,18,20,.92); position:sticky; top:0; z-index:3; }
    h1,h2,h3 { margin:0; letter-spacing:0; }
    h1 { font-size:24px; }
    h2 { font-size:18px; margin-bottom:12px; }
    h3 { font-size:14px; color:var(--muted); margin-bottom:8px; }
    .sub { color:var(--muted); margin-top:5px; font-size:13px; }
    main { display:grid; grid-template-columns:minmax(380px,.85fr) minmax(620px,1.45fr); gap:14px; padding:14px; }
    section,.drawer { background:rgba(20,27,31,.96); border:1px solid var(--line); border-radius:8px; padding:14px; min-width:0; }
    textarea,select,input,button { width:100%; border:1px solid var(--line); border-radius:6px; background:#0d1214; color:var(--text); padding:10px 12px; font:inherit; }
    textarea { min-height:108px; resize:vertical; line-height:1.5; }
    button { cursor:pointer; background:#17382f; border-color:#2c896b; font-weight:650; }
    button:hover { background:#1d493d; }
    button:disabled { cursor:wait; opacity:.65; }
    .header-actions { display:flex; gap:8px; width:auto; }
    .header-actions button { min-width:108px; }
    .stack,.chips,.files { display:flex; flex-wrap:wrap; gap:8px; }
    .chip { width:auto; margin:0; padding:8px 10px; background:#10181b; border:1px solid var(--line); color:var(--text); border-radius:999px; }
    .chip.active { border-color:var(--accent); color:var(--accent); box-shadow:0 0 0 1px rgba(53,212,157,.24) inset; }
    .card { border:1px solid var(--line); background:var(--panel2); border-radius:8px; padding:12px; }
    .meta-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-bottom:14px; }
    .metric b,.node b,.stack-item b { display:block; color:var(--muted); font-size:12px; margin-bottom:4px; }
    .metric { border:1px solid var(--line); border-radius:7px; padding:10px; background:#101619; min-height:62px; }
    .ok { color:var(--accent); } .fail { color:var(--bad); } .warn { color:var(--warn); } .muted { color:var(--muted); }
    .upload-row { display:grid; grid-template-columns:1fr auto; gap:8px; align-items:center; margin:10px 0; }
    .upload-row button { width:auto; min-width:92px; }
    .upload-preview { margin-top:8px; display:none; grid-template-columns:72px 1fr; gap:10px; align-items:center; }
    .upload-preview img { width:72px; height:52px; object-fit:cover; border:1px solid var(--line); border-radius:6px; }
    .expected { color:var(--muted); font-size:13px; margin-top:8px; line-height:1.5; }
    .graph { display:grid; grid-template-columns:repeat(4,minmax(130px,1fr)); gap:10px; margin-bottom:14px; }
    .node { border:1px solid var(--line); border-radius:8px; padding:10px; background:#0f1518; min-height:86px; position:relative; overflow:hidden; }
    .node::before { content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:var(--idle); }
    .node.pending { opacity:.72; }
    .node.running::before { background:var(--warn); }
    .node.completed::before,.node.approved::before { background:var(--accent); }
    .node.failed::before,.node.rejected::before { background:var(--bad); }
    .node.fallback::before,.node.repaired::before { background:var(--accent2); }
    .node span { display:block; font-size:12px; color:var(--muted); line-height:1.4; word-break:break-word; }
    .images { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:10px; }
    .images.single { grid-template-columns:1fr; }
    .images.single #annotated,.images.none { display:none; }
    .image-note { color:var(--muted); font-size:13px; margin:6px 0 8px; }
    .images img,.file-image { width:100%; max-height:340px; object-fit:contain; border:1px solid var(--line); border-radius:7px; background:#070a0b; }
    .results { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:8px; margin-top:10px; }
    .result-card { border:1px solid var(--line); border-radius:7px; padding:10px; background:#0f1518; }
    .result-card b { display:block; color:var(--accent); margin-bottom:4px; }
    .file-view { max-height:340px; overflow:auto; background:#070a0b; border:1px solid var(--line); border-radius:7px; padding:12px; white-space:pre-wrap; font-family:Consolas,ui-monospace,monospace; }
    .files button { width:auto; margin:0; padding:7px 10px; background:#10181b; border-color:var(--line); font-weight:500; }
    .stack-item { flex:1 1 150px; border:1px solid var(--line); background:#0f1518; border-radius:7px; padding:10px; }
    .stack-item.active { border-color:rgba(53,212,157,.72); }
    .stack-item.fallback { border-color:rgba(112,167,255,.72); }
    .stack-item.reserved { opacity:.66; }
    .drawer-backdrop { display:none; position:fixed; inset:0; background:rgba(0,0,0,.42); z-index:8; }
    .drawer-backdrop.open { display:block; }
    .drawer { position:fixed; right:0; top:0; bottom:0; width:min(520px,92vw); border-radius:0; z-index:9; transform:translateX(104%); transition:transform .2s ease; overflow:auto; }
    .drawer.open { transform:translateX(0); }
    .history-row { border:1px solid var(--line); border-radius:7px; padding:10px; margin-bottom:8px; background:#0f1518; }
    .history-row button { width:auto; margin-top:8px; margin-right:6px; padding:6px 9px; }
    .filters { display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; margin:10px 0; }
    @media (max-width:1080px){ main{grid-template-columns:1fr}.graph{grid-template-columns:repeat(2,1fr)}.meta-grid{grid-template-columns:repeat(2,1fr)} }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>RVClaw Agent Command Center</h1>
      <div class="sub">自然语言任务 -> Planner -> Safety Guard -> Skills -> Evidence Pack</div>
    </div>
    <div class="header-actions">
      <button onclick="openHistory()">历史记录</button>
      <button onclick="refreshHealth()">刷新状态</button>
    </div>
  </header>
  <main>
    <section>
      <h2>任务编排器</h2>
      <select id="taskTemplate" onchange="selectTemplate(this.value)"></select>
      <div id="templateInfo" class="expected"></div>
      <div id="presetChips" class="chips" style="margin:10px 0"></div>
      <textarea id="goal"></textarea>
      <div class="upload-row">
        <input id="imageFile" type="file" accept="image/png,image/jpeg,image/webp">
        <button onclick="uploadImage()">上传图片</button>
      </div>
      <div id="uploadPreview" class="upload-preview">
        <img id="uploadImagePreview" alt="uploaded image">
        <div><b id="uploadName">未上传</b><div id="uploadRef" class="sub">默认使用 K3 样例图或摄像头后端</div></div>
      </div>
      <select id="planner"><option value="llama_cpp">llama_cpp</option><option value="mock">mock</option></select>
      <input id="token" placeholder="RVCLAW_WEB_TOKEN" type="password">
      <button id="runButton" onclick="runTask()">准备运行</button>
      <h2 style="margin-top:16px">Runtime Stack Map</h2>
      <div id="runtimeStack" class="stack"></div>
    </section>
    <section>
      <div class="meta-grid">
        <div class="metric"><b>status</b><span id="status">idle</span></div>
        <div class="metric"><b>planner_mode</b><span id="plannerMode">-</span></div>
        <div class="metric"><b>tool calls</b><span id="toolCount">-</span></div>
        <div class="metric"><b>latency</b><span id="latency">-</span></div>
        <div class="metric"><b>vision</b><span id="visionBackend">-</span></div>
        <div class="metric"><b>model</b><span id="visionModel">-</span></div>
      </div>
      <h2>Agent 执行图</h2>
      <div id="agentGraph" class="graph"></div>
      <h2>CV 画面</h2>
      <div id="imageModeNote" class="image-note">等待视觉任务...</div>
      <div id="imageGrid" class="images none"><img id="capture" alt="capture"><img id="annotated" alt="annotated"></div>
      <h2>识别结论</h2>
      <div id="visionSummary" class="metric">等待视觉任务...</div>
      <div id="visionResults" class="results"></div>
      <h2 style="margin-top:16px">证据文件</h2>
      <div id="files" class="files"></div>
      <div id="fileView" class="file-view">等待运行...</div>
    </section>
  </main>
  <div id="historyBackdrop" class="drawer-backdrop" onclick="closeHistory()"></div>
  <aside id="historyDrawer" class="drawer">
    <div style="display:flex;justify-content:space-between;gap:12px;align-items:center">
      <h2>历史记录</h2><button style="width:auto" onclick="closeHistory()">关闭</button>
    </div>
    <div class="filters">
      <select id="historyStatus" onchange="renderHistory()"><option value="">全部状态</option><option>completed</option><option>failed</option><option>running</option></select>
      <select id="historyPlanner" onchange="renderHistory()"><option value="">全部 planner</option><option>llama_cpp</option><option>mock</option></select>
      <select id="historyType" onchange="renderHistory()"><option value="">全部任务</option><option value="vision">视觉</option><option value="inspection">巡检</option></select>
    </div>
    <div id="historyRows"></div>
  </aside>
  <script>
    const templates = {
      inspection:{label:'设备巡检', goal:'检查 A-03 区域设备状态并生成报告', chain:'memory_query -> move_to -> capture_image -> detect_status -> speak -> upload_report', kind:'inspection'},
      classification:{label:'图片分类', goal:'分类这张图片并说明结果', chain:'memory_query -> capture_image -> analyze_image(classification) -> speak -> upload_report', kind:'vision'},
      detection:{label:'目标检测', goal:'检测图片中的目标并生成结论', chain:'memory_query -> capture_image -> analyze_image(object_detection) -> speak -> upload_report', kind:'vision'},
      segmentation:{label:'语义分割', goal:'分割画面中的主要区域', chain:'memory_query -> capture_image -> analyze_image(segmentation) -> speak -> upload_report', kind:'vision'},
      face:{label:'人脸检测', goal:'检测画面中是否有人脸', chain:'memory_query -> capture_image -> analyze_image(face_detection) -> speak -> upload_report', kind:'vision'},
      unsafe:{label:'安全拒绝演示', goal:'移动到 Z-99 区域并拍照', chain:'planner -> Safety Guard(rejected) -> failed evidence pack', kind:'inspection'}
    };
    const graphNodes = [
      ['intake','Task Intake'], ['planner','Planner'], ['safety','Safety Guard'], ['memory','Memory'],
      ['capture','Capture'], ['vision','Vision'], ['speak','Speak'], ['report','Report']
    ];
    const stackItems = [
      ['k3','K3 Edge Box','active'], ['llama','llama.cpp Planner','active'], ['guard','Safety Guard','active'],
      ['skills','Skill Registry','active'], ['memory','SQLite Memory','active'], ['vision','CV/DemoZoo','active'],
      ['evidence','Evidence Pack','active'], ['mnn','MNN','reserved'], ['vllm','vLLM','reserved'],
      ['milvus','Milvus/Knowhere','reserved'], ['openclaw','ROS2/OpenClaw','reserved']
    ];
    let currentRun = null, currentImageRef = null, historyCache = [], pollTimer = null;

    function requestOptions(options) {
      const next = options || {}; const tokenValue = token.value.trim();
      if (tokenValue) { localStorage.setItem('rvclaw_web_token', tokenValue); next.headers = Object.assign({}, next.headers || {}, {'X-RVClaw-Token': tokenValue}); }
      return next;
    }
    async function api(path, options) { const r = await fetch(path, requestOptions(options)); if (!r.ok) throw new Error(await r.text()); return r; }
    function init(){
      taskTemplate.innerHTML = Object.entries(templates).map(([id,t]) => `<option value="${id}">${t.label}</option>`).join('');
      presetChips.innerHTML = Object.entries(templates).map(([id,t]) => `<button class="chip" id="chip-${id}" onclick="selectTemplate('${id}')">${t.label}</button>`).join('');
      runtimeStack.innerHTML = stackItems.map(([id,label,state]) => `<div id="stack-${id}" class="stack-item ${state}"><b>${state}</b>${label}</div>`).join('');
      token.value = localStorage.getItem('rvclaw_web_token') || '';
      selectTemplate('inspection'); renderGraph({summary:{status:'idle'}, metrics:{}, trace:[], files:[]}); refreshHealth();
    }
    async function refreshHealth(){ const h = await (await api('/api/health')).json(); document.querySelector('.sub').textContent = `${h.runs_dir} | uploads: ${h.uploads_dir}`; }
    function selectTemplate(id){
      taskTemplate.value = id; const t = templates[id]; goal.value = t.goal;
      templateInfo.textContent = `当前模板：${t.label} | 预期工具链：${t.chain}`;
      Object.keys(templates).forEach(key => document.getElementById(`chip-${key}`).classList.toggle('active', key === id));
    }
    async function uploadImage(){
      const f = imageFile.files[0]; if (!f) { uploadRef.textContent = '请先选择图片文件'; return; }
      const data = new FormData(); data.append('file', f);
      uploadRef.textContent = '上传中...';
      const payload = await (await api('/api/uploads', {method:'POST', body:data})).json();
      currentImageRef = payload.image_ref; uploadPreview.style.display = 'grid'; uploadName.textContent = payload.filename;
      uploadRef.textContent = payload.image_ref; uploadImagePreview.src = payload.url;
    }
    async function runTask(){
      runButton.disabled = true; runButton.textContent = 'Planner 生成中'; status.textContent = 'running'; status.className = 'warn';
      renderGraph({summary:{status:'running'}, metrics:{}, trace:[], files:[]});
      const payload = {goal:goal.value, planner:planner.value}; if (currentImageRef) payload.image_ref = currentImageRef;
      const created = await (await api('/api/runs', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)})).json();
      currentRun = created.run_id; runButton.textContent = '执行中'; pollRun(currentRun);
    }
    async function pollRun(runId){
      if (pollTimer) clearTimeout(pollTimer);
      const detail = await (await api(`/api/runs/${runId}`)).json(); renderRun(detail);
      const st = detail.summary.status;
      if (st === 'running') { pollTimer = setTimeout(() => pollRun(runId), 1000); return; }
      runButton.disabled = false; runButton.textContent = st === 'completed' ? '已完成，可再次运行' : '失败，可再次运行'; await refreshHistory();
    }
    async function loadRun(runId){ currentRun = runId; const detail = await (await api(`/api/runs/${runId}`)).json(); renderRun(detail); closeHistory(); }
    function renderRun(detail){
      const m = detail.metrics || {}; status.textContent = detail.summary.status; status.className = detail.summary.status === 'completed' ? 'ok' : detail.summary.status === 'failed' ? 'fail' : 'warn';
      plannerMode.textContent = m.planner_mode || detail.summary.planner_mode || '-'; toolCount.textContent = m.tool_call_count ?? detail.summary.tool_call_count ?? '-';
      latency.textContent = m.latency_ms ? `${m.latency_ms} ms` : '-'; visionBackend.textContent = m.vision_backend || '-'; visionModel.textContent = m.vision_model || '-';
      const visionOutput = findVisionOutput(detail);
      renderGraph(detail); renderStack(m); renderFiles(detail, visionOutput);
      if (m.worker_error) { visionSummary.textContent = `后台任务异常：${m.worker_error}`; visionResults.innerHTML = ''; }
      else renderVision(visionOutput);
    }
    function renderGraph(detail){
      const trace = detail.trace || [], m = detail.metrics || {}, status = detail.summary.status;
      const state = Object.fromEntries(graphNodes.map(([id]) => [id,{status:'pending',note:'等待'}]));
      state.intake = {status: status === 'idle' ? 'pending' : 'completed', note: detail.summary.run_id || '任务接收'};
      const plannerEvent = trace.find(e => e.event === 'planner.completed' || e.event === 'planner.failed');
      if (plannerEvent) state.planner = {status: plannerEvent.event.endsWith('failed') ? 'failed' : plannerStatus(m.planner_mode), note:m.planner_mode || plannerEvent.payload.planner || ''};
      else if (status === 'running') state.planner = {status:'running', note:'生成 tool_calls'};
      const rejected = trace.find(e => e.event === 'safety_guard.rejected');
      const approved = trace.find(e => e.event === 'safety_guard.approved');
      if (rejected) state.safety = {status:'rejected', note:rejected.payload.error || 'rejected'};
      else if (approved) state.safety = {status:'approved', note:'whitelist approved'};
      if (m.worker_error) state.planner = {status:'failed', note:m.worker_error};
      const completed = trace.filter(e => e.event === 'skill_call.completed').map(e => e.payload.call && e.payload.call.name);
      const failedCalls = trace.filter(e => e.event === 'skill_call.failed');
      const failedVision = failedCalls.find(e => e.payload && e.payload.call && ['analyze_image','detect_status'].includes(e.payload.call.name));
      if (completed.includes('memory_query')) state.memory = {status:'completed', note:'context loaded'};
      if (completed.includes('capture_image')) state.capture = {status:'completed', note:'image artifact'};
      if (completed.includes('detect_status') || completed.includes('analyze_image')) state.vision = {status:m.vision_backend === 'mock_fallback' ? 'fallback':'completed', note:m.vision_task || m.vision_backend || 'status detection'};
      if (failedVision) state.vision = {status:'failed', note:failedVision.payload.result.error || 'vision failed'};
      if (completed.includes('speak')) state.speak = {status:'completed', note:'status message'};
      if (completed.includes('upload_report')) state.report = {status:'completed', note:'report.md'};
      if (status === 'failed' && !state.report.status.includes('completed')) state.report = {status:'failed', note:'failed evidence kept'};
      agentGraph.innerHTML = graphNodes.map(([id,label]) => `<div class="node ${state[id].status}"><b>${label}</b><span>${state[id].status}</span><span>${state[id].note || ''}</span></div>`).join('');
    }
    function plannerStatus(mode){ if (!mode) return 'completed'; if (mode.includes('fallback')) return 'fallback'; if (mode.includes('repair')) return 'repaired'; if (mode === 'failed') return 'failed'; return 'completed'; }
    function renderStack(m){
      stackItems.forEach(([id,,base]) => { const el = document.getElementById(`stack-${id}`); if (!el) return; el.className = `stack-item ${base}`; });
      if (m.vision_backend === 'mock_fallback') document.getElementById('stack-vision').className = 'stack-item fallback';
      if (m.planner === 'mock') document.getElementById('stack-llama').className = 'stack-item fallback';
    }
    const imageDisplayPolicies = {classification:'single', object_detection:'compare', segmentation:'compare', face_detection:'compare', inspection:'compare', non_vision:'none'};
    function imagePolicyForTask(task, hasStatusDetection){
      if (hasStatusDetection) return imageDisplayPolicies.inspection;
      return imageDisplayPolicies[task] || imageDisplayPolicies.non_vision;
    }
    function renderFiles(detail, visionOutput){
      files.innerHTML = (detail.files || []).map(f => `<button onclick="showFile('${f}')">${f}</button>`).join('');
      const annotatedFile = (detail.files || []).find(f => f.endsWith('_annotated.png')); const captureFile = (detail.files || []).find(f => f.endsWith('_capture.png'));
      const hasStatusDetection = (detail.trace || []).some(e => e.payload && e.payload.call && e.payload.call.name === 'detect_status');
      const task = (visionOutput && visionOutput.task) || (hasStatusDetection ? 'inspection' : null);
      const policy = imagePolicyForTask(task, hasStatusDetection);
      imageGrid.className = `images ${policy}`;
      capture.removeAttribute('src'); annotated.removeAttribute('src');
      if (policy === 'none' || !captureFile) {
        imageGrid.className = 'images none';
        imageModeNote.textContent = detail.summary.status === 'failed' ? '当前 run 未进入图像处理阶段。' : '当前任务不需要显示图像。';
      } else if (policy === 'single') {
        imageModeNote.textContent = '图片分类展示输入图即可；分类结果在下方结构化卡片中呈现。';
        capture.src = fileUrl(detail.summary.run_id, captureFile);
      } else {
        imageModeNote.textContent = '该任务展示原图和处理后图，用于对比检测框、分割叠加或巡检标注。';
        capture.src = fileUrl(detail.summary.run_id, captureFile);
        if (annotatedFile) annotated.src = fileUrl(detail.summary.run_id, annotatedFile);
      }
      if ((detail.files || []).includes('report.md')) showFile('report.md');
    }
    function findVisionOutput(detail){ const rows = detail.trace || []; const ev = rows.findLast ? rows.findLast(e => e.payload && e.payload.call && e.payload.call.name === 'analyze_image') : [...rows].reverse().find(e => e.payload && e.payload.call && e.payload.call.name === 'analyze_image'); return ev && ev.payload.result && ev.payload.result.output; }
    function renderVision(output){
      if (!output) { visionSummary.textContent = '当前 run 没有 analyze_image 结果。'; visionResults.innerHTML = ''; return; }
      visionSummary.textContent = output.summary || '视觉任务完成。';
      const rows = [...(output.labels || []), ...(output.objects || []), ...(output.segments || []), ...(output.faces || [])];
      const warning = output.requires_real_model ? '<div class="result-card"><b>需要真实视觉后端</b><span>当前结果来自 cv_sample 启发式链路验证，不是模型推理。请启用 DemoZoo/MNN/ONNX。</span></div>' : '';
      visionResults.innerHTML = warning + (rows.map(item => `<div class="result-card"><b>${item.label || 'result'}</b><span>confidence: ${Number(item.confidence || 0).toFixed(2)}</span><br><span>${item.bbox ? 'bbox: ' + item.bbox.join(', ') : ''}</span></div>`).join('') || '<div class="result-card"><b>无目标</b><span>未返回可视对象。</span></div>');
    }
    function fileUrl(runId, name){ return `/api/runs/${encodeURIComponent(runId)}/files/${name.split('/').map(encodeURIComponent).join('/')}`; }
    async function showFile(name){ if(!currentRun) return; if (/\\.(png|jpg|jpeg|webp)$/i.test(name)) { fileView.innerHTML = `<img class="file-image" src="${fileUrl(currentRun,name)}" alt="${name}">`; return; } fileView.textContent = await (await api(fileUrl(currentRun,name))).text(); }
    async function refreshHistory(){ historyCache = await (await api('/api/runs')).json(); renderHistory(); }
    function openHistory(){ historyBackdrop.classList.add('open'); historyDrawer.classList.add('open'); refreshHistory(); }
    function closeHistory(){ historyBackdrop.classList.remove('open'); historyDrawer.classList.remove('open'); }
    function renderHistory(){
      const st = historyStatus.value, pl = historyPlanner.value, ty = historyType.value;
      const rows = historyCache.filter(r => (!st || r.status === st) && (!pl || r.planner === pl) && (!ty || (ty === 'vision' ? (r.vision_task || r.vision_backend) : !(r.vision_task || r.vision_backend))));
      historyRows.innerHTML = rows.map(r => `<div class="history-row"><b>${r.run_id}</b><div class="${r.status==='completed'?'ok':r.status==='failed'?'fail':'warn'}">${r.status}</div><div class="sub">${r.planner} | ${r.planner_mode || '-'}</div><button onclick="loadRun('${r.run_id}')">加载到主界面</button><button onclick="loadRun('${r.run_id}').then(()=>showFile('report.md'))">打开报告</button></div>`).join('') || '<div class="sub">暂无匹配历史。</div>';
    }
    init();
  </script>
</body>
</html>"""
