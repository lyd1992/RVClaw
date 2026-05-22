from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

from rvclaw.api import run_demo
from rvclaw.web.service import get_run_detail, list_run_files, list_runs, read_benchmark_rows, read_run_file


def create_app(runs_dir: str | Path, planner: str = "llama_cpp", web_token: str | None = None):
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
        from fastapi.responses import HTMLResponse, PlainTextResponse, Response
    except ImportError as exc:
        raise RuntimeError("FastAPI is not installed. Install RVClaw with: python3 -m pip install -e '.[api]'") from exc

    runs_root = Path(runs_dir)
    app = FastAPI(title="RVClaw K3 Web Demo", version="0.1.1")

    def check_token(x_rvclaw_token: str | None = Header(default=None)) -> None:
        if web_token and x_rvclaw_token != web_token:
            raise HTTPException(status_code=401, detail="missing or invalid RVClaw token")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _index_html()

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "runs_dir": str(runs_root), "default_planner": planner}

    @app.post("/api/runs")
    def create_run(payload: dict[str, Any], _: None = Depends(check_token)) -> dict[str, Any]:
        goal = str(payload.get("goal") or "检查 A-03 区域设备状态并生成报告")
        selected_planner = str(payload.get("planner") or planner)
        return run_demo(goal=goal, runs_dir=runs_root, planner_name=selected_planner).to_dict()

    @app.get("/api/runs")
    def runs() -> list[dict[str, Any]]:
        return list_runs(runs_root)

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str) -> dict[str, Any]:
        try:
            return get_run_detail(runs_root, run_id)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}/files")
    def run_files(run_id: str) -> list[str]:
        try:
            return list_run_files(runs_root, run_id)
        except (FileNotFoundError, ValueError) as exc:
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
            media_type = mimetypes.guess_type(name)[0] or file_payload["content_type"]
            return Response(file_payload["bytes"], media_type=media_type)
        media_type = file_payload["content_type"]
        if media_type.startswith("text/") or media_type in {"application/yaml", "application/jsonl"}:
            return PlainTextResponse(file_payload["content"], media_type=media_type)
        return Response(file_payload["content"], media_type=media_type)

    @app.get("/api/benchmarks")
    def benchmarks() -> list[dict[str, str]]:
        return read_benchmark_rows(runs_root)

    return app


def _index_html() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RVClaw K3 Console</title>
  <style>
    :root { color-scheme: dark; --bg: #111417; --panel: #181e22; --line: #2d373d; --text: #eaf0ee; --muted: #95a39f; --accent: #46d6a7; --warn: #f1b24a; --bad: #ef6a6a; }
    * { box-sizing: border-box; }
    body { margin: 0; background: radial-gradient(circle at 20% 0%, #1c332d, transparent 34%), var(--bg); color: var(--text); font-family: "Segoe UI", "Microsoft YaHei", sans-serif; }
    header { padding: 22px 28px 12px; border-bottom: 1px solid var(--line); display:flex; justify-content:space-between; align-items:end; gap:20px; }
    h1 { margin: 0; font-size: 24px; font-weight: 650; letter-spacing: 0; }
    .sub { color: var(--muted); margin-top: 6px; }
    main { display: grid; grid-template-columns: minmax(360px, 0.9fr) minmax(520px, 1.3fr); gap: 16px; padding: 16px; }
    section { background: color-mix(in srgb, var(--panel) 92%, transparent); border: 1px solid var(--line); border-radius: 8px; padding: 16px; min-width:0; }
    textarea, select, input, button { width: 100%; border: 1px solid var(--line); border-radius: 6px; background: #0f1315; color: var(--text); padding: 10px 12px; font: inherit; }
    textarea { min-height: 112px; resize: vertical; line-height: 1.5; }
    button { cursor: pointer; background: #17392f; border-color: #2f8d70; margin-top: 10px; font-weight: 650; }
    button:hover { background: #1e4b3f; }
    .grid { display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .metric { border:1px solid var(--line); border-radius:6px; padding:10px; background:#11171a; }
    .metric b { display:block; font-size:12px; color:var(--muted); margin-bottom:4px; }
    .timeline { display:flex; flex-direction:column; gap:8px; margin-top:12px; }
    .event { display:flex; justify-content:space-between; gap:10px; border:1px solid var(--line); border-radius:6px; padding:10px; background:#101517; }
    .ok { color: var(--accent); } .fail { color: var(--bad); } .warn { color: var(--warn); }
    .images { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:12px; }
    .images img, .file-image { width:100%; max-height:340px; object-fit:contain; border:1px solid var(--line); border-radius:6px; background:#090b0c; }
    .file-view { max-height: 340px; overflow:auto; background:#090b0c; border:1px solid var(--line); border-radius:6px; padding:12px; white-space:pre-wrap; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
    .files { display:flex; flex-wrap:wrap; gap:8px; margin:12px 0; }
    .files button { width:auto; margin:0; padding:7px 10px; background:#11171a; border-color:var(--line); font-weight:500; }
    @media (max-width: 980px) { main { grid-template-columns: 1fr; } .images { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header><div><h1>RVClaw K3 Console</h1><div class="sub">自然语言任务 · llama.cpp Planner · CV sample · RVBench evidence</div></div><div id="health" class="sub">checking...</div></header>
  <main>
    <section>
      <h2>任务输入</h2>
      <textarea id="goal">检查 A-03 区域设备状态并生成报告</textarea>
      <select id="planner"><option value="llama_cpp">llama_cpp</option><option value="mock">mock</option></select>
      <input id="token" placeholder="RVCLAW_WEB_TOKEN" type="password">
      <button onclick="runTask()">运行任务</button>
      <h2>历史运行</h2>
      <div id="runs" class="timeline"></div>
    </section>
    <section>
      <div class="grid">
        <div class="metric"><b>status</b><span id="status">idle</span></div>
        <div class="metric"><b>planner_mode</b><span id="plannerMode">-</span></div>
        <div class="metric"><b>tool calls</b><span id="toolCount">-</span></div>
        <div class="metric"><b>latency</b><span id="latency">-</span></div>
      </div>
      <h2>执行时间线</h2>
      <div id="timeline" class="timeline"></div>
      <h2>CV 画面</h2>
      <div class="images"><img id="capture" alt="capture"><img id="annotated" alt="annotated"></div>
      <h2>证据文件</h2>
      <div id="files" class="files"></div>
      <div id="fileView" class="file-view">等待运行...</div>
    </section>
  </main>
  <script>
    let currentRun = null;
    function requestOptions(options) {
      const next = options || {};
      const tokenValue = token.value.trim();
      if (tokenValue) {
        localStorage.setItem('rvclaw_web_token', tokenValue);
        next.headers = Object.assign({}, next.headers || {}, {'X-RVClaw-Token': tokenValue});
      }
      return next;
    }
    async function api(path, options) { const r = await fetch(path, requestOptions(options)); if (!r.ok) throw new Error(await r.text()); return r; }
    async function refreshHealth(){ const h = await (await api('/api/health')).json(); health.textContent = h.runs_dir; }
    async function refreshRuns(){ const rows = await (await api('/api/runs')).json(); runs.innerHTML = rows.map(r => `<div class="event" onclick="loadRun('${r.run_id}')"><span>${r.run_id}</span><span class="${r.status==='completed'?'ok':'fail'}">${r.status}</span></div>`).join(''); }
    async function runTask(){
      status.textContent = 'running'; timeline.innerHTML = '<div class="event"><span>planner</span><span class="warn">working</span></div>';
      const body = JSON.stringify({goal: goal.value, planner: planner.value});
      const result = await (await api('/api/runs', {method:'POST', headers:{'Content-Type':'application/json'}, body})).json();
      currentRun = result.run_id; await loadRun(currentRun); await refreshRuns();
    }
    async function loadRun(runId){
      currentRun = runId; const detail = await (await api(`/api/runs/${runId}`)).json(); const m = detail.metrics || {};
      status.textContent = m.status || detail.summary.status; status.className = status.textContent === 'completed' ? 'ok' : 'fail';
      plannerMode.textContent = m.planner_mode || '-'; toolCount.textContent = m.tool_call_count ?? '-'; latency.textContent = m.latency_ms ? `${m.latency_ms} ms` : '-';
      timeline.innerHTML = detail.trace.filter(e => e.event.includes('skill_call') || e.event.includes('planner')).map(e => `<div class="event"><span>${e.event}</span><span>${(e.payload.call && e.payload.call.name) || e.payload.planner || ''}</span></div>`).join('');
      files.innerHTML = detail.files.map(f => `<button onclick="showFile('${f}')">${f}</button>`).join('');
      const annotatedFile = detail.files.find(f => f.endsWith('_annotated.png')); const captureFile = detail.files.find(f => f.endsWith('_capture.png'));
      capture.src = captureFile ? fileUrl(runId, captureFile) : ''; annotated.src = annotatedFile ? fileUrl(runId, annotatedFile) : '';
      if (detail.files.includes('report.md')) showFile('report.md');
    }
    function fileUrl(runId, name) { return `/api/runs/${encodeURIComponent(runId)}/files/${name.split('/').map(encodeURIComponent).join('/')}`; }
    async function showFile(name){
      if(!currentRun) return;
      if (/\\.(png|jpg|jpeg|webp)$/i.test(name)) {
        fileView.innerHTML = `<img class="file-image" src="${fileUrl(currentRun, name)}" alt="${name}">`;
        return;
      }
      const text = await (await api(fileUrl(currentRun, name))).text();
      fileView.textContent = text;
    }
    token.value = localStorage.getItem('rvclaw_web_token') || '';
    refreshHealth().catch(e => health.textContent=e.message); refreshRuns();
  </script>
</body>
</html>"""
