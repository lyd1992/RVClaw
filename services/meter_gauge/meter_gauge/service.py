"""RVClaw 指针仪表盘识别 —— 上传网页 + 板端本地推理服务

设计约束（沿用本项目既有纪律）：
  * **独立服务、独立端口**（默认 8018），完全不触碰 rvclaw-studio(8017)。
    理由：studio 重启有 TCM 泄漏与内核 DMA 链表两个已知风险。
  * **零第三方 Web 依赖**：stdlib http.server + cv2/numpy/ncnn。
  * 模型**进程内常驻**，不用 C++ 子进程 —— C++ 可执行每次退出都走
    `__fput → dma_list_release`，那正是踩坏内核链表的入口。
  * 留档：每次请求的原图与标注图落盘（可追溯），写盘一律原子替换。

路由：
  GET  /                     网页
  GET  /sample.jpg           内置示例照片
  GET  /health               健康检查（含模型加载状态）
  GET  /records/<id>.jpg     留档原图
  GET  /records/<id>-ann.jpg 留档标注图
  POST /api/read             {"image_b64":..., "range_min":.., "range_max":.., "unit":..}
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import math
import os
import platform
import re
import socket
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from . import algo

PKG_ROOT = Path(__file__).resolve().parent.parent          # .../meter_gauge 包的上一层
WEB_DIR = PKG_ROOT / "web"
ASSET_DIR = PKG_ROOT / "assets"
MODEL_DIR = PKG_ROOT / "models"

DEFAULT_PORT = 8018
MAX_BODY = 48 * 1024 * 1024                                # 48 MB
MAX_SIDE = 1600                                            # 留档/预览的缩放口径
# ⚠️ 推理口径必须固定，别跟着客户端走。09-20 实证：原先推理用「客户端原始分辨率」、
#    留档只存缩到 1600 的预览 ⇒ 两套口径不是同一张图。用户报「示例图认不出来」时，
#    手上那张留档复现不出当时的推理输入，只能靠猜。现在推理与留档统一到 1600。
INFER_MAX_SIDE = int(os.environ.get("RVCLAW_METER_INFER_MAX_SIDE", "1600"))
JPEG_Q = 88
MAX_EVIDENCE_RAW_BYTES = 6 * 1024 * 1024                  # 超过就退回缩略预览，别把盘写满
VERSION = "1.1.0"                                         # /api/host 上报，页面顶栏显示真版本

_STATUS_ZH = {
    "readable": "识别成功",
    "no_meter": "未检测到仪表盘",
    "pointer_outside_scale": "表盘拍斜了，检测框不是接近正方形",
    "missing_pose_class": "未标出表盘刻度起点/终点或指针轴心",
    "unsupported": "不支持",
}

# 拒识时给用户看的中文解释。以前只有 status，于是 "missing keypoint(s): ['start']"
# 这类内部英文串会当读数大字显示 —— 用户既看不懂，也看不出该改什么。
# 注意措辞：旧文案断言「该表构图不在模型域内」，但 09-20 实测那只表**在**模型域内
# （同一张留档重跑就能读出 26.7 ℃），所以不能把偶发失手说成用户构图的问题。
_HINT_ZH = {
    "no_meter": "照片里没找到表盘。请让表盘占据画面主要部分，并避免反光。",
    "pointer_outside_scale": "表盘被拍成了明显的长方形（斜拍/透视），上游按宽高比直接拒识。请正对表盘、让圆形表盘完整入镜。",
    "missing_pose_class": "标准尺寸与原图两档都没标出刻度起点。请正对表盘、让 0 与满量程刻度都完整入镜；若反复失败，说明这块表的样式不在当前模型覆盖范围内。",
}

_LOCK = threading.Lock()


# ---------------------------------------------------------------- 留档
def _runtime_dir() -> Path:
    d = Path(os.environ.get("RVCLAW_METER_RUNTIME", str(PKG_ROOT / "runtime")))
    (d / "uploads").mkdir(parents=True, exist_ok=True)
    return d


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _keep_recent(upload_dir: Path, keep: int = 80) -> None:
    """按「一次请求」为单位清理：同一 rid 的 {rid}/-in/-ann 要么都留、要么都删。

    原先按文件数删，会把同一次请求的标注图删掉而留下原图（或反过来），
    页面上就会出现「有图无标注 / 有标注无图」的残记录。
    """
    try:
        groups: dict = {}
        for p in upload_dir.glob("*.jpg"):
            rid = p.name.rsplit(".jpg", 1)[0].removesuffix("-ann").removesuffix("-in")
            groups.setdefault(rid, []).append(p)
        order = sorted(groups.values(),
                       key=lambda g: max(f.stat().st_mtime for f in g), reverse=True)
        for group in order[keep:]:
            for f in group:
                f.unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------- 推理引擎
class Reader:
    """模型常驻的推理器；推理串行化（ncnn Net 不并发调用）。"""

    def __init__(self, model_dir: Path, threads: int = 4):
        self.model_dir = model_dir
        self.threads = threads
        self.det = None
        self.pose = None
        self.load_ms = None
        self.load()

    def _paths(self):
        m = self.model_dir
        return (m / "nanodet.param", m / "nanodet.bin",
                m / "yolov8s-pose-opt.param", m / "yolov8s-pose-opt.bin")

    def load(self) -> None:
        dp, db, pp, pb = self._paths()
        for p in (dp, db, pp, pb):
            if not p.exists():
                raise FileNotFoundError(f"model file missing: {p}")
        t0 = time.perf_counter()
        self.det = algo.Net(str(dp), str(db), threads=self.threads)
        self.pose = algo.Net(str(pp), str(pb), threads=self.threads)
        self.load_ms = (time.perf_counter() - t0) * 1000.0

    def read(self, image_bgr: np.ndarray, relax_gate: bool = False) -> dict:
        t0 = time.perf_counter()
        with _LOCK:
            res = algo.read_image(self.det, self.pose, image_bgr,
                                  relax_gate=relax_gate, want_vis=True)
        elapsed = (time.perf_counter() - t0) * 1000.0
        res["elapsed_ms"] = round(elapsed, 1)
        return res

    def pointer_pose(self, image_bgr: np.ndarray) -> dict:
        """Expose measured pointer geometry only, never a scale reading.

        A configured scale may use this when endpoint classes are confused.
        The normal reader and its aspect gate remain independent of this API.
        """
        started = time.perf_counter()
        with _LOCK:
            meters = algo.detect_meters(self.det, image_bgr)
            if len(meters) != 1:
                return {"status": "unreadable", "reason": "one dial required"}
            meter = meters[0]
            x, y, w, h = map(float, meter.rect)
            height, width = image_bgr.shape[:2]
            if not (meter.prob >= .65 and 0 <= x < x+w <= width
                    and 0 <= y < y+h <= height and min(w,h) >= 80):
                return {"status": "unreadable", "reason": "invalid dial locator"}
            xi, yi, wi, hi = map(int, (x,y,w,h))
            objects = algo.detect_pose(self.pose, image_bgr[yi:yi+hi,xi:xi+wi])
        pointers = [obj for obj in objects if obj.label == 0 and obj.prob >= .8
                    and obj.kps[2] >= .9 and obj.kps[5] >= .9]
        if len(pointers) != 1:
            return {"status": "unreadable", "reason": "one confident pointer required"}
        pointer = pointers[0]
        center = [float(pointer.kps[0]+xi), float(pointer.kps[1]+yi)]
        tip = [float(pointer.kps[3]+xi), float(pointer.kps[4]+yi)]
        if not all(math.isfinite(v) for v in center+tip):
            return {"status": "unreadable", "reason": "nonfinite pointer"}
        if not all(x <= p[0] <= x+w and y <= p[1] <= y+h for p in (center,tip)):
            return {"status": "unreadable", "reason": "pointer outside dial"}
        return {"status": "pointer_geometry", "center": center, "tip": tip,
                "score": float(pointer.prob),
                "keypoint_scores": [float(pointer.kps[2]),float(pointer.kps[5])],
                "meter_box": {"rect": meter.rect.tolist(), "score": float(meter.prob)},
                "image_size": [width,height], "n_meters": 1,
                "elapsed_ms": round((time.perf_counter()-started)*1000,1)}


_READER: Reader | None = None


def get_reader() -> Reader:
    global _READER
    if _READER is None:
        threads = int(os.environ.get("RVCLAW_METER_THREADS", "4"))
        _READER = Reader(MODEL_DIR, threads=threads)
    return _READER


# ---------------------------------------------------------------- 图像工具
def imread_u(path: Path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def imencode_u(ext: str, img: np.ndarray, quality: int = JPEG_Q) -> bytes:
    ok, buf = cv2.imencode(ext, img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("imencode failed")
    return buf.tobytes()


def shrink(img: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s >= 1.0:
        return img
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def decode_b64_image(data: str):
    """解码 dataURL / 裸 base64。任何格式问题一律返回 None（由调用方给出 400）。"""
    img, _, _ = decode_b64_image_ex(data)
    return img


def decode_b64_image_ex(data: str):
    """→ (image_bgr | None, 原始字节 | b"", 扩展名)

    原始字节要留下来：留档必须是"用户真的发了什么"，否则事后无法复现。
    """
    if not data:
        return None, b"", ".jpg"
    ext = ".jpg"
    if data.startswith("data:"):
        head = data.split(",", 1)[0].lower()
        for cand in (".png", ".webp", ".bmp", ".jpeg", ".jpg"):
            if cand in head:
                ext = ".jpg" if cand == ".jpeg" else cand
                break
    try:
        raw = base64.b64decode(data.split(",")[-1], validate=False)
    except (binascii.Error, ValueError):
        return None, b"", ext
    if not raw:
        return None, b"", ext
    try:
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        return None, b"", ext
    if img is None:
        return None, b"", ext
    return img, raw, ext


# ---------------------------------------------------------------- 业务：一次识别
def _contrast_normalize(image_bgr: np.ndarray) -> np.ndarray:
    """温和的对比度归一：只在 LAB 的 L 通道做线性拉伸。

    刻意不碰几何（不裁切/不缩放/不旋转）—— 这类改动会让画面比例变掉，
    而模型只输出比例，几何一动读数就直接错。
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    light, a, b = cv2.split(lab)
    light = cv2.normalize(light, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(cv2.merge((light, a, b)), cv2.COLOR_LAB2BGR)


def _summarize_read(out: dict) -> dict:
    """把 algo 的一次返回整理成「这次读数」的字段。

    顶层 status 是「整图流程」的，页面要的是「这一次读数」的，两者不是一回事
    （read_image 在 reads 非空时顶层恒为 ok，早期探针就因此误判过）。
    """
    reads = out.get("reads") or []
    if not reads:
        return {
            "status": out.get("status") or "unsupported",
            "gate_relaxed": False, "meter_box": None, "needle_method": None,
            "keypoints": None, "ratio": None, "annotated": None,
            "read_reason": out.get("reason") or "",
            "n_meters": len(out.get("meters") or []),
        }
    r = reads[0]
    readable = r.get("status") == "readable" and r.get("ratio") is not None
    return {
        "status": r.get("status") or "unsupported",
        "gate_relaxed": bool(r.get("gate_relaxed")),
        "meter_box": r.get("meter_box"),
        "needle_method": r.get("needle_method"),
        "keypoints": r.get("keypoints"),
        "ratio": float(r["ratio"]) if readable else None,
        "annotated": r.get("vis"),
        "read_reason": r.get("reason") or out.get("reason") or "",
        "n_meters": len(out.get("meters") or []),
    }


def run_read(image_bgr: np.ndarray, range_min: float, range_max: float,
             unit: str, relax_gate: bool = False,
             raw_bytes: bytes = b"", raw_ext: str = ".jpg",
             save: bool = True) -> dict:
    t0 = time.perf_counter()

    # 有界重试阶梯（最多三档、命中即停）：
    #   standard  固定 1600，与留档同一张图，可复现
    #   original  客户端原始分辨率 —— 覆盖「表盘在原图里很小、缩到 1600 就糊了」的相反情形
    #   contrast  只做对比度归一的 standard
    #   relaxed   仅当调用方显式要（上游会拒识的越界框围绕中心扩方），会失真，默认不启用
    canonical = shrink(image_bgr, INFER_MAX_SIDE)
    plan = [("standard", canonical, False)]
    if canonical.shape[:2] != image_bgr.shape[:2]:
        plan.append(("original", image_bgr, False))
    if relax_gate:
        plan.append(("relaxed", canonical, True))
    else:
        plan.append(("contrast", _contrast_normalize(canonical), False))

    attempts = []
    first = None
    picked = None
    for step, fed, relax_try in plan:
        out = get_reader().read(fed, relax_gate=relax_try)
        info = _summarize_read(out)
        attempts.append({
            "step": step,
            "size": [int(fed.shape[1]), int(fed.shape[0])],
            "status": info["status"],
            "relax_gate": bool(relax_try),
            "elapsed_ms": out.get("elapsed_ms"),
        })
        if info["status"] == "readable":
            picked = (step, fed, info, out)
            break
        if first is None:
            first = (step, fed, info, out)
    if picked is None:
        picked = first

    step, fed, info, out = picked
    status = info["status"]
    payload = {
        "status": status,
        "status_zh": _STATUS_ZH.get(status, status),
        "hint_zh": "" if status == "readable" else _HINT_ZH.get(status, ""),
        "reason": "" if status == "readable" else info["read_reason"],
        "image_size": [int(image_bgr.shape[1]), int(image_bgr.shape[0])],
        "infer_size": [int(fed.shape[1]), int(fed.shape[0])],
        "infer_step": step,
        "attempts": attempts,
        "n_meters": info["n_meters"],
        "range": [range_min, range_max],
        "unit": unit,
        "value": None,
        "ratio": None,
        "gate_relaxed": info["gate_relaxed"],
        "needle_method": info["needle_method"],
        "meter_box": info["meter_box"],
        # Detection is a locator even when the ROI reading gate rejects it.
        # Clients must run the unchanged model gates again on any new crop.
        "detected_meters": out.get("meters") or [],
        "keypoints": info["keypoints"],
        "elapsed_ms": out.get("elapsed_ms"),
        "model_load_ms": round(get_reader().load_ms or 0.0, 1),
    }
    if status == "readable":
        payload["ratio"] = info["ratio"]
        payload["value"] = range_min + (range_max - range_min) * info["ratio"]
        payload["value_round"] = round(payload["value"], 3)

    # 整图预览（画上表盘框），供页面左右对照
    overview = image_bgr.copy()
    rect = (info["meter_box"] or {}).get("rect")
    if rect:
        x, y, w, h = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
        cv2.rectangle(overview, (x, y), (x + w, y + h), (0, 0, 255), 3)

    annotated = info["annotated"]
    payload["annotated_b64"] = base64.b64encode(imencode_u(".jpg", shrink(annotated))).decode() \
        if annotated is not None else None
    payload["overview_b64"] = base64.b64encode(imencode_u(".jpg", shrink(overview, 1100))).decode()
    payload["annotated_original_size"] = (
        [int(annotated.shape[1]), int(annotated.shape[0])] if annotated is not None else None)

    # 留档：{rid}.jpg = 用户上传的原始字节（JPEG 且不过大时逐字节原样存），
    #       {rid}-in.jpg = 真正送进模型的那张图，{rid}-ann.jpg = 标注图。
    # 三者分开存，是为了「这次为什么没认出来」能被复现，而不是只剩一张缩过的预览。
    rid = time.strftime("%Y%m%d-%H%M%S") + "-" + os.urandom(2).hex()
    if save:
        try:
            up = _runtime_dir() / "uploads"
            exact = raw_bytes and raw_ext in (".jpg", ".png", ".webp", ".bmp") \
                and len(raw_bytes) <= MAX_EVIDENCE_RAW_BYTES
            if exact:
                _atomic_write(up / f"{rid}.jpg", raw_bytes)
            else:
                _atomic_write(up / f"{rid}.jpg", imencode_u(".jpg", shrink(image_bgr)))
            _atomic_write(up / f"{rid}-in.jpg", imencode_u(".jpg", fed))
            if annotated is not None:
                _atomic_write(up / f"{rid}-ann.jpg", imencode_u(".jpg", shrink(annotated)))
            payload["record_id"] = rid
            payload["record_exact"] = bool(exact)
            _keep_recent(up)
        except OSError:
            payload["record_id"] = None
            payload["record_exact"] = False

    payload["total_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    return payload


def host_info() -> dict:
    """/api/host：页面顶栏要展示的"真值"。原先顶栏那排（EdgeOne-01 / 192.168.11.93 /
    2d 14h 32m / v1.2.3）是写死的演示文案，没有一行是真的。"""
    def _read(path: str, default: str = "") -> str:
        try:
            return Path(path).read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return default

    try:
        uptime = float(_read("/proc/uptime", "0").split()[0])
    except (ValueError, IndexError):
        uptime = 0.0
    mem_mb = 0
    for line in _read("/proc/meminfo").splitlines():
        if line.startswith("MemTotal"):
            try:
                mem_mb = round(int(line.split()[1]) / 1024.0)
            except (ValueError, IndexError):
                pass
            break
    return {
        "ok": True,
        "service": "rvclaw-meter-gauge",
        "version": VERSION,
        "hostname": socket.gethostname(),
        "arch": platform.machine(),
        # os.cpu_count() 在这块板子上报 16，但 nproc（可用核）是 8 —— 顶栏要显示真能用的数
        "cpus": (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                 else (os.cpu_count() or 0)),
        "cpus_total": os.cpu_count() or 0,
        "mem_total_mb": mem_mb,
        "kernel": platform.release(),
        "board_model": _read("/proc/device-tree/model").replace("\x00", ""),
        "uptime_s": round(uptime),
        "boot_id": _read("/proc/sys/kernel/random/boot_id"),
        "infer_max_side": INFER_MAX_SIDE,
    }


def _warm_up() -> None:
    """开机先自己跑一张。

    冷启动的第一次推理不该由第一个用户来承担 —— 09-20 那次「示例图认不出来」
    第 1 次请求（服务启动后 4 秒）就是全场唯一或最早的一次异常。
    """
    try:
        img = imread_u(ASSET_DIR / "sample-01.jpg")
        if img is None:
            print("[meter-gauge] warm-up skipped: sample-01.jpg 读不出来", flush=True)
            return
        t0 = time.perf_counter()
        res = run_read(img, 0.0, 100.0, "\u2103", save=False)
        print("[meter-gauge] warm-up %.0f ms -> %s value=%s step=%s" % (
            (time.perf_counter() - t0) * 1000.0, res.get("status"),
            res.get("value"), res.get("infer_step")), flush=True)
    except Exception as exc:                      # 预热失败不该拦住启动
        print(f"[meter-gauge] warm-up failed: {exc!r}", flush=True)


# ---------------------------------------------------------------- HTTP
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".json": "application/json; charset=utf-8",
}


class Handler(BaseHTTPRequestHandler):
    server_version = "RVClawMeterGauge/1.0"
    protocol_version = "HTTP/1.1"

    # ---- 基础输出
    def _send(self, code: int, body: bytes, ctype: str, cache: str = "no-store"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code: int, obj: dict):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _file(self, path: Path, cache: str = "no-store"):
        if not path.is_file():
            self._json(404, {"error": "not found", "path": str(path)})
            return
        self._send(200, path.read_bytes(),
                   _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream"), cache)

    def log_message(self, fmt, *args):        # 走 journal，别刷屏
        if os.environ.get("RVCLAW_METER_VERBOSE"):
            super().log_message(fmt, *args)

    # ---- 路由
    def do_OPTIONS(self):
        self._send(204, b"", "text/plain")

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        try:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._file(WEB_DIR / "index.html")
            elif path == "/health":
                r = get_reader()
                self._json(200, {
                    "ok": True,
                    "model_loaded": r.det is not None and r.pose is not None,
                    "model_load_ms": round(r.load_ms or 0.0, 1),
                    "threads": r.threads,
                    "model_dir": str(r.model_dir),
                    "infer_max_side": INFER_MAX_SIDE,
                    "service_version": VERSION,
                    "uptime_s": host_info()["uptime_s"],
                    "uptime_note": "model resident in-process",
                })
            elif path == "/api/host":
                self._json(200, host_info())
            elif path == "/sample.jpg":
                self._file(ASSET_DIR / "sample-01.jpg", cache="no-store")
            elif path.startswith("/records/"):
                name = os.path.basename(path)
                if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{4}(-ann|-in)?\.jpg", name):
                    self._json(400, {"error": "bad record name"})
                    return
                self._file(_runtime_dir() / "uploads" / name)
            elif path in ("/app.js", "/app.css"):
                self._file(WEB_DIR / path.lstrip("/"))
            else:
                self._json(404, {"error": "not found", "path": path})
        except Exception:
            self._json(500, {"error": traceback.format_exc(limit=4)})

    def do_POST(self):
        try:
            path = self.path.split("?", 1)[0]
            if path not in ("/api/read", "/api/pointer-pose"):
                self._json(404, {"error": "not found", "path": path})
                return

            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY:
                self._json(413, {"error": f"bad content-length: {length}"})
                return
            body = json.loads(self.rfile.read(length).decode("utf-8"))

            b64 = body.get("image_b64") or ""
            img, raw_bytes, raw_ext = decode_b64_image_ex(b64)
            if img is None:
                self._json(400, {"error": "cannot decode image"})
                return

            if path == "/api/pointer-pose":
                self._json(200, get_reader().pointer_pose(img))
                return

            rmin = float(body.get("range_min", 0.0))
            rmax = float(body.get("range_max", 100.0))
            if rmax <= rmin:
                self._json(400, {"error": "range_max must be > range_min"})
                return
            unit = str(body.get("unit", ""))[:16]
            relax = bool(int(body.get("relax_gate", 0)))

            res = run_read(img, rmin, rmax, unit, relax_gate=relax,
                           raw_bytes=raw_bytes, raw_ext=raw_ext)
            self._json(200, res)
        except Exception:
            self._json(500, {"error": traceback.format_exc(limit=6)})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("RVCLAW_METER_PORT", DEFAULT_PORT)))
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args(argv)

    r = get_reader()
    print(f"[meter-gauge] models loaded in {r.load_ms:.0f} ms from {r.model_dir}", flush=True)
    _warm_up()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(f"[meter-gauge] listening on http://{args.host}:{args.port}/", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0
