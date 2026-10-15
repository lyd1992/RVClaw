import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import base64
import json
import math
import mimetypes
import os
import signal
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np

from .person_tracking import get_person_tracking_payload
from .sample_flow import run_sample_flow
from .video_detection import get_video_detection_payload
from .vision_state import get_vision_state
from rvclaw_algo import MeterReader

# The in-process PP-OCRv5 engine.  Same two ONNX models as the standalone
# binary, run inside this process on onnxruntime's CPU provider -- see
# rvclaw_algo.ppocr_onnx for why the subprocess had to go.  Missing module (an
# older board) just means the upload path keeps using the binary.
try:
    from rvclaw_algo.ppocr_onnx import PpocrOnnxEngine
except Exception as _ppocr_import_error:  # pragma: no cover - deployment dependent
    PpocrOnnxEngine = None  # type: ignore[assignment]
    PPOCR_IMPORT_ERROR = "%s: %s" % (type(_ppocr_import_error).__name__, _ppocr_import_error)
else:
    PPOCR_IMPORT_ERROR = None

# The pointer-gauge reader: the ETH Zurich `analog_gauge_reader` models, run as
# ONNX inside this process.  See rvclaw_algo/ethz_gauge_reader.py for the route
# (the board cannot run upstream's torch/mmocr stack) and for the deviations
# from upstream's pipeline.  Missing package or missing model files means the
# gauge path reports itself unsupported -- it never falls back to the withdrawn
# in-house geometry.
try:
    from rvclaw_algo.ethz_gauge_reader import EthzGaugeReader
except Exception as _gauge_import_error:  # pragma: no cover - deployment dependent
    EthzGaugeReader = None  # type: ignore[assignment]
    GAUGE_IMPORT_ERROR = "%s: %s" % (type(_gauge_import_error).__name__, _gauge_import_error)
else:
    GAUGE_IMPORT_ERROR = None
from .meter_config import MAX_REFERENCE_IMAGE_SIZE, MeterConfigError, MeterConfigStore, meter_is_ready
from .meter_trial import TrialError, run_trial, runtime_reading
from . import rvbench_store


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IMAGE = REPO_ROOT / "samples" / "sample_meter.svg"
DEFAULT_RUNS = REPO_ROOT / "runs"
STUDIO_ROOT = Path(
    os.getenv("RVCLAW_STUDIO_ROOT", str(REPO_ROOT / "studio" / "web"))
)
METER_TRIAL_ROOT = REPO_ROOT / "runs" / "meter-trials"
METER_EVIDENCE_ROOT = REPO_ROOT / "runs" / "meter-uploads"
METER_EVIDENCE_INDEX = METER_EVIDENCE_ROOT / "index.json"
INSPECTION_PHOTO_ROOT = Path(
    os.getenv("RVCLAW_INSPECTION_PHOTO_ROOT", "/data/inspection_photos")
)
INSPECTION_EVIDENCE_TOKEN = re.compile(r"^[A-Za-z0-9_-]{1,96}$")
# The upload path used to crop a fixed normalised rectangle here
# (0.412, 0.310, 0.160, 0.100), which only ever matched the Aclara meter it was
# tuned for and silently mis-cropped every other upload -- the display usually
# fell outside it entirely.  The display is now located per photo: in pixels for
# a screen (see _read_display_region) or by PP-OCR text detection for printed
# and mechanical counters (see _select_digital_reading).
METER_MUTATION_LOCK = threading.Lock()
METER_STORE = MeterConfigStore(
    REPO_ROOT / "configs" / "meter" / "meters.json",
    REPO_ROOT / "configs" / "meter" / "reference-images",
)
METER_CONFIGURATION_PATH = re.compile(r"^/api/meter-configurations/([^/]+)$")
METER_REFERENCE_UPLOAD_PATH = re.compile(r"^/api/meter-configurations/([^/]+)/reference-image$")
METER_ACTIVATION_PATH = re.compile(r"^/api/meter-configurations/([^/]+)/activate$")
METER_CONFIG_PATH = Path(
    os.getenv("RVCLAW_METER_CONFIG", str(REPO_ROOT / "configs" / "meter" / "meters.json"))
)


def _device_snapshot():
    """Combine board telemetry, camera state, and the active visual model."""
    telemetry = _benchmark_status()
    camera = CAMERA_CONTROLLER.status()
    vision = get_vision_state()
    item = (camera.get("cameras") or [{}])[0]
    return {
        "sampled_at": telemetry.get("sampled_at"),
        "host": telemetry.get("host"),
        "network": telemetry.get("network", {}),
        "cpu": telemetry.get("cpu", {}),
        "inference": telemetry.get("inference", {}),
        "stability": telemetry.get("stability", {}),
        "camera": {"active_mode": camera.get("active_mode"), "switching": camera.get("switching"), "generation": camera.get("generation"), "name": item.get("name"), "online": item.get("online"), "stream_url": item.get("stream_url"), "width": item.get("width"), "height": item.get("height"), "fps": item.get("fps")},
        "model": vision.get("model", {}),
    }


def _recent_logs(limit=80):
    candidates = [
        Path("/data/RVClaw/logs/studio-meter.log"),
        REPO_ROOT / "logs" / "studio-meter.log",
    ]
    for path in candidates:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
            rows = []
            for line in lines:
                rows.append({"time": "", "level": "INFO", "source": "studio", "message": line[-1200:]})
            return {"logs": rows, "source": str(path)}
        except OSError:
            continue
    return {"logs": [], "source": None}


def _load_digital_meter_config():
    """Load the selected ready meter calibration for the K3 process command."""
    try:
        document = json.loads(METER_CONFIG_PATH.read_text(encoding="utf-8"))
        meters = document.get("meters", []) if isinstance(document, dict) else []
        persisted_id = document.get("active_meter_id", "") if isinstance(document, dict) else ""
    except (OSError, json.JSONDecodeError):
        meters = []
        persisted_id = ""
    meters = [item for item in meters if isinstance(item, dict)]
    selected_id = (
        persisted_id.strip() if isinstance(persisted_id, str) else ""
    ) or os.getenv("RVCLAW_METER_ID", "").strip()
    def is_ready(item):
        return bool(
            item.get("display_region")
            and all(item.get("alignment_markers") or [])
            and item.get("reference_image")
        )

    selected = next(
        (item for item in meters if item.get("meter_id") == selected_id and is_ready(item)),
        None,
    )
    if selected is None:
        selected = next((item for item in meters if is_ready(item)), None)
    if selected is None:
        return {
            "meter_id": None,
            "name": None,
            "display_region": [0.10, 0.15, 0.80, 0.70],
            "unit": None,
            "decimal_places": 0,
            "digit_count": 0,
            "min_confidence": 0.58,
            "consistency_count": 3,
            "normal_min": None,
            "normal_max": None,
            "max_change": None,
            "reference_image": None,
            "alignment_markers": [None, None],
            "configured": False,
        }
    region = selected.get("display_region") or {}
    return {
        "meter_id": selected.get("meter_id"),
        "name": selected.get("name"),
        "display_region": [
            float(region.get("x", 0.10)),
            float(region.get("y", 0.15)),
            float(region.get("width", 0.80)),
            float(region.get("height", 0.70)),
        ],
        "unit": selected.get("unit") or None,
        "decimal_places": int(selected.get("decimal_places") or 0),
        "digit_count": int(selected.get("digit_count") or 0),
        "min_confidence": float(selected.get("min_confidence", 0.58)),
        "consistency_count": int(selected.get("consistency_count", 3)),
        "normal_min": selected.get("normal_min"),
        "normal_max": selected.get("normal_max"),
        "max_change": selected.get("max_change"),
        "reference_image": (
            str(METER_STORE.reference_image_dir / selected["reference_image"])
            if selected.get("reference_image") else None
        ),
        "alignment_markers": selected.get("alignment_markers") or [None, None],
        "configured": is_ready(selected),
    }


DIGITAL_METER_CONFIG = _load_digital_meter_config()
DIGITAL_METER_ID = DIGITAL_METER_CONFIG["meter_id"]
DIGITAL_METER_REGION = DIGITAL_METER_CONFIG["display_region"]
DIGITAL_METER_UNIT = DIGITAL_METER_CONFIG["unit"]


def _empty_inference_metrics():
    return {
        "detections": 0,
        "persons": 0,
        "helmet": 0,
        "no_helmet": 0,
        "checking": 0,
        "inference_ms": None,
        "inferences": 0,
        "ppe_ready": False,
        "ppe_inference_ms": None,
        "ppe_inferences": 0,
        "meter_type": None,
        "meter_status": None,
        "meter_value": None,
        "meter_angle_deg": None,
        "meter_confidence": None,
        "meter_raw_text": "",
        "meter_value_method": "",
        "meter_text_items": [],
    }


class CameraStream:
    """Capture one UVC camera once and fan its latest JPEG frame out to clients."""

    def __init__(
        self,
        camera_id,
        name,
        device,
        width,
        height,
        fps,
        stream_fps=None,
        command=None,
        working_directory=None,
        inference=None,
        required_paths=None,
    ):
        self.camera_id = camera_id
        self.name = name
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        # The capture/inference loop remains at full speed.  Only cap the
        # browser-facing MJPEG fan-out so slow remote clients cannot build an
        # unbounded socket backlog and starve the other camera/API requests.
        self.stream_fps = max(
            1,
            min(int(stream_fps if stream_fps is not None else fps), int(fps)),
        )
        self.command = command
        self.working_directory = working_directory
        self.inference = inference
        self.required_paths = [Path(path) for path in (required_paths or [])]
        self._condition = threading.Condition()
        self._frame = None
        self._sequence = 0
        self._last_frame_at = None
        self._frame_times = deque(maxlen=120)
        self._last_error = None
        self._stderr_lines = deque(maxlen=20)
        self._inference_metrics = _empty_inference_metrics()
        self._inference_active = False
        self._inference_error = None
        self._process = None
        self._stopping = threading.Event()
        self._thread = None
        self._lifecycle_lock = threading.Lock()

    def start(self):
        with self._lifecycle_lock:
            if self._thread and self._thread.is_alive():
                return
            with self._condition:
                self._stopping.clear()
                self._frame = None
                self._sequence += 1
                self._last_frame_at = None
                self._frame_times.clear()
                self._last_error = None
                self._stderr_lines.clear()
                self._inference_metrics = _empty_inference_metrics()
                self._inference_active = False
                self._inference_error = None
                self._condition.notify_all()
            self._thread = threading.Thread(
                target=self._capture_loop,
                name=f"camera-{self.camera_id}",
                daemon=True,
            )
            self._thread.start()

    def stop(self):
        with self._lifecycle_lock:
            self._stopping.set()
            with self._condition:
                self._frame = None
                self._sequence += 1
                self._last_frame_at = None
                self._frame_times.clear()
                self._inference_active = False
                self._condition.notify_all()
            process = self._process
            if process and process.poll() is None:
                # Let the native decoder release its TCM buffers before the
                # next mode claims the shared camera. Killing it immediately
                # can exhaust the decoder pool after repeated switches.
                process.terminate()
                try:
                    grace = float(os.getenv("RVCLAW_CAMERA_STOP_GRACE_SECONDS", "0.25"))
                    process.wait(timeout=max(0.05, min(grace, 1.0)))
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1)
            thread = self._thread
            if thread and thread.is_alive():
                # The controller starts the next pipeline only after this
                # worker has exited; waiting here prevents two native decoder
                # processes from briefly sharing the camera/TCM resources.
                thread.join(timeout=3)
            self._thread = None
            self._process = None

    def is_running(self):
        thread = self._thread
        return bool(thread and thread.is_alive() and not self._stopping.is_set())

    def wait_for_frame(self, previous_sequence, timeout=5.0):
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence != previous_sequence or self._stopping.is_set(),
                timeout=timeout,
            )
            return self._frame, self._sequence

    def fresh_frame(self, timeout=3.0, max_age_seconds=1.5):
        """Return a newly captured JPEG without opening or switching the camera."""

        with self._condition:
            previous_sequence = self._sequence
            ready = self._condition.wait_for(
                lambda: (
                    self._frame is not None
                    and self._sequence != previous_sequence
                )
                or self._stopping.is_set(),
                timeout=max(0.05, float(timeout)),
            )
            if not ready or self._frame is None or self._last_frame_at is None:
                raise TimeoutError("camera did not provide a fresh frame in time")
            frame = bytes(self._frame)
            sequence = int(self._sequence)
            captured_at = float(self._last_frame_at)
        age_seconds = max(0.0, time.time() - captured_at)
        if age_seconds > max(0.05, float(max_age_seconds)):
            raise TimeoutError("camera frame is stale")
        return frame, sequence, captured_at

    def status(self):
        now = time.time()
        with self._condition:
            last_frame_at = self._last_frame_at
            frame_times = list(self._frame_times)
            last_error = self._last_error
            frame_ready = self._frame is not None
        age_ms = None
        if last_frame_at is not None:
            age_ms = round((now - last_frame_at) * 1000)
        recent_times = [stamp for stamp in frame_times if now - stamp <= 3.0]
        measured_fps = 0.0
        if len(recent_times) > 1:
            measured_fps = (len(recent_times) - 1) / (recent_times[-1] - recent_times[0])
        return {
            "id": self.camera_id,
            "name": self.name,
            "device": self.device,
            "stream_url": f"/stream/{self.camera_id}.mjpg",
            "width": self.width,
            "height": self.height,
            "fps": round(measured_fps, 1),
            "target_fps": self.fps,
            "stream_fps": self.stream_fps,
            "active": self.is_running(),
            "online": frame_ready and age_ms is not None and age_ms < 3000,
            "frame_age_ms": age_ms,
            "last_error": last_error,
            "inference": {
                **(self.inference or {"enabled": False}),
                **self._inference_metrics,
                "ready": bool(self._inference_active and self._frame is not None),
                "error": self._inference_error,
            },
        }

    def _capture_loop(self):
        while not self._stopping.is_set():
            try:
                self._capture_once()
            except Exception as error:  # keep the web service alive if a camera is unplugged
                if not self._stopping.is_set():
                    with self._condition:
                        self._last_error = str(error)
            if not self._stopping.wait(1.0):
                continue

    def _capture_once(self):
        missing_paths = [str(path) for path in self.required_paths if not path.exists()]
        self._inference_active = bool(self.command and not missing_paths)
        self._inference_error = (
            None if self._inference_active else
            f"Missing detector dependency: {', '.join(missing_paths)}" if self.command else None
        )
        command = self.command if self._inference_active else self._ffmpeg_command()
        self._process = subprocess.Popen(
            command,
            cwd=self.working_directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(self._process.stderr,),
            name=f"camera-{self.camera_id}-stderr",
            daemon=True,
        )
        stderr_thread.start()
        buffer = bytearray()
        try:
            while not self._stopping.is_set():
                chunk = self._process.stdout.read(65536)
                if not chunk:
                    if self._stopping.is_set():
                        return
                    stderr = " | ".join(self._stderr_lines)
                    raise RuntimeError(stderr or f"capture process exited with code {self._process.poll()}")
                buffer.extend(chunk)
                self._publish_complete_frames(buffer)
        finally:
            process = self._process
            self._process = None
            if process is not None:
                # Close our end of the pipe before asking the child to stop.  A
                # child whose write is blocked on a pipe nobody drains cannot act
                # on SIGTERM at all, which is precisely how these encoders ended
                # up being SIGKILLed; closing the pipe makes that write fail at
                # once so they can exit on their own.  Only a child that still
                # refuses after the grace period is force-killed, because a
                # forced kill hands the DMA-buffer release to the kernel -- the
                # path that corrupts the global DMA list on this board.
                if process.stdout is not None:
                    try:
                        process.stdout.close()
                    except Exception:
                        pass
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=CAPTURE_TERMINATE_GRACE)
                except subprocess.TimeoutExpired:
                    process.kill()
                    try:
                        process.wait(timeout=CAPTURE_KILL_GRACE)
                    except subprocess.TimeoutExpired:
                        with self._condition:
                            self._last_error = (
                                "capture process %s ignored SIGTERM and SIGKILL" % process.pid)

    def _ffmpeg_command(self):
        return [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "v4l2",
            "-input_format",
            "mjpeg",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            str(self.fps),
            "-i",
            self.device,
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "copy",
            "-f",
            "mjpeg",
            "pipe:1",
        ]

    def _drain_stderr(self, stream):
        for raw_line in iter(stream.readline, b""):
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if line.startswith("RVCLAW_METRICS "):
                try:
                    metrics = json.loads(line.removeprefix("RVCLAW_METRICS "))
                    with self._condition:
                        self._inference_metrics.update(metrics)
                    continue
                except json.JSONDecodeError:
                    pass
            with self._condition:
                self._stderr_lines.append(line)

    def _publish_complete_frames(self, buffer):
        while True:
            start = buffer.find(b"\xff\xd8")
            if start < 0:
                if len(buffer) > 1:
                    del buffer[:-1]
                return
            end = buffer.find(b"\xff\xd9", start + 2)
            if end < 0:
                if start > 0:
                    del buffer[:start]
                return
            frame = bytes(buffer[start : end + 2])
            del buffer[: end + 2]
            with self._condition:
                self._frame = frame
                self._sequence += 1
                self._last_frame_at = time.time()
                self._frame_times.append(self._last_frame_at)
                self._last_error = None
                self._condition.notify_all()


YOLOV8_BINARY = Path(
    os.getenv("RVCLAW_YOLOV8_BINARY", str(REPO_ROOT / "build" / "yolov8_mjpeg"))
)
YOLOV8_MODEL = Path(os.path.expanduser("~/.cache/models/vision/yolov8/yolov8n_no_dfl.q.onnx"))
VISION_ROOT = Path(os.getenv("SPACEMIT_VISION_ROOT", "/data/spacemit_robot/components/model_zoo/vision"))
YOLOV8_CONFIG = Path(
    os.getenv(
        "RVCLAW_YOLOV8_CONFIG",
        str(REPO_ROOT / "configs" / "thresholds" / "general_yolov8.yaml"),
    )
)
SAFETYVISION_MODEL = Path(
    os.path.expanduser(
        os.getenv(
            "RVCLAW_PPE_MODEL",
            "~/.cache/models/vision/ppe/safetyvision_yolov8n/best_fp16.onnx",
        )
    )
)
SAFETYVISION_CONFIG = Path(
    os.getenv(
        "RVCLAW_PPE_CONFIG",
        str(REPO_ROOT / "configs" / "thresholds" / "helmet_yolov8.yaml"),
    )
)
ASTRA_PERSON_CONFIG = Path(
    os.getenv(
        "RVCLAW_ASTRA_PERSON_CONFIG",
        str(REPO_ROOT / "configs" / "thresholds" / "astra_person_bytetrack.yaml"),
    )
)
HARDHAT_MODEL = Path(
    os.path.expanduser(
        os.getenv(
            "RVCLAW_HARDHAT_MODEL",
            "~/.cache/models/vision/ppe/rvclaw_hardhat_yolov8n/best_fp16.onnx",
        )
    )
)
HARDHAT_CONFIG = Path(
    os.getenv(
        "RVCLAW_HARDHAT_CONFIG",
        str(REPO_ROOT / "configs" / "thresholds" / "hardhat_yolov8.yaml"),
    )
)
ASTRA_DEVICE = "/dev/v4l/by-id/usb-Astra_Pro_HD_Camera_Astra_Pro_HD_Camera-video-index0"
# Astra Pro's RGB (Red Green Blue) stream supports 1280x720 at 30 FPS (frames per second). Use
# this only for the digital display crop; pointer and safety modes retain the
# lower-bandwidth profile used by their existing models.
ASTRA_DIGITAL_WIDTH = 1280
ASTRA_DIGITAL_HEIGHT = 720
ASTRA_DIGITAL_FPS = 30
# Digital-meter input is a pre-configured display region. It uses the same
# PP-OCRv5 recognition model but bypasses DBNet text detection.
PPOCR_CONFIG = REPO_ROOT / "configs" / "vision" / "ppocr_meter_rec.yaml"
PPOCR_MODEL = Path(os.path.expanduser("~/.cache/models/vision/ppocr/PP-OCRv5_mobile_det.onnx"))
PPOCR_REC_MODEL = Path(os.path.expanduser("~/.cache/models/vision/ppocr/PP-OCRv5_mobile_rec.onnx"))
PPOCR_UPLOAD_BINARY = Path(os.getenv("RVCLAW_PPOCR_UPLOAD_BINARY", "/data/RVClaw/build/meter-paddleocr/meter_paddleocr"))
# Uploaded photos carry no calibrated display region, so the upload path runs
# PP-OCR with DBNet text detection enabled and locates the reading itself.
# The camera pipeline keeps the rec-only profile because its input is already
# cropped to the display area recorded in configs/meter/meters.json.
PPOCR_UPLOAD_CONFIG = Path(os.getenv(
    "RVCLAW_PPOCR_UPLOAD_CONFIG",
    str(REPO_ROOT / "configs" / "vision" / "ppocr_meter_upload.yaml"),
))
UPLOAD_TEXT_MIN_SCORE = float(os.getenv("RVCLAW_UPLOAD_TEXT_MIN_SCORE", "0.5"))
UPLOAD_MAX_DIGIT_BOXES = int(os.getenv("RVCLAW_UPLOAD_MAX_DIGIT_BOXES", "6"))
UPLOAD_CANDIDATE_LIMIT = 5

# --- analogue gauge ---------------------------------------------------------
# Pointer-gauge reading runs the ETH Zurich `analog_gauge_reader` models -- gauge
# face detection, needle segmentation and DINOv2 key points -- exported to ONNX
# and executed in-process, with this board's PP-OCRv5 supplying the dial numbers
# in place of MMOCR.  `_read_pointer_gauge` below is the only upload entry point.
# The earlier in-house readers are gone from the upload path: the Hough-circle
# one that `MeterReader` still exposes for the live camera modes, and the
# OCR-label-fitted scale that used to live in `rvclaw_algo.gauge_reader` (kept
# under output/retired-pointer-reader-20260918/ for rollback).

# Grace given to a capture child (yolov8_mjpeg / ffmpeg) between SIGTERM and the
# SIGKILL that forces it out.  2 s was not enough: the child is usually blocked
# writing to a pipe nobody drains, so it cannot even see SIGTERM in that window,
# and a SIGKILLed child leaves the kernel to release its V4L2 + SpacemiT DMA
# buffers during task exit -- the path that corrupts the global DMA list on this
# board.  Every "X"-state ffmpeg in the kernel oops trail arrived there by a
# forced kill.  See docs/板端OCR卡死-内核DMA链表损坏-20260917.md.
CAPTURE_TERMINATE_GRACE = float(os.getenv("RVCLAW_CAPTURE_TERMINATE_GRACE", "8"))
CAPTURE_KILL_GRACE = float(os.getenv("RVCLAW_CAPTURE_KILL_GRACE", "5"))

# --- PP-OCRv5 execution engine ---------------------------------------------
# Every upload used to shell out to build/meter-paddleocr/meter_paddleocr.  On a
# board whose kernel DMA list has been corrupted (__fput -> dma_list_release
# faults on LIST_POISON1) that helper never returns, so each upload burns its
# whole timeout and answers nothing; measured on 2026-09-17, it stalls on a
# full-size photo while a 300x176 crop still finishes in ~1 s.  The identical
# ONNX models run in-process on the CPU provider, which touches no SpacemiT DMA
# and also drops the per-call model load.  Set RVCLAW_PPOCR_ENGINE=legacy to go
# back to the subprocess.
PPOCR_ENGINE_KIND = os.getenv("RVCLAW_PPOCR_ENGINE", "onnx").strip().lower()
_PPOCR_ENGINE = None
_PPOCR_ENGINE_LOCK = threading.Lock()
PPOCR_ENGINE_ERROR = PPOCR_IMPORT_ERROR


def _ppocr_engine():
    """The shared in-process engine, or ``None`` to use the standalone binary."""
    global _PPOCR_ENGINE, PPOCR_ENGINE_ERROR
    if PPOCR_ENGINE_KIND == "legacy" or PpocrOnnxEngine is None:
        return None
    if _PPOCR_ENGINE is not None:
        return _PPOCR_ENGINE
    with _PPOCR_ENGINE_LOCK:
        if _PPOCR_ENGINE is None:
            try:
                _PPOCR_ENGINE = PpocrOnnxEngine()
            except Exception as error:
                # Session creation happens on first use, so a failure here is
                # not fatal to the upload path -- the binary still exists.
                PPOCR_ENGINE_ERROR = "%s: %s" % (type(error).__name__, error)
                return None
    return _PPOCR_ENGINE


def _ppocr_available():
    """Whether PP-OCRv5 can run at all: the in-process engine, else the binary.

    Cheap by design -- it only checks that the profile's model files exist, so
    it is safe to call on every upload.
    """
    engine = _ppocr_engine()
    if engine is not None:
        return bool(engine.models_present(PPOCR_UPLOAD_CONFIG))
    return PPOCR_UPLOAD_BINARY.is_file()


def _ppocr_payload(image, config_path, timeout):
    """Run PP-OCRv5 and return the JSON envelope the binary used to print.

    ``image`` is either a BGR ndarray or already-encoded bytes; an ndarray is
    encoded to JPEG (quality 95) first so both engines see exactly the same
    pixels.  Returns ``None`` when nothing usable came back -- callers decide
    whether that means "refuse" or "report an error".
    """
    engine = _ppocr_engine()
    if engine is not None:
        try:
            return engine.recognize(image, str(config_path))
        except Exception:
            return None
    return _ppocr_payload_via_binary(image, config_path, timeout)


def _ppocr_payload_via_binary(image, config_path, timeout):
    """Original path: write a JPEG to disk and run the native helper."""
    if isinstance(image, np.ndarray):
        ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            return None
        image_bytes = buffer.tobytes()
    else:
        image_bytes = image
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="rvclaw-ocr-", suffix=".jpg", delete=False) as temp:
            temp.write(image_bytes)
            temp_path = temp.name
        completed = subprocess.run(
            [str(PPOCR_UPLOAD_BINARY), "--image", temp_path, "--config", str(config_path)],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
        if completed.returncode != 0:
            return None
        payload_line = next(
            (line.strip() for line in reversed(completed.stdout.splitlines())
             if line.strip().startswith("{")), "")
        if not payload_line:
            return None
        return json.loads(payload_line)
    except Exception:
        return None
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
# Display-screen reader.  DBNet does not recognise seven-segment / dotted-matrix
# glyphs as text (measured: no box is produced even with det_db_thresh lowered to
# 0.10), so detection cannot locate an LCD reading.  The recogniser can read those
# glyphs perfectly well, so the upload path locates the display in pixels instead:
# dark ink forming a row whose local background is darker than the housing panel.
# Crops are then fed straight to the recogniser, skipping detection.
UPLOAD_DISPLAY_REGION_ENABLED = os.getenv(
    "RVCLAW_UPLOAD_DISPLAY_REGION", "1").strip().lower() not in {"0", "false", "no"}
UPLOAD_DISPLAY_MAX_REGIONS = int(os.getenv("RVCLAW_UPLOAD_DISPLAY_MAX_REGIONS", "3"))
# A region reading is trusted on its own only above this recogniser score; below
# it the path falls back to the whole-frame detector, which handles printed and
# mechanical counters.  Measured: a true LCD reading scored 0.886, while the
# spurious reading of a multimeter dial scored 0.488.
UPLOAD_DISPLAY_MIN_SCORE = float(os.getenv("RVCLAW_UPLOAD_DISPLAY_MIN_SCORE", "0.55"))
# Crop geometry matters more than expected: cropping mid-glyph or leaving too much
# vertical margin makes the recogniser merge or invent digits ("74590" -> "7459",
# -> "14594").  The three variants below must therefore agree before a reading is
# used.  Demanding perfect unanimity turned out to be too strict, though: on the
# JPEG the page's canvas produces, the tightest variant reads "74599" while the
# other two read "74590", and discarding that 2-of-3 majority hands the answer to
# the whole-frame detector, which then reports a printed nameplate number.  A
# strict majority is used instead -- the winner needs at least
# UPLOAD_DISPLAY_MIN_AGREEMENT votes *and* more than half of all numeric votes, so a
# genuinely contested crop is still refused.
UPLOAD_DISPLAY_CROP_VARIANTS = (
    (0.20, 0.05, 1.0),
    (0.20, 0.05, 1.5),
    (0.06, 0.06, 1.5),
)
# A dark-screen proposal often outlines the whole LCD, including the unit and
# status text below the large digits.  These crops retain only the upper value
# band.  They join the ordinary glyph-row crops in the vote instead of
# replacing them, so an unusual screen still fails closed when no view agrees.
UPLOAD_DISPLAY_SCREEN_CROP_VARIANTS = (
    (0.10, 0.04, 0.03, 0.30, 1.0),
    (0.17, 0.10, 0.07, 0.35, 1.0),
    (0.14, 0.08, 0.05, 0.32, 1.5),
    # A wider, higher-resolution value band keeps the leading 7 and trailing 0
    # of a perspective-skewed five-digit LCD. It is still only one vote; the
    # strict-majority gate below remains the authority.
    (0.08, 0.02, 0.02, 0.26, 2.0),
)
# The low-contrast reflective LCD proposal encloses part of the pale meter body
# as well as the screen. These three tighter crops isolate the value band. On
# the four PHONE-B field frames they independently agree on 74590; retaining a
# three-view majority keeps a single clipped leading digit from being accepted.
UPLOAD_REFLECTIVE_LCD_CROP_VARIANTS = (
    (0.08, 0.12, 0.30, 0.45, 1.5),
    (0.08, 0.12, 0.35, 0.45, 1.5),
    (0.10, 0.10, 0.35, 0.50, 1.5),
)
UPLOAD_DISPLAY_MIN_AGREEMENT = int(os.getenv("RVCLAW_UPLOAD_DISPLAY_MIN_AGREEMENT", "2"))
# DBNet sometimes detects most of a small seven-segment value but clips a thin
# final digit. Recognition-only retries around that detected row recover the
# missing edge without guessing a decimal point. Three nearby crops must reach
# a strict majority before the detected result may be refined.
UPLOAD_DETECTED_ROW_CROP_VARIANTS = (
    (0.25, 0.40, 0.30, 1.5),
    (0.25, 0.50, 0.30, 1.5),
    (0.25, 0.60, 0.30, 1.5),
)
UPLOAD_DETECTED_ROW_MIN_AGREEMENT = int(
    os.getenv("RVCLAW_UPLOAD_DETECTED_ROW_MIN_AGREEMENT", "2")
)
UPLOAD_DETECTED_FRAGMENT_MIN_SCORE = float(
    os.getenv("RVCLAW_UPLOAD_DETECTED_FRAGMENT_MIN_SCORE", "0.80")
)
UPLOAD_DETECTED_EDGE_RECOVERY_MIN_SCORE = float(
    os.getenv("RVCLAW_UPLOAD_DETECTED_EDGE_RECOVERY_MIN_SCORE", "0.95")
)
METER_TYPE = os.getenv("RVCLAW_METER_TYPE", "pointer")
METER_START_ANGLE = float(os.getenv("RVCLAW_METER_START_ANGLE", "225"))
METER_END_ANGLE = float(os.getenv("RVCLAW_METER_END_ANGLE", "315"))
METER_MIN = float(os.getenv("RVCLAW_METER_MIN", "0"))
METER_MAX = float(os.getenv("RVCLAW_METER_MAX", "1"))
METER_DIRECTION = os.getenv("RVCLAW_METER_DIRECTION", "counterclockwise")
METER_UNIT = os.getenv("RVCLAW_METER_UNIT", "") or None
METER_ROI = [float(value) for value in os.getenv("RVCLAW_METER_ROI", "0,0,1,1").split(",")]
if len(METER_ROI) != 4:
    raise ValueError("RVCLAW_METER_ROI must contain x,y,width,height")
DIGITAL_METER_ROI = [
    float(value)
    for value in os.getenv("RVCLAW_DIGITAL_METER_ROI", "0.10,0.15,0.80,0.70").split(",")
]
if len(DIGITAL_METER_ROI) != 4:
    raise ValueError("RVCLAW_DIGITAL_METER_ROI must contain x,y,width,height")
DIGITAL_DISPLAY_ROI = [
    float(value)
    for value in os.getenv("RVCLAW_DIGITAL_DISPLAY_ROI", "0.10,0.15,0.80,0.45").split(",")
]
if len(DIGITAL_DISPLAY_ROI) != 4:
    raise ValueError("RVCLAW_DIGITAL_DISPLAY_ROI must contain x,y,width,height")


def _nested_roi(parent, child):
    return {
        "x": parent[0] + child[0] * parent[2],
        "y": parent[1] + child[1] * parent[3],
        "width": child[2] * parent[2],
        "height": child[3] * parent[3],
    }


DIGITAL_DISPLAY_BBOX = _nested_roi(DIGITAL_METER_ROI, DIGITAL_DISPLAY_ROI)
DIGITAL_CONFIGURED_BBOX = {
    "x": DIGITAL_METER_REGION[0],
    "y": DIGITAL_METER_REGION[1],
    "width": DIGITAL_METER_REGION[2],
    "height": DIGITAL_METER_REGION[3],
}


def refresh_digital_meter_camera():
    """Rebuild the digital camera command after a web configuration change."""
    global DIGITAL_METER_CONFIG, DIGITAL_METER_ID, DIGITAL_METER_REGION
    global DIGITAL_METER_UNIT, DIGITAL_CONFIGURED_BBOX
    DIGITAL_METER_CONFIG = _load_digital_meter_config()
    DIGITAL_METER_ID = DIGITAL_METER_CONFIG["meter_id"]
    DIGITAL_METER_REGION = DIGITAL_METER_CONFIG["display_region"]
    DIGITAL_METER_UNIT = DIGITAL_METER_CONFIG["unit"]
    DIGITAL_CONFIGURED_BBOX = {
        "x": DIGITAL_METER_REGION[0],
        "y": DIGITAL_METER_REGION[1],
        "width": DIGITAL_METER_REGION[2],
        "height": DIGITAL_METER_REGION[3],
    }
    camera = CAMERA_MODES.get("astra_meter_digital") if "CAMERA_MODES" in globals() else None
    if camera is not None:
        command = _digital_meter_command()
        command_changed = command != camera.command
        camera.command = command
        camera.inference["display_roi"] = DIGITAL_METER_REGION
        camera.inference["meter_id"] = DIGITAL_METER_ID
        camera.inference["meter_name"] = DIGITAL_METER_CONFIG.get("name")
        camera.inference["unit"] = DIGITAL_METER_UNIT
        camera.inference["configured"] = DIGITAL_METER_CONFIG["configured"]
        controller = globals().get("CAMERA_CONTROLLER")
        if command_changed and controller is not None and controller.status().get("active_mode") == "astra_meter_digital" and camera.is_running():
            camera.stop()
            camera.start()

def _detector_command(
    device,
    width,
    height,
    fps,
    config,
    model,
    hardhat_only=False,
    ppe_only=False,
    ppe_config=None,
    ppe_model=None,
    ppe_fps=5,
    head_roi_ppe=False,
    meter_only=False,
    meter_type="pointer",
    meter_fps=5,
    meter_start_angle=225.0,
    meter_end_angle=315.0,
    meter_min=0.0,
    meter_max=1.0,
    meter_direction="counterclockwise",
    meter_roi=(0.0, 0.0, 1.0, 1.0),
    display_roi=(0.10, 0.15, 0.80, 0.45),
    meter_decimal_places=0,
    meter_digit_count=0,
    meter_min_confidence=0.58,
    meter_consistency_count=3,
    meter_normal_min=None,
    meter_normal_max=None,
    meter_max_change=None,
):
    command = [str(YOLOV8_BINARY)]
    if config and model:
        command.extend([
        "--config", str(config), "--model", str(model),
        ])
    command.extend([
        "--device",
        device,
        "--width",
        str(width),
        "--height",
        str(height),
        "--fps",
        str(fps),
    ])
    if meter_only:
        command.extend([
            "--meter-only", "--meter-type", meter_type,
            "--meter-fps", str(meter_fps),
            "--meter-start-angle", str(meter_start_angle),
            "--meter-end-angle", str(meter_end_angle),
            "--meter-min", str(meter_min),
            "--meter-max", str(meter_max),
            "--meter-direction", meter_direction,
            "--meter-roi-x", str(meter_roi[0]),
            "--meter-roi-y", str(meter_roi[1]),
            "--meter-roi-width", str(meter_roi[2]),
            "--meter-roi-height", str(meter_roi[3]),
            "--display-roi-x", str(display_roi[0]),
            "--display-roi-y", str(display_roi[1]),
            "--display-roi-width", str(display_roi[2]),
            "--display-roi-height", str(display_roi[3]),
            "--meter-decimal-places", str(meter_decimal_places),
            "--meter-digit-count", str(meter_digit_count),
            "--meter-min-confidence", str(meter_min_confidence),
            "--meter-consistency-count", str(meter_consistency_count),
        ])
        if meter_normal_min is not None:
            command.extend(["--meter-normal-min", str(meter_normal_min)])
        if meter_normal_max is not None:
            command.extend(["--meter-normal-max", str(meter_normal_max)])
        if meter_max_change is not None:
            command.extend(["--meter-max-change", str(meter_max_change)])
    if hardhat_only:
        command.append("--hardhat-only")
    if ppe_only:
        command.append("--ppe-only")
    if ppe_config is not None and ppe_model is not None:
        command.extend(
            [
                "--ppe-config",
                str(ppe_config),
                "--ppe-model",
                str(ppe_model),
                "--ppe-fps",
                str(ppe_fps),
            ]
        )
    if head_roi_ppe:
        command.append("--head-roi-ppe")
    return command


def _digital_meter_command():
    markers = DIGITAL_METER_CONFIG.get("alignment_markers") or [None, None]
    command = _detector_command(
        ASTRA_DEVICE, ASTRA_DIGITAL_WIDTH, ASTRA_DIGITAL_HEIGHT, ASTRA_DIGITAL_FPS,
        PPOCR_CONFIG, PPOCR_MODEL,
        meter_only=True, meter_type="digital", meter_fps=5,
        meter_roi=DIGITAL_METER_REGION,
        display_roi=(0.0, 0.0, 1.0, 1.0),
        meter_decimal_places=DIGITAL_METER_CONFIG["decimal_places"],
        meter_digit_count=DIGITAL_METER_CONFIG["digit_count"],
        meter_min_confidence=DIGITAL_METER_CONFIG["min_confidence"],
        meter_consistency_count=DIGITAL_METER_CONFIG["consistency_count"],
        meter_normal_min=DIGITAL_METER_CONFIG["normal_min"],
        meter_normal_max=DIGITAL_METER_CONFIG["normal_max"],
        meter_max_change=DIGITAL_METER_CONFIG["max_change"],
    )
    if DIGITAL_METER_CONFIG.get("reference_image"):
        command.extend(["--meter-reference-image", DIGITAL_METER_CONFIG["reference_image"]])
    command.extend(["--meter-trial-dir", str(METER_TRIAL_ROOT)])
    if len(markers) == 2 and all(isinstance(marker, dict) for marker in markers):
        command.extend([
            "--meter-marker-a-x", str(markers[0]["x"]),
            "--meter-marker-a-y", str(markers[0]["y"]),
            "--meter-marker-b-x", str(markers[1]["x"]),
            "--meter-marker-b-y", str(markers[1]["y"]),
        ])
    return command


def _read_meter_evidence() -> list:
    try:
        value = json.loads(METER_EVIDENCE_INDEX.read_text(encoding="utf-8"))
        return value if isinstance(value, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _save_meter_evidence(record: dict, image_bytes: bytes, crop_bytes: bytes | None = None) -> dict:
    """Persist one uploaded meter image and its inference result for review."""
    METER_EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    evidence_id = uuid.uuid4().hex
    image_name = f"{evidence_id}.jpg"
    (METER_EVIDENCE_ROOT / image_name).write_bytes(image_bytes)
    image_url = f"/api/meter-upload-records/{evidence_id}/image.jpg"
    crop_url = None
    if crop_bytes:
        (METER_EVIDENCE_ROOT / f"{evidence_id}-crop.jpg").write_bytes(crop_bytes)
        crop_url = f"/api/meter-upload-records/{evidence_id}/crop.jpg"
    record = {**record, "evidence_id": evidence_id,
              "image_url": image_url, "evidence_url": image_url, "crop_url": crop_url}
    records = [record, *_read_meter_evidence()]
    # Keep the page history bounded while retaining recent evidence.
    records = records[:100]
    METER_EVIDENCE_INDEX.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return record


# --- uploaded digital photos: locating the reading --------------------------

# A reading is digits only.  Units ("200m", "CAT 600V"), status words ("OFF",
# "HOLD") and model codes ("UT33B") are dropped here, so the selector can never
# report one of them as a value -- which is exactly what the previous
# whole-crop recogniser did when it answered "C" for a multimeter photo.
_NUMERIC_READING = re.compile(r"^[+-]?\d+(?:[.,]\d+)?$")


def _polygon_bbox(polygon):
    """Axis-aligned bounding box of a PP-OCR quadrilateral, in image pixels."""
    xs = []
    ys = []
    for point in polygon or []:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            try:
                xs.append(float(point[0]))
                ys.append(float(point[1]))
            except (TypeError, ValueError):
                continue
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _text_entries(items):
    """Normalise PP-OCR detections into entries carrying a pixel box.

    Both whole-frame consumers need the same thing from the detector: a text
    string, a score and a position.  The digital selector uses the box to show
    the operator what it read; the gauge reader uses the box midpoint to work
    out which angle each printed number sits at.  A detection without a usable
    polygon is dropped -- with no position it cannot be attributed to either a
    display or a dial.
    """
    entries = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        bbox = _polygon_bbox(item.get("poly"))
        if bbox is None:
            continue
        try:
            score = float(item.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        x0, y0, x1, y1 = bbox
        entries.append({"text": text, "score": score, "bbox": bbox,
                        "height": y1 - y0, "width": x1 - x0})
    return entries


_DISPLAY_REGION_CONFIG = {
    # Ink cutoff as a fraction of the panel's median grey level.  A screen is
    # darker than the housing but never near-black, which is what separates a
    # display from a shadow or a dark image border.
    "cutoff_ratios": (0.50, 0.58),
    "kernel_w_ratio": 0.010,
    "kernel_h_ratio": 0.003,
    "min_height_ratio": 0.012,
    "max_height_ratio": 0.35,
    "min_width_ratio": 0.02,
    "aspect_range": (1.2, 6.0),
    "min_fill": 0.12,
    "min_background_drop": 15.0,
    "min_background_ratio": 0.28,
    # A screen is markedly darker than the housing panel; text printed on the panel
    # is only slightly darker.  Measured: an LCD screen sat at 0.61 of the panel's
    # median grey, while a false "display" on a multimeter (printed dial text, not a
    # screen at all) sat at 0.80 -- and fed the recogniser a confident wrong value
    # ("8.8" on a meter reading 0.00).  The ratio below is the midpoint of those two
    # measurements, so it separates them without being tuned to either.
    "max_background_ratio": float(os.getenv("RVCLAW_UPLOAD_DISPLAY_MAX_BG_RATIO", "0.70")),
    "min_glyphs": 3,
    "glyph_min_height_ratio": 0.35,
    "padding_ratio": 0.18,
}


def _propose_emissive_display_regions(image):
    """Locate coloured LED/segment rows on a dark instrument window.

    The ordinary display locator searches for dark ink on a dim LCD.  A red,
    green or blue LED display is the inverse: saturated luminous segments sit
    on an almost black window, so treating the window itself as ink collapses
    the whole panel into one unusable component.  This colour-aware path finds
    the segment row first and still leaves recognition to PP-OCRv5.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    panel = float(np.median(gray))
    colour_masks = (
        cv2.bitwise_or(
            cv2.inRange(hsv, (0, 90, 80), (18, 255, 255)),
            cv2.inRange(hsv, (165, 90, 80), (179, 255, 255)),
        ),
        cv2.inRange(hsv, (30, 80, 70), (100, 255, 255)),
        cv2.inRange(hsv, (100, 80, 70), (145, 255, 255)),
    )
    proposals = []
    for raw_mask in colour_masks:
        # A narrow seven-segment "1" can sit noticeably farther from its
        # neighbour than the other digits.  The wider horizontal close keeps
        # the complete number in one row (95.31 used to be truncated to 95.3).
        kernel_w = max(9, int(width * 0.09)) | 1
        kernel_h = max(3, int(height * 0.006)) | 1
        joined = cv2.morphologyEx(
            raw_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h)),
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
        for index in range(1, count):
            x, y, row_width, row_height, area = (int(value) for value in stats[index])
            if row_width < width * 0.12 or row_height < height * 0.025:
                continue
            if row_height > height * 0.30:
                continue
            aspect = row_width / float(max(1, row_height))
            if not 1.5 <= aspect <= 10.0:
                continue
            patch = raw_mask[y:y + row_height, x:x + row_width]
            raw_count, _, raw_stats, _ = cv2.connectedComponentsWithStats(patch, 8)
            glyph_parts = sum(
                1
                for component in range(1, raw_count)
                if raw_stats[component, cv2.CC_STAT_AREA] >= 12
                and raw_stats[component, cv2.CC_STAT_HEIGHT] >= row_height * 0.18
            )
            if glyph_parts < 3:
                continue
            pad_x = max(6, int(row_height * 0.20))
            pad_y = max(3, int(row_height * 0.08))
            px0, py0 = max(0, x - pad_x), max(0, y - pad_y)
            px1, py1 = min(width, x + row_width + pad_x), min(height, y + row_height + pad_y)
            background = float(np.median(gray[py0:py1, px0:px1]))
            if background > min(120.0, panel * 0.75):
                continue
            proposals.append({
                "bbox": (px0, py0, px1, py1),
                "ink_bbox": (x, y, x + row_width, y + row_height),
                "line_height": row_height,
                "background": round(background, 1),
                "glyphs": glyph_parts,
                "source": "emissive",
            })
    return proposals


def _propose_large_dark_panel_regions(image):
    """Locate a large monochrome seven-segment instrument face.

    The field PHONE-A print has grey segments on a broad black fascia.  Neither
    DBNet nor the colour-emissive locator sees that as text, but the recognition
    model reads the value reliably once the upper value band is cropped.  This
    proposal is deliberately geometric and conservative: the dark component
    must be large, rectangular, dense, and surrounded by a substantially
    brighter ring.  Recognition still requires a strict multi-crop majority.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    proposals = []
    for threshold_rank, threshold in enumerate((70, 90, 110)):
        mask = np.where(gray < threshold, 255, 0).astype(np.uint8)
        kernel_w = max(9, int(width * 0.02)) | 1
        kernel_h = max(5, int(height * 0.008)) | 1
        joined = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h)),
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
        for index in range(1, count):
            x, y, panel_width, panel_height, area = (
                int(value) for value in stats[index]
            )
            if not width * 0.25 <= panel_width <= width * 0.92:
                continue
            if not height * 0.08 <= panel_height <= height * 0.62:
                continue
            aspect = panel_width / float(max(1, panel_height))
            if not 1.2 <= aspect <= 6.0:
                continue
            fill = area / float(max(1, panel_width * panel_height))
            if fill < 0.45:
                continue

            pad = max(8, int(min(panel_width, panel_height) * 0.10))
            ox0, oy0 = max(0, x - pad), max(0, y - pad)
            ox1 = min(width, x + panel_width + pad)
            oy1 = min(height, y + panel_height + pad)
            outer = gray[oy0:oy1, ox0:ox1]
            ring = np.ones(outer.shape, dtype=bool)
            ring[y - oy0:y - oy0 + panel_height,
                 x - ox0:x - ox0 + panel_width] = False
            if not np.any(ring):
                continue
            background = float(np.median(gray[y:y + panel_height, x:x + panel_width]))
            surround = float(np.median(outer[ring]))
            contrast = surround - background
            if background > 130.0 or contrast < 40.0:
                continue
            proposals.append(
                {
                    "bbox": (x, y, x + panel_width, y + panel_height),
                    "ink_bbox": (x, y, x + panel_width, y + panel_height),
                    "line_height": panel_height,
                    "background": round(background, 1),
                    "surround": round(surround, 1),
                    "contrast": round(contrast, 1),
                    "glyphs": 0,
                    "source": "large_dark_panel",
                    "threshold_rank": threshold_rank,
                }
            )
    return proposals


def _propose_reflective_lcd_regions(image):
    """Locate a low-contrast reflective LCD inside a pale instrument face.

    A reflective LCD does not satisfy the bright-ring contrast used by
    _propose_large_dark_panel_regions: in the PHONE-B field frames the LCD
    assembly is only 8--12 grey levels darker than its immediate surround. A
    thresholded component still gives a repeatable enclosure, though. Keep
    this proposal deliberately narrow (upper part of the frame, instrument-
    sized rectangle, bounded fill) and leave the final decision to the existing
    multi-crop OCR majority. That prevents a printed serial-number row from
    becoming a reading merely because it is dark.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    proposals = []
    # Threshold 80 gives the most stable enclosure across the field sequence;
    # 70 and 90 remain fallbacks for illumination changes.
    for threshold_rank, threshold in enumerate((80, 70, 90)):
        mask = np.where(gray < threshold, 255, 0).astype(np.uint8)
        kernel_w = max(9, int(width * 0.02)) | 1
        kernel_h = max(5, int(height * 0.008)) | 1
        joined = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h)),
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
        for index in range(1, count):
            x, y, panel_width, panel_height, area = (
                int(value) for value in stats[index]
            )
            if y >= height * 0.45:
                continue
            if not width * 0.18 <= panel_width <= width * 0.50:
                continue
            if not height * 0.10 <= panel_height <= height * 0.45:
                continue
            aspect = panel_width / float(max(1, panel_height))
            if not 1.2 <= aspect <= 2.5:
                continue
            fill = area / float(max(1, panel_width * panel_height))
            if not 0.32 <= fill <= 0.58:
                continue

            pad = max(8, int(min(panel_width, panel_height) * 0.10))
            ox0, oy0 = max(0, x - pad), max(0, y - pad)
            ox1 = min(width, x + panel_width + pad)
            oy1 = min(height, y + panel_height + pad)
            outer = gray[oy0:oy1, ox0:ox1]
            ring = np.ones(outer.shape, dtype=bool)
            ring[
                y - oy0:y - oy0 + panel_height,
                x - ox0:x - ox0 + panel_width,
            ] = False
            if not np.any(ring):
                continue
            background = float(
                np.median(gray[y:y + panel_height, x:x + panel_width])
            )
            surround = float(np.median(outer[ring]))
            contrast = surround - background
            if background > 130.0 or not -5.0 <= contrast <= 35.0:
                continue
            proposals.append(
                {
                    "bbox": (x, y, x + panel_width, y + panel_height),
                    "ink_bbox": (x, y, x + panel_width, y + panel_height),
                    "line_height": panel_height,
                    "background": round(background, 1),
                    "surround": round(surround, 1),
                    "contrast": round(contrast, 1),
                    "glyphs": 0,
                    "source": "reflective_lcd_panel",
                    "threshold_rank": threshold_rank,
                }
            )
    return proposals


def _propose_display_regions(image, limit=UPLOAD_DISPLAY_MAX_REGIONS):
    """Locate likely instrument displays in an uploaded photo.

    Returns a list, tallest row first, of dicts with ``bbox`` (what the page
    outlines), ``ink_bbox`` (the tight glyph row) and ``line_height``.

    This exists because DBNet cannot see seven-segment glyphs at all, so on a
    photo of an LCD the detector returns the printed labels around the screen and
    never the reading itself.
    """
    config = _DISPLAY_REGION_CONFIG
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    panel = float(np.median(gray))

    raw = _propose_emissive_display_regions(image)
    raw.extend(_propose_large_dark_panel_regions(image))
    raw.extend(_propose_reflective_lcd_regions(image))
    for ratio in config["cutoff_ratios"]:
        ink_raw = np.where(gray < panel * ratio, 255, 0).astype(np.uint8)
        kernel_w = max(9, int(width * config["kernel_w_ratio"])) | 1
        kernel_h = max(3, int(height * config["kernel_h_ratio"])) | 1
        # Closing groups the glyphs into rows.  Glyph counting must use the raw
        # ink though: closing bridges neighbouring digits, so a closed row is
        # always a single connected component.
        ink = cv2.morphologyEx(
            ink_raw, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h)))

        count, _, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
        for index in range(1, count):
            x, y, w, h, area = (int(value) for value in stats[index])
            if h < height * config["min_height_ratio"] or h > height * config["max_height_ratio"]:
                continue
            if w < width * config["min_width_ratio"]:
                continue
            if area / float(w * h) < config["min_fill"]:
                continue

            for band_y0, band_y1 in _ink_row_bands(ink, (x, y, w, h)):
                band_h = band_y1 - band_y0
                if band_h < height * config["min_height_ratio"]:
                    continue
                if band_h > height * config["max_height_ratio"]:
                    continue
                tightened = _tighten_columns(ink, x, x + w, band_y0, band_y1)
                if tightened is None:
                    continue
                band_x0, band_x1 = tightened
                band_w = band_x1 - band_x0
                aspect = band_w / float(band_h)
                if not (config["aspect_range"][0] <= aspect <= config["aspect_range"][1]):
                    continue
                fill = ink_raw[band_y0:band_y1, band_x0:band_x1].sum() / float(255 * band_w * band_h)
                if fill < config["min_fill"]:
                    continue

                pad = max(4, int(band_h * config["padding_ratio"]))
                px0, py0 = max(0, band_x0 - pad), max(0, band_y0 - pad)
                px1, py1 = min(width, band_x1 + pad), min(height, band_y1 + pad)
                background = float(np.median(gray[py0:py1, px0:px1]))
                if panel - background < config["min_background_drop"]:
                    continue
                if background < panel * config["min_background_ratio"]:
                    continue
                # Reject "regions" that are only marginally darker than the panel:
                # those are printed text or texture on the housing, not a screen.
                if background > panel * config["max_background_ratio"]:
                    continue

                glyphs = _count_glyphs(ink_raw, band_x0, band_y0, band_x1, band_y1, band_h)
                if glyphs < config["min_glyphs"]:
                    continue

                raw.append({
                    "bbox": (px0, py0, px1, py1),
                    "ink_bbox": (band_x0, band_y0, band_x1, band_y1),
                    "line_height": band_h,
                    "background": round(background, 1),
                    "glyphs": glyphs,
                    "source": "dark_screen",
                })

    source_priority = {
        "emissive": 0,
        "large_dark_panel": 1,
        "reflective_lcd_panel": 2,
        "dark_screen": 3,
    }
    raw.sort(key=lambda item: (
        source_priority.get(item.get("source"), 3),
        item.get("threshold_rank", 0),
        -item["line_height"],
    ))
    kept = []
    for item in raw:
        if any(_box_iou(item["bbox"], other["bbox"]) > 0.4 for other in kept):
            continue
        kept.append(item)
        if len(kept) >= limit:
            break
    return kept


def _ink_row_bands(ink, box):
    """Split a component into text rows via its horizontal projection."""
    x, y, w, h = box
    patch = ink[y:y + h, x:x + w]
    profile = patch.sum(axis=1) / float(255 * max(1, w))
    bands = []
    start = None
    for index, value in enumerate(profile):
        if value >= 0.03 and start is None:
            start = index
        elif value < 0.03 and start is not None:
            if index - start >= 2:
                bands.append((y + start, y + index))
            start = None
    if start is not None:
        bands.append((y + start, y + h))
    return bands


def _tighten_columns(ink, x0, x1, y0, y1):
    """Shrink a row to the columns that actually carry ink."""
    columns = ink[y0:y1, x0:x1].sum(axis=0)
    indexes = np.nonzero(columns)[0]
    if len(indexes) == 0:
        return None
    return x0 + int(indexes[0]), x0 + int(indexes[-1]) + 1


def _count_glyphs(ink, x0, y0, x1, y1, line_height):
    """Count glyph-sized ink blobs in a row; a reading is several of them."""
    patch = ink[y0:y1, x0:x1]
    count, _, stats, _ = cv2.connectedComponentsWithStats(patch, 8)
    glyphs = 0
    for index in range(1, count):
        _, _, gw, gh, area = (int(value) for value in stats[index])
        if gh < line_height * _DISPLAY_REGION_CONFIG["glyph_min_height_ratio"]:
            continue
        if gw < 3 or area < 12:
            continue
        glyphs += 1
    return glyphs


def _box_iou(first, second):
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = float((ix1 - ix0) * (iy1 - iy0))
    union = float((ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0)) - inter
    return inter / union if union > 0 else 0.0


def _recognise_crop(image, timeout=20):
    """Run PP-OCRv5 recognition over an in-memory crop.

    Uses the recognition-only profile: the caller has already located the
    display, and detection is exactly what fails on seven-segment glyphs.
    Returns ``{"text": ..., "score": ...}`` or ``None``.

    With the in-process engine a run costs ~0.15 s instead of the ~0.9 s the
    native helper needed, almost all of which was loading the model again in a
    fresh process.  The timeout stays as a stall guard for the legacy path.
    Refusing a variant is safe (the agreement gate then refuses the reading);
    stalling the page is not.
    """
    # PP-OCR's recogniser is substantially more stable on instrument digits
    # after colour is removed: the supplied red LED reads 95.31 instead of
    # punctuation, and the Aclara LCD's last zero no longer becomes a nine.
    if image is not None and image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    parsed = _ppocr_payload(image, PPOCR_CONFIG, timeout)
    if not parsed:
        return None
    try:
        items = [item for item in (parsed.get("items") or [])
                 if isinstance(item, dict) and str(item.get("text", "")).strip()]
        if not items:
            return None
        text = "".join(str(item.get("text", "")).strip() for item in items)
        score = sum(float(item.get("score", 0.0) or 0.0) for item in items) / len(items)
        return {"text": text, "score": score}
    except Exception:
        return None


def _recognise_crops(crops):
    """Recognise several crops, one subprocess each, all at once.

    The recogniser binary is started afresh per crop, and on this board a run
    costs a nearly fixed ~0.85 s no matter how small the input is -- measured, a
    75x44 crop takes as long as a 300x176 one, so almost all of it is ONNX model
    loading rather than arithmetic.  Reading the crop variants in sequence
    therefore cost ~2.8 s per photo.  Running them together brings the wall clock
    down to ~1.2 s while leaving the variants -- and so the agreement gate --
    exactly as they were.
    """
    if not crops:
        return []
    if len(crops) == 1:
        return [_recognise_crop(crops[0])]
    with ThreadPoolExecutor(max_workers=len(crops)) as pool:
        return list(pool.map(_recognise_crop, crops))


def _display_crop_box(image, region, pad_x_ratio, pad_y_ratio):
    """Where a padded display crop lands in the original image."""
    height, width = image.shape[:2]
    x0, y0, x1, y1 = region["ink_bbox"]
    pad_x = int(region["line_height"] * pad_x_ratio)
    pad_y = int(region["line_height"] * pad_y_ratio)
    return (max(0, x0 - pad_x), max(0, y0 - pad_y),
            min(width, x1 + pad_x), min(height, y1 + pad_y))


def _crop_display_region(image, region, pad_x_ratio, pad_y_ratio, scale):
    """Crop a proposed display region and optionally upscale it."""
    cx0, cy0, cx1, cy1 = _display_crop_box(image, region, pad_x_ratio, pad_y_ratio)
    if cx1 - cx0 < 8 or cy1 - cy0 < 8:
        return None
    crop = image[cy0:cy1, cx0:cx1]
    if scale != 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop


def _crop_display_screen_value(image, region, left, top, right, bottom, scale):
    """Crop the large-value band from a proposed whole display window."""
    x0, y0, x1, y1 = region["bbox"]
    width = x1 - x0
    height = y1 - y0
    cx0 = int(round(x0 + width * left))
    cy0 = int(round(y0 + height * top))
    cx1 = int(round(x1 - width * right))
    cy1 = int(round(y1 - height * bottom))
    if cx1 - cx0 < 8 or cy1 - cy0 < 8:
        return None
    crop = image[cy0:cy1, cx0:cx1]
    if scale != 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop


def _recover_contested_display(image, region, tally):
    """Verify a unique high-confidence value with nearby crop geometries.

    These are same-frame checks, not independent camera observations. The route
    still requires separate frames before classifying a meter.
    """
    seeds = [(text, item) for text, item in tally.items()
             if item["score"] >= 0.95 and len(item["crop"] or ()) == 5]
    if len(seeds) != 1:
        return None
    text, seed = seeds[0]
    left, top, right, bottom, scale = seed["crop"]
    specs = [(left, top + .02, right, bottom, scale),
             (left, top, max(0., right - .02), bottom, scale),
             (left, top, right + .02, bottom, scale)]
    crops = [_crop_display_screen_value(image, region, *spec) for spec in specs]
    if any(crop is None for crop in crops):
        return None
    checks = _recognise_crops(crops)
    if len(checks) != 3 or any(
        not item or item["score"] < .95
        or not _NUMERIC_READING.match(item["text"])
        or item["text"].replace(",", ".") != text for item in checks
    ):
        return None
    return text, {"votes": 3, "score": min(item["score"] for item in checks),
                  "crop": seed["crop"], "recovery": "verified_nearby_crop_geometry"}


def _read_display_region(image):
    """Read the value off a display screen in the photo.

    Returns ``(result, bbox, diagnostics)`` when a display reading clears the score
    gate *and* every crop variant that cleared it agrees.  Otherwise the result is
    ``None`` and the caller falls back to whole-frame detection; ``diagnostics``
    records what was proposed and why it was refused, so a refusal is explicable
    rather than silent.  Rejecting a boundary-sensitive reading is deliberate: it
    is what keeps a wrong value from being reported with a confident score.
    """
    diagnostics = {"regions": 0, "refused": None, "recognised": False}
    for region in _propose_display_regions(image):
        diagnostics["regions"] += 1
        variants = []
        if region.get("source") not in {
            "large_dark_panel",
            "reflective_lcd_panel",
        }:
            for variant in UPLOAD_DISPLAY_CROP_VARIANTS:
                crop = _crop_display_region(image, region, *variant)
                if crop is not None:
                    variants.append((variant, crop))
        # Both proposal paths may outline a whole LCD. The live Aclara image is
        # classified as ``emissive`` even though it is a reflective LCD; limiting
        # value-band crops to ``dark_screen`` clipped its leading/trailing digits.
        if region.get("source") == "reflective_lcd_panel":
            for variant in UPLOAD_REFLECTIVE_LCD_CROP_VARIANTS:
                crop = _crop_display_screen_value(image, region, *variant)
                if crop is not None:
                    variants.append((variant, crop))
        elif region.get("source") in {
            "dark_screen",
            "emissive",
            "large_dark_panel",
        }:
            for variant in UPLOAD_DISPLAY_SCREEN_CROP_VARIANTS:
                crop = _crop_display_screen_value(image, region, *variant)
                if crop is not None:
                    variants.append((variant, crop))
        if not variants:
            continue
        # Every variant is an independent read of the same region, so they are read
        # together rather than one after another -- see _recognise_crops.
        found_all = _recognise_crops([crop for _, crop in variants])
        tally = {}
        for (variant, _crop), found in zip(variants, found_all):
            if not found:
                continue
            if found["score"] < UPLOAD_DISPLAY_MIN_SCORE:
                continue
            if not _NUMERIC_READING.match(found["text"]):
                continue
            value = found["text"].replace(",", ".")
            # Digits were read off a located screen.  Whether or not the crops end
            # up agreeing, this tells the caller that a display was really there.
            diagnostics["recognised"] = True
            entry = tally.setdefault(value, {"votes": 0, "score": 0.0, "crop": None})
            entry["votes"] += 1
            if found["score"] > entry["score"]:
                entry["score"] = found["score"]
                entry["crop"] = tuple(variant)

        if not tally:
            continue
        # Record what this region read and where, so that a refusal downstream can
        # name the competing values instead of being a bare "no reading".
        diagnostics["bbox"] = [round(float(value), 1) for value in region["bbox"]]
        diagnostics["tally"] = [
            {"text": text, "votes": item["votes"], "score": round(item["score"], 4),
             "line_height": round(float(region["line_height"]), 1)}
            for text, item in sorted(tally.items(), key=lambda pair: -pair[1]["votes"])
        ]
        # A plurality such as 2/1/1 is not a majority. Refuse it unless a unique
        # high-confidence seed also survives three distinct nearby crop cuts.
        value, best = max(tally.items(), key=lambda item: (item[1]["votes"], item[1]["score"]))
        total_votes = sum(item["votes"] for item in tally.values())
        if best["votes"] < UPLOAD_DISPLAY_MIN_AGREEMENT or best["votes"] * 2 <= total_votes:
            recovered = _recover_contested_display(image, region, tally) if len(tally) > 1 else None
            if recovered is None:
                diagnostics["refused"] = "no strict crop majority: " + ", ".join(
                    "%s x%d" % (text, item["votes"]) for text, item in sorted(tally.items()))
                continue
            value, best = recovered
            diagnostics["recovery"] = best["recovery"]
        return {
            "status": "stable",
            "value": value,
            "raw_text": value,
            "unit": DIGITAL_METER_UNIT,
            "confidence": round(best["score"], 4),
            "value_method": "paddleocr_display_region",
            "model": "PaddleOCR PP-OCRv5 mobile_rec (display region)",
            "display_region": {"bbox": [round(float(v), 1) for v in region["bbox"]],
                               "line_height": region["line_height"],
                               "glyphs": region["glyphs"],
                               "background": region["background"],
                               "crop_variant": list(best["crop"]),
                               # The padded box the winning variant actually read, so
                               # the page's zoomed evidence shows the same pixels the
                               # value came from rather than a tight box that clips
                               # the last digit.
                               "crop_bbox": [round(float(v), 1) for v in (
                                   _display_crop_box(image, region, best["crop"][0], best["crop"][1])
                                   if len(best["crop"]) == 3
                                   else (
                                       region["bbox"][0] + (region["bbox"][2] - region["bbox"][0]) * best["crop"][0],
                                       region["bbox"][1] + (region["bbox"][3] - region["bbox"][1]) * best["crop"][1],
                                       region["bbox"][2] - (region["bbox"][2] - region["bbox"][0]) * best["crop"][2],
                                       region["bbox"][3] - (region["bbox"][3] - region["bbox"][1]) * best["crop"][3],
                                   )
                               )],
                               "agreeing_variants": best["votes"],
                               "recovery": best.get("recovery")},
            "candidates": [{
                "text": value,
                "score": round(best["score"], 4),
                "height": round(float(region["line_height"]), 1),
                "bbox": [round(float(v), 1) for v in region["bbox"]],
                "source": "display-region",
            }],
            "text_items": [],
            "reason": "",
        }, region["bbox"], diagnostics
    return None, None, diagnostics


def _read_full_frame(image_bytes):
    """Whole-frame DBNet detection + recognition, then pick the reading.

    This is the path that works for printed and mechanical counters; it cannot
    see an LCD (see ``_read_display_region``).
    """
    upload_config = PPOCR_UPLOAD_CONFIG if PPOCR_UPLOAD_CONFIG.is_file() else PPOCR_CONFIG
    parsed = _ppocr_payload(image_bytes, upload_config, 90)
    if not parsed:
        raise RuntimeError("PaddleOCR 推理失败")
    result, bbox = _select_digital_reading(parsed)
    # A low-confidence fragment can still locate a row. It is never a reading:
    # recovery below requires unanimous enlarged crops and compatible digits.
    if result.get("status") == "unreadable":
        fragments = [item for item in _text_entries(parsed.get("items"))
                     if _NUMERIC_READING.fullmatch(item["text"])
                     and 0.25 <= item["score"] < UPLOAD_TEXT_MIN_SCORE]
        if len(fragments) == 1:
            fragment = fragments[0]
            if len(_numeric_signature(fragment["text"])[1]) <= 3:
                result["detected_fragment"] = {
                    "text": fragment["text"], "score": fragment["score"],
                    "bbox": list(fragment["bbox"]),
                }
                bbox = fragment["bbox"]
    return result, bbox


def _select_digital_reading(payload):
    """Choose the most likely reading among the PP-OCR detections.

    Returns ``(result, bbox)`` where ``bbox`` is the box the page highlights.
    The selector never invents a value: when several plausible numbers compete,
    or none clears the score gate, it says so instead of picking one, so a wrong
    reading cannot masquerade as a successful one.
    """
    items = payload.get("items") if isinstance(payload, dict) else None
    entries = _text_entries(items)

    base = {
        "unit": DIGITAL_METER_UNIT,
        "value_method": "paddleocr_det_rec",
        "model": "PaddleOCR PP-OCRv5 det+rec",
        "text_items": [{"text": entry["text"], "score": round(entry["score"], 6)}
                       for entry in entries],
    }
    digits = [entry for entry in entries
              if entry["score"] >= UPLOAD_TEXT_MIN_SCORE
              and _NUMERIC_READING.match(entry["text"])]

    if not digits:
        return {**base, "status": "unreadable", "value": None, "raw_text": None,
                "confidence": 0.0, "candidates": [],
                "reason": "未检测到可信的数字读数，请让显示区清晰并占据画面主要部分"}, None

    # The display normally uses the largest glyphs on the panel, so glyph height
    # discriminates better than the recogniser score: scale marks score high but
    # are small, while the reading is large and occasionally scores lower.
    digits.sort(key=lambda entry: (-entry["height"], -entry["score"]))
    top = digits[0]
    candidates = [{
        "text": entry["text"].replace(",", "."),
        "score": round(entry["score"], 4),
        "height": round(entry["height"], 1),
        "bbox": [round(value, 1) for value in entry["bbox"]],
    } for entry in digits[:UPLOAD_CANDIDATE_LIMIT]]

    if len(digits) > UPLOAD_MAX_DIGIT_BOXES:
        # Panel-style instruments (a multimeter dial, a calibrated scale) carry
        # many valid numbers at once.  Refusing beats guessing which one the
        # operator meant.
        return {**base, "status": "ambiguous", "value": None, "raw_text": None,
                # No reading is reported, so no confidence may be implied.  The
                # per-candidate scores stay in "candidates" for inspection.
                "confidence": 0.0, "candidates": candidates,
                "reason": (f"画面中检测到 {len(digits)} 处数字，无法确定唯一读数，"
                           f"请对准仪表显示区重新拍照")}, top["bbox"]

    value = top["text"].replace(",", ".")
    return {**base, "status": "stable", "value": value, "raw_text": value,
            "confidence": round(top["score"], 4), "candidates": candidates,
            "reason": ""}, top["bbox"]


def _expanded_detected_row_crop(
    image, bbox, left_pad_ratio, right_pad_ratio, vertical_pad_ratio, scale
):
    """Expand a DBNet number row before recognition-only retry."""

    height, width = image.shape[:2]
    x0, y0, x1, y1 = (int(round(value)) for value in bbox)
    row_height = max(1, y1 - y0)
    cx0 = max(0, x0 - int(row_height * left_pad_ratio))
    cy0 = max(0, y0 - int(row_height * vertical_pad_ratio))
    cx1 = min(width, x1 + int(row_height * right_pad_ratio))
    cy1 = min(height, y1 + int(row_height * vertical_pad_ratio))
    if cx1 - cx0 < 8 or cy1 - cy0 < 8:
        return None, None
    crop = image[cy0:cy1, cx0:cx1]
    if scale != 1.0:
        crop = cv2.resize(
            crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
    return crop, (cx0, cy0, cx1, cy1)


def _numeric_signature(text):
    value = str(text or "").strip().replace(",", ".")
    sign = "-" if value.startswith("-") else "+"
    return sign, "".join(character for character in value if character.isdigit())


def _matching_suffix_length(first, second):
    matched = 0
    for left, right in zip(reversed(first), reversed(second)):
        if left != right:
            break
        matched += 1
    return matched


def _normalise_numeric_crop_text(text):
    text = str(text or "").strip().replace(",", ".")
    # The row recognizer can separate a narrow final decimal digit with a
    # space. Preserve the observed decimal point and refuse other separated
    # numbers; never join multiple counters or insert missing punctuation.
    separated = re.fullmatch(r"([+-]?\d+\.\d+)\s+(\d)", text)
    return "".join(separated.groups()) if separated else text


def _refine_detected_numeric_reading(image, result, bbox):
    """Verify and safely repair a tight whole-frame OCR detection.

    A narrow trailing digit can fall just outside DBNet's box at live-camera
    resolution. Multiple expanded crops must agree. A tight box may retain
    only an internal fragment (for example ``53`` from ``95.31``), so a
    unanimous, high-confidence expanded result may repair that fragment. A
    conflicting full-length value still fails closed instead of becoming a
    false successful inspection.
    """

    fragment_hint = result.get("detected_fragment") or {}
    if (result.get("status") != "stable" and not fragment_hint) or bbox is None:
        return result, bbox
    variants = []
    for variant in UPLOAD_DETECTED_ROW_CROP_VARIANTS:
        crop, crop_bbox = _expanded_detected_row_crop(image, bbox, *variant)
        if crop is not None:
            variants.append((variant, crop_bbox, crop))
    diagnostics = {
        "attempted": bool(variants),
        "base_text": str(result.get("raw_text") or fragment_hint.get("text") or ""),
        "tally": [],
        "accepted": False,
    }
    if not variants:
        refined = dict(result)
        refined["detected_region_scan"] = diagnostics
        return refined, bbox

    tally = {}
    recognised = _recognise_crops([item[2] for item in variants])
    for (variant, crop_bbox, _crop), found in zip(variants, recognised):
        if not found:
            continue
        score = float(found.get("score", 0.0) or 0.0)
        if score < UPLOAD_DISPLAY_MIN_SCORE:
            continue
        text = _normalise_numeric_crop_text(found.get("text", ""))
        if _NUMERIC_READING.fullmatch(text) is None:
            continue
        entry = tally.setdefault(
            text, {"votes": 0, "score": 0.0, "variant": None, "bbox": None}
        )
        entry["votes"] += 1
        if score > entry["score"]:
            entry["score"] = score
            entry["variant"] = tuple(variant)
            entry["bbox"] = tuple(crop_bbox)

    diagnostics["tally"] = [
        {"text": text, "votes": item["votes"], "score": round(item["score"], 4)}
        for text, item in sorted(
            tally.items(), key=lambda pair: (-pair[1]["votes"], -pair[1]["score"])
        )
    ]
    if not tally:
        refined = dict(result)
        refined["detected_region_scan"] = diagnostics
        return refined, bbox

    recovered_text, recovered = max(
        tally.items(), key=lambda item: (item[1]["votes"], item[1]["score"])
    )
    rivals = max(
        (item["votes"] for text, item in tally.items() if text != recovered_text),
        default=0,
    )
    if (
        recovered["votes"] < UPLOAD_DETECTED_ROW_MIN_AGREEMENT
        or recovered["votes"] <= rivals
    ):
        diagnostics["refused"] = "expanded crop variants did not reach a strict majority"
        refined = dict(result)
        refined["detected_region_scan"] = diagnostics
        return refined, bbox

    base_sign, base_digits = _numeric_signature(diagnostics["base_text"])
    recovered_sign, recovered_digits = _numeric_signature(recovered_text)
    direct_recovery = (
        base_sign == recovered_sign
        and bool(base_digits)
        and (
            recovered_digits == base_digits
            or (
                recovered_digits.startswith(base_digits)
                and len(recovered_digits) == len(base_digits) + 1
            )
        )
    )
    fragment_recovery = (
        base_sign == recovered_sign
        and bool(base_digits)
        and len(base_digits) < len(recovered_digits)
        and base_digits in recovered_digits
        and recovered["votes"] == len(variants)
        and recovered["score"] >= UPLOAD_DETECTED_FRAGMENT_MIN_SCORE
    )
    # A tight DBNet box may clip the left edge of a small LCD row.  In the
    # 2026-09-22 PHONE-B field frame the tight box read 19590 while all three
    # expanded crops independently read 74590 at 0.9866 confidence.  Permit
    # correction only when every expanded crop agrees, confidence is very high,
    # lengths match, and disagreement is confined to at most the first two
    # digits.  Other full-value conflicts remain fail-closed.
    shared_suffix = _matching_suffix_length(base_digits, recovered_digits)
    leading_edge_recovery = (
        base_sign == recovered_sign
        and len(base_digits) == len(recovered_digits)
        and len(recovered_digits) >= 4
        and shared_suffix >= len(recovered_digits) - 2
        and recovered["votes"] == len(variants)
        and recovered["score"] >= UPLOAD_DETECTED_EDGE_RECOVERY_MIN_SCORE
    )
    # A weak partial row can have both a mistaken first segment and a clipped
    # trailing digit. Require a high-confidence unanimous initial result,
    # overlap across the remaining prefix, and a second geometry check below.
    two_edge_fragment = (
        base_sign == recovered_sign
        and len(base_digits) >= 3
        and len(recovered_digits) == len(base_digits) + 1
        and _matching_suffix_length(base_digits, recovered_digits[:-1]) >= len(base_digits) - 1
        and float(result.get("confidence", 0.)) < .65
        and recovered["votes"] == len(variants)
        and recovered["score"] >= UPLOAD_DETECTED_EDGE_RECOVERY_MIN_SCORE
    )
    compatible = direct_recovery or fragment_recovery or leading_edge_recovery
    if fragment_hint:
        # A locating fragment below the original score gate cannot be accepted
        # by direct agreement with itself.
        compatible = fragment_recovery
    if not compatible and (two_edge_fragment or (
            base_sign == recovered_sign and base_digits
            and len(base_digits) < len(recovered_digits)
            and base_digits in recovered_digits)):
        # DBNet crops can clip the top/bottom of slanted seven-segment rows.
        # Verify three wider vertical margins before repairing a fragment;
        # preserve the existing unanimous-vote and confidence requirements.
        margin_variants = [(left, right, max(vertical, .40), scale)
                           for left, right, vertical, scale in UPLOAD_DETECTED_ROW_CROP_VARIANTS]
        margin_crops = [_expanded_detected_row_crop(image, bbox, *v)
                        for v in margin_variants]
        verified = _recognise_crops([crop for crop, box in margin_crops
                                    if crop is not None])
        diagnostics["vertical_margin_verification"] = [
            {"text": (item or {}).get("text"), "score": (item or {}).get("score", 0.)}
            for item in verified]
        if (len(verified) == len(margin_variants)
                and all(item and _normalise_numeric_crop_text(item.get("text", "")) == recovered_text
                        and float(item.get("score", 0.)) >= UPLOAD_DETECTED_FRAGMENT_MIN_SCORE
                        for item in verified)):
            compatible = fragment_recovery = True
            recovered = dict(recovered, votes=len(verified),
                             score=min(float(item["score"]) for item in verified),
                             variant=margin_variants[0], bbox=margin_crops[0][1])
    if not compatible:
        diagnostics["refused"] = "detected row and expanded crops disagree"
        ambiguous = dict(result)
        ambiguous.update(
            {
                "status": "ambiguous",
                "value": None,
                "raw_text": None,
                "confidence": 0.0,
                "reason": "检测框读数与扩大裁剪复核结果不一致，拒绝给出读数",
                "detected_region_scan": diagnostics,
            }
        )
        return ambiguous, None

    diagnostics.update(
        {
            "accepted": True,
            "text": recovered_text,
            "votes": recovered["votes"],
            "crop_variant": list(recovered["variant"]),
            "crop_bbox": [round(float(value), 1) for value in recovered["bbox"]],
            "recovery_mode": (
                "direct" if direct_recovery
                else "two_edge_fragment" if two_edge_fragment
                else "fragment" if fragment_recovery
                else "leading_edge"
            ),
        }
    )
    refined = dict(result)
    refined.update(
        {
            "status": "stable",
            "value": recovered_text,
            "raw_text": recovered_text,
            "reason": "",
            "confidence": round(recovered["score"], 4),
            "value_method": "paddleocr_detected_region_consensus",
            "model": "PaddleOCR PP-OCRv5 mobile_rec (expanded detected row)",
            "detected_region_scan": diagnostics,
        }
    )
    return refined, recovered["bbox"]


# --- uploaded pointer photos: the ETH Zurich gauge reader -------------------

# One reader, built on first use.  Upstream's own `pipeline.py` builds its four
# ONNX sessions the same way, and they are expensive to construct, so the object
# is shared across uploads exactly like the PP-OCRv5 engine above.
_GAUGE_READER = None
_GAUGE_READER_LOCK = threading.Lock()


def _gauge_reader():
    """The shared ETH Zurich analogue-gauge reader, or ``None`` if unusable.

    Construction is cheap -- the reader opens its ONNX sessions lazily on first
    inference -- so the only failure mode here is a missing package, which is
    recorded in ``GAUGE_IMPORT_ERROR`` so the upload can explain itself.
    """
    global _GAUGE_READER, GAUGE_IMPORT_ERROR
    if EthzGaugeReader is None:
        return None
    if _GAUGE_READER is not None:
        return _GAUGE_READER
    with _GAUGE_READER_LOCK:
        if _GAUGE_READER is None:
            try:
                _GAUGE_READER = EthzGaugeReader()
            except Exception as error:  # noqa: BLE001 - reported, not raised
                GAUGE_IMPORT_ERROR = "%s: %s" % (type(error).__name__, error)
                return None
    return _GAUGE_READER


# The reader reports its refusals in upstream's English.  They are shown to the
# operator verbatim, and the digital card already answers in Chinese, so the
# common ones are translated here.  An unmapped reason passes through unchanged
# rather than being hidden: a failure mode nobody has seen yet should still be
# readable in the evidence record.
_GAUGE_REASON_ZH = (
    (r"^empty image$", "图片为空"),
    (r"^no gauge face detected$", "没有在照片里找到仪表表盘，请把表盘拍满画面"),
    (r"^only (\d+) key points", "表盘刻度弧上只找到 %s 个特征点，刻度不清或拍摄角度太斜"),
    (r"^ellipse fit failed$", "表盘轮廓拟合失败"),
    (r"^no needle found$", "没有找到指针"),
    (r"^needle mask only (\d+) pixels$", "指针像素太少（%s），指针太细或对比度太低"),
    (r"^needle line does not meet the dial arc$", "指针方向与表盘弧线没有交点"),
    (r"^ocr_unavailable$", "OCR 不可用，无法识别表盘上的数字"),
    (r"^only (\d+) dial number", "表盘数字只识别出 %s 个（至少需要 2 个）"),
    (r"^angle/value fit failed: (.*)$", "角度与读数的拟合失败：%s"),
    (r"^fit collapsed \(inliers=(\d+)\)$", "角度与读数的拟合崩塌（内点 %s 个）"),
)


def _gauge_reason_zh(reason):
    """Translate a reader refusal for the operator; unknown text is kept."""
    text = reason or ""
    for pattern, template in _GAUGE_REASON_ZH:
        matched = re.match(pattern, text)
        if matched:
            groups = matched.groups()
            return template % groups if groups else template
    return text


def _gauge_refusal(reason, status="unsupported", **extra):
    """The upload result for a pointer photo that could not be read at all.

    Every key the success path sets is set here too -- with ``None`` where there
    is nothing to report -- so a stored evidence record always has the same
    shape and the page never has to special-case a refusal.  ``status`` is
    ``unsupported`` when a capability is missing (the models or the package) and
    ``error`` when something in the stack actually threw, so the two are not
    confused during triage.
    """
    return {
        "kind": "pointer", "status": status, "value": None,
        "raw_text": None, "unit": METER_UNIT, "confidence": 0.0,
        "angle_deg": None, "dial": None, "scale": None, "candidates": [],
        "gauge_box": None, "value_method": "unsupported", "model": "none",
        "reason": reason, "notes": {}, **extra,
    }


def _read_pointer_gauge(image, calibration=None):
    """Read one uploaded pointer-gauge photo with the ETH Zurich pipeline.

    The upstream `analog_gauge_reader` cannot run on this board -- it needs
    torch 2.0 plus MMCV/MMOCR -- so its three models run here as ONNX on
    onnxruntime's CPU provider and its geometry is re-implemented in numpy; see
    ``rvclaw_algo/ethz_gauge_reader.py`` for the port and for the few places it
    deliberately departs from upstream.  Dial numbers come from the same
    PP-OCRv5 engine the digital path uses, which replaces upstream's MMOCR.

    Returns the page's result schema directly.  ``angle_deg`` is the needle's
    polar angle on the dial (the page prints it next to the value), and the
    reader's ``notes`` travel with the evidence so a disputed reading can be
    traced back through detection, ellipse fit and angle->value fit.
    """
    reader = _gauge_reader()
    if reader is None:
        return _gauge_refusal(
            "指针表读数不可用：%s" % (GAUGE_IMPORT_ERROR or "ETH Zurich reader 未安装"))
    if not reader.models_present():
        return _gauge_refusal(
            "指针表模型缺失（%s）：请先部署 analog_gauge_reader 的 3 个 ONNX 模型"
            % reader.model_dir)

    # PP-OCRv5 reads the dial numbers.  `_ppocr_payload` is used rather than the
    # engine directly so that a board still on the standalone binary keeps
    # working; a failed OCR call yields no readings, which the reader reports as
    # an unreadable dial rather than as a wrong value.
    ocr_failures = []

    def ocr(frame):
        payload = _ppocr_payload(frame, PPOCR_UPLOAD_CONFIG, 90)
        if not payload:
            ocr_failures.append("PP-OCRv5 无返回")
            return []
        return payload.get("items") or []

    try:
        reading = reader.read(image, ocr=ocr, calibration=calibration)
    except Exception as error:  # noqa: BLE001 - a reader crash is a refusal
        return _gauge_refusal("指针表读取失败：%s: %s" % (type(error).__name__, error),
                              status="error")

    notes = reading.get("notes") or {}
    if ocr_failures:
        notes["ocr_failures"] = ocr_failures[:3]
    box = reading.get("gauge_box")
    value = reading.get("value")
    raw_reason = reading.get("reason") or ""
    if raw_reason:
        # Keep upstream's wording in the evidence record; show the operator the
        # translated one.
        notes["reason_en"] = raw_reason
    candidates = [
        {"text": item.get("text"), "value": item.get("value"),
         "score": item.get("score"), "bbox": item.get("bbox"),
         "source": "dial-number", "angle_deg": item.get("angle_deg"),
         "inlier": item.get("inlier")}
        for item in reading.get("candidates") or []
    ]
    status = reading.get("status") or "unreadable"
    if status == "stable":
        reason_out = ""
    elif status == "out_of_range":
        # The angle->value line is being extrapolated past the last label.  The
        # value is still reported -- it is where the needle actually points --
        # but the record says plainly that it was not measured.
        reason_out = "指针指向刻度标注范围之外，读数为外推值，仅供参考"
    else:
        reason_out = _gauge_reason_zh(raw_reason)
    configured_scale = notes.get("configured_scale") or {}
    return {
        "kind": "pointer",
        "status": status,
        "value": value,
        # The page falls back to raw_text when value is absent, and shows the
        # reason when both are; keeping the number as text too means a refused
        # reading can never be mistaken for a measured one.
        "raw_text": None if value is None else str(value),
        "unit": reading.get("unit") or METER_UNIT,
        "confidence": reading.get("confidence") or 0.0,
        "angle_deg": reading.get("screen_angle_deg", reading.get("needle_angle_deg")),
        "dial": {"bbox": box, "score": reading.get("gauge_score")},
        "scale": {"span": reading.get("scale_span"),
                  "zero_deg": configured_scale.get("zero_angle_deg", notes.get("theta_zero_deg")),
                  "zero_source": "configured_screen" if configured_scale else notes.get("theta_zero_source")},
        "candidates": candidates,
        "gauge_box": box,
        "value_method": reading.get("value_method") or "ethz_analog_gauge_reader",
        "model": "ETH Zurich analog_gauge_reader (ONNX, CPU)",
        "reason": reason_out,
        "notes": notes,
    }


def _read_configured_meter_frame(image, meter_type, meter_id, calibration=None):
    """Use this meter's configured display geometry on current pixels only."""
    record = METER_STORE.get(meter_id)
    if not record or not record.get("reference_image") or not record.get("display_region"):
        return _read_meter_frame(image, meter_type, calibration=calibration)
    from .meter_reference import ReferenceRegistrationError, read_registered_display

    filename = record["reference_image"]
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise MeterConfigError("invalid meter reference filename")
    reference = cv2.imread(str(METER_STORE.reference_image_dir / filename))
    if reference is None:
        raise MeterConfigError("configured meter reference image is unavailable")
    try:
        if meter_type == "pointer":
            if not record.get("pointer_scale"):
                return _read_meter_frame(image, meter_type, calibration=calibration)
            from .pointer_reference import read_registered_pointer
            return read_registered_pointer(reference, image, record, calibration)
        return read_registered_display(
            reference, image, record, _recognise_crops, _expanded_detected_row_crop,
            _normalise_numeric_crop_text, min_score=UPLOAD_DISPLAY_MIN_SCORE,
            min_agreement=UPLOAD_DISPLAY_MIN_AGREEMENT, unit=record.get("unit", ""),
        )
    except (ReferenceRegistrationError, cv2.error, np.linalg.LinAlgError) as error:
        return {"kind": meter_type, "status": "unreadable", "value": None,
                "raw_text": None, "confidence": 0.0, "candidates": [],
                "reason": "current display registration refused: " + str(error),
                "model": "PaddleOCR PP-OCRv5 mobile_rec (registered display)",
                "value_method": "paddleocr_registered_current_display"}


def _read_meter_frame(image, meter_type, calibration=None):
    """Run the proven upload readers on one already-captured camera frame."""

    if meter_type == "pointer":
        from .pointer_gauge_adapter import read_pointer_gauge

        return read_pointer_gauge(image, calibration)
    if meter_type != "digital":
        raise ValueError("meter_type must be digital or pointer")

    if not _ppocr_available():
        reading = MeterReader().read(
            image,
            "digital",
            {"display_kind": "ocr", "unit": DIGITAL_METER_UNIT},
        )
        return reading.as_dict()

    result, _reading_bbox = (None, None)
    display_scan = None
    if UPLOAD_DISPLAY_REGION_ENABLED:
        result, _reading_bbox, display_scan = _read_display_region(image)
    if result is None:
        scan_bbox = (display_scan or {}).get("bbox")
        if scan_bbox:
            result = {
                "status": "ambiguous",
                "value": None,
                "raw_text": None,
                "unit": DIGITAL_METER_UNIT,
                "confidence": 0.0,
                "value_method": "paddleocr_display_region",
                "model": "PaddleOCR PP-OCRv5 mobile_rec (display region)",
                "candidates": [
                    {
                        "text": item["text"],
                        "score": item["score"],
                        "height": item["line_height"],
                        "bbox": scan_bbox,
                        "source": "display-region",
                    }
                    for item in (display_scan.get("tally") or [])
                ],
                "text_items": [],
                "reason": "display located but crop variants disagreed",
            }
        else:
            encoded = _encode_jpeg(image, 92)
            if not encoded:
                raise ValueError("image encoding failed")
            result, _reading_bbox = _read_full_frame(encoded)
            result, _reading_bbox = _refine_detected_numeric_reading(
                image, result, _reading_bbox
            )
    result["display_scan"] = display_scan
    if display_scan and display_scan.get("refused") and result.get("status") != "stable":
        existing = result.get("reason") or ""
        result["reason"] = (existing + "; " + display_scan["refused"]).lstrip("; ")
    return result


def _encode_jpeg(image, quality=92):
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buffer.tobytes() if ok else None


def _annotate_reading(image, highlights):
    """Return a JPEG of the upload with the given boxes outlined.

    ``highlights`` is a sequence of ``(bbox, colour, thickness)``.  A reading is
    marked with one thick green box; an ambiguous frame gets a thin box around
    every competing candidate, so the operator can see what the detector found
    instead of being shown an unexplained refusal.
    """
    annotated = image.copy()
    height, width = annotated.shape[:2]
    for bbox, colour, thickness in highlights:
        if bbox is None:
            continue
        x0, y0, x1, y1 = (int(round(value)) for value in bbox)
        x0 = max(0, min(width - 1, x0))
        y0 = max(0, min(height - 1, y0))
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        cv2.rectangle(annotated, (x0, y0), (x1, y1), colour, thickness)
    return _encode_jpeg(annotated, 92)


def _crop_reading(image, bbox):
    """Return a JPEG of the chosen box, used as the page's zoomed evidence."""
    if bbox is None:
        return None
    height, width = image.shape[:2]
    x0, y0, x1, y1 = (int(round(value)) for value in bbox)
    x0 = max(0, min(width, x0))
    y0 = max(0, min(height, y0))
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    ok, buffer = cv2.imencode(".jpg", image[y0:y1, x0:x1],
                              [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return buffer.tobytes() if ok else None


def _atomic_write_bytes(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as stream:
        stream.write(payload)
        stream.flush()
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _atomic_write_json(path, payload):
    encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    _atomic_write_bytes(path, encoded)


def _inspection_evidence_token(value, label):
    text = str(value or "").strip()
    if not INSPECTION_EVIDENCE_TOKEN.fullmatch(text):
        raise MeterConfigError(
            f"{label} must contain 1-96 letters, digits, _ or -"
        )
    return text


def _result_bbox(result):
    gauge_box = result.get("gauge_box")
    if isinstance(gauge_box, (list, tuple)) and len(gauge_box) == 4:
        return gauge_box
    detected_region_scan = result.get("detected_region_scan")
    if isinstance(detected_region_scan, dict):
        bbox = detected_region_scan.get("crop_bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            return bbox
    display_scan = result.get("display_scan")
    if isinstance(display_scan, dict):
        bbox = display_scan.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            return bbox
    for candidate in result.get("candidates") or []:
        if isinstance(candidate, dict):
            bbox = candidate.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                return bbox
    return None


def _persist_inspection_evidence(
    image_bytes,
    image,
    result,
    *,
    session_id,
    waypoint,
    capture_index,
):
    session = _inspection_evidence_token(session_id, "evidence_session")
    point = _inspection_evidence_token(waypoint, "waypoint").upper()
    try:
        index = int(capture_index)
    except (TypeError, ValueError) as error:
        raise MeterConfigError("capture_index must be an integer") from error
    if not 1 <= index <= 9999:
        raise MeterConfigError("capture_index must be in [1, 9999]")

    directory = INSPECTION_PHOTO_ROOT / session / point
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    stem = f"{index:02d}-{timestamp}"
    scene_path = directory / f"{stem}-scene.jpg"
    crop_path = directory / f"{stem}-meter.jpg"
    result_path = directory / f"{stem}-result.json"

    _atomic_write_bytes(scene_path, image_bytes)
    crop_bytes = _crop_reading(image, _result_bbox(result))
    if crop_bytes:
        _atomic_write_bytes(crop_path, crop_bytes)

    evidence = {
        "session_id": session,
        "waypoint": point,
        "capture_index": index,
        "scene_path": str(scene_path),
        "meter_crop_path": str(crop_path) if crop_bytes else None,
        "result_path": str(result_path),
    }
    result["evidence"] = evidence
    _atomic_write_json(
        result_path,
        {
            "schema_version": 1,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "evidence": evidence,
            "result": result,
        },
    )
    return evidence


CAMERA_MODES = {
    "astra_raw": CameraStream(
        camera_id="astra",
        name="Astra Pro RGB",
        device=ASTRA_DEVICE,
        width=1280,
        height=720,
        fps=30,
        stream_fps=15,
        command=None,
        inference={
            "enabled": False,
            "task": "raw_capture",
            "model": "none",
            "backend": "ffmpeg",
            "capture_profile": "Astra Pro RGB 1280x720@30",
        },
        required_paths=[ASTRA_DEVICE],
    ),
    "c70_general": CameraStream(
        # Keep the existing mode id for API compatibility.  Both modes now
        # read from Astra; only the detector pipeline changes.
        camera_id="astra",
        name="Astra Pro RGB",
        device=ASTRA_DEVICE,
        width=640,
        height=480,
        fps=10,
        stream_fps=10,
        command=_detector_command(
            ASTRA_DEVICE, 640, 480, 10, YOLOV8_CONFIG, YOLOV8_MODEL
        ),
        working_directory=str(VISION_ROOT),
        inference={
            "enabled": True,
            "task": "general_detection",
            "model": "YOLOv8n General Detection",
            "ppe_only": False,
            "ppe_enabled": False,
            "backend": "SpaceMITExecutionProvider",
        },
        required_paths=[YOLOV8_BINARY, YOLOV8_MODEL, YOLOV8_CONFIG, ASTRA_DEVICE],
    ),
    "astra_hardhat": CameraStream(
        camera_id="astra",
        name="Astra Pro RGB",
        device=ASTRA_DEVICE,
        width=640,
        height=480,
        fps=30,
        stream_fps=30,
        command=_detector_command(
            ASTRA_DEVICE,
            640,
            480,
            30,
            ASTRA_PERSON_CONFIG,
            YOLOV8_MODEL,
            ppe_config=HARDHAT_CONFIG,
            ppe_model=HARDHAT_MODEL,
            ppe_fps=5,
            head_roi_ppe=True,
        ),
        working_directory=str(VISION_ROOT),
        inference={
            "enabled": True,
            "task": "helmet_compliance",
            "model": "YOLOv8n ByteTrack + RVClaw Hard-Hat",
            "ppe_only": False,
            "ppe_enabled": True,
            "ppe_model": "RVClaw YOLOv8n Hard-Hat",
            "ppe_target_fps": 5,
            "tracking": "ByteTrack",
            "head_roi": True,
            "backend": "SpaceMITExecutionProvider",
        },
        required_paths=[
            YOLOV8_BINARY,
            YOLOV8_MODEL,
            ASTRA_PERSON_CONFIG,
            HARDHAT_MODEL,
            HARDHAT_CONFIG,
            ASTRA_DEVICE,
        ],
    ),
    "astra_meter": CameraStream(
        camera_id="astra",
        name="Astra Pro RGB",
        device=ASTRA_DEVICE,
        width=640,
        height=480,
        fps=30,
        stream_fps=30,
        command=_detector_command(
            ASTRA_DEVICE, 640, 480, 30,
            PPOCR_CONFIG if METER_TYPE == "digital" else "",
            PPOCR_MODEL if METER_TYPE == "digital" else "",
            meter_only=True,
            meter_type=METER_TYPE,
            meter_start_angle=METER_START_ANGLE,
            meter_end_angle=METER_END_ANGLE,
            meter_min=METER_MIN,
            meter_max=METER_MAX,
            meter_direction=METER_DIRECTION,
            meter_roi=METER_ROI,
        ),
        working_directory=str(VISION_ROOT),
        inference={
            "enabled": True,
            "task": "meter_reading",
            "model": "PP-OCRv5 Mobile" if METER_TYPE == "digital" else "OpenCV Pointer Reader",
            "meter_type": METER_TYPE,
            "backend": "CPUExecutionProvider" if METER_TYPE == "digital" else "OpenCV",
        },
        required_paths=[
            YOLOV8_BINARY,
            ASTRA_DEVICE,
            *([PPOCR_CONFIG, PPOCR_MODEL, PPOCR_REC_MODEL] if METER_TYPE == "digital" else []),
        ],
    ),
    "astra_meter_digital": CameraStream(
        camera_id="astra",
        name="Astra Pro RGB",
        device=ASTRA_DEVICE,
        width=ASTRA_DIGITAL_WIDTH,
        height=ASTRA_DIGITAL_HEIGHT,
        fps=ASTRA_DIGITAL_FPS,
        stream_fps=ASTRA_DIGITAL_FPS,
        command=_digital_meter_command(),
        working_directory=str(VISION_ROOT),
        inference={
            "enabled": True,
            "task": "meter_reading",
            "model": "PP-OCRv5 Mobile",
            "meter_type": "digital",
            "backend": "CPUExecutionProvider",
            "capture_profile": "Astra Pro RGB 1280x720@30",
            "display_roi": DIGITAL_METER_REGION,
            "meter_id": DIGITAL_METER_ID,
            "meter_name": DIGITAL_METER_CONFIG.get("name"),
            "unit": DIGITAL_METER_UNIT,
            "configured": DIGITAL_METER_CONFIG["configured"],
            "text_items": [],
        },
        required_paths=[YOLOV8_BINARY, ASTRA_DEVICE, PPOCR_CONFIG, PPOCR_MODEL, PPOCR_REC_MODEL],
    ),
    "astra_meter_pointer": CameraStream(
        camera_id="astra",
        name="Astra Pro RGB",
        device=ASTRA_DEVICE,
        width=640,
        height=480,
        fps=30,
        stream_fps=30,
        command=_detector_command(
            ASTRA_DEVICE, 640, 480, 30, "", "", meter_only=True, meter_type="pointer",
            meter_start_angle=METER_START_ANGLE, meter_end_angle=METER_END_ANGLE,
            meter_min=METER_MIN, meter_max=METER_MAX,
            meter_direction=METER_DIRECTION, meter_roi=METER_ROI,
        ),
        working_directory=str(VISION_ROOT),
        inference={
            "enabled": True,
            "task": "meter_reading",
            "model": "OpenCV Pointer Reader",
            "meter_type": "pointer",
            "backend": "OpenCV",
        },
        required_paths=[YOLOV8_BINARY, ASTRA_DEVICE],
    ),
}

CAMERA_MODE_INFO = {
    "astra_raw": {
        "label": "Astra 原始画面",
        "task": "raw_capture",
    },
    "c70_general": {
        "label": "Astra 目标检测",
        "task": "general_detection",
    },
    "astra_hardhat": {
        "label": "Astra 安全帽检测",
        "task": "helmet_compliance",
    },
    "astra_meter": {
        "label": "Astra 仪表盘检测",
        "task": "meter_reading",
    },
    "astra_meter_digital": {
        "label": "Astra 数字仪表盘",
        "task": "meter_reading",
    },
    "astra_meter_pointer": {
        "label": "Astra 指针仪表盘",
        "task": "meter_reading",
    },
}


class CameraModeController:
    """Own exactly one camera/model pipeline and switch it atomically."""

    def __init__(self, cameras, default_mode):
        if default_mode not in cameras:
            raise ValueError(f"Unknown default camera mode: {default_mode}")
        self._cameras = cameras
        self._active_mode = default_mode
        self._switching = False
        self._generation = 0
        self._lock = threading.RLock()
        self._switch_thread = None

    def start(self):
        with self._lock:
            self._cameras[self._active_mode].start()

    def stop(self):
        with self._lock:
            if self._switching:
                raise RuntimeError("相机正在切换，请稍后重试")
            self._cameras[self._active_mode].stop()
            self._generation += 1

    def switch(self, mode):
        if mode not in self._cameras:
            raise ValueError(f"Unknown camera mode: {mode}")
        with self._lock:
            current = self._cameras[self._active_mode]
            if mode == self._active_mode and current.is_running() and not self._switching:
                return self._status_locked()
            if self._switching:
                if mode == self._active_mode:
                    return self._status_locked()
                raise RuntimeError("camera switch already in progress")
            self._switching = True
            self._active_mode = mode
            self._generation += 1
            worker = threading.Thread(
                target=self._switch_worker,
                args=(current, self._cameras[mode]),
                name="camera-mode-switch",
                daemon=True,
            )
            self._switch_thread = worker
            worker.start()
            return self._status_locked()

    def _switch_worker(self, current, target):
        try:
            # CameraStream.stop() fully reaps the native process before the
            # target pipeline is allowed to open the shared camera device.
            current.stop()
            target.start()
        except Exception as error:
            with target._condition:
                target._last_error = str(error)
        finally:
            with self._lock:
                self._switching = False
                self._switch_thread = None

    def status(self):
        with self._lock:
            return self._status_locked()

    def active_camera(self, camera_id, required_mode=None):
        with self._lock:
            if self._switching:
                return None
            if required_mode is not None and self._active_mode != required_mode:
                return None
            camera = self._cameras[self._active_mode]
            if camera.camera_id != camera_id or not camera.is_running():
                return None
            return camera

    def _status_locked(self):
        camera = self._cameras[self._active_mode]
        return {
            "active_mode": self._active_mode,
            "switching": self._switching,
            "generation": self._generation,
            "modes": [
                {"id": mode, **CAMERA_MODE_INFO[mode]}
                for mode in self._cameras
            ],
            "cameras": [camera.status()],
        }


DEFAULT_CAMERA_MODE = os.getenv("RVCLAW_CAMERA_MODE", "c70_general")
CAMERA_CONTROLLER = CameraModeController(CAMERA_MODES, DEFAULT_CAMERA_MODE)


def _benchmark_status():
    """Read-only board telemetry for the RVBench dashboard."""
    thermal = []
    for sensor in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
        try:
            value = int(sensor.read_text().strip()) / 1000.0
            sensor_type = (sensor.parent / 'type').read_text().strip()
        except (OSError, ValueError):
            continue
        thermal.append({'sensor': sensor.parent.name, 'type': sensor_type, 'celsius': round(value, 1)})
    policies = []
    frequencies = []
    for policy in sorted(Path('/sys/devices/system/cpu/cpufreq').glob('policy*')):
        try:
            current = int((policy / 'scaling_cur_freq').read_text().strip()) / 1000.0
            related = (policy / 'related_cpus').read_text().strip()
            governor = (policy / 'scaling_governor').read_text().strip()
            policies.append({'policy': policy.name, 'related_cpus': related, 'frequency_mhz': round(current, 1), 'governor': governor})
            frequencies.extend([current] * max(1, len(related.split())))
        except (OSError, ValueError):
            continue
    interfaces = []
    try:
        for line in Path('/proc/net/dev').read_text().splitlines()[2:]:
            name, counters = line.split(':', 1)
            fields = counters.split()
            if len(fields) >= 9:
                interfaces.append({'name': name.strip(), 'rx_bytes': int(fields[0]), 'tx_bytes': int(fields[8])})
    except (OSError, ValueError):
        pass
    camera = CAMERA_CONTROLLER.status()
    item = (camera.get('cameras') or [{}])[0]
    inference = item.get('inference') or {}
    host = socket.gethostname()
    return {
        'generated_at': time.time(),
        'sampled_at': datetime.now(timezone.utc).isoformat(),
        'host': host,
        'network': {'interfaces': interfaces, 'online': bool(item.get('online')), 'frame_age_ms': item.get('frame_age_ms')},
        'cpu': {'temperatures': thermal, 'temperature_celsius': max((x['celsius'] for x in thermal), default=None),
                'temperature_sensor': max(thermal, key=lambda x: x['celsius'], default=None),
                'frequencies_mhz': frequencies, 'frequency_mhz': round(sum(frequencies) / len(frequencies), 1) if frequencies else None,
                'policies': policies},
        'inference': {'enabled': bool(inference.get('enabled')), 'latency_ms': inference.get('inference_ms'),
                      'inferences': inference.get('inferences', 0), 'model': inference.get('model'), 'status': inference.get('error') or inference.get('task')},
        'stability': {'camera_online': bool(item.get('online')), 'frame_age_ms': item.get('frame_age_ms'), 'generation': camera.get('generation')}
    }


class RVClawRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        parsed = urlparse(self.path)
        decoded_path = unquote(parsed.path)
        if rvbench_store.is_path(decoded_path):
            try:
                value = rvbench_store.get(REPO_ROOT, STUDIO_ROOT, decoded_path, parsed.query)
                if isinstance(value, bytes):
                    self._write_bytes(value, _content_type(Path(decoded_path)), cache_control="no-store")
                elif value is not None:
                    self._write_json(value)
                else:
                    self.send_error(404)
            except rvbench_store.RvbenchError as error:
                self._write_json({"error": str(error)}, status=error.status)
            return
        if decoded_path == "/api/device/snapshot":
            self._write_json(_device_snapshot())
            return
        if decoded_path == "/api/logs":
            self._write_json(_recent_logs())
            return
        if decoded_path == "/api/benchmark-status":
            self._write_json(_benchmark_status())
            return
        if decoded_path == "/api/meter-current-read":
            self._write_json({
                "api_version": 2,
                "read_only": True,
                "switches_camera_mode": False,
                "supports_evidence_capture": True,
                "evidence_root": str(INSPECTION_PHOTO_ROOT),
                "supported_meter_types": ["digital", "pointer"],
                "requires_active_camera_id": "astra",
                "required_camera_mode": "astra_raw",
            })
            return
        if decoded_path == "/api/meter-configurations":
            self._write_json({
                "meters": METER_STORE.list(),
                "active_meter_id": METER_STORE.active_meter_id(),
            })
            return
        if decoded_path.startswith("/api/meter-reference-images/"):
            self._serve_meter_reference_image(decoded_path.removeprefix("/api/meter-reference-images/"))
            return
        if parsed.path == "/api/sample-flow":
            self._write_json(self._run_sample_flow())
            return
        if parsed.path == "/api/person-tracking":
            self._write_json(get_person_tracking_payload())
            return
        if parsed.path == "/api/vision-state":
            self._write_json(get_vision_state())
            return
        if parsed.path == "/api/video-detections":
            self._write_json(get_video_detection_payload())
            return
        if parsed.path == "/api/cameras":
            self._write_json(CAMERA_CONTROLLER.status())
            return
        if parsed.path == "/api/camera-mode":
            self._write_json(CAMERA_CONTROLLER.status())
            return
        if parsed.path == "/api/meter-reading":
            status = CAMERA_CONTROLLER.status()
            self._write_json(runtime_reading(status))
            return
        if parsed.path in {"/api/meter-upload-records", "/api/meter-evidence"}:
            records = _read_meter_evidence()
            params = parse_qs(parsed.query)
            meter_type = (params.get("meter_type") or [""])[0]
            status = (params.get("status") or [""])[0]
            if meter_type:
                records = [item for item in records if item.get("meter_type") == meter_type or item.get("kind") == meter_type]
            if status:
                records = [item for item in records if item.get("status") == status]
            total = len(records)
            try:
                offset = max(0, int((params.get("offset") or [0])[0]))
                limit = min(100, max(1, int((params.get("limit") or [100])[0])))
            except ValueError:
                offset, limit = 0, 100
            self._write_json({"records": records[offset:offset + limit], "total": total, "offset": offset, "limit": limit})
            return
        evidence_match = re.fullmatch(r"/api/meter-(?:upload-records|evidence)/([a-f0-9]{32})/(image|crop)\.jpg", parsed.path)
        if evidence_match:
            suffix = "-crop" if evidence_match[2] == "crop" else ""
            target = METER_EVIDENCE_ROOT / f"{evidence_match[1]}{suffix}.jpg"
            if not target.is_file():
                self.send_error(404)
            else:
                self._write_bytes(target.read_bytes(), "image/jpeg", cache_control="no-store")
            return
        result_match = re.fullmatch(r"/api/meter-upload-records/([a-f0-9]{32})/result\.json", parsed.path)
        if result_match:
            target = METER_EVIDENCE_ROOT / f"{result_match[1]}.json"
            if not target.is_file():
                self.send_error(404)
            else:
                self._write_bytes(target.read_bytes(), "application/json; charset=utf-8", cache_control="no-store")
            return
        if parsed.path == "/api/meter-test-read":
            self._write_json({"error": "请刷新网页，使用新版拍照试读"}, status=410)
            return
        artifact = re.fullmatch(r"/api/meter-trials/([a-f0-9]{32})/(scene.jpg|crop.png|result.json)", parsed.path)
        if artifact:
            target = METER_TRIAL_ROOT / f"{artifact[1]}-{artifact[2]}"
            if not target.is_file():
                self.send_error(404)
            else:
                self._write_bytes(target.read_bytes(), mimetypes.guess_type(target.name)[0] or "application/octet-stream")
            return
        if parsed.path.startswith("/stream/") and parsed.path.endswith(".mjpg"):
            camera_id = parsed.path.removeprefix("/stream/").removesuffix(".mjpg")
            camera = CAMERA_CONTROLLER.active_camera(camera_id)
            if camera is None:
                self.send_error(404)
            else:
                self._serve_camera(camera)
            return
        if parsed.path.startswith("/artifacts/"):
            self._serve_artifact(parsed.path.removeprefix("/artifacts/"))
            return
        if parsed.path.startswith("/samples/"):
            self._serve_sample(parsed.path.removeprefix("/samples/"))
            return
        self._serve_studio(parsed.path)

    def do_POST(self):
        if not METER_MUTATION_LOCK.acquire(blocking=False):
            self.close_connection = True
            self._write_json({"error": "检测操作进行中，请稍后重试"}, status=409)
            return
        try:
            if CAMERA_CONTROLLER.status().get("switching"):
                self.close_connection = True
                self._write_json({"error": "相机正在切换，请稍后重试"}, status=409)
                return
            self._do_post()
        except TrialError as error:
            self._write_json({"error": str(error)}, status=error.status)
        except (OSError, ValueError, RuntimeError) as error:
            self._write_json({"error": str(error)}, status=503)
        finally:
            METER_MUTATION_LOCK.release()

    def _do_post(self):
        parsed = urlparse(self.path)
        decoded_path = unquote(parsed.path)
        if rvbench_store.is_path(decoded_path):
            try:
                payload = self._read_json_body(max_size=16 * 1024 * 1024)
                self._write_json(rvbench_store.post(REPO_ROOT, STUDIO_ROOT, decoded_path, payload))
            except rvbench_store.RvbenchError as error:
                self._write_json({"error": str(error)}, status=error.status)
            return
        if decoded_path == "/api/meter-current-read":
            self._read_current_meter_frame()
            return
        if parsed.path in {"/api/meter-image-read", "/api/meter/image-read", "/api/meter/paddleocr-read"}:
            self._read_uploaded_meter_image()
            return
        action = re.fullmatch(r"/api/meter-configurations/([A-Za-z0-9_-]+)/(test-read|stop)", decoded_path)
        if action:
            record = METER_STORE.get(action[1])
            if not record:
                self._write_json({"error": "仪表不存在"}, status=404)
                return
            if action[2] == "test-read":
                if not meter_is_ready(record):
                    raise TrialError("请先完成并保存仪表标定", 400)
                self._write_json({"ok": True, "trial": run_trial(CAMERA_CONTROLLER, record, METER_TRIAL_ROOT)})
            else:
                current = runtime_reading(CAMERA_CONTROLLER.status())
                if current["mode"] != "astra_meter_digital" or current["meter_id"] != action[1]:
                    raise TrialError("当前运行的不是所选仪表，不会停止其他检测")
                CAMERA_CONTROLLER.stop()
                self._write_json(runtime_reading(CAMERA_CONTROLLER.status()))
            return
        if decoded_path == "/api/meter-configurations":
            self._save_meter_configuration()
            return
        activation_match = METER_ACTIVATION_PATH.fullmatch(decoded_path)
        if activation_match:
            self._activate_meter_configuration(activation_match.group(1))
            return
        reference_match = METER_REFERENCE_UPLOAD_PATH.fullmatch(decoded_path)
        if reference_match:
            self._upload_meter_reference_image(reference_match.group(1))
            return
        if parsed.path == "/api/camera-stop":
            CAMERA_CONTROLLER.stop()
            self._write_json(CAMERA_CONTROLLER.status())
            return
        if parsed.path != "/api/camera-mode":
            self.send_error(404)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._write_json({"error": "Invalid Content-Length"}, status=400)
            return
        if content_length <= 0 or content_length > 4096:
            self._write_json({"error": "Invalid request body"}, status=400)
            return
        try:
            payload = json.loads(self.rfile.read(content_length))
            mode = payload.get("mode")
            result = CAMERA_CONTROLLER.switch(mode)
        except (json.JSONDecodeError, AttributeError):
            self._write_json({"error": "Expected JSON object with mode"}, status=400)
            return
        except ValueError as error:
            self._write_json({"error": str(error)}, status=400)
            return
        except RuntimeError as error:
            self._write_json({"error": str(error)}, status=409)
            return
        self._write_json(result)

    def do_DELETE(self):
        if not METER_MUTATION_LOCK.acquire(blocking=False):
            self._write_json({"error": "检测操作进行中，请稍后重试"}, status=409)
            return
        try:
            if CAMERA_CONTROLLER.status().get("switching"):
                self._write_json({"error": "相机正在切换，请稍后重试"}, status=409)
                return
            self._do_delete()
        finally:
            METER_MUTATION_LOCK.release()

    def _do_delete(self):
        parsed = urlparse(self.path)
        decoded_path = unquote(parsed.path)
        if rvbench_store.is_path(decoded_path):
            try:
                self._write_json(rvbench_store.delete(REPO_ROOT, decoded_path))
            except rvbench_store.RvbenchError as error:
                self._write_json({"error": str(error)}, status=error.status)
            return
        evidence = re.fullmatch(r"/api/meter-upload-records/([a-f0-9]{32})", decoded_path)
        if evidence:
            evidence_id = evidence.group(1)
            records = _read_meter_evidence()
            kept = [item for item in records if item.get("evidence_id") != evidence_id]
            if len(kept) == len(records):
                self._write_json({"error": "记录不存在"}, status=404)
                return
            METER_EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
            METER_EVIDENCE_INDEX.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
            for suffix in (".jpg", "-crop.jpg", ".json"):
                try:
                    (METER_EVIDENCE_ROOT / f"{evidence_id}{suffix}").unlink()
                except FileNotFoundError:
                    pass
            self._write_json({"deleted": True})
            return
        match = METER_CONFIGURATION_PATH.fullmatch(unquote(parsed.path))
        if not match:
            self.send_error(404)
            return
        try:
            if not METER_STORE.delete(match.group(1)):
                self._write_json({"error": "Meter configuration was not found"}, status=404)
                return
        except MeterConfigError as error:
            self._write_json({"error": str(error)}, status=400)
            return
        refresh_digital_meter_camera()
        self._write_json({"deleted": True})

    def _read_json_body(self, max_size=256 * 1024):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise MeterConfigError("Invalid request body") from error
        if content_length <= 0 or content_length > max_size:
            raise MeterConfigError("Invalid request body")
        try:
            value = json.loads(self.rfile.read(content_length))
        except json.JSONDecodeError as error:
            raise MeterConfigError("Invalid JSON body") from error
        if not isinstance(value, dict):
            raise MeterConfigError("Expected a JSON object")
        return value

    def _read_current_meter_frame(self):
        """Read one fresh frame from Studio's existing Astra stream."""

        payload = self._read_json_body(max_size=64 * 1024)
        meter_type = str(payload.get("meter_type", "")).strip().lower()
        if meter_type not in {"digital", "pointer"}:
            raise MeterConfigError("meter_type must be digital or pointer")
        meter_id = str(payload.get("meter_id", "")).strip()
        if not meter_id or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", meter_id):
            raise MeterConfigError("meter_id must contain 1-64 letters, digits, _ or -")
        required_camera_mode = str(
            payload.get("required_camera_mode", "astra_raw")
        ).strip()
        if required_camera_mode != "astra_raw":
            raise MeterConfigError("required_camera_mode must be astra_raw")
        calibration = payload.get("calibration")
        if calibration is not None and not isinstance(calibration, dict):
            raise MeterConfigError("calibration must be a JSON object")
        try:
            timeout_s = float(payload.get("timeout_s", 3.0))
            max_frame_age_ms = float(payload.get("max_frame_age_ms", 1500.0))
        except (TypeError, ValueError) as error:
            raise MeterConfigError("timeout_s and max_frame_age_ms must be numeric") from error
        if not 0.05 <= timeout_s <= 10.0:
            raise MeterConfigError("timeout_s must be in [0.05, 10.0]")
        if not 50.0 <= max_frame_age_ms <= 3000.0:
            raise MeterConfigError("max_frame_age_ms must be in [50, 3000]")

        evidence_values = (
            payload.get("evidence_session"),
            payload.get("waypoint"),
            payload.get("capture_index"),
        )
        evidence_requested = any(value is not None for value in evidence_values)
        if evidence_requested and not all(value is not None for value in evidence_values):
            raise MeterConfigError(
                "evidence_session, waypoint and capture_index must be provided together"
            )
        if evidence_requested:
            evidence_session = _inspection_evidence_token(
                payload.get("evidence_session"), "evidence_session"
            )
            waypoint = _inspection_evidence_token(payload.get("waypoint"), "waypoint")
            try:
                capture_index = int(payload.get("capture_index"))
            except (TypeError, ValueError) as error:
                raise MeterConfigError("capture_index must be an integer") from error
            if not 1 <= capture_index <= 9999:
                raise MeterConfigError("capture_index must be in [1, 9999]")

        camera_status = CAMERA_CONTROLLER.status()
        if camera_status.get("active_mode") != required_camera_mode:
            raise TrialError(
                f"current camera mode must be {required_camera_mode}",
                409,
            )
        camera = CAMERA_CONTROLLER.active_camera(
            "astra", required_mode=required_camera_mode
        )
        if camera is None:
            raise TrialError("Astra raw camera is not running or is switching", 503)
        capture_started = time.monotonic()
        image_bytes, sequence, captured_at = camera.fresh_frame(
            timeout=timeout_s,
            max_age_seconds=max_frame_age_ms / 1000.0,
        )
        post_capture_status = CAMERA_CONTROLLER.status()
        if (
            post_capture_status.get("active_mode") != required_camera_mode
            or post_capture_status.get("switching") is True
        ):
            raise TrialError("camera mode changed during meter capture", 409)
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise TrialError("current camera frame could not be decoded", 502)

        started = time.monotonic()
        try:
            if meter_type == "pointer":
                from .pointer_gauge_adapter import pointer_gauge_inference_budget

                remaining = timeout_s - (time.monotonic() - capture_started) - 0.25
                if remaining <= 0:
                    result = {"kind": "pointer", "status": "unreadable", "value": None,
                              "confidence": 0.0, "gauge_box": None,
                              "reason": "pointer gauge request budget exhausted after capture"}
                else:
                    with pointer_gauge_inference_budget(remaining):
                        result = _read_configured_meter_frame(image, meter_type, meter_id, calibration=calibration)
            else:
                result = _read_configured_meter_frame(
                    image, meter_type, meter_id, calibration=calibration
                )
        except Exception as error:
            if evidence_requested:
                _persist_inspection_evidence(
                    image_bytes,
                    image,
                    {
                        "status": "error",
                        "reason": str(error) or type(error).__name__,
                        "meter_id": meter_id,
                        "meter_type": meter_type,
                    },
                    session_id=evidence_session,
                    waypoint=waypoint,
                    capture_index=capture_index,
                )
            raise
        result.update({
            "meter_id": meter_id,
            "meter_type": meter_type,
            "camera_mode": CAMERA_CONTROLLER.status().get("active_mode"),
            "frame_sequence": sequence,
            "captured_at": datetime.fromtimestamp(captured_at, timezone.utc).isoformat(),
            "frame_age_ms": round(max(0.0, time.time() - captured_at) * 1000.0, 1),
            "inference_ms": round((time.monotonic() - started) * 1000.0, 1),
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        if evidence_requested:
            _persist_inspection_evidence(
                image_bytes,
                image,
                result,
                session_id=evidence_session,
                waypoint=waypoint,
                capture_index=capture_index,
            )
        self._write_json({"ok": True, "trial": result})

    def _read_uploaded_meter_image(self):
        """Infer one user-uploaded image and persist it as page evidence."""
        try:
            payload = self._read_json_body(max_size=16 * 1024 * 1024)
            meter_type = str(payload.get("meter_type", "")).strip().lower()
            if meter_type not in {"digital", "pointer"}:
                raise MeterConfigError("meter_type must be digital or pointer")
            calibration = payload.get("calibration")
            if calibration is not None and not isinstance(calibration, dict):
                raise MeterConfigError("calibration must be a JSON object")
            encoded = payload.get("image")
            if not isinstance(encoded, str) or not encoded:
                raise MeterConfigError("image is required")
            raw = encoded.split(",", 1)[1] if "," in encoded else encoded
            image_bytes = base64.b64decode(raw, validate=True)
            if not image_bytes or len(image_bytes) > 10 * 1024 * 1024:
                raise MeterConfigError("image must be between 1 byte and 10 MB")
            image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise MeterConfigError("无法读取图片")
            encoded_jpg = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if not encoded_jpg[0]:
                raise MeterConfigError("图片编码失败")
            image_bytes = encoded_jpg[1].tobytes()

            original_jpg = image_bytes
            crop_bytes = None
            # An uploaded photo has no calibrated display region, so the frame is
            # kept whole and the reader locates the display itself.  Cropping a
            # hard-coded rectangle here is what made the previous upload path read
            # whatever happened to occupy one fixed spot on a 1920x1080 frame,
            # regardless of what was uploaded.
            config = {"display_kind": "ocr", "unit": DIGITAL_METER_UNIT}
            started = time.monotonic()
            if meter_type == "pointer":
                # Pointer gauges are read by the ETH Zurich `analog_gauge_reader`
                # models, running as ONNX inside this process.  Both readers that
                # used to answer this branch were in-house geometry -- first a
                # Hough circle against a hard-coded angle range, then a scale
                # fitted from PP-OCR labels -- and both are retired.  A missing
                # package or missing model files report themselves unsupported
                # rather than silently falling back to either of them.
                result = _read_pointer_gauge(image, calibration=calibration)
                gauge_box = result.get("gauge_box")
                if gauge_box:
                    # Show what the reader saw: green when it produced a value,
                    # amber when the face was found but not read (a needle out of
                    # range, or too few dial numbers), so the operator can tell a
                    # miss from a mis-aimed photo.
                    colour = (0, 200, 80) if result["status"] == "stable" else (0, 165, 255)
                    annotated = _annotate_reading(image, [(gauge_box, colour, 3)])
                    if annotated:
                        image_bytes = annotated
                        original_jpg = annotated
                    crop_bytes = _crop_reading(image, gauge_box) or crop_bytes
            elif meter_type == "digital" and _ppocr_available():
                # Two readers, one result.  The display reader locates a screen in
                # the photo's pixels and reads it with the recogniser alone; it is
                # tried first because it is fast and because it is the only one
                # that can see an LCD.  Whole-frame detection then covers printed
                # and mechanical counters, which the display reader ignores by
                # design (their background is not darker than the housing).
                result, reading_bbox = (None, None)
                display_scan = None
                if UPLOAD_DISPLAY_REGION_ENABLED:
                    result, reading_bbox, display_scan = _read_display_region(image)
                if result is None:
                    scan_bbox = (display_scan or {}).get("bbox")
                    if scan_bbox:
                        # A screen was located and digits were read off it, but the
                        # crop variants disagreed, so the reading was refused.  The
                        # whole-frame detector cannot help here: DBNet does not see
                        # seven-segment glyphs at all (measured), so a whole-frame
                        # "stable" reading is housing text -- an Aclara meter's
                        # nameplate "85.941540" instead of the LCD's "74590" -- and
                        # running it anyway costs 3-6 s only to be downgraded below.
                        # Answer straight from the display scan and skip it.
                        result = {
                            "status": "ambiguous",
                            "value": None,
                            "raw_text": None,
                            "unit": DIGITAL_METER_UNIT,
                            "confidence": 0.0,
                            "value_method": "paddleocr_display_region",
                            "model": "PaddleOCR PP-OCRv5 mobile_rec (display region)",
                            "candidates": [
                                {"text": item["text"],
                                 "score": item["score"],
                                 "height": item["line_height"],
                                 "bbox": scan_bbox,
                                 "source": "display-region"}
                                for item in (display_scan.get("tally") or [])
                            ],
                            "text_items": [],
                            "reason": "已定位到显示屏，但裁剪变体读数不一致，拒绝给出读数",
                        }
                        reading_bbox = None
                    else:
                        # No screen anywhere in the frame: this is a printed or
                        # mechanical counter, which is what whole-frame detection is
                        # for.
                        result, reading_bbox = _read_full_frame(image_bytes)
                        result, reading_bbox = _refine_detected_numeric_reading(
                            image, result, reading_bbox
                        )
                # Keep the display scan in the response: when no screen reading was
                # produced, this is what says whether a display was even found and
                # why its reading was refused.
                result["display_scan"] = display_scan
                if display_scan and display_scan.get("refused") and result.get("status") != "stable":
                    existing = result.get("reason") or ""
                    result["reason"] = (existing + "；" + display_scan["refused"]).lstrip("；")

                # Show the same evidence the reader used: the chosen box in green
                # when a reading was produced, every competing candidate in amber
                # when the frame held too many numbers to pick one.
                if result["status"] == "stable" and reading_bbox is not None:
                    highlights = [(reading_bbox, (0, 200, 80), 3)]
                    # Zoom the page's evidence to the crop the value was actually read
                    # from, not the tight box: the tight box clips the last digit, so
                    # the saved crop read "7459" for a meter showing "74590".
                    crop_box = (result.get("display_region") or {}).get("crop_bbox")
                    crop_bytes = _crop_reading(image, crop_box or reading_bbox) or crop_bytes
                elif result["status"] == "ambiguous":
                    highlights = [(candidate["bbox"], (0, 165, 255), 2)
                                  for candidate in result.get("candidates", [])]
                else:
                    highlights = []
                annotated = _annotate_reading(image, highlights)
                if annotated:
                    image_bytes = annotated
                    original_jpg = annotated
            else:
                reading = MeterReader().read(image, meter_type, config)
                result = reading.as_dict()
            result.update({
                "model": result.get("model") or "RVClaw MeterReader",
                "inference_ms": round((time.monotonic() - started) * 1000, 1),
                "filename": str(payload.get("filename") or "uploaded-image"),
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
            # Uploads never use the calibrated meters.json region; the box the
            # reading came from travels in "candidates" (and, for a screen, in
            # "display_region") so the page can outline it.
            result["roi"] = None
            evidence = _save_meter_evidence(result, original_jpg, crop_bytes)
            evidence_result = {**result, **evidence}
            evidence_path = METER_EVIDENCE_ROOT / f"{evidence['evidence_id']}.json"
            evidence_path.write_text(json.dumps(evidence_result, ensure_ascii=False, indent=2), encoding="utf-8")
            self._write_json(evidence_result)
        except (MeterConfigError, ValueError, base64.binascii.Error) as error:
            self._write_json({"status": "invalid_request", "value": None, "reason": str(error)}, status=400)

    def _save_meter_configuration(self):
        try:
            record = METER_STORE.upsert(self._read_json_body())
        except MeterConfigError as error:
            self._write_json({"error": str(error)}, status=400)
            return
        if METER_STORE.active_meter_id() == record["meter_id"]:
            refresh_digital_meter_camera()
        self._write_json({"meter": record})

    def _activate_meter_configuration(self, meter_id):
        try:
            record = METER_STORE.activate(meter_id)
        except MeterConfigError as error:
            self._write_json({"error": str(error)}, status=400)
            return
        if METER_STORE.active_meter_id() == record["meter_id"]:
            refresh_digital_meter_camera()
        try:
            camera_status = CAMERA_CONTROLLER.switch("astra_meter_digital")
        except (ValueError, RuntimeError) as error:
            self._write_json({"error": str(error)}, status=409)
            return
        self._write_json({
            "meter": record,
            "active_meter_id": METER_STORE.active_meter_id(),
            "camera": camera_status,
        })

    def _upload_meter_reference_image(self, meter_id):
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if content_length <= 0 or content_length > MAX_REFERENCE_IMAGE_SIZE:
            self._write_json({"error": "Reference image must be between 1 byte and 10 MB"}, status=400)
            return
        try:
            record = METER_STORE.save_reference_image(
                meter_id, content_type, self.rfile.read(content_length)
            )
        except MeterConfigError as error:
            self._write_json({"error": str(error)}, status=400)
            return
        if METER_STORE.active_meter_id() == record["meter_id"]:
            refresh_digital_meter_camera()
        self._write_json({"meter": record})

    def _serve_meter_reference_image(self, filename):
        if Path(filename).name != filename:
            self.send_error(404)
            return
        target = (METER_STORE.reference_image_dir / filename).resolve()
        root = METER_STORE.reference_image_dir.resolve()
        try:
            target.relative_to(root)
        except ValueError:
            self.send_error(404)
            return
        if not target.is_file():
            self.send_error(404)
            return
        self._write_bytes(target.read_bytes(), _content_type(target), cache_control="no-store")

    def _run_sample_flow(self):
        return run_sample_flow(
            image_path=DEFAULT_IMAGE,
            output_root=DEFAULT_RUNS,
            task_id="studio-sample-001",
            point_id="A-03",
            category="meter_reading",
        )

    def _serve_camera(self, camera):
        try:
            # A browser tab can disappear without completing a TCP close.  A
            # finite send timeout prevents that stale MJPEG writer from living
            # forever with megabytes queued in the kernel.
            self.connection.settimeout(3.0)
            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("X-RVClaw-Stream-FPS", str(camera.stream_fps))
            self.end_headers()

            sequence = -1
            # Full-rate mode must forward every captured frame. Applying a
            # 1/30-second software gate to a nominal 30 fps source causes
            # near-boundary frames to be skipped because of scheduler jitter.
            frame_interval = (
                0.0 if camera.stream_fps >= camera.fps else 1.0 / camera.stream_fps
            )
            next_emit_at = 0.0
            while True:
                frame, next_sequence = camera.wait_for_frame(sequence)
                if not camera.is_running():
                    break
                if frame is None or next_sequence == sequence:
                    continue
                sequence = next_sequence
                if frame_interval > 0.0:
                    now = time.monotonic()
                    if now < next_emit_at:
                        continue
                    next_emit_at = now + frame_interval
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, socket.timeout, OSError):
            return
        finally:
            self.close_connection = True

    def _serve_artifact(self, relative_path):
        target = (DEFAULT_RUNS / relative_path).resolve()
        runs_root = DEFAULT_RUNS.resolve()
        if not str(target).startswith(str(runs_root)) or not target.exists() or target.is_dir():
            self.send_error(404)
            return
        self._write_bytes(target.read_bytes(), _content_type(target))

    def _serve_sample(self, relative_path):
        samples_root = (REPO_ROOT / "samples").resolve()
        target = (samples_root / relative_path).resolve()
        if not str(target).startswith(str(samples_root)) or not target.exists() or target.is_dir():
            self.send_error(404)
            return
        self._write_bytes(target.read_bytes(), _content_type(target))

    def _serve_studio(self, request_path):
        relative = request_path.lstrip("/") or "index.html"
        target = (STUDIO_ROOT / relative).resolve()
        studio_root = STUDIO_ROOT.resolve()
        if not str(target).startswith(str(studio_root)) or not target.exists() or target.is_dir():
            self.send_error(404)
            return
        self._write_bytes(
            target.read_bytes(),
            _content_type(target),
            cache_control="no-store, no-cache, must-revalidate, max-age=0",
        )

    def _write_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self._write_bytes(body, "application/json; charset=utf-8", status=status)

    def _write_bytes(self, body, content_type, status=200, cache_control=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if cache_control:
            self.send_header("Cache-Control", cache_control)
        self.end_headers()
        self.wfile.write(body)


class RVClawHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64


def _content_type(path):
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def main():
    parser = argparse.ArgumentParser(description="RVClaw local Studio API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8017, type=int)
    args = parser.parse_args()

    CAMERA_CONTROLLER.start()

    server = RVClawHTTPServer((args.host, args.port), RVClawRequestHandler)
    signal.signal(signal.SIGTERM, lambda _signum, _frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    print(f"RVClaw Studio API listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        CAMERA_CONTROLLER.stop()


if __name__ == "__main__":
    main()
