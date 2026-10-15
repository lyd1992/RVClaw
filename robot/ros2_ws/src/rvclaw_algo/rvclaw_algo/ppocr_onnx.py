"""In-process PP-OCRv5: DBNet text detection + CRNN/CTC recognition.

Why this exists
===============
Studio used to shell out to ``build/meter-paddleocr/meter_paddleocr`` for every
uploaded photo.  On this board that helper eventually hangs forever: once the
kernel's DMA list has been corrupted, ``__fput -> dma_list_release`` faults with
``LIST_POISON1`` (``dead000000000100``), and a process must walk that very path
while it exits.  Measured twice on the K3, the binary prints its ONNX graph
warnings and then never emits a single JSON byte.

Running the same two ONNX models *inside* the Studio process, with
onnxruntime's pure-CPU execution provider, allocates no SpacemiT DMA or TCM
memory at all -- so the upload path keeps working on a board where the C++
helper hangs.  It also deletes the ~0.9 s model load that a fresh process paid
on every single call, even though the model never changed.

Fidelity
========
Everything below mirrors ``build/meter-paddleocr-src/ppocr_detector.cpp``
parameter for parameter -- same resize policy, same 32-px alignment, same
``db_thresh``/``box_thresh``/``unclip_ratio``, same "short side >= 3 px" filter,
same TL/TR/BR/BL ordering, same ``x/127.5 - 1`` recognition normalisation, same
zero-padded width, same CTC greedy decode with blank == index 0.  That is
deliberate: readings must stay comparable with the ones already recorded in
``runs/meter-uploads/index.json``.
"""

from __future__ import annotations

import math
import os
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

__all__ = ["PpocrOnnxEngine", "load_ppocr_config", "default_config_path"]

# PaddleOCR's ImageNet statistics, used by the DBNet detector.
_DET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_DET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

_DET_SIZE_DIVISOR = 32
_MIN_BOX_SIDE_PX = 3.0
_MAX_REC_WIDTH = 3200
_REC_NORMALISE_SCALE = 1.0 / 127.5
_REC_NORMALISE_SHIFT = -1.0

# Where the dict lives on the K3 image; the yaml points here too.
DEFAULT_DICT_PATH = (
    "/data/spacemit_robot/components/model_zoo/vision/assets/labels/ppocr_keys.txt"
)
# The upload profile keeps DBNet enabled; the camera profile is rec-only.
DEFAULT_UPLOAD_CONFIG = "/data/RVClaw/configs/vision/ppocr_meter_upload.yaml"
DEFAULT_REC_CONFIG = "/data/RVClaw/configs/vision/ppocr_meter_rec.yaml"

ImageLike = Union[np.ndarray, bytes, bytearray, str, "os.PathLike[str]"]

_CACHE_LOCK = threading.Lock()
_PIPELINE_CACHE: Dict[Tuple[str, str], "_Pipeline"] = {}


def default_config_path(rec_only: bool = False) -> str:
    """Path of the shipped profile that matches ``rec_only``."""
    return DEFAULT_REC_CONFIG if rec_only else DEFAULT_UPLOAD_CONFIG


def as_image(source: ImageLike) -> np.ndarray:
    """Accept a path, encoded bytes or an ndarray; always return BGR uint8.

    Callers hand over a path or JPEG bytes as often as a decoded frame -- the
    standalone binary took a path, so accepting one keeps this engine a drop-in
    replacement -- so all three are normalised here rather than at each caller.
    """
    if isinstance(source, np.ndarray):
        return source
    if isinstance(source, (str, os.PathLike)):
        image = cv2.imread(os.fspath(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("cannot read image file: %s" % source)
        return image
    if isinstance(source, (bytes, bytearray)):
        buffer = np.frombuffer(bytes(source), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("cannot decode image bytes")
        return image
    raise TypeError("expected a path, image bytes or an ndarray, got %r" % type(source))


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def _parse_yaml_subset(text: str) -> Dict[str, Any]:
    """Parse just the shape the PP-OCR profiles actually use.

    Two levels deep: ``key: value`` at the top, ``default_params:`` introducing
    an indented block, and indented ``- item`` sequences.  Used only when PyYAML
    is missing, so the profiles keep working on a minimal image.
    """
    result: Dict[str, Any] = {}
    section: Optional[Dict[str, Any]] = None
    pending_list_key: Optional[str] = None

    def coerce(raw: str) -> Any:
        raw = raw.strip()
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
            return raw[1:-1]
        lowered = raw.lower()
        if lowered in ("true", "false"):
            return lowered == "true"
        if lowered in ("null", "~", ""):
            return None
        try:
            return int(raw)
        except ValueError:
            pass
        try:
            return float(raw)
        except ValueError:
            pass
        return raw

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- "):
            if section is not None and pending_list_key is not None:
                section.setdefault(pending_list_key, []).append(coerce(stripped[2:]))
            continue
        if ":" not in stripped:
            continue
        key, _, raw_value = stripped.partition(":")
        key = key.strip()
        raw_value = raw_value.strip()
        if not line.startswith((" ", "\t")):
            if raw_value == "":
                section = {}
                result[key] = section
                pending_list_key = None
            else:
                result[key] = coerce(raw_value)
                section = None
                pending_list_key = None
            continue
        if section is None:
            continue
        if raw_value == "":
            pending_list_key = key
            section[key] = []
        else:
            section[key] = coerce(raw_value)
            pending_list_key = None
    return result


def load_ppocr_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Read a PP-OCR profile, returning ``{model_path, default_params...}``."""
    if not config_path or not os.path.isfile(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            raw = handle.read()
    except OSError:
        return {}
    try:
        import yaml  # type: ignore

        parsed = yaml.safe_load(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return _parse_yaml_subset(raw)


def _expand(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("~"):
        return os.path.expanduser(value)
    return value


# ---------------------------------------------------------------------------
# one config = two sessions (detector + recogniser)
# ---------------------------------------------------------------------------


class _Pipeline:
    """Detector and recogniser sessions described by a single yaml profile."""

    def __init__(self, config_path: str, provider: str):
        config = load_ppocr_config(config_path)
        params = config.get("default_params") or {}
        self.config_path = config_path
        self.det_model_path = _expand(config.get("model_path")) or _expand(
            params.get("model_path")
        )
        self.rec_model_path = _expand(params.get("rec_model_path"))
        self.dict_path = _expand(params.get("dict_path")) or DEFAULT_DICT_PATH
        self.det_limit_side_len = int(params.get("det_limit_side_len", 960))
        self.det_db_thresh = float(params.get("det_db_thresh", 0.3))
        self.det_db_box_thresh = float(params.get("det_db_box_thresh", 0.6))
        self.det_db_unclip_ratio = float(params.get("det_db_unclip_ratio", 2.0))
        self.rec_img_h = int(params.get("rec_img_h", 48))
        self.rec_img_w_max = int(params.get("rec_img_w_max", 320))
        self.num_threads = int(params.get("num_threads", 4))
        self.rec_only = bool(params.get("rec_only", False))
        self.provider = provider
        self.dict: List[str] = []
        self._det_session = None
        self._rec_session = None
        self._det_input = None
        self._rec_input = None
        self._lock = threading.Lock()

    # -- sessions ---------------------------------------------------------

    def _session_options(self):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, self.num_threads)
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return options

    def _make_session(self, model_path: Optional[str]):
        import onnxruntime as ort

        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError("ONNX model not found: %s" % model_path)
        session = ort.InferenceSession(
            model_path, self._session_options(), providers=[self.provider]
        )
        return session

    def ensure_loaded(self):
        """Create both sessions on first use (and load the dict once)."""
        if self._rec_session is not None and (self.rec_only or self._det_session is not None):
            return
        with self._lock:
            if not self.dict:
                self.dict = self._load_dict(self.dict_path)
            if self._rec_session is None:
                self._rec_session = self._make_session(self.rec_model_path)
                self._rec_input = self._rec_session.get_inputs()[0].name
            if not self.rec_only and self._det_session is None:
                self._det_session = self._make_session(self.det_model_path)
                self._det_input = self._det_session.get_inputs()[0].name

    @staticmethod
    def _load_dict(dict_path: str) -> List[str]:
        """Load the character table exactly as the C++ loader does.

        Index 0 is the CTC blank and already present at line 0 of the shipped
        file, so no extra blank is inserted -- shifting every index would turn
        correct text into garbage.  Empty lines mean a space, and PaddleOCR's
        ``use_space_char`` appends one more space class at the end.
        """
        chars: List[str] = []
        with open(dict_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.rstrip("\r\n")
                chars.append(line if line else " ")
        if len(chars) <= 1:
            raise ValueError("dict file is empty: %s" % dict_path)
        chars.append(" ")
        return chars

    # -- detection --------------------------------------------------------

    def _det_preprocess(self, image: np.ndarray):
        ori_h, ori_w = image.shape[:2]
        ratio = float(max(ori_h, ori_w)) / float(self.det_limit_side_len)
        net_h, net_w = ori_h, ori_w
        if ratio > 1.0:
            net_h = int(ori_h / ratio)
            net_w = int(ori_w / ratio)
        net_h = max(_DET_SIZE_DIVISOR, (net_h + 31) // 32 * 32)
        net_w = max(_DET_SIZE_DIVISOR, (net_w + 31) // 32 * 32)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (net_w, net_h), interpolation=cv2.INTER_LINEAR)
        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)
        blob = (blob - _DET_MEAN) / _DET_STD
        return np.ascontiguousarray(blob[None, ...]), net_h, net_w

    @staticmethod
    def _sort_box_points(points: np.ndarray) -> np.ndarray:
        """Order four corners as TL, TR, BR, BL (PaddleOCR's get_mini_boxes)."""
        order = np.argsort(points[:, 0], kind="stable")
        pts = points[order]
        tl = 0 if pts[0][1] <= pts[1][1] else 1
        bl = 1 - tl
        tr = 2 if pts[2][1] <= pts[3][1] else 3
        br = 3 if tr == 2 else 2
        return np.array([pts[tl], pts[tr], pts[br], pts[bl]], dtype=np.int32)

    @staticmethod
    def _box_score(prob: np.ndarray, box: np.ndarray) -> float:
        """Mean probability inside ``box`` -- the DBNet box-confidence filter."""
        height, width = prob.shape[:2]
        pts = np.clip(box.astype(np.int32), (0, 0), (width - 1, height - 1))
        x, y, w, h = cv2.boundingRect(pts)
        if w <= 0 or h <= 0:
            return 0.0
        mask = np.zeros((h, w), dtype=np.uint8)
        shifted = (pts - np.array([x, y], dtype=np.int32)).astype(np.int32)
        cv2.fillPoly(mask, [shifted], 1)
        return float(cv2.mean(prob[y:y + h, x:x + w], mask=mask)[0])

    def _unclip(self, polygon: np.ndarray) -> Optional[np.ndarray]:
        """Expand a polygon by ``area * ratio / perimeter`` along its normals."""
        poly = polygon.astype(np.float32).reshape(-1, 1, 2)
        area = abs(float(cv2.contourArea(poly)))
        length = float(cv2.arcLength(poly, True))
        if length < 1e-6:
            return polygon
        distance = area * self.det_db_unclip_ratio / length

        count = len(poly)
        offsets: List[Tuple[int, int]] = []
        for index in range(count):
            p1 = poly[index][0]
            p2 = poly[(index + 1) % count][0]
            edge_x = float(p2[0] - p1[0])
            edge_y = float(p2[1] - p1[1])
            norm = math.hypot(edge_x, edge_y)
            if norm < 1e-6:
                continue
            normal = np.array([-edge_y / norm, edge_x / norm], dtype=np.float32)
            for point in (p1, p2):
                moved = point + normal * distance
                offsets.append((int(moved[0]), int(moved[1])))
        if not offsets:
            return polygon
        hull = cv2.convexHull(np.array(offsets, dtype=np.int32).reshape(-1, 1, 2))
        return hull.reshape(-1, 2)

    def _db_postprocess(self, prob: np.ndarray, ori_h: int, ori_w: int,
                        net_h: int, net_w: int) -> List[Dict[str, Any]]:
        binary = np.where(prob > self.det_db_thresh, 255, 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(binary, kernel)

        found = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = found[-2] if len(found) == 3 else found[0]

        scale_x = float(ori_w) / float(net_w)
        scale_y = float(ori_h) / float(net_h)

        boxes: List[Dict[str, Any]] = []
        for contour in contours:
            if contour is None or len(contour) < 4:
                continue
            rect = cv2.minAreaRect(contour)
            if min(rect[1][0], rect[1][1]) < _MIN_BOX_SIDE_PX:
                continue
            box_pts = cv2.boxPoints(rect).astype(np.int32)
            score = self._box_score(prob, box_pts)
            if score < self.det_db_box_thresh:
                continue
            expanded = self._unclip(box_pts)
            if expanded is None or len(expanded) < 4:
                continue
            expanded_rect = cv2.minAreaRect(expanded.reshape(-1, 1, 2).astype(np.float32))
            expanded_pts = cv2.boxPoints(expanded_rect)
            final = np.empty((4, 2), dtype=np.int32)
            final[:, 0] = np.clip(np.round(expanded_pts[:, 0] * scale_x), 0, ori_w - 1)
            final[:, 1] = np.clip(np.round(expanded_pts[:, 1] * scale_y), 0, ori_h - 1)
            boxes.append({"points": self._sort_box_points(final), "score": score})
        return boxes

    # -- recognition ------------------------------------------------------

    @staticmethod
    def _crop_text_box(image: np.ndarray, box: np.ndarray) -> np.ndarray:
        """Perspective-warp a TL/TR/BR/BL quad to an axis-aligned crop."""
        src = box.astype(np.float32)
        w1 = math.hypot(src[1][0] - src[0][0], src[1][1] - src[0][1])
        w2 = math.hypot(src[2][0] - src[3][0], src[2][1] - src[3][1])
        h1 = math.hypot(src[3][0] - src[0][0], src[3][1] - src[0][1])
        h2 = math.hypot(src[2][0] - src[1][0], src[2][1] - src[1][1])
        crop_w = max(1, int(max(w1, w2)))
        crop_h = max(1, int(max(h1, h2)))
        dst = np.array(
            [[0, 0], [crop_w - 1, 0], [crop_w - 1, crop_h - 1], [0, crop_h - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(src, dst)
        crop = cv2.warpPerspective(image, matrix, (crop_w, crop_h))
        # Vertical text: rotate upright, as PaddleOCR does.
        if crop_h > crop_w * 1.5:
            crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return crop

    def _rec_tensor(self, crop: np.ndarray) -> np.ndarray:
        """Build the ``1x3x48xW`` recognition input, zero-padded on the right."""
        rows, cols = crop.shape[:2]
        aspect = float(cols) / float(rows)
        wanted = int(math.ceil(self.rec_img_h * aspect))
        input_w = min(_MAX_REC_WIDTH, max(self.rec_img_w_max, wanted))
        target_w = max(1, min(input_w, wanted))

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (target_w, self.rec_img_h), interpolation=cv2.INTER_LINEAR)
        normalised = resized.astype(np.float32) * _REC_NORMALISE_SCALE + _REC_NORMALISE_SHIFT

        tensor = np.zeros((1, 3, self.rec_img_h, input_w), dtype=np.float32)
        tensor[0, :, :, :target_w] = normalised.transpose(2, 0, 1)
        return tensor

    def _rec_decode(self, logits: np.ndarray) -> Tuple[str, float]:
        """Greedy CTC decode; index 0 is blank, repeats collapse."""
        probabilities = logits[0]
        best_path = probabilities.argmax(axis=1)
        pieces: List[str] = []
        score_sum = 0.0
        emitted = 0
        last = -1
        for step, best in enumerate(best_path.tolist()):
            if best != 0 and best != last and best < len(self.dict):
                pieces.append(self.dict[best])
                score_sum += float(probabilities[step, best])
                emitted += 1
            last = best
        return "".join(pieces), (score_sum / emitted if emitted else 0.0)

    def rec_run(self, crop: np.ndarray) -> Tuple[str, float]:
        if crop is None or crop.size == 0:
            return "", 0.0
        self.ensure_loaded()
        tensor = self._rec_tensor(crop)
        logits = self._rec_session.run(None, {self._rec_input: tensor})[0]
        return self._rec_decode(np.asarray(logits))

    # -- pipeline ---------------------------------------------------------

    def detect_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect text boxes and recognise each, in reading order of detection."""
        self.ensure_loaded()

        if self.rec_only:
            text, score = self.rec_run(image)
            if not text:
                return []
            height, width = image.shape[:2]
            return [{
                "text": text,
                "score": score,
                "poly": [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            }]

        blob, net_h, net_w = self._det_preprocess(image)
        probability = self._det_session.run(None, {self._det_input: blob})[0]
        prob_map = np.asarray(probability)
        while prob_map.ndim > 2:
            prob_map = prob_map[0]

        boxes = self._db_postprocess(prob_map, image.shape[0], image.shape[1], net_h, net_w)

        results: List[Dict[str, Any]] = []
        for box in boxes:
            crop = self._crop_text_box(image, box["points"])
            text, score = self.rec_run(crop)
            if not text:
                continue
            results.append({
                "text": text,
                "score": score,
                "poly": [[int(x), int(y)] for x, y in box["points"]],
            })
        return results


# ---------------------------------------------------------------------------
# public engine
# ---------------------------------------------------------------------------


class PpocrOnnxEngine:
    """Thread-safe front end over one or more PP-OCR profiles.

    ``recognize`` returns exactly the JSON payload the standalone binary used to
    print, so callers keep their existing parsing code.
    """

    def __init__(self, provider: str = "CPUExecutionProvider"):
        self.provider = provider

    def _pipeline(self, config_path: Optional[str]) -> _Pipeline:
        key = (os.path.abspath(config_path or DEFAULT_UPLOAD_CONFIG), self.provider)
        with _CACHE_LOCK:
            pipeline = _PIPELINE_CACHE.get(key)
            if pipeline is None:
                pipeline = _Pipeline(key[0], self.provider)
                _PIPELINE_CACHE[key] = pipeline
        return pipeline

    def warmup(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Create the sessions up front so the first upload is not the slow one."""
        pipeline = self._pipeline(config_path)
        pipeline.ensure_loaded()
        return {
            "config": pipeline.config_path,
            "det_model": pipeline.det_model_path,
            "rec_model": pipeline.rec_model_path,
            "rec_only": pipeline.rec_only,
            "dict_classes": len(pipeline.dict),
            "provider": pipeline.provider,
        }

    def models_present(self, config_path: Optional[str] = None) -> bool:
        """Whether the files a profile needs exist -- without creating a session.

        Callers gate on this before committing to the OCR path, so it must stay
        cheap; a real load happens on the first photograph, not on every request.
        """
        try:
            pipeline = self._pipeline(config_path)
        except Exception:
            return False
        if not pipeline.rec_model_path or not os.path.isfile(pipeline.rec_model_path):
            return False
        if not pipeline.rec_only:
            if not pipeline.det_model_path or not os.path.isfile(pipeline.det_model_path):
                return False
        return os.path.isfile(pipeline.dict_path)

    def recognize(self, source: ImageLike, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Run detection + recognition, mirroring the binary's JSON envelope."""
        image = as_image(source)
        if image is None or image.size == 0:
            raise ValueError("empty image")
        pipeline = self._pipeline(config_path)
        items = pipeline.detect_text(image)
        return {
            "ok": True,
            "image_width": int(image.shape[1]),
            "image_height": int(image.shape[0]),
            "items": [
                {"text": item["text"], "score": float(item["score"]), "poly": item["poly"]}
                for item in items
            ],
        }

    def recognize_crop(self, source: ImageLike,
                       config_path: Optional[str] = None) -> Dict[str, Any]:
        """Recognise one pre-cropped region (the rec-only profile)."""
        image = as_image(source)
        if image is None or image.size == 0:
            raise ValueError("empty image")
        pipeline = self._pipeline(config_path or DEFAULT_REC_CONFIG)
        text, score = pipeline.rec_run(image)
        return {"text": text, "score": float(score)}
