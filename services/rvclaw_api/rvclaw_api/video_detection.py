import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from rvclaw_algo import YoloV8nDetector


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VIDEO = REPO_ROOT / "samples" / "factory_people_demo.mp4"
SAMPLE_INTERVAL_SEC = float(os.getenv("RVCLAW_VIDEO_SAMPLE_SEC", "0.25"))
MAX_SAMPLED_FRAMES = int(os.getenv("RVCLAW_VIDEO_MAX_FRAMES", "320"))


def get_video_detection_payload(video_path=None, inference_video_path=None):
    video_path = Path(video_path or os.getenv("RVCLAW_VIDEO_DISPLAY_PATH", DEFAULT_VIDEO))
    inference_video_path = _resolve_inference_video_path(video_path, inference_video_path)
    detector = YoloV8nDetector()
    runtime = detector.runtime()
    has_video = video_path.exists()
    has_inference_video = inference_video_path.exists()
    stream = _video_stream(inference_video_path) if has_inference_video else _fallback_stream()
    frames = []

    if runtime.get("available") and not has_inference_video:
        runtime = {**runtime, "video_inference_error": f"Inference video not found: {inference_video_path}"}
    elif has_inference_video and runtime.get("available"):
        try:
            frames = _load_or_build_frame_detections(inference_video_path, detector, runtime, stream)
        except Exception as exc:
            runtime = {**runtime, "video_inference_error": str(exc)}

    overlay_mode = _overlay_mode(runtime)
    detections = _nearest_non_empty_detections(frames)
    return {
        "source": {
            "type": "sample-video" if has_video else "browser-canvas",
            "name": video_path.name if has_video else "factory-aisle-canvas-demo",
            "url": f"/samples/{video_path.name}" if has_video else None,
            "replaceable_with": "ros2_camera_topic",
            "topic": "/camera/color/image_raw",
            "adapter": "rvclaw_perception.camera_stream",
            "transport": "mp4-file" if has_video else "browser-canvas",
        },
        "stream": {
            **stream,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "frame_id": "camera_color_optical_frame",
            "encoding": "rgb8",
        },
        "runtime": {
            **runtime,
            "flow": "video-frame -> yolov8n -> per-frame-boxes -> studio-overlay",
            "overlay_mode": overlay_mode,
            "sample_interval_sec": SAMPLE_INTERVAL_SEC,
            "display_video": str(video_path),
            "inference_video": str(inference_video_path),
            "inference_video_exists": has_inference_video,
        },
        "detections": detections,
        "frames": frames,
        "tracks": [],
    }


def _resolve_inference_video_path(video_path, inference_video_path=None):
    if inference_video_path:
        return Path(inference_video_path)
    configured = os.getenv("RVCLAW_VIDEO_INFERENCE_PATH")
    if configured:
        return Path(configured)
    sidecar = video_path.with_name(f"{video_path.stem}_cv2{video_path.suffix}")
    if sidecar.exists():
        return sidecar
    return video_path


def _load_or_build_frame_detections(video_path, detector, runtime, stream):
    cache_path = video_path.with_suffix(".yolov8n.json")
    cache_key = {
        "video": str(video_path),
        "video_mtime": video_path.stat().st_mtime,
        "model_path": runtime.get("model_path"),
        "confidence": runtime.get("confidence"),
        "sample_interval_sec": SAMPLE_INTERVAL_SEC,
        "max_sampled_frames": MAX_SAMPLED_FRAMES,
        "decoder": "cv2-or-ffmpeg-v2",
    }
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        cached_frames = cached.get("frames", [])
        if cached.get("cache_key") == cache_key and cached_frames:
            return cached_frames

    frames = _build_frame_detections(video_path, detector, stream)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump({"cache_key": cache_key, "frames": frames}, handle, indent=2)
    return frames


def _build_frame_detections(video_path, detector, stream):
    try:
        return _build_frame_detections_with_cv2(video_path, detector, stream)
    except RuntimeError as cv2_error:
        return _build_frame_detections_with_ffmpeg(video_path, detector, stream, cv2_error)


def _build_frame_detections_with_cv2(video_path, detector, stream):
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video with cv2: {video_path}")

    fps = max(float(stream["fps"]), 1.0)
    sample_step = max(int(round(fps * SAMPLE_INTERVAL_SEC)), 1)
    frames = []
    frame_index = 0
    sampled = 0
    try:
        while sampled < MAX_SAMPLED_FRAMES:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % sample_step == 0:
                detections = detector.detect_frame(frame)
                frames.append(
                    {
                        "time_sec": round(frame_index / fps, 3),
                        "frame_index": frame_index,
                        "decode_backend": "cv2",
                        "detections": detections,
                    }
                )
                sampled += 1
            frame_index += 1
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded with cv2 from video: {video_path}")
    return frames


def _build_frame_detections_with_ffmpeg(video_path, detector, stream, cv2_error):
    frame_dir = video_path.with_suffix(".ffmpeg_frames")
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frame_dir.glob("frame_*.jpg"):
        old_frame.unlink()

    frame_rate = 1.0 / max(SAMPLE_INTERVAL_SEC, 0.001)
    output_pattern = frame_dir / "frame_%06d.jpg"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={frame_rate:.6f}",
        "-frames:v",
        str(MAX_SAMPLED_FRAMES),
        "-q:v",
        "3",
        str(output_pattern),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"{cv2_error}; ffmpeg not found") from exc

    if completed.returncode != 0:
        detail = (completed.stderr or "ffmpeg returned a non-zero exit code").strip()
        raise RuntimeError(f"{cv2_error}; ffmpeg fallback failed: {detail}")

    image_paths = sorted(frame_dir.glob("frame_*.jpg"))
    if not image_paths:
        raise RuntimeError(f"{cv2_error}; ffmpeg fallback produced no frames: {video_path}")

    fps = max(float(stream.get("fps", 15)), 1.0)
    frames = []
    for index, image_path in enumerate(image_paths[:MAX_SAMPLED_FRAMES]):
        time_sec = round(index * SAMPLE_INTERVAL_SEC, 3)
        frames.append(
            {
                "time_sec": time_sec,
                "frame_index": int(round(time_sec * fps)),
                "decode_backend": "ffmpeg",
                "detections": detector.detect_image(image_path),
            }
        )
    return frames


def _video_stream(video_path):
    try:
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return _fallback_stream()
        try:
            fps = capture.get(cv2.CAP_PROP_FPS) or 15
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 960)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 540)
            frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            duration = frame_count / fps if fps else 12
            return {
                "width": width,
                "height": height,
                "fps": round(float(fps), 2),
                "duration_sec": round(float(duration), 2),
            }
        finally:
            capture.release()
    except Exception:
        return _fallback_stream()


def _fallback_stream():
    return {"width": 960, "height": 540, "fps": 15, "duration_sec": 12}


def _overlay_mode(runtime):
    if not runtime.get("available"):
        return "model-unavailable"
    if runtime.get("video_inference_error"):
        return "video-decode-unavailable"
    return "model-output"


def _nearest_non_empty_detections(frames):
    for frame in frames:
        if frame.get("detections"):
            return frame["detections"]
    return []
