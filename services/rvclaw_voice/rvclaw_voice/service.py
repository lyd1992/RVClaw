"""On-device speech service with optional SpacemiT SDK adapters.

The service is intentionally import-safe when model wheels are not installed;
``--health`` can then report the missing runtime without touching audio devices.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
import logging
import os
from pathlib import Path
from queue import Empty, Full, Queue
import signal
import subprocess
from threading import Event, Lock, Thread
import tempfile
import time
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple
from types import SimpleNamespace
import wave

import numpy as np

from .router import (
    ACK_TEXT,
    CommandHandlerError,
    FAILURE_TEXT,
    CommandRouter,
    _last_result_response,
    _status_response,
)
from .robot_client import (
    ChassisMotionClient,
    ChassisRouteClient,
    InspectionCoordinator,
    ManualMappingClient,
)
from .voice_command import MOTION_INTENTS, is_complete_command, parse_transcript, strip_wake_phrase
from .acoustic import AcousticGate, AcousticUtterance, NativeSpeech, acoustic_command_text, command_text, save_diagnostic
from .input_level import InputLevelControl
from .synthesis import DEFAULT_TTS_BACKEND, SpeechSynthesizer
from .task_gateway import TaskGateway
from .inspection_speech_events import prompt_for_event as inspection_prompt_for_event

LOG = logging.getLogger("rvclaw_voice")


def _optional_import(name: str) -> Optional[Any]:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


class _StreamingAsrCallback:
    """Forward only final SDK ASR events to the command worker."""

    def __init__(self, runtime: "SpeechRuntime"):
        self.runtime = runtime

    def on_open(self) -> None:
        LOG.debug("streaming ASR session opened")

    def on_event(self, result: Any) -> None:
        sentences = getattr(result, "sentences", ()) or ()
        is_final = bool(getattr(sentences[-1], "is_final", True)) if sentences else True
        if is_final and self.runtime._asr_stream_active:
            try:
                self.runtime._asr_result_queue.put_nowait(result)
                self.runtime._stream_final_count += 1
            except Full:
                LOG.warning("dropping final streaming ASR result because queue is full")
        else:
            LOG.debug("streaming ASR partial transcript=%r", getattr(result, "text", ""))

    def on_complete(self) -> None:
        LOG.debug("streaming ASR session completed")

    def on_error(self, result: Any) -> None:
        self.runtime._stream_failed = True
        LOG.warning("streaming ASR error: %s", getattr(result, "message", result))

    def on_close(self) -> None:
        LOG.debug("streaming ASR session closed")


class SpeechRuntime:
    """Realtime microphone, VAD and ASR pipeline for voice commands."""

    SAMPLE_RATE = 16000
    VAD_SAMPLES = 512
    # The K3 USB speaker advertises both rates, but the PortAudio(Portable
    # Audio) ALSA(Advanced Linux Sound Architecture) path can reject 48000 Hz
    # intermittently.  Try the alternate advertised rate before reporting a
    # playback failure.
    PLAYBACK_SAMPLE_RATE = 48000
    PLAYBACK_SAMPLE_RATES = (48000, 44100)
    PLAYBACK_CHANNELS = 2
    PLAYBACK_TIMEOUT_SECONDS = 3.0
    TTS_TIMEOUT_SECONDS = 8.0
    ASR_LOCK_TIMEOUT_SECONDS = 1.0
    DEFAULT_VAD_MIN_SILENCE_MS = 800
    RESPONSE_PROMPTS = {
        ACK_TEXT: "accepted.wav",
        FAILURE_TEXT: "failed.wav",
        "巡检开始": "inspection-started.wav",
        "巡检结束": "inspection-finished.wav",
        "小车已到达第一个巡检点": "arrived-a.wav",
        "小车已到达第二个巡检点": "arrived-b.wav",
        "小车已到达第三个巡检点": "arrived-c.wav",
        "当前没有任务": "status-idle.wav",
        "当前正在执行巡检任务": "status-inspection.wav",
        "当前正在执行左转任务": "status-left.wav",
        "当前正在执行仪表检测任务": "status-meter.wav",
        "当前正在执行安全帽检测任务": "status-hardhat.wav",
        "当前正在执行检测任务": "status-detection.wav",
        "仪表检测开始": "meter-started.wav",
        "安全帽检测开始": "hardhat-started.wav",
        "小车已停止": "stopped.wav",
        "左转完成": "left-finished.wav",
        "巡检任务已启动": "inspection-launched.wav",
        "小车执行成功": "succeeded.wav",
        "没有听清，请再说一次": "please-repeat.wav",
        "上次仪表检测已完成": "meter-finished.wav",
        "上次安全帽检测已完成": "hardhat-finished.wav",
        "上次检测结果已获取": "result-ready.wav",
        "建图已开始，请使用手柄低速巡视环境": "mapping-started.wav",
        "正在保存地图，请保持小车静止": "mapping-saving.wav",
        "地图保存成功": "mapping-saved.wav",
        "地图保存失败": "mapping-save-failed.wav",
        "当前没有正在进行的建图任务": "mapping-not-running.wav",
        "建图正在进行，请先结束当前建图": "mapping-already-running.wav",
        "小车仍在移动，请关闭手柄并保持静止后再次说建图结束": "mapping-not-stationary.wav",
        "建图已取消，地图没有保存": "mapping-cancelled.wav",
        "当前正在建图": "status-mapping.wav",
    }

    def __init__(
        self,
        *,
        router: Optional[CommandRouter] = None,
        capture_device: Optional[int] = None,
        player_device: Optional[int] = None,
    ):
        self.router = router or build_router()
        self.capture_device = capture_device
        self.player_device = player_device
        self.audio = _optional_import("spacemit_audio")
        self.vad = _optional_import("spacemit_vad")
        self.asr = _optional_import("spacemit_asr")
        self.tts = _optional_import("spacemit_tts")
        self._synthesizer = None
        self._running = False
        self._capture = None
        self._vad_engine = None
        self._asr_engine = None
        self._native_speech = None
        self._acoustic_config = os.getenv("RVCLAW_ACOUSTIC_CONFIG", "")
        self._asr_backend = os.getenv("RVCLAW_ASR_BACKEND", "sensevoice")
        if self._asr_backend not in {"sensevoice", "zipformer"}:
            raise ValueError("unsupported RVCLAW_ASR_BACKEND")
        self._asr_engine_lock = Lock()
        self._asr_stream_state_lock = Lock()
        self._asr_stream_lock_held = False
        self._player = None
        self._playback_lock = Lock()
        self._stop_event = Event()
        self._playback_event = Event()
        self._playback_generation = 0
        self._audio_queue: Queue[bytes] = Queue(maxsize=256)
        self._utterance_queue: Queue[bytes] = Queue(maxsize=4)
        self._asr_result_queue: Queue[Any] = Queue(maxsize=4)
        self._vad_thread: Optional[Thread] = None
        self._asr_thread: Optional[Thread] = None
        self._asr_streaming_requested = os.getenv("RVCLAW_ASR_STREAMING", "1").lower() in {
            "1", "true", "yes", "on"
        }
        self._asr_stream_active = False
        self._stream_failed = False
        self._stream_final_count = 0
        self._stream_expected_final_count = 0
        self._arrival_thread: Optional[Thread] = None
        self._arrival_event_dir = Path(
            os.getenv("RVCLAW_ARRIVAL_EVENT_DIR", "/data/RVClaw/runtime/waypoint-arrivals")
        )
        self._arrival_seen = set()
        self._command_threads = set()
        self._command_threads_lock = Lock()
        self._capture_name: Optional[str] = None
        self._fault: Optional[str] = None
        self._playback_timeout_seconds = float(
            os.getenv("RVCLAW_PLAYBACK_TIMEOUT_SECONDS", str(self.PLAYBACK_TIMEOUT_SECONDS))
        )
        self._tts_timeout_seconds = float(
            os.getenv("RVCLAW_TTS_TIMEOUT_SECONDS", str(self.TTS_TIMEOUT_SECONDS))
        )
        self._asr_lock_timeout_seconds = float(
            os.getenv("RVCLAW_ASR_LOCK_TIMEOUT_SECONDS", str(self.ASR_LOCK_TIMEOUT_SECONDS))
        )
        if min(
            self._playback_timeout_seconds,
            self._tts_timeout_seconds,
            self._asr_lock_timeout_seconds,
        ) <= 0:
            raise ValueError("voice playback, TTS and ASR lock timeouts must be positive")
        self._recognized_count = 0
        self._dropped_audio_chunks = 0
        self._speech_segments = 0
        self._last_input_rms = 0.0
        self._last_input_peak = 0.0
        self._pending_wake_until = 0.0
        self._input_gain = float(os.getenv("RVCLAW_INPUT_GAIN", "1.0"))
        self._auto_input_level = os.getenv("RVCLAW_AUTO_INPUT_LEVEL", "0") == "1"
        self._input_level = InputLevelControl()
        if not 1.0 <= self._input_gain <= 4.0:
            raise ValueError("RVCLAW_INPUT_GAIN must be between 1.0 and 4.0")
        self._vad_trigger_threshold = float(os.getenv("RVCLAW_VAD_TRIGGER_THRESHOLD", "0.35"))
        self._vad_stop_threshold = float(os.getenv("RVCLAW_VAD_STOP_THRESHOLD", "0.20"))
        self._vad_min_silence_ms = int(
            os.getenv("RVCLAW_VAD_MIN_SILENCE_MS", str(self.DEFAULT_VAD_MIN_SILENCE_MS))
        )
        self._vad_preroll_frames = int(os.getenv("RVCLAW_VAD_PREROLL_FRAMES", "40"))
        if not 0.0 < self._vad_stop_threshold <= self._vad_trigger_threshold <= 1.0:
            raise ValueError("VAD thresholds must satisfy 0 < stop <= trigger <= 1")
        if self._vad_min_silence_ms < 200:
            raise ValueError("RVCLAW_VAD_MIN_SILENCE_MS must be at least 200")
        if not 1 <= self._vad_preroll_frames <= 60:
            raise ValueError("RVCLAW_VAD_PREROLL_FRAMES must be between 1 and 60")
        prompt_dir = Path(
            os.getenv(
                "RVCLAW_VOICE_PROMPT_DIR",
                str(Path(__file__).resolve().parent.parent / "prompts"),
            )
        )
        self._prompt_paths = {
            text: prompt_dir / filename for text, filename in self.RESPONSE_PROMPTS.items()
        }

    def _select_asr_model_dir(self) -> str:
        configured = os.getenv("RVCLAW_ASR_MODEL_DIR")
        if configured:
            return os.path.expanduser(configured)
        return os.path.expanduser("~/.cache/models/asr/" + self._asr_backend)

    @staticmethod
    def _tts_model_dir() -> str:
        return os.path.expanduser(
            os.getenv("RVCLAW_TTS_MODEL_DIR", "~/.cache/models/tts/matcha-tts")
        )

    @staticmethod
    def _tts_backend() -> str:
        return os.getenv("RVCLAW_TTS_BACKEND", DEFAULT_TTS_BACKEND)

    def health(self) -> Dict[str, Any]:
        tts_model_dir = self._tts_model_dir()
        tts_ready = bool(self.tts) and Path(tts_model_dir).is_dir()
        required_ready = all((self.audio, self.vad, self.asr)) and tts_ready
        prompts_ready = all(path.is_file() for path in self._prompt_paths.values())
        return {
            "ready": required_ready and prompts_ready and self._fault is None,
            "devices": {
                "capture": self.capture_device,
                "capture_name": self._capture_name,
                "player": self.player_device,
            },
            "modules": {
                "spacemit_audio": bool(self.audio),
                "spacemit_vad": bool(self.vad),
                "spacemit_asr": bool(self.asr),
                "spacemit_tts": bool(self.tts),
            },
            "asr": {
                "model_dir": getattr(self, "_asr_model_dir", None),
                "backend": self._asr_backend,
                "acoustic_config": self._acoustic_config or None,
                "primary_backend": "sherpa-onnx-zipformer" if self._acoustic_config else self._asr_backend,
                "acoustic_process_alive": bool(self._native_speech and self._native_speech.process.poll() is None),
                "streaming_requested": self._asr_streaming_requested,
                "streaming_active": self._asr_stream_active,
            },
            "tts": {
                "model_dir": tts_model_dir,
                "ready": tts_ready,
                "backend": self._tts_backend().lower(),
                "process_isolated": True,
                "process_alive": bool(self._synthesizer and self._synthesizer.process
                                      and self._synthesizer.process.poll() is None),
            },
            "sample_format": "PCM16_LE/16000Hz/mono",
            "tts_playback": "try_48000Hz_then_44100Hz_stereo",
            "response_prompts_ready": prompts_ready,
            "running": self._running,
            "recognized_count": self._recognized_count,
            "dropped_audio_chunks": self._dropped_audio_chunks,
            "speech_segments": self._speech_segments,
            "last_input_rms": round(self._last_input_rms, 6),
            "last_input_peak": round(self._last_input_peak, 6),
            "audio_tuning": {
                "input_gain": self._input_gain,
                "auto_input_level": self._auto_input_level,
                "effective_input_gain": self._input_level.gain if self._auto_input_level else self._input_gain,
                "vad_trigger_threshold": self._vad_trigger_threshold,
                "vad_stop_threshold": self._vad_stop_threshold,
                "vad_min_silence_ms": self._vad_min_silence_ms,
                "vad_preroll_frames": self._vad_preroll_frames,
                "vad_preroll_ms": self._vad_preroll_frames * 32,
            },
            "response_timeouts": {
                "playback_seconds": self._playback_timeout_seconds,
                "tts_seconds": self._tts_timeout_seconds,
                "asr_lock_seconds": self._asr_lock_timeout_seconds,
            },
            "fault": self._fault,
        }

    def start(self) -> None:
        if self._running:
            return
        missing = [
            name
            for name, module in (
                ("audio", self.audio),
                ("vad", self.vad),
                ("asr", self.asr),
                ("tts", self.tts),
            )
            if module is None
        ]
        if missing:
            raise RuntimeError("missing speech runtime: " + ", ".join(missing))
        tts_model_dir = self._tts_model_dir()
        if not Path(tts_model_dir).is_dir():
            raise RuntimeError("missing TTS model directory: " + tts_model_dir)
        missing_prompts = [str(path) for path in self._prompt_paths.values() if not path.is_file()]
        if missing_prompts:
            raise RuntimeError("missing voice response prompts: " + ", ".join(missing_prompts))

        devices = self.audio.AudioCapture.list_devices()
        requested = self.capture_device
        configured = os.getenv("RVCLAW_CAPTURE_DEVICE")
        if configured is not None:
            requested = int(configured)
        self.capture_device, self._capture_name = self.select_capture_device(devices, requested)
        player_devices = self.audio.AudioPlayer.list_devices()
        requested_player = self.player_device
        configured_player = os.getenv("RVCLAW_PLAYER_DEVICE")
        if configured_player is not None:
            requested_player = int(configured_player)
        self.player_device, _ = self.select_player_device(player_devices, requested_player)

        self._fault = None
        self._stop_event.clear()
        self._audio_queue = Queue(maxsize=256)
        self._utterance_queue = Queue(maxsize=4)
        self._asr_result_queue = Queue(maxsize=4)
        capture_chunk_bytes = max(
            self.VAD_SAMPLES * 2,
            int(os.getenv("RVCLAW_CAPTURE_CHUNK_BYTES", "3200")),
        )

        try:
            self._synthesizer = SpeechSynthesizer(tts_model_dir, self._tts_backend())
            LOG.info("speech synthesizer ready process_isolated=true")
            vad_config = (
                self.vad.VadConfig.preset("silero")
                .with_trigger_threshold(self._vad_trigger_threshold)
                .with_stop_threshold(self._vad_stop_threshold)
                .with_min_silence_duration(self._vad_min_silence_ms)
            )
            self._vad_engine = self.vad.VadEngine(vad_config)

            model_dir = self._select_asr_model_dir()
            self._asr_model_dir = model_dir
            asr_config = self.asr.Config(model_dir, backend=self._asr_backend)
            asr_config.language = self.asr.Language.ZH
            asr_config.punctuation_enabled = True
            asr_config.provider = os.getenv("RVCLAW_ASR_PROVIDER", "spacemit")
            self._asr_engine = self.asr.Engine(asr_config).initialize()
            LOG.info("recognition verifier backend=%s model=%s provider=%s", self._asr_backend, model_dir, asr_config.provider)
            if self._acoustic_config:
                self._native_speech = NativeSpeech(self._acoustic_config)
                LOG.info("acoustic frontend ready config=%s primary=sherpa-onnx-zipformer wake=hello_car", self._acoustic_config)
            self._asr_stream_active = False
            self._stream_failed = False
            self.audio.init(
                sample_rate=self.SAMPLE_RATE,
                channels=1,
                chunk_size=capture_chunk_bytes,
                capture_device=self.capture_device,
            )
            self._capture = self.audio.AudioCapture(self.capture_device)
            self._capture.set_callback(self._on_audio)

            self._running = True
            self._vad_thread = Thread(target=self._vad_loop, name="rvclaw-vad", daemon=True)
            self._asr_thread = Thread(target=self._asr_loop, name="rvclaw-asr", daemon=True)
            self._vad_thread.start()
            self._asr_thread.start()
            self._arrival_event_dir.mkdir(parents=True, exist_ok=True)
            self._arrival_seen = {path.name for path in self._arrival_event_dir.glob("*.json")}
            self._arrival_thread = Thread(target=self._arrival_loop, name="rvclaw-arrivals", daemon=True)
            self._arrival_thread.start()
            if not self._capture.start(self.SAMPLE_RATE, 1, capture_chunk_bytes):
                raise RuntimeError("failed to start microphone capture")
            LOG.info(
                "voice listener started capture_device=%s capture_name=%r wake_phrase=%r",
                self.capture_device,
                self._capture_name,
                "小微小微|你好，小车",
            )
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        self._stop_event.set()
        capture = self._capture
        self._capture = None
        if capture is not None:
            try:
                capture.stop()
            finally:
                capture.close()
        self._abort_stream("shutdown")

        for thread in (self._vad_thread, self._asr_thread, self._arrival_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=30)
        self._vad_thread = None
        self._asr_thread = None
        self._arrival_thread = None
        if self._native_speech is not None:
            self._native_speech.close()
            self._native_speech = None
        with self._command_threads_lock:
            command_threads = list(self._command_threads)
            self._command_threads.clear()
        for thread in command_threads:
            if thread.is_alive():
                thread.join(timeout=2)

        if self._synthesizer is not None:
            self._synthesizer.close()
            self._synthesizer = None

        if self._asr_engine is not None:
            self._asr_engine.shutdown()
            self._asr_engine = None
        self._player = None
        self._vad_engine = None
        self._running = False

    @staticmethod
    def select_capture_device(
        devices: Iterable[Sequence[Any]], requested: Optional[int] = None
    ) -> Tuple[int, str]:
        available = [(int(device[0]), str(device[1])) for device in devices]
        if not available:
            raise RuntimeError("no microphone capture device found")
        if requested is not None:
            for index, name in available:
                if index == requested:
                    return index, name
            raise RuntimeError(f"configured capture device {requested} is unavailable")
        for preferred in ("XFM-DP", "ASTRA Pro"):
            for index, name in available:
                if preferred.lower() in name.lower():
                    return index, name
        return available[0]

    @staticmethod
    def select_player_device(
        devices: Iterable[Sequence[Any]], requested: Optional[int] = None
    ) -> Tuple[int, str]:
        available = [(int(device[0]), str(device[1])) for device in devices]
        if not available:
            raise RuntimeError("no speaker playback device found")
        if requested is not None:
            for index, name in available:
                if index == requested:
                    return index, name
            raise RuntimeError(f"configured player device {requested} is unavailable")
        for preferred in ("USB Audio Device", "pipewire", "default"):
            for index, name in available:
                if preferred.lower() in name.lower():
                    return index, name
        return available[0]

    @staticmethod
    def result_confidence(sentences: Iterable[Any]) -> float:
        scores = [
            max(0.0, min(1.0, float(sentence.confidence)))
            for sentence in sentences
            if getattr(sentence, "confidence", None) is not None
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def _on_audio(self, data: bytes) -> None:
        if self._stop_event.is_set() or self._playback_event.is_set():
            return
        try:
            self._audio_queue.put_nowait(bytes(data))
        except Full:
            self._dropped_audio_chunks += 1

    def _vad_loop(self) -> None:
        if getattr(self, "_native_speech", None) is not None:
            self._acoustic_vad_loop()
            return
        frame_bytes = self.VAD_SAMPLES * 2
        pending = bytearray()
        pre_roll = deque(maxlen=self._vad_preroll_frames)
        utterance = []
        in_speech = False
        max_frames = int(
            float(os.getenv("RVCLAW_MAX_UTTERANCE_SECONDS", "12"))
            * self.SAMPLE_RATE
            / self.VAD_SAMPLES
        )
        min_frames = int(
            float(os.getenv("RVCLAW_MIN_UTTERANCE_SECONDS", "0.35"))
            * self.SAMPLE_RATE
            / self.VAD_SAMPLES
        )

        try:
            while not self._stop_event.is_set() or not self._audio_queue.empty():
                if self._playback_event.is_set():
                    self._abort_stream("playback")
                    pending.clear()
                    pre_roll.clear()
                    utterance = []
                    in_speech = False
                    self._vad_engine.reset()
                    time.sleep(0.05)
                    continue
                try:
                    pending.extend(self._audio_queue.get(timeout=0.2))
                except Empty:
                    continue
                while len(pending) >= frame_bytes:
                    if self._playback_event.is_set():
                        self._abort_stream("playback")
                        pending.clear()
                        pre_roll.clear()
                        utterance = []
                        in_speech = False
                        self._vad_engine.reset()
                        break
                    frame = self.apply_input_gain(bytes(pending[:frame_bytes]), self._input_gain)
                    del pending[:frame_bytes]
                    samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
                    self._last_input_rms = float(np.sqrt(np.mean(samples * samples)))
                    self._last_input_peak = float(np.max(np.abs(samples)))
                    result = self._vad_engine.detect(samples, self.SAMPLE_RATE)

                    if not in_speech:
                        pre_roll.append(frame)
                        if result.is_speech_start or result.is_speech:
                            utterance = list(pre_roll)
                            pre_roll.clear()
                            in_speech = True
                            self._speech_segments += 1
                            LOG.info(
                                "speech started probability=%.3f preroll_frames=%d",
                                float(getattr(result, "smoothed_probability", getattr(result, "probability", 0.0))),
                                self._vad_preroll_frames,
                            )
                            self._stream_begin()
                            for buffered in utterance:
                                self._stream_send(buffered)
                    else:
                        utterance.append(frame)
                        self._stream_send(frame)
                        if result.is_speech_end or len(utterance) >= max_frames:
                            stream_ok = self._stream_end()
                            if len(utterance) >= min_frames and not stream_ok:
                                self._enqueue_utterance(b"".join(utterance))
                            utterance = []
                            in_speech = False
                            self._vad_engine.reset()
                            LOG.info("speech ended probability=%.3f streaming=%s", float(getattr(result, "smoothed_probability", getattr(result, "probability", 0.0))), stream_ok)
            if in_speech and len(utterance) >= min_frames:
                if not self._stream_end():
                    self._enqueue_utterance(b"".join(utterance))
        except Exception as exc:
            self._set_fault("vad", exc)

    def _enqueue_utterance(self, audio: bytes) -> None:
        try:
            self._utterance_queue.put_nowait(audio)
        except Full:
            LOG.warning("dropping utterance because ASR queue is full")

    def _acoustic_vad_loop(self) -> None:
        native = self._native_speech
        self._input_level.reset()
        LOG.info("input level control automatic=%s fixed_gain=%.2f", self._auto_input_level, self._input_gain)
        gate = AcousticGate()
        pending = bytearray()
        preroll = deque(maxlen=self._vad_preroll_frames)
        history = deque(maxlen=375)
        received_samples = 0
        segment_start = 0
        segment_wake = None
        consumed_until = 0
        diagnostic_prefix = b""
        frames = []
        active = False
        paused = False
        generation = self._playback_generation
        dropped = self._dropped_audio_chunks
        partial = ""
        silence_samples = 0
        max_frames = int(12 * self.SAMPLE_RATE / self.VAD_SAMPLES)
        try:
            while not self._stop_event.is_set():
                if generation != self._playback_generation or dropped != self._dropped_audio_chunks:
                    native.request("R")
                    gate.reset()
                    self._input_level.reset()
                    pending.clear()
                    preroll.clear()
                    history.clear()
                    received_samples = 0
                    segment_wake = None
                    consumed_until = 0
                    diagnostic_prefix = b""
                    frames = []
                    active = False
                    self._vad_engine.reset()
                    generation = self._playback_generation
                    dropped = self._dropped_audio_chunks
                    LOG.info("acoustic session reset playback_generation=%d dropped_audio_chunks=%d", generation, dropped)
                if self._playback_event.is_set():
                    if not paused:
                        native.request("R")
                        gate.reset()
                        self._input_level.reset()
                        pending.clear()
                        preroll.clear()
                        history.clear()
                        received_samples = 0
                        segment_wake = None
                        consumed_until = 0
                        diagnostic_prefix = b""
                        frames = []
                        active = False
                        self._vad_engine.reset()
                        paused = True
                    self._stop_event.wait(0.02)
                    continue
                paused = False
                if not active and self._audio_queue.empty() and gate.expire():
                    LOG.info("acoustic wake expired without a command; requesting repeat")
                    self._enqueue_utterance(AcousticUtterance(b"", "", "", {"reason": "command_timeout"}))
                try:
                    pending.extend(self._audio_queue.get(timeout=0.2))
                except Empty:
                    continue
                while len(pending) >= self.VAD_SAMPLES * 2:
                    if self._playback_event.is_set() or self._stop_event.is_set() or generation != self._playback_generation:
                        break
                    raw_frame = bytes(pending[:self.VAD_SAMPLES * 2])
                    frame = (self._input_level.process(raw_frame) if self._auto_input_level
                             else self.apply_input_gain(raw_frame, self._input_gain))
                    del pending[:self.VAD_SAMPLES * 2]
                    history.append(frame)
                    received_samples += self.VAD_SAMPLES
                    wake_event = native.request("K", frame).get("wake")
                    samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
                    self._last_input_rms = float(np.sqrt(np.mean(samples * samples)))
                    self._last_input_peak = float(np.max(np.abs(samples)))
                    vad = self._vad_engine.detect(samples, self.SAMPLE_RATE)
                    if wake_event and segment_wake is None:
                        history_start = received_samples - len(history) * self.VAD_SAMPLES
                        # Keyword timestamps restart after the backend's internal
                        # silence reset. Recover context on our own sample clock;
                        # the gate validates the complete prefix before dispatch.
                        desired_start = max(consumed_until, received_samples - 4 * self.SAMPLE_RATE)
                        if active:
                            desired_start = max(consumed_until, min(desired_start, segment_start))
                        offset = max(0, (desired_start - history_start) // self.VAD_SAMPLES)
                        replay_start = history_start + offset * self.VAD_SAMPLES
                        restart_recognizer = not active or replay_start != segment_start
                        frames = list(history)[offset:]
                        segment_start = replay_start
                        segment_wake = dict(wake_event)
                        segment_wake["continuous"] = True
                        diagnostic_prefix = b""
                        if restart_recognizer:
                            native.request("C")
                            for buffered in frames:
                                native.request("D", buffered)
                        else:
                            native.request("D", frame)
                        if not active:
                            started_at = time.monotonic()
                            self._speech_segments += 1
                        active = True
                        silence_samples = 0
                        partial = ""
                        preroll.clear()
                        LOG.info("continuous acoustic wake detected recovered_frames=%d", len(frames))
                    elif not active:
                        preroll.append(frame)
                        if not (vad.is_speech_start or vad.is_speech):
                            continue
                        native.request("C")
                        frames = list(preroll)
                        segment_start = received_samples - len(frames) * self.VAD_SAMPLES
                        segment_wake = None
                        diagnostic_prefix = b"".join(list(history)[:-len(frames)])
                        preroll.clear()
                        active = True
                        silence_samples = 0
                        started_at = time.monotonic()
                        partial = ""
                        self._speech_segments += 1
                        LOG.info("speech started acoustic=true preroll_frames=%d", len(frames))
                        for buffered in frames:
                            native.request("D", buffered)
                    else:
                        frames.append(frame)
                        interim = native.request("D", frame)
                        text = interim.get("asr", {}).get("text", "")
                        if text and text != partial:
                            partial = text
                            LOG.info("acoustic partial transcript=%r", text)
                    # The installed Silero backend emits speech_end immediately
                    # below its stop threshold, ignoring min_silence_duration_ms.
                    # Count audio samples here so short pauses retain one stream.
                    probability = getattr(vad, "smoothed_probability", None)
                    quiet = (probability < self._vad_stop_threshold if probability is not None
                             else vad.is_speech_end or not vad.is_speech)
                    silence_samples = silence_samples + self.VAD_SAMPLES if quiet else 0
                    silence_complete = silence_samples * 1000 >= self._vad_min_silence_ms * self.SAMPLE_RATE
                    if silence_complete or len(frames) >= max_frames:
                        final_started = time.monotonic()
                        result = native.request("F")
                        LOG.info("voice timing final_decode_seconds=%.3f silence_audio_seconds=%.3f queued_audio_chunks=%d",
                                 time.monotonic() - final_started, silence_samples / self.SAMPLE_RATE,
                                 self._audio_queue.qsize())
                        result["wake"] = segment_wake
                        if generation != self._playback_generation or self._playback_event.is_set() or self._stop_event.is_set():
                            gate.reset()
                        elif len(frames) >= max_frames:
                            gate.reset()
                            LOG.warning("acoustic utterance rejected: maximum duration")
                        else:
                            utterance = gate.finish(result, b"".join(frames), started_at=started_at)
                            LOG.info("acoustic final raw=%r wake=%s command=%r", result.get("asr", {}).get("text"), result.get("wake"), utterance.transcript if utterance else None)
                            self._record_acoustic_diagnostic(result, frames, diagnostic_prefix, utterance)
                            if utterance is not None:
                                self._enqueue_utterance(utterance)
                        if segment_wake:
                            consumed_until = received_samples
                        active = False
                        frames = []
                        segment_wake = None
                        self._vad_engine.reset()
        except Exception as exc:
            self._set_fault("acoustic", exc)

    def _record_acoustic_diagnostic(self, result, frames, prefix, utterance):
        raw = result.get("asr", {}).get("text", "")
        samples = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768
        LOG.info("acoustic segment seconds=%.3f rms=%.5f peak=%.5f preroll_limit_ms=%d",
                 len(samples) / self.SAMPLE_RATE, float(np.sqrt(np.mean(samples * samples))),
                 float(np.max(np.abs(samples))), self._vad_preroll_frames * 32)
        compact = "".join(c for c in raw if c.isalnum())
        suspected_miss = result.get("wake") is None and (
            is_complete_command(command_text(raw)) or compact.startswith("你好小")
            or compact in {"好小车", "小车"}
        )
        directory = os.getenv("RVCLAW_VOICE_DIAGNOSTIC_DIR")
        if not directory or (utterance is None and not suspected_miss and not result.get("wake")):
            return
        if utterance is None:
            pcm = b"".join(frames)
            utterance = AcousticUtterance(pcm, "", raw,
                {"rejected": True, "reason": "command_missing" if result.get("wake") else "wake_missing",
                 "detected_wake": result.get("wake"), "context_seconds": len(prefix) / 32000},
                prefix + pcm)
        try:
            utterance.wake = dict(utterance.wake, input_level_mode="automatic" if self._auto_input_level else "fixed",
                                  fixed_input_gain=self._input_gain)
            save_diagnostic(directory, utterance)
        except OSError:
            LOG.exception("failed to save acoustic diagnostic")

    def _verify_acoustic_utterance(self, utterance: AcousticUtterance) -> Optional[Any]:
        if not is_complete_command(utterance.transcript):
            LOG.warning("acoustic command rejected: incomplete or unknown text=%r", utterance.transcript)
            return None
        verification_started = time.monotonic()
        verification = self._recognize_whole_utterance(utterance.pcm)
        LOG.info("voice timing verification_seconds=%.3f audio_seconds=%.3f command=%r",
                 time.monotonic() - verification_started, len(utterance.pcm) / (2 * self.SAMPLE_RATE),
                 utterance.transcript)
        if verification is None:
            return None
        if utterance.wake.get("continuous"):
            verified = acoustic_command_text(verification.text)
            if verified is None:
                LOG.warning("continuous wake rejected: verifier prefix unavailable text=%r", verification.text)
                return None
        else:
            verified = command_text(verification.text)
        if verified != utterance.transcript:
            LOG.warning("acoustic command rejected: models disagree primary=%r verifier=%r", utterance.transcript, verified)
            return None
        LOG.info("acoustic command verified primary=%r verifier=%r confidence_source=%s_backend_uncalibrated", utterance.transcript, verified, self._asr_backend)
        return SimpleNamespace(text="你好，小车，" + utterance.transcript, sentences=verification.sentences)

    def _get_asr_stream_state_lock(self) -> Lock:
        lock = getattr(self, "_asr_stream_state_lock", None)
        if lock is None:
            lock = Lock()
            self._asr_stream_state_lock = lock
        return lock

    def _stream_begin(self) -> None:
        with self._get_asr_stream_state_lock():
            self._asr_stream_active = False
            self._stream_failed = False
            self._stream_expected_final_count = self._stream_final_count + 1
            if not self._asr_streaming_requested or self._asr_engine is None:
                return
            if not self._asr_engine_lock.acquire(blocking=False):
                LOG.warning("streaming ASR busy; using whole-utterance fallback")
                return
            self._asr_stream_lock_held = True
            try:
                callback_type = getattr(self.asr, "AsrCallback", None)
                if callback_type is None:
                    raise RuntimeError("SDK does not expose AsrCallback")
                handler = _StreamingAsrCallback(self)

                class SdkStreamingAsrCallback(callback_type):
                    def __init__(self) -> None:
                        super().__init__()

                    def on_open(self) -> None:
                        handler.on_open()

                    def on_event(self, result: Any) -> None:
                        handler.on_event(result)

                    def on_complete(self) -> None:
                        handler.on_complete()

                    def on_error(self, result: Any) -> None:
                        handler.on_error(result)

                    def on_close(self) -> None:
                        handler.on_close()

                self._asr_stream_active = True
                self._asr_engine.start(callback=SdkStreamingAsrCallback())
            except Exception as exc:
                self._stream_failed = True
                self._asr_stream_active = False
                LOG.warning("streaming ASR unavailable; using whole-utterance fallback: %s", exc)
                if self._asr_stream_lock_held:
                    self._asr_engine_lock.release()
                    self._asr_stream_lock_held = False

    def _stream_send(self, frame: bytes) -> None:
        with self._get_asr_stream_state_lock():
            if not self._asr_stream_active or self._stream_failed:
                return
            try:
                self._asr_engine.send_audio_frame(frame)
            except Exception as exc:
                self._stream_failed = True
                LOG.warning("streaming ASR frame failed; using fallback: %s", exc)

    def _stream_end(self) -> bool:
        with self._get_asr_stream_state_lock():
            if not self._asr_stream_active:
                return False
            try:
                self._asr_engine.flush()
                deadline = time.monotonic() + 0.6
                while (
                    self._stream_final_count < self._stream_expected_final_count
                    and time.monotonic() < deadline
                    and not self._stream_failed
                ):
                    time.sleep(0.02)
                return (
                    not self._stream_failed
                    and self._stream_final_count >= self._stream_expected_final_count
                )
            except Exception as exc:
                LOG.warning("streaming ASR flush failed; using fallback: %s", exc)
                self._stream_failed = True
                return False
            finally:
                self._asr_stream_active = False
                try:
                    self._asr_engine.stop()
                except Exception:
                    LOG.exception("failed to stop streaming ASR session")
                if self._asr_stream_lock_held:
                    self._asr_engine_lock.release()
                    self._asr_stream_lock_held = False

    def _abort_stream(self, reason: str) -> bool:
        """Stop an unfinished streaming session without producing a command."""
        with self._get_asr_stream_state_lock():
            was_active = bool(getattr(self, "_asr_stream_active", False))
            lock_held = bool(getattr(self, "_asr_stream_lock_held", False))
            if not was_active and not lock_held:
                return False
            self._asr_stream_active = False
            self._stream_failed = True
            try:
                if was_active and self._asr_engine is not None:
                    self._asr_engine.stop()
            except Exception:
                LOG.exception("failed to abort streaming ASR session reason=%s", reason)
            finally:
                if self._asr_stream_lock_held:
                    self._asr_engine_lock.release()
                    self._asr_stream_lock_held = False
            LOG.info("streaming ASR session aborted reason=%s", reason)
            return True

    def _recognize_whole_utterance(self, pcm: bytes) -> Optional[Any]:
        timeout = float(
            getattr(self, "_asr_lock_timeout_seconds", self.ASR_LOCK_TIMEOUT_SECONDS)
        )
        if not self._asr_engine_lock.acquire(timeout=max(0.01, timeout)):
            LOG.warning(
                "dropping whole utterance because ASR remained busy timeout_seconds=%.2f",
                timeout,
            )
            return None
        try:
            audio = np.frombuffer(pcm, dtype=np.int16).copy()
            return self._asr_engine.recognize(audio)
        finally:
            self._asr_engine_lock.release()

    def _arrival_loop(self) -> None:
        """Speak only atomic PASS events emitted after waypoint safety checks."""
        while not self._stop_event.is_set():
            try:
                for path in sorted(self._arrival_event_dir.glob("*.json")):
                    if path.name in self._arrival_seen:
                        continue
                    try:
                        event = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        continue
                    self._arrival_seen.add(path.name)
                    response = self.event_response(event)
                    if response is None:
                        continue
                    if self._speak(response):
                        try:
                            path.unlink()
                        except OSError:
                            LOG.exception("failed to remove handled waypoint arrival event path=%s", path)
                        LOG.info(
                            "verified waypoint arrival announced response=%s event=%s",
                            response,
                            path.name,
                        )
                    else:
                        LOG.error(
                            "verified waypoint arrival announcement failed response=%s event=%s",
                            response,
                            path.name,
                        )
            except Exception:
                LOG.exception("waypoint arrival watcher failed")
            self._stop_event.wait(0.2)

    @staticmethod
    def event_response(event: Dict[str, Any]) -> Optional[str]:
        prompt = inspection_prompt_for_event(event)
        if prompt is not None:
            return prompt.text
        if event.get("event") == "inspection_started" and event.get("status") == "PASS":
            return "巡检开始"
        if event.get("event") == "inspection_finished" and event.get("status") == "PASS":
            return "巡检结束"
        if event.get("event") == "inspection_failed" and event.get("status") == "FAIL":
            return FAILURE_TEXT
        return SpeechRuntime.arrival_response(event)

    @staticmethod
    def arrival_response(event: Dict[str, Any]) -> Optional[str]:
        if (
            event.get("event") != "waypoint_arrived"
            or event.get("status") != "PASS"
            or str(event.get("waypoint", "")).upper() not in {"A", "B", "C"}
        ):
            return None
        return "小车已到达%s点" % str(event["waypoint"]).upper()

    @staticmethod
    def apply_input_gain(data: bytes, gain: float) -> bytes:
        if gain == 1.0:
            return data
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        return np.clip(samples * gain, -32768, 32767).astype(np.int16).tobytes()

    def prepare_transcript(self, transcript: str, now: Optional[float] = None) -> Optional[str]:
        """Join a wake-only segment to one natural follow-up command."""
        current = time.monotonic() if now is None else now
        command_text = strip_wake_phrase(transcript)
        if command_text is not None:
            self._pending_wake_until = 0.0
            if not command_text.strip(" ，,。.!！啊呀呢"):
                self._pending_wake_until = current + float(
                    os.getenv("RVCLAW_WAKE_CONTINUATION_SECONDS", "3.0")
                )
                return None
            return transcript
        wake_fragment = "".join(character for character in transcript if character.isalnum())
        if wake_fragment in {
            "小车",
            "小车啊",
            "你好小车",
            "你好小车啊",
            "小微小微",
            "小微小微啊",
            "你好小程啊",
            "你好潇什啊",
        }:
            self._pending_wake_until = current + float(
                os.getenv("RVCLAW_WAKE_CONTINUATION_SECONDS", "3.0")
            )
            return None
        if current <= self._pending_wake_until:
            self._pending_wake_until = 0.0
            return "你好，小车，" + transcript
        self._pending_wake_until = 0.0
        return transcript

    def _asr_loop(self) -> None:
        try:
            while (
                not self._stop_event.is_set()
                or not self._utterance_queue.empty()
                or not self._asr_result_queue.empty()
            ):
                early_acknowledged = None
                try:
                    result = self._asr_result_queue.get_nowait()
                except Empty:
                    try:
                        pcm = self._utterance_queue.get(timeout=0.2)
                    except Empty:
                        continue
                    if isinstance(pcm, AcousticUtterance) and self._stop_event.is_set():
                        continue
                    if isinstance(pcm, AcousticUtterance) and is_complete_command(pcm.transcript):
                        # Receipt is audible while the immutable final audio is
                        # verified. Dispatch still waits for both operations.
                        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="rvclaw-verifier") as verifier:
                            verification = verifier.submit(self._verify_acoustic_utterance, pcm)
                            early_acknowledged = self._speak(ACK_TEXT)
                            result = verification.result()
                        if self._stop_event.is_set():
                            continue
                    else:
                        result = (
                            self._verify_acoustic_utterance(pcm)
                            if isinstance(pcm, AcousticUtterance)
                            else self._recognize_whole_utterance(pcm)
                        )
                    if result is None:
                        if isinstance(pcm, AcousticUtterance) and not self._stop_event.is_set():
                            self._speak("没有听清，请再说一次")
                        continue
                transcript = result.text.strip()
                if not transcript:
                    continue
                transcript = self.prepare_transcript(transcript)
                if transcript is None:
                    LOG.info("recognized wake phrase; waiting for command continuation")
                    continue
                confidence = self.result_confidence(result.sentences)
                self._recognized_count += 1
                if self.should_acknowledge(transcript, confidence, self.router):
                    # The acknowledgement is deliberately finished before work
                    # starts, so the spoken contract has a deterministic order.
                    acknowledged = (self._speak(ACK_TEXT) if early_acknowledged is None
                                    else early_acknowledged)
                    command = parse_transcript(transcript, confidence)
                    # A silent motion command is unsafe: the operator must hear
                    # the acknowledgement before the wheels can be armed.  For
                    # read-only commands, still dispatch after an audio fault so
                    # a transient speaker error does not discard the request.
                    if acknowledged or (command is not None and command.intent not in MOTION_INTENTS):
                        self._dispatch_in_background(transcript, confidence)
                    else:
                        LOG.error(
                            "motion command was not dispatched because acknowledgement failed transcript=%r",
                            transcript,
                        )
                        self._speak(FAILURE_TEXT)
                else:
                    command_result = self.router.dispatch_transcript(transcript, confidence)
                    # Only a recognized command that included the wake phrase gets
                    # an audible response. Ambient speech remains silent.
                    response_text = self.audible_response(transcript, command_result)
                    if response_text:
                        self._speak(response_text)
                    self._log_command(transcript, confidence, command_result)
        except Exception as exc:
            self._set_fault("asr", exc)

    @staticmethod
    def should_acknowledge(
        transcript: str, confidence: float, router: CommandRouter
    ) -> bool:
        command = parse_transcript(transcript, confidence)
        return command is not None

    def _dispatch_in_background(self, transcript: str, confidence: float) -> None:
        def run() -> None:
            try:
                command_result = self.router.dispatch_transcript(transcript, confidence)
                if (
                    not command_result.get("accepted")
                    and command_result.get("status") != "confirmation_required"
                    and not self._stop_event.is_set()
                ):
                    self._speak(FAILURE_TEXT)
                elif command_result.get("follow_up_text") and not self._stop_event.is_set():
                    self._speak(command_result["follow_up_text"])
                self._log_command(transcript, confidence, command_result)
            except Exception:
                LOG.exception("background command dispatch failed transcript=%r", transcript)
                if not self._stop_event.is_set():
                    self._speak(FAILURE_TEXT)
            finally:
                with self._command_threads_lock:
                    self._command_threads.discard(thread)

        thread = Thread(target=run, name="rvclaw-command", daemon=True)
        with self._command_threads_lock:
            self._command_threads.add(thread)
        thread.start()

    @staticmethod
    def _log_command(transcript: str, confidence: float, command_result: Dict[str, Any]) -> None:
        LOG.info(
            "recognized transcript=%r confidence=%.3f command=%s",
            transcript,
            confidence,
            json.dumps(command_result, ensure_ascii=False, sort_keys=True),
        )

    @staticmethod
    def audible_response(transcript: str, command_result: Dict[str, Any]) -> Optional[str]:
        if strip_wake_phrase(transcript) is None:
            return None
        return command_result.get("response_text")

    def _playback_devices(self) -> Sequence[int]:
        """Return the configured device first, followed by current devices."""
        candidates = []
        if self.player_device is not None:
            candidates.append(int(self.player_device))
        try:
            devices = self.audio.AudioPlayer.list_devices()
        except Exception as exc:
            LOG.warning("failed to refresh speaker devices: %s", exc)
            devices = ()
        for device in devices or ():
            try:
                index = int(device[0])
            except (TypeError, ValueError, IndexError):
                continue
            if index not in candidates:
                candidates.append(index)
        return tuple(candidates)

    def _player_call_with_timeout(self, player: Any, method_name: str) -> bool:
        """Run player cleanup without allowing a broken binding to hang the listener."""

        method = getattr(player, method_name, None)
        if method is None:
            return True
        outcome: Dict[str, Any] = {"error": None}

        def call() -> None:
            try:
                method()
            except Exception as exc:
                outcome["error"] = exc

        worker = Thread(target=call, name="rvclaw-player-%s" % method_name, daemon=True)
        worker.start()
        timeout = float(getattr(self, "_playback_timeout_seconds", self.PLAYBACK_TIMEOUT_SECONDS))
        worker.join(timeout=max(0.01, timeout))
        if worker.is_alive():
            LOG.error(
                "speaker player %s timed out timeout_seconds=%.2f",
                method_name,
                timeout,
            )
            return False
        if outcome["error"] is not None:
            raise outcome["error"]
        return True

    def _play_file_with_timeout(self, player: Any, playback_path: Path) -> bool:
        """Keep a blocking audio binding call from stopping command recognition."""

        outcome: Dict[str, Any] = {"value": False, "error": None}

        def play() -> None:
            try:
                outcome["value"] = bool(player.play_file(str(playback_path)))
            except Exception as exc:
                outcome["error"] = exc

        worker = Thread(target=play, name="rvclaw-playback-call", daemon=True)
        worker.start()
        timeout = float(getattr(self, "_playback_timeout_seconds", self.PLAYBACK_TIMEOUT_SECONDS))
        worker.join(timeout=max(0.01, timeout))
        if worker.is_alive():
            LOG.error(
                "speaker playback timed out path=%s timeout_seconds=%.2f",
                playback_path,
                timeout,
            )
            try:
                self._player_call_with_timeout(player, "stop")
            except Exception:
                LOG.warning("failed to stop timed-out speaker playback", exc_info=True)
            return False
        if outcome["error"] is not None:
            raise outcome["error"]
        return bool(outcome["value"])

    def _speak(self, text: str) -> bool:
        """Play a short response in half-duplex mode and resume listening."""
        response_started = time.monotonic()
        if self.audio is None:
            LOG.warning("voice response unavailable: audio player is missing")
            return False
        prompt_path = self._prompt_paths.get(text)
        temporary_prompt: Optional[Path] = None
        if prompt_path is None:
            temporary_prompt = self._synthesize_prompt(text)
            if temporary_prompt is None:
                return False
            prompt_path = temporary_prompt
        elif not prompt_path.is_file():
            LOG.error("voice response prompt is missing text=%r path=%s", text, prompt_path)
            return False

        try:
            with self._playback_lock:
                capture = self._capture
                capture_was_running = bool(capture is not None and capture.is_running())
                self._playback_event.set()
                self._playback_generation = getattr(self, "_playback_generation", 0) + 1
                self._abort_stream("playback")
                try:
                    if capture_was_running:
                        capture.stop()
                    while not self._audio_queue.empty():
                        try:
                            self._audio_queue.get_nowait()
                        except Empty:
                            break
                    retries = max(1, int(os.getenv("RVCLAW_PLAYBACK_RETRIES", "2")))
                    devices = self._playback_devices()
                    if not devices:
                        LOG.error("no speaker devices available while speaking text=%r", text)
                        return False
                    for attempt in range(1, retries + 1):
                        for device in devices:
                            for sample_rate in self.PLAYBACK_SAMPLE_RATES:
                                converted_prompt: Optional[Path] = None
                                player = None
                                try:
                                    playback_path, converted_prompt = self._prepare_playback_prompt(
                                        prompt_path, sample_rate=sample_rate
                                    )
                                    player = self.audio.AudioPlayer(device)
                                    self._player = player
                                    LOG.info("voice timing playback_call_seconds=%.3f text=%r device=%s sample_rate=%s",
                                             time.monotonic() - response_started, text, device, sample_rate)
                                    played = self._play_file_with_timeout(player, playback_path)
                                    if played:
                                        self.player_device = device
                                        LOG.info(
                                            "voice response played text=%r device=%s sample_rate=%s attempt=%s",
                                            text,
                                            device,
                                            sample_rate,
                                            attempt,
                                        )
                                        return True
                                    LOG.warning(
                                        "speaker rejected response text=%r device=%s sample_rate=%s attempt=%s",
                                        text,
                                        device,
                                        sample_rate,
                                        attempt,
                                    )
                                except Exception as exc:
                                    LOG.warning(
                                        "voice response attempt failed text=%r device=%s sample_rate=%s attempt=%s error=%s",
                                        text,
                                        device,
                                        sample_rate,
                                        attempt,
                                        exc,
                                    )
                                finally:
                                    if player is not None:
                                        try:
                                            self._player_call_with_timeout(player, "close")
                                        except Exception:
                                            LOG.exception("speaker cleanup failed device=%s", device)
                                    self._player = None
                                    if converted_prompt is not None:
                                        converted_prompt.unlink(missing_ok=True)
                    LOG.error(
                        "failed to play response after retries text=%r devices=%s sample_rates=%s",
                        text,
                        devices,
                        self.PLAYBACK_SAMPLE_RATES,
                    )
                    return False
                except Exception:
                    LOG.exception("voice response failed text=%r", text)
                    try:
                        if self._player is not None:
                            self._player_call_with_timeout(self._player, "stop")
                            self._player_call_with_timeout(self._player, "close")
                    except Exception:
                        LOG.exception("speaker cleanup failed")
                    self._player = None
                    return False
                finally:
                    time.sleep(float(os.getenv("RVCLAW_TTS_RESUME_DELAY", "0.15")))
                    self._playback_event.clear()
                    if capture_was_running and capture is not None and not self._stop_event.is_set():
                        capture_chunk_bytes = max(
                            self.VAD_SAMPLES * 2,
                            int(os.getenv("RVCLAW_CAPTURE_CHUNK_BYTES", "3200")),
                        )
                        try:
                            # The board audio binding does not reliably restart a
                            # stopped stream; recreate the capture object instead.
                            capture.close()
                            resumed = self.audio.AudioCapture(self.capture_device)
                            resumed.set_callback(self._on_audio)
                            self._capture = resumed
                            if not resumed.start(self.SAMPLE_RATE, 1, capture_chunk_bytes):
                                raise RuntimeError("failed to resume microphone capture")
                        except Exception as exc:
                            self._capture = None
                            self._set_fault("capture_resume", exc)
        finally:
            if temporary_prompt is not None:
                try:
                    temporary_prompt.unlink()
                except OSError:
                    LOG.exception("failed to remove temporary voice prompt path=%s", temporary_prompt)

    @classmethod
    def _prepare_playback_prompt(
        cls, prompt_path: Path, *, sample_rate: Optional[int] = None
    ) -> Tuple[Path, Optional[Path]]:
        """Return a PCM16 WAV compatible with the K3 USB speaker."""

        target_sample_rate = int(sample_rate or cls.PLAYBACK_SAMPLE_RATE)

        with wave.open(str(prompt_path), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            source_sample_rate = source.getframerate()
            compression = source.getcomptype()
            frame_count = source.getnframes()
            frames = source.readframes(frame_count)
        if sample_width != 2 or compression != "NONE" or channels not in {1, 2}:
            raise ValueError("voice prompt must be uncompressed PCM16 mono or stereo WAV")
        if frame_count <= 0:
            raise ValueError("voice prompt is empty")
        if source_sample_rate == target_sample_rate and channels == cls.PLAYBACK_CHANNELS:
            return prompt_path, None

        samples = np.frombuffer(frames, dtype="<i2").reshape(-1, channels)
        if channels == 1:
            samples = np.repeat(samples, cls.PLAYBACK_CHANNELS, axis=1)
        if source_sample_rate != target_sample_rate:
            target_count = max(
                1,
                int(round(samples.shape[0] * target_sample_rate / source_sample_rate)),
            )
            source_positions = np.arange(samples.shape[0], dtype=np.float64)
            target_positions = np.arange(target_count, dtype=np.float64) * (
                source_sample_rate / target_sample_rate
            )
            samples = np.stack(
                [
                    np.interp(target_positions, source_positions, samples[:, channel])
                    for channel in range(cls.PLAYBACK_CHANNELS)
                ],
                axis=1,
            )
            samples = np.rint(samples).clip(-32768, 32767).astype("<i2")

        descriptor, filename = tempfile.mkstemp(
            prefix="rvclaw-voice-playback-", suffix=".wav"
        )
        os.close(descriptor)
        converted_path = Path(filename)
        try:
            with wave.open(str(converted_path), "wb") as target:
                target.setnchannels(cls.PLAYBACK_CHANNELS)
                target.setsampwidth(2)
                target.setframerate(target_sample_rate)
                target.writeframes(samples.astype("<i2", copy=False).tobytes())
        except Exception:
            converted_path.unlink(missing_ok=True)
            raise
        LOG.info(
            "voice prompt converted for playback source_rate=%d source_channels=%d path=%s",
            source_sample_rate,
            channels,
            prompt_path,
        )
        return converted_path, converted_path

    def _synthesize_prompt(self, text: str) -> Optional[Path]:
        """Generate variable content in the already initialized child process."""
        synthesizer = getattr(self, "_synthesizer", None)
        if synthesizer is None:
            LOG.error("dynamic voice response unavailable: synthesizer is not running")
            return None
        descriptor, filename = tempfile.mkstemp(prefix="rvclaw-voice-", suffix=".wav")
        os.close(descriptor)
        prompt_path = Path(filename)
        started = time.monotonic()
        try:
            if not synthesizer.synthesize(text, prompt_path, self._tts_timeout_seconds):
                raise RuntimeError("speech synthesis failed")
            if not prompt_path.is_file() or not prompt_path.stat().st_size:
                raise RuntimeError("speech synthesis produced no audio")
            LOG.info("dynamic voice generated seconds=%.3f text=%r", time.monotonic() - started, text)
            return prompt_path
        except Exception:
            prompt_path.unlink(missing_ok=True)
            LOG.exception("isolated speech synthesis failed text=%r", text)
            return None

    def _set_fault(self, component: str, exc: Exception) -> None:
        self._fault = f"{component}: {type(exc).__name__}: {exc}"
        self._running = False
        self._stop_event.set()
        LOG.exception("voice listener fault in %s", component)


def _route_runner_active() -> bool:
    """True when a route or voice mapping chassis wrapper is alive.

    Startup reconciliation must never close a task that a live runner is still
    working on. When liveness cannot be determined we report True so the sweep is
    skipped: leaving a stale row behind is harmless, closing a live one is not.
    """
    try:
        completed = subprocess.run(
            [
                "pgrep",
                "-f",
                "rvclaw_waypoint_route_client.py|slam_mapping_session.sh --voice-run",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    return completed.returncode == 0


def build_router(
    coordinator: Optional[InspectionCoordinator] = None,
    motion_client: Optional[ChassisMotionClient] = None,
    route_client: Optional[ChassisRouteClient] = None,
    mapping_client: Optional[ManualMappingClient] = None,
) -> CommandRouter:
    task_db = os.getenv("RVCLAW_TASK_DB", "").strip()
    gateway = TaskGateway(task_db) if task_db else None
    if gateway is not None:
        # A task left "running" by a process that died can never be finished by
        # anyone: the watcher that would have published its terminal event died
        # with it (harness teardown, service restart, power cut). Such rows are
        # indistinguishable from real work in progress, so sweep them once, here,
        # where a fresh process by definition owns no route yet.
        if _route_runner_active():
            LOG.info("skipping stale-task reconciliation: a route runner is alive")
        else:
            gateway.reconcile_stale_tasks()
    router = CommandRouter(min_confidence=float(os.getenv("RVCLAW_VOICE_MIN_CONFIDENCE", "0.72")), gateway=gateway)
    coordinator = coordinator or InspectionCoordinator()
    motion_enabled = os.getenv("RVCLAW_VOICE_ENABLE_MOTION", "0").lower() in {"1", "true", "yes", "on"}
    motion = motion_client or ChassisMotionClient()
    route_enabled = os.getenv("RVCLAW_VOICE_ENABLE_INSPECTION", "0").lower() in {
        "1", "true", "yes", "on"
    }
    route = route_client or ChassisRouteClient()
    mapping_enabled = os.getenv("RVCLAW_VOICE_ENABLE_MAPPING", "0").lower() in {
        "1", "true", "yes", "on"
    }
    mapping = mapping_client or ManualMappingClient()
    # The route runner is asynchronous, so let it close its own task row when
    # the real terminal state is observed.  Without this binding a finished
    # route keeps its task "running" forever and never emits TASK_COMPLETED.
    if getattr(route, "task_recorder", None) is None:
        try:
            route.task_recorder = router.record_terminal_state
        except AttributeError:
            pass
    if getattr(mapping, "task_recorder", None) is None:
        try:
            mapping.task_recorder = router.record_terminal_state
        except AttributeError:
            pass
    mapping_attached = False
    try:
        mapping_attached = bool(mapping.start_monitor_if_active())
    except Exception:
        LOG.exception("failed to attach existing manual mapping session")
    if mapping_attached and gateway is not None:
        router.adopt_active_task(gateway.latest_active_task_id({"start_mapping"}))
    chassis_reservation_lock = Lock()
    chassis_reservation = {"intent": None, "stop_requested": False}

    def reserve_chassis(intent: str) -> None:
        """Reject concurrent or externally started chassis work before dispatch."""

        with chassis_reservation_lock:
            if chassis_reservation["intent"] is not None:
                raise RuntimeError("another chassis task is being dispatched")
            motion_status = motion.status()
            route_status = route.status()
            mapping_status = mapping.status()
            active_states = {"running", "localizing", "finishing"}
            if str(motion_status.get("state", "")).lower() in active_states:
                raise RuntimeError("left-turn task is already running")
            if str(route_status.get("state", "")).lower() in active_states:
                raise RuntimeError("inspection route is already running")
            if str(mapping_status.get("state", "")).lower() in active_states | {"starting"}:
                raise CommandHandlerError(
                    "manual mapping is already running",
                    "建图正在进行，请先结束当前建图",
                )
            chassis_reservation["intent"] = intent
            chassis_reservation["stop_requested"] = False

    def release_chassis(intent: str) -> bool:
        with chassis_reservation_lock:
            if chassis_reservation["intent"] == intent:
                stop_requested = bool(chassis_reservation["stop_requested"])
                chassis_reservation["intent"] = None
                chassis_reservation["stop_requested"] = False
                return stop_requested
        return False

    def run_chassis(intent: str, operation, stop_operation):
        reserve_chassis(intent)
        released = False
        try:
            result = operation()
            stop_requested = release_chassis(intent)
            released = True
            if stop_requested and str((result or {}).get("state", "")).lower() in {
                "running", "localizing", "finishing"
            }:
                try:
                    stop_operation()
                finally:
                    raise RuntimeError("chassis task was stopped during startup")
            return result
        finally:
            if not released:
                stop_requested = release_chassis(intent)
                if stop_requested:
                    try:
                        stop_operation()
                    except Exception:
                        pass

    def status(_command):
        vision = coordinator.status()
        motion_status = motion.status()
        route_status = route.status()
        mapping_status = mapping.status()
        return {
            **vision,
            "motion": motion_status,
            "route": route_status,
            "mapping": mapping_status,
            "follow_up_text": _status_response(
                vision, motion_status, route_status, mapping_status
            ),
        }

    def inspect(_command):
        result = coordinator.start("inspect_meter")
        return {**result, "follow_up_text": "仪表检测开始"}

    def check_ppe(_command):
        result = coordinator.start("check_ppe")
        return {**result, "follow_up_text": "安全帽检测开始"}

    def repeat(_command):
        result = coordinator.last_result()
        return {
            "state": "completed",
            "result": result,
            "follow_up_text": _last_result_response(result),
        }

    def stop(_command):
        with chassis_reservation_lock:
            if chassis_reservation["intent"] is not None:
                chassis_reservation["stop_requested"] = True
        results = {}
        failures = []
        for name, operation in (
            ("inspection", coordinator.stop),
            ("motion", motion.stop),
            ("route", route.stop),
            ("mapping", mapping.stop),
        ):
            try:
                results[name] = operation()
            except Exception as exc:
                failures.append(name)
                results[name] = {"state": "error", "error_type": type(exc).__name__}
        if failures:
            raise RuntimeError("failed to stop: " + ", ".join(failures))
        return {"state": "stopped", "results": results, "follow_up_text": "小车已停止"}

    def turn_left(command):
        if not motion_enabled and motion_client is None:
            raise RuntimeError("voice motion is disabled")
        result = run_chassis("turn_left", lambda: motion.left_turn(command), motion.stop)
        return {**result, "follow_up_text": "左转完成"}

    def start_inspection(command):
        if not route_enabled and route_client is None:
            raise RuntimeError("voice inspection is disabled")
        result = run_chassis("start_inspection", lambda: route.start(command), route.stop)
        return {**result, "follow_up_text": "巡检任务已启动"}

    def resume_inspection(command):
        if not route_enabled and route_client is None:
            raise RuntimeError("voice inspection is disabled")
        result = run_chassis(
            "resume_inspection",
            lambda: route.resume(command, force_relocalization=False),
            route.stop,
        )
        return {**result, "follow_up_text": "巡检恢复任务已启动"}

    def resume_inspection_relocalize(command):
        if not route_enabled and route_client is None:
            raise RuntimeError("voice inspection is disabled")
        result = run_chassis(
            "resume_inspection_relocalize",
            lambda: route.resume(command, force_relocalization=True),
            route.stop,
        )
        return {**result, "follow_up_text": "重新定位并恢复巡检任务已启动"}

    def start_mapping(command):
        if not mapping_enabled and mapping_client is None:
            raise RuntimeError("voice mapping is disabled")
        if str(mapping.status().get("state", "")).lower() in {
            "starting", "running", "finishing"
        }:
            raise CommandHandlerError(
                "manual mapping is already running",
                "建图正在进行，请先结束当前建图",
            )
        result = run_chassis("start_mapping", lambda: mapping.start(command), mapping.stop)
        return {
            **result,
            "follow_up_text": "建图已开始，请使用手柄低速巡视环境",
        }

    def finish_mapping(command):
        if not mapping_enabled and mapping_client is None:
            raise RuntimeError("voice mapping is disabled")
        return mapping.finish(command)

    router.register("query_status", status)
    router.register("inspect_meter", inspect)
    router.register("check_ppe", check_ppe)
    router.register("repeat_last_result", repeat)
    router.register("stop", stop)
    if route_enabled or route_client is not None:
        router.register("start_inspection", start_inspection)
        router.register("resume_inspection", resume_inspection)
        router.register(
            "resume_inspection_relocalize",
            resume_inspection_relocalize,
        )
    if motion_enabled or motion_client is not None:
        router.register("turn_left", turn_left)
    if mapping_enabled or mapping_client is not None:
        router.register("start_mapping", start_mapping)
        router.register("finish_mapping", finish_mapping)
    return router


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="RVClaw offline voice service")
    parser.add_argument("--health", action="store_true", help="print runtime health without opening devices")
    parser.add_argument("--run", action="store_true", help="start streaming service")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    router = build_router()
    runtime = SpeechRuntime(router=router)
    if args.health:
        print({"runtime": runtime.health(), "router": router.health()})
        return 0 if runtime.health()["ready"] else 2
    if args.run:
        runtime.start()

        def _request_stop(_signum, _frame):
            runtime.stop()

        signal.signal(signal.SIGTERM, _request_stop)
        signal.signal(signal.SIGINT, _request_stop)
        try:
            while runtime.health()["running"]:
                time.sleep(1)
        finally:
            runtime.stop()
        return 0
    parser.error("choose --health or --run")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
