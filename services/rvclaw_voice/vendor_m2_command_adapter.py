#!/usr/bin/env python3
"""Use the vendor M2 wake event to gate the local RISC-V command recognizer.

The M2 remains the only process that owns ``/dev/ttyACM2``.  After a vendor
wake event, PCM frames are sent to the already deployed local sherpa-onnx
recognizer.  Recognized text is dispatched through the existing RVClaw router,
so chassis safety checks remain in one place.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import socket
import subprocess
import time
import struct
import threading
import uuid
import base64
import re
import wave
from pathlib import Path
from typing import Optional

from rvclaw_voice.service import build_router
from rvclaw_voice.inspection_speech_events import (
    prompt_for_event as inspection_prompt_for_event,
    write_playback_ack,
)
from rvclaw_voice.voice_command import MOTION_INTENTS, parse_command_text, strip_wake_phrase
from vendor_m2 import M2DeviceError, VendorM2Reader

LOG = logging.getLogger("vendor_m2_command_adapter")

# Fixed prompts are audible to the same room microphone used by the local wake
# fallback. Mark playback globally so the capture loop can discard self-audio
# instead of feeding it back into VAD/ASR. The lock also prevents two prompt
# producers (the command loop and waypoint watcher) from talking over each
# other.
_PROMPT_PLAYBACK_LOCK = threading.Lock()
_PROMPT_PLAYBACK_ACTIVE = threading.Event()
_PROMPT_ECHO_SUPPRESS_UNTIL = 0.0


def _prompt_audio_suppressed(*, clock=time.monotonic) -> bool:
    return _PROMPT_PLAYBACK_ACTIVE.is_set() or float(clock()) < _PROMPT_ECHO_SUPPRESS_UNTIL


def _save_diagnostic_audio(directory: str | None, pcm: bytes, *, raw_text: str,
                           remote_text: str, request_id: str | None) -> None:
    """Persist one post-wake segment for offline replay, with bounded retention."""
    if not directory or not pcm:
        return
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    stem = root / ("voice-" + str(time.time_ns()))
    with wave.open(str(stem.with_suffix(".wav")), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(pcm)
    stem.with_suffix(".json").write_text(json.dumps({
        "raw_transcript": raw_text, "remote_transcript": remote_text,
        "request_id": request_id, "sample_rate": 16000,
    }, ensure_ascii=False), encoding="utf-8")
    files = sorted(root.glob("voice-*.wav"))
    for old in files[:-30]:
        old.unlink(missing_ok=True)
        old.with_suffix(".json").unlink(missing_ok=True)


class AIUIStream:
    """Stream board PCM to the computer AIUI gateway and collect final text."""

    def __init__(self, host: str, port: int, required: bool = False):
        self.host, self.port, self.required = host, port, required
        self.wake_ack_timeout = max(0.5, float(os.getenv("RVCLAW_AIUI_WAKE_TIMEOUT", "3.0")))
        self.socket = None
        self.file = None
        self.send_lock = threading.Lock()
        self.result_lock = threading.Lock()
        self.final = {}
        self.final_event = threading.Event()
        self.wake_ack_event = threading.Event()
        self.request_id = None
        self.connected = False
        self.reader_thread = None

    def open(self, timeout: float = 5.0) -> None:
        """Open one online session; callers may safely retry after a disconnect."""
        self.close()
        self.socket = socket.create_connection((self.host, self.port), timeout=timeout)
        self.socket.settimeout(1.0)
        self.file = self.socket.makefile("rb")
        hello = json.loads(self.file.readline(131072))
        if hello.get("type") != "hello" or hello.get("backend") != "fifo-cmd-write":
            raise RuntimeError("电脑端在线识别网关协议不匹配")
        # A buffered file object becomes unusable after a socket timeout. Use a
        # finite timeout only for the handshake, then close the socket from
        # close() to wake the dedicated result reader.
        self.socket.settimeout(None)
        self.connected = True
        self.reader_thread = threading.Thread(target=self._read_results, daemon=True)
        self.reader_thread.start()
        LOG.info("computer AIUI gateway connected host=%s port=%d", self.host, self.port)

    def _send(self, event: dict) -> None:
        event = {"protocol": "rvclaw-m2-v1", **event}
        if "pcm" in event:
            event["pcm_b64"] = base64.b64encode(event.pop("pcm")).decode("ascii")
        with self.send_lock:
            self.socket.sendall((json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode())

    def _read_results(self) -> None:
        try:
            while self.connected:
                try:
                    raw = self.file.readline(131072)
                except socket.timeout:
                    continue
                if not raw:
                    break
                result = json.loads(raw)
                if result.get("type") == "wake_ack" and result.get("request_id") == self.request_id:
                    self.wake_ack_event.set()
                    continue
                if result.get("type") == "result" and result.get("final") is True:
                    if result.get("request_id") == self.request_id:
                        with self.result_lock:
                            self.final = result
                        self.final_event.set()
                elif result.get("type") == "error":
                    LOG.warning("computer AIUI error code=%s: %s", result.get("code"), result.get("message", "unknown"))
                    if result.get("request_id") == self.request_id:
                        self.final_event.set()
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            LOG.warning("computer AIUI result stream stopped: %s", exc)
        finally:
            self.connected = False
            self.final_event.set()

    def wake(self) -> str:
        if not self.connected:
            raise RuntimeError("电脑端在线识别未连接")
        self.request_id = uuid.uuid4().hex
        with self.result_lock:
            self.final = {}
        self.final_event.clear()
        self.wake_ack_event.clear()
        self._send({"type": "wake", "request_id": self.request_id})
        if not self.wake_ack_event.wait(self.wake_ack_timeout):
            raise RuntimeError("电脑端在线识别唤醒确认超时")
        return self.request_id

    def audio(self, pcm: bytes) -> None:
        if not self.connected or self.request_id is None:
            raise RuntimeError("在线识别会话未唤醒")
        # Keep each network record small and aligned to 40 milliseconds.
        for offset in range(0, len(pcm), 1280):
            frame = pcm[offset:offset + 1280]
            if len(frame) % 2:
                frame = frame[:-1]
            if frame:
                self._send({"type": "audio", "request_id": self.request_id, "pcm": frame})
                time.sleep(0.04)

    def end(self, timeout: float = 10.0) -> dict:
        if self.connected and self.request_id is not None:
            self._send({"type": "end_audio", "request_id": self.request_id})
        self.final_event.wait(timeout)
        with self.result_lock:
            return dict(self.final)

    def cancel(self) -> None:
        if self.connected and self.request_id is not None:
            try:
                self._send({"type": "cancel", "request_id": self.request_id})
            except OSError:
                pass

    def close(self) -> None:
        self.connected = False
        if self.socket is not None:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.socket.close()
            self.socket = None
        self.file = None
        self.reader_thread = None
        self.request_id = None
        self.final_event.clear()
        self.wake_ack_event.clear()


def _command_text(text: str) -> str:
    """Normalize only for comparing independent recognizer results."""
    text = text.strip().lower()
    # The online gateway commonly returns only the command suffix while the
    # local recognizer returns the wake phrase plus suffix. Compare equivalent
    # command portions instead of requiring byte-for-byte transcripts.
    command = strip_wake_phrase(text)
    if command is not None:
        text = command
    return re.sub(r"[^\w]+", "", text)


def _result_confidence(result) -> float | None:
    """Use the board verifier's reported sentence scores only."""
    scores = []
    for sentence in getattr(result, "sentences", ()) or ():
        value = getattr(sentence, "confidence", None)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if 0.0 <= value <= 1.0:
            scores.append(value)
    return sum(scores) / len(scores) if scores else None


def _is_wake_event(event: dict) -> bool:
    """Only vendor AIUI eventType=4 records are wake events.

    Control replies are also sent with message type 0x04; treating every such
    record as a wake caused an acknowledgement loop when the firmware rejected
    the audio control packet.
    """
    message = event.get("message") or {}
    content = message.get("content") if isinstance(message, dict) else None
    return isinstance(content, dict) and str(content.get("eventType")) == "4"


def _serial_fd_stale(reader) -> bool:
    """Detect a USB serial descriptor that survived device re-enumeration."""
    fd = getattr(reader, "fd", None)
    if fd is None:
        port = getattr(reader, "port", None)
        try:
            fd = port.fileno() if port is not None else None
        except (AttributeError, OSError, ValueError):
            fd = None
    if fd is None:
        return False
    try:
        target = os.readlink(f"/proc/self/fd/{fd}")
    except OSError:
        return False
    return "(deleted)" in target


def _rearm_wake_channel(reader, control, *, delay_seconds: float, reason: str) -> None:
    """Briefly reopen both M260C serial descriptors to restore wake readiness.

    The module can remain responsive to handshake frames while its acoustic
    wake engine no longer emits eventType=4 records.  Closing both host-side
    descriptors and reopening them restored the field unit without touching
    navigation, the microphone capture process, or the ASR model.
    """
    control.close()
    reader.close()
    delay = max(0.0, float(delay_seconds))
    if delay:
        time.sleep(delay)
    buffer = getattr(reader, "buffer", None)
    if buffer is not None and hasattr(buffer, "clear"):
        buffer.clear()
    reader.open()
    try:
        control.open()
    except Exception:
        reader.close()
        raise
    LOG.info("vendor wake channel rearmed reason_code=%s delay=%.3fs", reason, delay)


def _restart_capture(capture, *, delay_seconds: float, reason: str) -> None:
    """Restart ALSA capture in place instead of taking down the voice service."""
    capture.close()
    delay = max(0.0, float(delay_seconds))
    if delay:
        time.sleep(delay)
    capture.open()
    LOG.info("voice capture restarted reason_code=%s delay=%.3fs", reason, delay)


def _flush_capture_after_prompt(capture, *, reason: str) -> str:
    """Discard prompt echo cheaply, restarting capture only as a fallback."""
    discard = getattr(capture, "discard_pending", None)
    if callable(discard):
        try:
            if discard():
                LOG.info("voice capture flushed reason_code=%s method=drain", reason)
                return "drained"
        except (OSError, RuntimeError):
            LOG.exception("voice_failure reason_code=capture_drain_failed stage=%s", reason)
    _restart_capture(capture, delay_seconds=0.0, reason=f"{reason}_fallback_restart")
    return "restarted"


class CrossSourceWakeDeduplicator:
    """Suppress one delayed M260C event for a wake already handled locally."""

    def __init__(self, window_seconds: float, *, clock=time.monotonic):
        self.window_seconds = max(0.0, float(window_seconds))
        self._clock = clock
        self._pending_local_wake_at: float | None = None

    def note_local_wake(self) -> None:
        self._pending_local_wake_at = float(self._clock())

    def consume_vendor_duplicate(self) -> tuple[bool, float | None]:
        if self._pending_local_wake_at is None:
            return False, None
        age = max(0.0, float(self._clock()) - self._pending_local_wake_at)
        self._pending_local_wake_at = None
        return age <= self.window_seconds, age


def _vendor_wake_identity(event: dict) -> tuple | None:
    """Return a stable identity for one M260C IVW event when available."""
    message = event.get("message") or {}
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, dict):
        return None
    info = content.get("info")
    if isinstance(info, str):
        try:
            info = json.loads(info)
        except (TypeError, ValueError, json.JSONDecodeError):
            info = None
    ivw = info.get("ivw") if isinstance(info, dict) else None
    if not isinstance(ivw, dict):
        return None
    start_ms = ivw.get("start_ms")
    end_ms = ivw.get("end_ms")
    if start_ms is None or end_ms is None:
        return None
    return (str(start_ms), str(end_ms), str(ivw.get("keyword") or ""))


class VendorWakeReplayDeduplicator:
    """Ignore an exact IVW event replayed after reopening the serial port."""

    def __init__(self, window_seconds: float, *, clock=time.monotonic):
        self.window_seconds = max(0.0, float(window_seconds))
        self._clock = clock
        self._seen: dict[tuple, float] = {}

    def is_replay(self, event: dict) -> bool:
        identity = _vendor_wake_identity(event)
        if identity is None:
            return False
        now = float(self._clock())
        cutoff = now - self.window_seconds
        self._seen = {key: seen_at for key, seen_at in self._seen.items() if seen_at >= cutoff}
        if identity in self._seen:
            return True
        self._seen[identity] = now
        return False


class M2ControlWriter:
    """Send the vendor control packet that enables original audio output."""

    def __init__(self, port: str):
        self.port = port
        self.fd = None
        self.sid = 1

    def open(self) -> None:
        self.fd = os.open(self.port, os.O_WRONLY | os.O_NOCTTY)

    def set_audio(self, enabled: bool) -> None:
        if self.fd is None:
            return
        payload = json.dumps(
            {"type": "get_original_audio", "content": {"audio": 1 if enabled else 0}},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self._write(0x05, self.sid, payload)
        LOG.info("vendor audio %s requested sid=%d", "start" if enabled else "stop", self.sid)
        self.sid = (self.sid + 1) & 0xFFFF or 1

    def confirm_handshake(self, sid: int) -> None:
        # The vendor module keeps sending type 0x01 handshake frames until
        # the host returns a type 0xFF confirmation with the same sequence ID.
        self._write(0xFF, sid, b"\xA5\x00\x00\x00")
        LOG.info("vendor handshake confirmed sid=%d", sid)

    def set_wake_keywords(self, keyword: str, threshold: int) -> None:
        """Request the vendor-documented acoustic keyword/threshold setting.

        The vendor M2 example uses type 0x05 and wakeup_keywords; its guide
        specifies a threshold range of 0..5000. Writing the request alone is
        not evidence that the firmware accepted it.
        """
        if not isinstance(threshold, int) or isinstance(threshold, bool) or not 0 <= threshold <= 5000:
            raise ValueError("M260C wake threshold must be an integer in 0..5000")
        keyword = str(keyword).strip()
        if not keyword or len(keyword) > 256 or any(ord(char) < 32 for char in keyword):
            raise ValueError("M260C wake keyword must be a nonempty single line")
        if self.fd is None:
            raise RuntimeError("M260C control port is closed")
        payload = json.dumps(
            {"type": "wakeup_keywords", "content": {"keyword": keyword, "threshold": str(threshold)}},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self._write(0x05, self.sid, payload)
        LOG.info("vendor wake configuration requested keyword=%r threshold=%d sid=%d", keyword, threshold, self.sid)
        self.sid = (self.sid + 1) & 0xFFFF or 1

    def _write(self, msg_type: int, sid: int, payload: bytes) -> None:
        frame = bytearray([0xA5, 0x01, msg_type])
        frame.extend(struct.pack("<HH", len(payload), sid))
        frame.extend(payload)
        frame.append((-sum(frame)) & 0xFF)
        os.write(self.fd, frame)

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class AlsaCapture:
    """Continuously capture the M2 USB microphone through ALSA."""

    def __init__(self, device: str):
        self.device = device
        self.process = None

    def open(self) -> None:
        self.process = subprocess.Popen(
            ["arecord", "-q", "-D", self.device, "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

    def read(self, timeout: float = 0.1) -> bytes:
        if self.process is None or self.process.stdout is None:
            return b""
        exit_code = self.process.poll()
        if exit_code is not None:
            raise RuntimeError(f"arecord exited with status {exit_code}")
        import select

        if not select.select([self.process.stdout], [], [], timeout)[0]:
            return b""
        return os.read(self.process.stdout.fileno(), 65536)

    def discard_pending(
        self,
        *,
        max_bytes: int = 1048576,
        max_seconds: float = 0.45,
        quiet_seconds: float | None = None,
    ) -> bool:
        """Drain prompt echo until the capture pipe stays quiet briefly.

        Returning on the first empty poll races the USB audio pipeline: the
        speaker tail can arrive a few milliseconds later and become the first
        frames of the command session.  Keep the capture process open, but
        require a bounded quiet interval before declaring the pipe clean.
        """
        if self.process is None or self.process.stdout is None:
            return False
        exit_code = self.process.poll()
        if exit_code is not None:
            raise RuntimeError(f"arecord exited with status {exit_code}")

        import select

        if quiet_seconds is None:
            # Drain only bytes that already exist when playback returns. The
            # previous 160 ms future-wait ate the beginning of short commands
            # spoken immediately after "小车，收到" in field use.
            quiet_seconds = float(os.getenv("RVCLAW_AUDIO_DRAIN_QUIET_SECONDS", "0.0"))
        quiet_seconds = max(0.0, float(quiet_seconds))
        started = time.monotonic()
        settle_deadline = started + quiet_seconds
        deadline = started + max(quiet_seconds, max(0.0, float(max_seconds)))
        remaining = max(0, int(max_bytes))
        drained = 0
        while remaining > 0 and time.monotonic() < settle_deadline:
            now = time.monotonic()
            wait = min(0.02, max(0.0, settle_deadline - now))
            if not select.select([self.process.stdout], [], [], wait)[0]:
                continue
            chunk = os.read(self.process.stdout.fileno(), min(65536, remaining))
            if not chunk:
                return False
            drained += len(chunk)
            remaining -= len(chunk)

        while remaining > 0 and time.monotonic() <= deadline:
            if not select.select([self.process.stdout], [], [], 0)[0]:
                LOG.info(
                    "voice capture backlog drained bytes=%d settle=%.3fs",
                    drained,
                    quiet_seconds,
                )
                return True
            chunk = os.read(self.process.stdout.fileno(), min(65536, remaining))
            if not chunk:
                return False
            drained += len(chunk)
            remaining -= len(chunk)

        LOG.warning(
            "voice capture backlog drain incomplete bytes=%d budget_bytes=%d budget_seconds=%.3f",
            drained,
            max_bytes,
            max_seconds,
        )
        return False

    def close(self) -> None:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
            self.process = None


REPEAT_TEXT = "没有听清，请再说一次"
CONFIRM_PROMPT = "请确认开始巡检"
CONFIRM_REQUIRED_STATUS = "confirmation_required"


class EndpointDetector:
    """Decide when the utterance that follows a vendor wake event has ended.

    The vendor firmware reports only the wake moment; nothing ever reports the
    end of speech, so the adapter has to find the endpoint itself.  The first
    implementation treated ``peak > --silence-peak`` as speech, but the
    XFM-DP-V0.0.18 capture floor already peaks above the configured value
    (measured min 716 / p10 815 against a threshold of 700), so silence was
    never observed: every session ran to the hard ceiling, the transcript was
    discarded and the adapter announced a command failure.  The detector now
    uses the board VAD when it is available and otherwise calibrates an energy
    threshold from the measured capture floor.

    Every frame is exactly ``FRAME_SAMPLES`` samples so the verdict is a pure
    function of the PCM stream and can be replayed offline.
    """

    FRAME_SAMPLES = 512

    def __init__(
        self,
        *,
        vad_model_dir: str,
        detector: str = "auto",
        sample_rate: int = 16000,
        silence_seconds: float = 0.8,
        minimum_seconds: float = 0.35,
        maximum_seconds: float = 8.0,
        peak_floor: int = 700,
        no_speech_seconds: float = 4.0,
    ):
        self.sample_rate = int(sample_rate)
        self.silence_seconds = float(silence_seconds)
        self.minimum_seconds = float(minimum_seconds)
        self.maximum_seconds = float(maximum_seconds)
        self.peak_floor = int(peak_floor)
        self.no_speech_seconds = float(no_speech_seconds)
        self._vad_model_dir = vad_model_dir
        mode = str(detector or "auto").strip().lower()
        self._force_energy = mode == "energy"
        self._force_vad = mode == "vad"
        self._engine = None
        self._vad_error = RuntimeError("vad disabled by --endpoint-detector energy") \
            if self._force_energy else None
        self.elapsed = 0.0
        self.speech_seconds = 0.0
        self.last_speech_at = 0.0
        self.reset()

    # -- detector construction ------------------------------------------------
    def _build_engine(self):
        """Return a usable VAD engine, or None to keep the energy detector."""
        if self._engine is not None or self._vad_error is not None:
            return self._engine
        try:
            import spacemit_vad

            config = spacemit_vad.VadConfig.preset("silero")
            config.model_dir = self._vad_model_dir
            engine = spacemit_vad.VadEngine(config)
            if not engine.is_initialized:
                raise RuntimeError("VAD engine reported uninitialized")
            self._engine = engine
            LOG.info(
                "voice_endpoint detector=vad engine=%s window=%s model_dir=%s",
                engine.engine_name, getattr(config, "window_size", None), self._vad_model_dir,
            )
        except Exception as exc:  # noqa: BLE001 - any failure means "use energy"
            self._vad_error = exc
            if self._force_vad:
                raise
            LOG.warning("voice_endpoint detector=energy vad_unavailable=%s", exc)
        return self._engine

    @property
    def detector_name(self) -> str:
        return "vad" if self._engine is not None else "energy"

    def warm_up(self) -> None:
        """Resolve the detector at startup so readiness is visible in the log.

        Building the engine on the first utterance would hide a missing VAD
        model until a real command arrived; resolving it here makes the active
        detector an explicit readiness fact instead.
        """
        self._build_engine()
        LOG.info(
            "voice_endpoint ready detector=%s silence=%.2fs no_speech=%.2fs maximum=%.2fs",
            self.detector_name, self.silence_seconds, self.no_speech_seconds, self.maximum_seconds,
        )

    # -- session control ------------------------------------------------------
    def reset(self) -> None:
        self._buffer = b""
        self._frames = 0
        self._speech_seen = False
        self._last_speech = 0.0
        self._noise_peak = None
        self._loud = 0
        self.elapsed = 0.0
        self.speech_seconds = 0.0
        self.last_speech_at = 0.0
        engine = self._engine
        if engine is not None:
            try:
                engine.reset()
            except Exception:  # noqa: BLE001 - a failed reset must not stop capture
                LOG.debug("vad reset failed", exc_info=True)

    # -- per-frame classification --------------------------------------------
    def _frame_is_speech(self, frame: bytes) -> bool:
        engine = self._build_engine()
        if engine is not None:
            try:
                import numpy as np

                samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
                return bool(engine.detect(samples, self.sample_rate).is_speech)
            except Exception as exc:  # noqa: BLE001 - degrade to energy detection
                self._vad_error = exc
                self._engine = None
                LOG.warning("voice_endpoint detector=energy vad_failed=%s", exc)
        return self._energy_is_speech(frame)

    def _energy_is_speech(self, frame: bytes) -> bool:
        samples = memoryview(frame).cast("h")
        peak = max((abs(int(value)) for value in samples), default=0)
        if self._noise_peak is None:
            self._noise_peak = float(peak)
        threshold = max(self.peak_floor, int(self._noise_peak * 2.5))
        if peak > threshold:
            # Require two consecutive loud frames so a single noise spike cannot
            # open a session that then waits for a silence that never arrives.
            self._loud += 1
        else:
            self._loud = 0
            # Only quiet frames may re-teach the capture floor; letting a loud
            # utterance raise the threshold would hide the rest of the sentence.
            self._noise_peak = 0.95 * self._noise_peak + 0.05 * peak
        return self._loud >= 2

    # -- public entry point ---------------------------------------------------
    def feed(self, pcm: bytes) -> str | None:
        """Absorb PCM and return an endpoint verdict, or None to continue."""
        self._buffer += pcm
        frame_bytes = self.FRAME_SAMPLES * 2
        while len(self._buffer) >= frame_bytes:
            frame = self._buffer[:frame_bytes]
            self._buffer = self._buffer[frame_bytes:]
            speech = self._frame_is_speech(frame)
            self._frames += 1
            elapsed = self._frames * self.FRAME_SAMPLES / self.sample_rate
            self.elapsed = elapsed
            if speech:
                self._speech_seen = True
                self._last_speech = elapsed
                self.last_speech_at = elapsed
                self.speech_seconds += self.FRAME_SAMPLES / self.sample_rate
            if self._speech_seen:
                silence = elapsed - self._last_speech
                if silence >= self.silence_seconds and elapsed >= self.minimum_seconds:
                    return "silence"
            elif elapsed >= self.no_speech_seconds:
                return "no_speech"
            if elapsed >= self.maximum_seconds:
                return "max_duration"
        return None


def _play_prompt(
    text: str, prompt_dir: Path, filename_override: Optional[str] = None
) -> bool:
    names = {
        "小车，收到": "accepted.wav",
        "小车执行失败": "failed.wav",
        REPEAT_TEXT: "please-repeat.wav",
        CONFIRM_PROMPT: "confirm-inspection.wav",
        "当前没有任务": "status-idle.wav",
        "当前正在执行巡检任务": "status-inspection.wav",
        "当前正在执行左转任务": "status-left.wav",
        "当前正在执行仪表检测任务": "status-meter.wav",
        "当前正在执行安全帽检测任务": "status-hardhat.wav",
        "当前正在执行检测任务": "status-detection.wav",
        "小车已停止": "stopped.wav",
        "巡检任务已启动": "inspection-launched.wav",
        "请确认继续巡检": "confirm-resume-inspection.wav",
        "请确认重新定位并继续巡检": "confirm-relocalize-resume-inspection.wav",
        "巡检恢复任务已启动": "inspection-resume-launched.wav",
        "重新定位并恢复巡检任务已启动": "inspection-relocalize-resume-launched.wav",
        # Ordinal wording instead of "A/B/C 点": MATCHA_ZH (the only backend that
        # renders at a full 22050 Hz) cannot pronounce Latin letters, and MATCHA_ZH_EN
        # renders at 16 kHz, which audibly muddies the announcement. Chinese ordinals
        # keep the waypoints distinguishable *and* keep the original voice quality.
        "小车已到达第一个巡检点": "arrived-a.wav",
        "小车已到达第二个巡检点": "arrived-b.wav",
        "小车已到达第三个巡检点": "arrived-c.wav",
        "小车执行成功": "succeeded.wav",
        "仪表检测开始": "meter-started.wav",
        "安全帽检测开始": "hardhat-started.wav",
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
    global _PROMPT_ECHO_SUPPRESS_UNTIL

    filename = filename_override or names.get(text)
    with _PROMPT_PLAYBACK_LOCK:
        _PROMPT_PLAYBACK_ACTIVE.set()
        try:
            if not filename:
                LOG.warning("prompt is not registered text=%r", text)
                return False
            path = prompt_dir / filename
            if path.is_file():
                try:
                    completed = subprocess.run(
                        ["aplay", "-q", str(path)],
                        timeout=float(os.getenv("RVCLAW_PROMPT_PLAYBACK_TIMEOUT", "20")),
                        check=False,
                    )
                    if completed.returncode != 0:
                        LOG.warning(
                            "prompt playback returned non-zero text=%r file=%s return_code=%s",
                            text,
                            filename,
                            completed.returncode,
                        )
                        return False
                    LOG.info("prompt played text=%r file=%s", text, filename)
                    return True
                except (OSError, subprocess.TimeoutExpired) as exc:
                    LOG.warning("prompt playback failed text=%r error=%s", text, exc)
                    return False
            # A missing fixed reply was previously silent, which made a dropped
            # prompt indistinguishable from a decision not to speak.
            LOG.warning("prompt missing text=%r file=%s", text, filename)
            return False
        finally:
            tail = max(0.0, float(os.getenv("RVCLAW_PROMPT_ECHO_SUPPRESS_SECONDS", "0.12")))
            _PROMPT_ECHO_SUPPRESS_UNTIL = max(
                _PROMPT_ECHO_SUPPRESS_UNTIL,
                time.monotonic() + tail,
            )
            _PROMPT_PLAYBACK_ACTIVE.clear()


def _acknowledge_wake_and_flush(capture, prompt_dir: Path, *, source: str) -> bool:
    """Acknowledge a wake event and discard the acknowledgement echo.

    ALSA capture remains open while the speaker plays fixed prompts. Discarding
    its already-buffered backlog gives the following command a clean recording
    window without paying the process-restart delay on every wake.
    """
    played = _play_prompt("小车，收到", prompt_dir)
    _flush_capture_after_prompt(capture, reason=f"{source}_wake_ack_flush")
    return played


def _is_execution_failure(dispatched: dict) -> bool:
    """True only when a parsed command reached an executor and then failed."""
    status = str(dispatched.get("status") or "").strip().lower()
    reason = str(dispatched.get("reason") or "").strip().lower()
    return status == "error" or reason in {
        "handler_error",
        "handler_not_configured",
        "task_rejected",
        "task_start_failed",
    }


def _play_dispatch_feedback(
    dispatched: dict,
    prompt_dir: Path,
    *,
    router_called: bool,
    speech_seconds: float | None = None,
    unparsed_prompt_min_speech_seconds: float = 0.60,
    transcript: str | None = None,
) -> str:
    """Speak feedback without calling an unparsed utterance an execution failure."""
    if dispatched.get("accepted"):
        response_text = dispatched.get("response_text")
        follow_up_text = dispatched.get("follow_up_text")
        if response_text:
            _play_prompt(response_text, prompt_dir)
        if follow_up_text:
            _play_prompt(follow_up_text, prompt_dir)
        return "accepted"
    if dispatched.get("status") == CONFIRM_REQUIRED_STATUS:
        _play_prompt(dispatched.get("response_text") or CONFIRM_PROMPT, prompt_dir)
        return "confirmation_required"
    if router_called and _is_execution_failure(dispatched):
        _play_prompt(dispatched.get("response_text") or "小车执行失败", prompt_dir)
        return "execution_failed"

    normalised = _command_text(transcript or "")
    # VAD can mark only a fraction of a clearly transcribed command as speech.
    # A command attempt deserves a repeat prompt even below the noise-duration
    # gate. This affects feedback only; parsing and motion policy stay closed.
    command_attempt = any(word in normalised for word in (
        "继续", "恢复", "巡检", "开始", "启动", "执行", "暂停", "返航",
        "仪表", "识别", "状态", "建图", "检测", "左转", "停止",
    ))
    if (not command_attempt and speech_seconds is not None
            and speech_seconds < float(unparsed_prompt_min_speech_seconds)):
        LOG.info(
            "unparsed short command window expired silently speech=%.2fs threshold=%.2fs reason=%s",
            speech_seconds,
            unparsed_prompt_min_speech_seconds,
            dispatched.get("reason"),
        )
        return "silent_unparsed"
    if transcript and not router_called and len(normalised) <= 2 and not command_attempt:
        LOG.info(
            "unparsed short transcript expired silently text=%r reason=%s",
            transcript,
            dispatched.get("reason"),
        )
        return "silent_unparsed"
    _play_prompt(REPEAT_TEXT, prompt_dir)
    return "repeat"


def _trim_pcm_for_asr(pcm: bytes, *, sample_rate: int = 16000) -> bytes:
    """Remove long quiet margins while retaining conservative speech padding."""
    if not pcm:
        return pcm
    import numpy as np

    samples = np.frombuffer(pcm, dtype=np.int16)
    frame_samples = max(1, int(sample_rate * 0.02))
    frame_count = len(samples) // frame_samples
    if frame_count < 3:
        return pcm
    frames = samples[: frame_count * frame_samples].reshape(frame_count, frame_samples).astype(np.float32)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    noise = float(np.percentile(rms, 20))
    threshold = max(180.0, noise * 2.2)
    active = np.flatnonzero(rms >= threshold)
    if active.size == 0:
        return pcm
    pad_frames = max(1, int(round(0.35 / 0.02)))
    first = max(0, int(active[0]) - pad_frames)
    last = min(frame_count, int(active[-1]) + pad_frames + 1)
    trimmed = samples[first * frame_samples : last * frame_samples]
    # Very short crops usually mean a transient noise pulse or a clipped word.
    if len(trimmed) < int(sample_rate * 0.50):
        return pcm
    return trimmed.astype(np.int16, copy=False).tobytes()


def _peak_normalize_pcm(pcm: bytes) -> bytes:
    """Return a bounded louder copy for a conservative second ASR pass."""
    if not pcm:
        return pcm
    import numpy as np

    samples = np.frombuffer(pcm, dtype=np.int16)
    peak = int(np.max(np.abs(samples.astype(np.int32)))) if samples.size else 0
    if peak < 1:
        return pcm
    gain = min(8.0, 28000.0 / peak)
    if gain <= 1.05:
        return pcm
    return np.clip(samples.astype(np.float32) * gain, -32768, 32767).astype(np.int16).tobytes()


def _retain_pcm_tail(frames: list[bytes], *, seconds: float, sample_rate: int = 16000) -> list[bytes]:
    """Keep a bounded pre-roll across idle VAD windows to avoid clipped wakes."""
    keep_bytes = max(0, int(float(seconds) * sample_rate * 2))
    if keep_bytes == 0 or not frames:
        return []
    tail = b"".join(frames)[-keep_bytes:]
    return [tail] if tail else []


WAKE_ONLY_RECOVERY_PHRASES = (
    # Field-only recovery aliases.  They acknowledge a wake but are never
    # accepted as prefixes for an executable command.
    "小黑小黑",
    # Exact field ASR substitutions observed on 2026-09-22. These aliases are
    # wake-only: they can acknowledge the operator, but cannot authorize a
    # command prefix or move the chassis.
    "小微小薇",
    "小回小薇",
    "小米小威",
    # Exact isolated wake substitutions from the 2026-10-09 field recordings.
    # These only open a continuation session, never an executable prefix.
    "小伟小飞",
    "小梅小梅",
    "小回小鬼",
    "小龟小黑",
    "小雨晓的",
    "小微小月",
    "小是小伙人",
    "小黑小薇",
)


def _normalise_wake_text(text: str) -> str:
    return re.sub(r"[^\w]+", "", str(text or "").strip().lower())


def _is_wake_only_transcript(text: str) -> bool:
    suffix = strip_wake_phrase(text)
    if suffix is not None:
        return not suffix.strip()
    normalised = _normalise_wake_text(text)
    # SenseVoice can choose a different homophone for each occurrence of wei.
    # Match the complete four-character wake only; do not rewrite commands or
    # accept these extra spellings as motion-command prefixes.
    if re.fullmatch(r"小[微薇伟威维韦]小[微薇伟威维韦]", normalised):
        return True
    return any(normalised == _normalise_wake_text(item) for item in WAKE_ONLY_RECOVERY_PHRASES)


def _recognize_local_wake_candidate(pcm: bytes, *, asr) -> tuple[str, float | None, str]:
    """Try full, trimmed and normalized PCM without broad fuzzy matching."""
    if not pcm:
        return "", None, "empty"
    import numpy as np

    trimmed = _trim_pcm_for_asr(pcm)
    normalized = _peak_normalize_pcm(trimmed)
    variants = (("full", pcm), ("trimmed", trimmed), ("trimmed_normalized", normalized))
    unique: list[tuple[str, bytes]] = []
    for label, candidate in variants:
        if candidate and all(candidate != previous for _, previous in unique):
            unique.append((label, candidate))

    first: tuple[str, float | None, str] | None = None
    for label, candidate in unique:
        audio = np.frombuffer(candidate, dtype=np.int16).astype(np.float32) / 32768.0
        result = asr.recognize(audio)
        raw = str(result.text or "")
        confidence = _result_confidence(result)
        if first is None:
            first = (raw, confidence, label)
        if strip_wake_phrase(raw) is not None or _is_wake_only_transcript(raw):
            if label != "full":
                LOG.info(
                    "local wake fallback recovered on alternate audio variant=%s first=%r recovered=%r",
                    label,
                    first[0],
                    raw,
                )
            return raw, confidence, label
    return first or ("", None, "unavailable")


def _active_wake_probe_due(
    detector,
    *,
    checked_speech_seconds: float,
    silence_seconds: float,
) -> bool:
    """Probe a possible repeated wake before the longer command endpoint."""
    threshold = max(0.0, float(silence_seconds))
    return bool(
        threshold > 0.0
        and detector.speech_seconds > checked_speech_seconds + 1e-9
        and detector.last_speech_at > 0.0
        and detector.elapsed - detector.last_speech_at >= threshold
    )


def _dispatch_local_wake_fallback(
    pcm: bytes,
    *,
    asr,
    router,
    prompt_dir: Path,
    diagnostic_dir: str | None,
    min_confidence: float,
) -> str:
    """Recognize a complete wake-prefixed utterance from the live microphone.

    This is a fail-safe path for an M260C that still exchanges serial
    handshakes but has stopped emitting acoustic ``eventType=4`` records.  It
    never accepts a bare command: the ASR transcript must begin with one of the
    closed wake phrases and the remaining text must be one exact registered
    command.  The normal router therefore retains all confirmation and motion
    policy checks.

    Returns ``ignored`` for ambient speech, ``wake_only`` when only the wake
    phrase was spoken, and ``handled`` once an audible response was attempted.
    """
    if not pcm:
        return "ignored"
    try:
        raw, confidence, recognition_variant = _recognize_local_wake_candidate(pcm, asr=asr)
    except Exception:  # noqa: BLE001 - an idle recognition fault must not kill the listener
        LOG.exception("voice_failure reason_code=local_wake_asr_failed")
        return "ignored"
    try:
        _save_diagnostic_audio(
            diagnostic_dir,
            pcm,
            raw_text=raw,
            remote_text="",
            request_id=None,
        )
    except OSError:
        LOG.exception("voice_failure reason_code=diagnostic_save_failed source=local_wake_fallback")

    wake_only = _is_wake_only_transcript(raw)
    command_suffix = strip_wake_phrase(raw)
    if command_suffix is None and not wake_only:
        LOG.info("local wake fallback ignored ambient transcript=%r confidence=%r", raw, confidence)
        return "ignored"
    if confidence is None or confidence < float(min_confidence):
        LOG.warning(
            "voice_failure reason_code=local_wake_low_confidence text=%r confidence=%r threshold=%.3f",
            raw,
            confidence,
            min_confidence,
        )
        _play_prompt(REPEAT_TEXT, prompt_dir)
        return "handled"
    if wake_only:
        LOG.info(
            "local wake fallback detected wake-only transcript=%r confidence=%.3f variant=%s",
            raw,
            confidence,
            recognition_variant,
        )
        _play_prompt("小车，收到", prompt_dir)
        return "wake_only"

    command = parse_command_text(
        command_suffix,
        confidence,
        source="voice",
        request_id=uuid.uuid4().hex,
    )
    if command is None:
        LOG.warning("voice_failure reason_code=local_wake_command_unparsed text=%r", raw)
        _play_prompt(REPEAT_TEXT, prompt_dir)
        return "handled"
    try:
        dispatched = router.dispatch(command)
    except Exception:  # noqa: BLE001 - preserve the always-online listener
        LOG.exception("voice_failure reason_code=local_wake_dispatch_failed intent=%s", command.intent)
        _play_prompt("小车执行失败", prompt_dir)
        return "handled"
    LOG.info(
        "local wake fallback command text=%r confidence=%.3f dispatch=%s",
        raw,
        confidence,
        json.dumps(dispatched, ensure_ascii=False),
    )
    _play_dispatch_feedback(dispatched, prompt_dir, router_called=True)
    return "handled"


ARRIVAL_RESPONSES = {
    "A": "小车已到达第一个巡检点",
    "B": "小车已到达第二个巡检点",
    "C": "小车已到达第三个巡检点",
}

# Route lifecycle events the chassis executor publishes besides the arrivals.
# Only a clean finish is announced: the failure path is already covered by the
# immediate "小车执行失败" reply, and speaking it again here is what produced
# the duplicated failure broadcast.
TERMINAL_RESPONSES = {
    ("inspection_finished", "PASS"): "小车执行成功",
}


def _response_for_event(event: dict) -> Optional[str]:
    """Return the fixed reply this adapter owns for a spool event, or None."""
    try:
        name = str(event.get("event", ""))
        status = str(event.get("status", "")).upper()
    except AttributeError:
        return None
    if name == "waypoint_arrived" and status == "PASS":
        return ARRIVAL_RESPONSES.get(str(event.get("waypoint", "")).upper())
    inspection_prompt = inspection_prompt_for_event(event)
    if inspection_prompt is not None:
        if inspection_prompt.silent:
            return None
        return inspection_prompt.text
    return TERMINAL_RESPONSES.get((name, status))


def _prompt_for_event(event: dict) -> Optional[tuple[str, Optional[str]]]:
    inspection_prompt = inspection_prompt_for_event(event)
    if inspection_prompt is not None:
        return inspection_prompt.text, inspection_prompt.filename
    response = _response_for_event(event)
    if response is None:
        return None
    return response, None


def _event_requires_silent_ack(event: dict) -> bool:
    prompt = inspection_prompt_for_event(event)
    return bool(prompt is not None and prompt.silent)


def _ack_playback(
    path: Path,
    event: dict,
    *,
    played: bool,
    error: Optional[str] = None,
) -> None:
    try:
        ack_path = write_playback_ack(path, event, played=played, error=error)
        if ack_path is not None:
            LOG.info(
                "speech event acknowledged event=%s played=%s ack=%s",
                event.get("event"),
                played,
                ack_path.name,
            )
    except Exception:
        LOG.exception("failed to acknowledge speech event file=%s", path.name)


def _arrival_age(event: dict, path: Path, now: float) -> float:
    try:
        published = float(event.get("timestamp") or 0.0)
    except (TypeError, ValueError):
        published = 0.0
    if published <= 0.0:
        try:
            published = path.stat().st_mtime
        except OSError:
            return 0.0
    return max(0.0, now - published)


def _sweep_arrival_spool(event_dir: Path, prompt_dir: Path) -> None:
    """Announce arrivals that are still fresh and retire the rest.

    Events published while the adapter was down used to be added to ``seen`` at
    thread start and were then ignored forever, so a route that ran during a
    service restart was silently never announced.  Arrivals published within
    ``RVCLAW_ARRIVAL_FRESH_SECONDS`` are announced now; anything older is stale
    news and is retired.  Events the adapter never acted on are retired once
    they pass ``RVCLAW_ARRIVAL_RETENTION_SECONDS`` so the spool cannot grow
    without bound.
    """
    fresh_seconds = float(os.getenv("RVCLAW_ARRIVAL_FRESH_SECONDS", "120"))
    retention_seconds = float(os.getenv("RVCLAW_ARRIVAL_RETENTION_SECONDS", "86400"))
    now = time.time()
    announced = silently_acknowledged = retired = retained = 0
    for path in sorted(event_dir.glob("*.json")):
        try:
            event = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            event = None
        age = _arrival_age(event, path, now) if event is not None else retention_seconds + 1
        silent_ack = _event_requires_silent_ack(event) if event is not None else False
        prompt = _prompt_for_event(event) if event is not None else None
        if silent_ack:
            _ack_playback(path, event, played=True)
            path.unlink(missing_ok=True)
            silently_acknowledged += 1
            LOG.info("spool event silently acknowledged event=%s", path.name)
        elif prompt and age <= fresh_seconds:
            response, filename = prompt
            played = _play_prompt(response, prompt_dir, filename)
            _ack_playback(
                path,
                event,
                played=played,
                error=None if played else "prompt playback failed",
            )
            path.unlink(missing_ok=True)
            announced += 1
            LOG.info("spool event announced response=%r age=%.1fs event=%s",
                     response, age, path.name)
        elif event is not None and prompt:
            response, _filename = prompt
            _ack_playback(path, event, played=False, error="speech event was stale")
            path.unlink(missing_ok=True)
            retired += 1
            LOG.info("spool event retired as stale response=%r age=%.1fs event=%s",
                     response, age, path.name)
        elif event is not None and event.get("requires_ack") is True:
            _ack_playback(path, event, played=False, error="unsupported speech event")
            path.unlink(missing_ok=True)
            retired += 1
        elif age >= retention_seconds:
            path.unlink(missing_ok=True)
            retired += 1
        else:
            retained += 1
    for ack_path in event_dir.glob("*.json.ack"):
        try:
            if now - ack_path.stat().st_mtime >= retention_seconds:
                ack_path.unlink(missing_ok=True)
                retired += 1
        except OSError:
            pass
    if announced or silently_acknowledged or retired or retained:
        LOG.info(
            "arrival spool swept announced=%d silent_ack=%d retired=%d retained=%d dir=%s",
            announced,
            silently_acknowledged,
            retired,
            retained,
            event_dir,
        )


def _watch_waypoint_arrivals(
    event_dir: Path, prompt_dir: Path, stop_event: threading.Event
) -> None:
    """Announce verified, newly published A/B/C arrival events."""
    event_dir.mkdir(parents=True, exist_ok=True)
    try:
        _sweep_arrival_spool(event_dir, prompt_dir)
    except Exception:  # noqa: BLE001 - a sweep failure must not stop the watcher
        LOG.exception("waypoint arrival sweep failed")
    seen = {path.name for path in event_dir.glob("*.json")}
    last_sweep = time.monotonic()
    while not stop_event.is_set():
        try:
            for path in sorted(event_dir.glob("*.json")):
                if path.name in seen:
                    continue
                try:
                    event = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                seen.add(path.name)
                if _event_requires_silent_ack(event):
                    _ack_playback(path, event, played=True)
                    path.unlink(missing_ok=True)
                    LOG.info("route event silently acknowledged event=%s", path.name)
                    continue
                prompt = _prompt_for_event(event)
                if prompt is None:
                    # inspection_started / inspection_failed and any malformed
                    # event have no prompt of their own, so consume the file
                    # instead of re-scanning it on every restart.
                    _ack_playback(
                        path,
                        event,
                        played=False,
                        error="unsupported speech event",
                    )
                    path.unlink(missing_ok=True)
                    continue
                response, filename = prompt
                played = _play_prompt(response, prompt_dir, filename)
                _ack_playback(
                    path,
                    event,
                    played=played,
                    error=None if played else "prompt playback failed",
                )
                path.unlink(missing_ok=True)
                if event.get("event") == "waypoint_arrived":
                    LOG.info("waypoint arrival announced waypoint=%s event=%s",
                             event.get("waypoint"), path.name)
                else:
                    LOG.info("route terminal announced response=%r event=%s",
                             response, path.name)
            if time.monotonic() - last_sweep >= 60.0:
                last_sweep = time.monotonic()
                _sweep_arrival_spool(event_dir, prompt_dir)
        except Exception:
            LOG.exception("waypoint arrival watcher failed")
        stop_event.wait(0.2)


def run(args: argparse.Namespace) -> int:
    reader = VendorM2Reader(args.port_device, args.baud)
    capture = AlsaCapture(args.capture_device)
    control = M2ControlWriter(args.port_device)
    aiui = AIUIStream(args.aiui_host, args.aiui_port, required=args.aiui_required)
    # Use the existing local ASR (Automatic Speech Recognition) model on CPU.
    # The board's camera and speech accelerators share limited hardware buffers;
    # CPU inference keeps this adapter independent of the camera pipeline.
    import numpy as np
    import spacemit_asr

    config = spacemit_asr.Config(args.asr_model_dir, backend=args.asr_backend)
    config.language = spacemit_asr.Language.ZH
    config.punctuation_enabled = True
    config.provider = "cpu"
    asr = spacemit_asr.Engine(config).initialize()
    router = build_router()
    prompt_dir = Path(args.prompt_dir)
    wake_deduplicator = CrossSourceWakeDeduplicator(args.cross_source_wake_dedup_seconds)
    vendor_wake_deduplicator = VendorWakeReplayDeduplicator(
        args.vendor_wake_replay_dedup_seconds
    )
    active = False
    frames: list[bytes] = []
    active_wake_probe_speech_seconds = 0.0
    request_id = None
    remote_audio_error = None
    diagnostic_dir = os.getenv("RVCLAW_VOICE_DIAGNOSTIC_DIR", "").strip() or None
    detector = EndpointDetector(
        vad_model_dir=args.vad_model_dir,
        detector=args.endpoint_detector,
        sample_rate=16000,
        silence_seconds=args.silence_seconds,
        minimum_seconds=args.minimum_seconds,
        maximum_seconds=args.maximum_seconds,
        peak_floor=args.silence_peak,
        no_speech_seconds=args.no_speech_seconds,
    )
    fallback_detector = None
    fallback_frames: list[bytes] = []
    if args.local_wake_fallback:
        fallback_detector = EndpointDetector(
            vad_model_dir=args.vad_model_dir,
            detector=args.endpoint_detector,
            sample_rate=16000,
            silence_seconds=args.fallback_silence_seconds,
            minimum_seconds=args.minimum_seconds,
            maximum_seconds=args.fallback_maximum_seconds,
            peak_floor=args.silence_peak,
            no_speech_seconds=args.fallback_no_speech_seconds,
        )
    arrival_event_dir = Path(
        os.getenv("RVCLAW_ARRIVAL_EVENT_DIR", "/data/RVClaw/runtime/waypoint-arrivals")
    )
    arrival_stop = threading.Event()
    arrival_thread = threading.Thread(
        target=_watch_waypoint_arrivals,
        args=(arrival_event_dir, prompt_dir, arrival_stop),
        name="rvclaw-waypoint-arrivals",
        daemon=True,
    )
    try:
        reader.open()
        capture.open()
        control.open()
        # Opt in explicitly. This keeps a known acoustic setting across host
        # service restarts without changing unrelated firmware parameters.
        wake_threshold = os.getenv("RVCLAW_M260C_WAKE_THRESHOLD", "").strip()
        if wake_threshold:
            try:
                control.set_wake_keywords(
                    os.getenv("RVCLAW_M260C_WAKE_KEYWORD", "xiao3 wei1 xiao3 wei1"),
                    int(wake_threshold),
                )
            except (ValueError, OSError, RuntimeError):
                LOG.exception("voice_failure reason_code=vendor_wake_configuration_failed")
        try:
            aiui.open()
        except (OSError, RuntimeError) as exc:
            if args.aiui_required:
                raise RuntimeError(f"电脑端在线识别不可用: {exc}") from exc
            LOG.warning("voice_failure reason_code=cloud_unavailable error=%s", exc)
        LOG.info("vendor wake adapter ready reason_code=ready wake_port=%s audio_port=%s", reader.port, args.audio_port)
        detector.warm_up()
        if fallback_detector is not None:
            fallback_detector.warm_up()
            LOG.info(
                "local wake fallback ready wake_phrases=closed_set silence=%.2fs maximum=%.2fs confidence=%.2f",
                args.fallback_silence_seconds,
                args.fallback_maximum_seconds,
                args.fallback_min_confidence,
            )
        arrival_thread.start()
        last_device_check = 0.0
        last_handshake = time.monotonic()
        last_rearm = last_handshake
        rearm_reason: str | None = None
        while True:
            now = time.monotonic()
            if not active:
                if (
                    rearm_reason is None
                    and args.idle_rearm_seconds > 0.0
                    and now - last_rearm >= args.idle_rearm_seconds
                ):
                    rearm_reason = "idle_timer"
                if rearm_reason is not None:
                    _rearm_wake_channel(
                        reader,
                        control,
                        delay_seconds=args.rearm_delay_seconds,
                        reason=rearm_reason,
                    )
                    last_rearm = time.monotonic()
                    last_handshake = last_rearm
                    last_device_check = last_rearm
                    rearm_reason = None
                    continue
            if now - last_device_check >= 1.0:
                last_device_check = now
                if _serial_fd_stale(reader):
                    LOG.error("voice_failure reason_code=serial_device_stale port=%s", reader.port)
                    raise OSError(f"serial device descriptor became stale: {reader.port}")
                if (
                    args.handshake_timeout_seconds > 0.0
                    and now - last_handshake >= args.handshake_timeout_seconds
                ):
                    LOG.warning(
                        "voice_failure reason_code=handshake_timeout age=%.2fs",
                        now - last_handshake,
                    )
                    rearm_reason = "handshake_timeout"
                    if not active:
                        continue
            try:
                events = reader.read(0.02)
            except TypeError:
                # The deployed pyserial reader has a no-argument read API;
                # the nonblocking SDK reader accepts an optional timeout.
                events = reader.read()
            for event in events:
                kind = event.get("type")
                if kind == "message" and event.get("message_type") == 0x01:
                    control.confirm_handshake(int(event.get("sid", 0)))
                    last_handshake = time.monotonic()
                    continue
                if kind == "wake":
                    if not _is_wake_event(event):
                        LOG.warning("ignoring non-wake vendor 0x04 message=%r", event.get("message"))
                        continue
                    if vendor_wake_deduplicator.is_replay(event):
                        LOG.info(
                            "duplicate vendor wake suppressed reason_code=vendor_event_replay message=%r",
                            event.get("message"),
                        )
                        continue
                    duplicate, duplicate_age = wake_deduplicator.consume_vendor_duplicate()
                    if duplicate:
                        LOG.info(
                            "duplicate vendor wake suppressed reason_code=cross_source_duplicate age=%.3fs message=%r",
                            duplicate_age,
                            event.get("message"),
                        )
                        continue
                    if active:
                        LOG.info("new vendor wake resets unfinished command")
                    active = True
                    frames = []
                    active_wake_probe_speech_seconds = 0.0
                    fallback_frames = []
                    if fallback_detector is not None:
                        fallback_detector.reset()
                    remote_audio_error = None
                    detector.reset()
                    # Every acoustic session gets a fresh id even when the
                    # online gateway is unavailable; otherwise a stale cloud
                    # id could replay a prior local command after reconnect.
                    request_id = uuid.uuid4().hex
                    # Online recognition is an enhancement. A broken tunnel
                    # must never prevent the local command path from running.
                    if not aiui.connected and not args.aiui_required:
                        try:
                            aiui.open(timeout=min(2.0, args.aiui_result_timeout))
                            LOG.info("computer AIUI reconnected")
                        except (OSError, RuntimeError) as exc:
                            LOG.debug("computer AIUI reconnect deferred: %s", exc)
                    if aiui.connected:
                        try:
                            request_id = aiui.wake()
                        except (OSError, RuntimeError) as exc:
                            remote_audio_error = exc
                            LOG.warning("voice_failure reason_code=cloud_unavailable stage=wake error=%s", exc)
                    # The current XFM firmware rejects get_original_audio
                    # control packets. ALSA capture is already continuous,
                    # so gating it here only creates an error response.
                    LOG.info("vendor wake detected message=%r", event.get("message"))
                    _acknowledge_wake_and_flush(
                        capture,
                        prompt_dir,
                        source="vendor",
                    )
                    if args.rearm_after_wake_ack:
                        _rearm_wake_channel(
                            reader,
                            control,
                            delay_seconds=args.rearm_delay_seconds,
                            reason="vendor_wake_ack",
                        )
                        last_rearm = time.monotonic()
                        last_handshake = last_rearm
                        last_device_check = last_rearm
                    detector.reset()
                    frames = []
            try:
                pcm = capture.read(0.18)
            except (OSError, RuntimeError) as exc:
                LOG.warning("voice_failure reason_code=capture_stopped error=%s", exc)
                active = False
                frames = []
                active_wake_probe_speech_seconds = 0.0
                remote_audio_error = None
                detector.reset()
                fallback_frames = []
                if fallback_detector is not None:
                    fallback_detector.reset()
                aiui.cancel()
                _restart_capture(
                    capture,
                    delay_seconds=args.capture_retry_seconds,
                    reason="arecord_exited",
                )
                continue
            if pcm and _prompt_audio_suppressed():
                # A waypoint/status prompt may be played by the watcher thread
                # while the main capture loop is idle. Consume that self-audio
                # without allowing it to open or contaminate a wake/command
                # session. If a command session was already open, keep the
                # authorization window but restart its acoustic endpoint.
                frames = []
                active_wake_probe_speech_seconds = 0.0
                detector.reset()
                fallback_frames = []
                if fallback_detector is not None:
                    fallback_detector.reset()
                continue
            if not active and pcm and fallback_detector is not None:
                fallback_frames.append(pcm)
                fallback_endpoint = fallback_detector.feed(pcm)
                if fallback_endpoint == "no_speech":
                    fallback_frames = _retain_pcm_tail(
                        fallback_frames,
                        seconds=args.fallback_preroll_seconds,
                    )
                    fallback_detector.reset()
                elif fallback_endpoint:
                    fallback_pcm = b"".join(fallback_frames)
                    fallback_speech_seconds = fallback_detector.speech_seconds
                    fallback_frames = []
                    fallback_detector.reset()
                    LOG.info(
                        "local wake fallback endpoint reason_code=%s utterance=%.2fs speech=%.2fs",
                        fallback_endpoint,
                        len(fallback_pcm) / 32000.0,
                        fallback_speech_seconds,
                    )
                    fallback_action = _dispatch_local_wake_fallback(
                        fallback_pcm,
                        asr=asr,
                        router=router,
                        prompt_dir=prompt_dir,
                        diagnostic_dir=diagnostic_dir,
                        min_confidence=args.fallback_min_confidence,
                    )
                    if fallback_action != "ignored":
                        wake_deduplicator.note_local_wake()
                        # Prompt playback is audible to the XFM microphone while
                        # arecord keeps running. Drain the captured echo without
                        # restarting the process on the normal path.
                        _flush_capture_after_prompt(capture, reason="local_wake_prompt_flush")
                        if args.rearm_after_wake_ack:
                            _rearm_wake_channel(
                                reader,
                                control,
                                delay_seconds=args.rearm_delay_seconds,
                                reason="local_wake_ack",
                            )
                            last_rearm = time.monotonic()
                            last_handshake = last_rearm
                            last_device_check = last_rearm
                    if fallback_action == "wake_only":
                        active = True
                        frames = []
                        active_wake_probe_speech_seconds = 0.0
                        remote_audio_error = None
                        detector.reset()
                        request_id = uuid.uuid4().hex
                        LOG.info("local wake fallback opened command continuation session")
                continue
            if active and pcm:
                    frames.append(pcm)
                    if aiui.connected and remote_audio_error is None:
                        try:
                            aiui.audio(pcm)
                        except (OSError, RuntimeError) as exc:
                            remote_audio_error = exc
                            LOG.warning("online AIUI audio failed reason_code=cloud_unavailable: %s", exc)
                    samples = memoryview(pcm).cast("h")
                    peak = max((abs(int(value)) for value in samples), default=0)
                    endpoint = detector.feed(pcm)
                    LOG.debug("endpoint frame peak=%d verdict=%s elapsed=%.2f", peak, endpoint, detector.elapsed)
                    if endpoint is None and _active_wake_probe_due(
                        detector,
                        checked_speech_seconds=active_wake_probe_speech_seconds,
                        silence_seconds=args.active_wake_probe_silence_seconds,
                    ):
                        active_wake_probe_speech_seconds = detector.speech_seconds
                        try:
                            probe_raw, probe_confidence, probe_variant = _recognize_local_wake_candidate(
                                b"".join(frames),
                                asr=asr,
                            )
                        except Exception:  # noqa: BLE001 - keep the command session alive
                            LOG.exception("voice_failure reason_code=active_wake_probe_asr_failed")
                        else:
                            if (
                                _is_wake_only_transcript(probe_raw)
                                and probe_confidence is not None
                                and probe_confidence >= float(args.fallback_min_confidence)
                            ):
                                LOG.info(
                                    "repeated wake detected by active probe text=%r confidence=%.3f variant=%s",
                                    probe_raw,
                                    probe_confidence,
                                    probe_variant,
                                )
                                aiui.cancel()
                                _acknowledge_wake_and_flush(
                                    capture,
                                    prompt_dir,
                                    source="active_probe",
                                )
                                wake_deduplicator.note_local_wake()
                                if args.rearm_after_wake_ack:
                                    _rearm_wake_channel(
                                        reader,
                                        control,
                                        delay_seconds=args.rearm_delay_seconds,
                                        reason="active_probe_wake_ack",
                                    )
                                    last_rearm = time.monotonic()
                                    last_handshake = last_rearm
                                    last_device_check = last_rearm
                                active = True
                                frames = []
                                active_wake_probe_speech_seconds = 0.0
                                remote_audio_error = None
                                detector.reset()
                                request_id = uuid.uuid4().hex
                                continue
                    if endpoint == "no_speech":
                        # The wake event fired but nobody spoke a command. This
                        # is not a failed command. Expire quietly so a wake-only
                        # health check produces exactly one acknowledgement.
                        active = False
                        aiui.cancel()
                        LOG.info("voice_endpoint reason_code=no_speech elapsed=%.2fs detector=%s",
                                 detector.elapsed, detector.detector_name)
                        frames = []
                        active_wake_probe_speech_seconds = 0.0
                        rearm_reason = None if args.rearm_after_wake_ack else "session_no_speech"
                    elif endpoint:
                        active = False
                        active_wake_probe_speech_seconds = 0.0
                        LOG.info("voice_endpoint reason_code=%s utterance=%.2fs speech=%.2fs detector=%s",
                                 endpoint, detector.elapsed, detector.speech_seconds,
                                 detector.detector_name)
                        # Keep ALSA capture open between wake sessions.
                        session_pcm = b"".join(frames)
                        recognition_pcm = _trim_pcm_for_asr(session_pcm)
                        audio = np.frombuffer(recognition_pcm, dtype=np.int16).astype(np.float32) / 32768.0
                        result = asr.recognize(audio)
                        raw = str(result.text or "")
                        confidence = _result_confidence(result)
                        remote_result = {}
                        if aiui.connected and remote_audio_error is None:
                            # Do not hold the local path hostage to a slow
                            # cloud final. Keep a short diagnostic wait when
                            # local ASR already produced a usable transcript.
                            remote_wait = 0.8 if raw else args.aiui_result_timeout
                            remote_result = aiui.end(timeout=remote_wait)
                        remote_text = str(remote_result.get("text") or "")
                        try:
                            _save_diagnostic_audio(diagnostic_dir, session_pcm,
                                                   raw_text=raw, remote_text=remote_text,
                                                   request_id=request_id)
                        except OSError:
                            LOG.exception("voice_failure reason_code=diagnostic_save_failed")
                        if not raw:
                            LOG.warning("voice_failure reason_code=local_asr_empty cloud_text=%r", remote_text)
                        elif confidence is None:
                            LOG.warning("voice_failure reason_code=local_confidence_missing text=%r", raw)
                        if _is_wake_only_transcript(raw):
                            LOG.info("repeated wake received during command window text=%r", raw)
                            aiui.cancel()
                            _acknowledge_wake_and_flush(
                                capture,
                                prompt_dir,
                                source="repeated",
                            )
                            if args.rearm_after_wake_ack:
                                _rearm_wake_channel(
                                    reader,
                                    control,
                                    delay_seconds=args.rearm_delay_seconds,
                                    reason="repeated_wake_ack",
                                )
                                last_rearm = time.monotonic()
                                last_handshake = last_rearm
                                last_device_check = last_rearm
                            wake_deduplicator.note_local_wake()
                            active = True
                            frames = []
                            active_wake_probe_speech_seconds = 0.0
                            remote_audio_error = None
                            detector.reset()
                            request_id = uuid.uuid4().hex
                            continue
                        # The M2 wake event is the authorization boundary. Do
                        # not require the ASR transcript to repeat that wake
                        # phrase; parse only its complete command suffix.
                        command_suffix = strip_wake_phrase(raw) if raw else None
                        if command_suffix is None:
                            command_suffix = raw.strip()
                        local_command = parse_command_text(command_suffix, confidence, source="voice", request_id=request_id) if command_suffix and confidence is not None else None
                        texts_match = bool(raw and remote_text and _command_text(raw) == _command_text(remote_text))
                        remote_suffix = strip_wake_phrase(remote_text) or remote_text.strip()
                        # Treat two exact spellings of the same registered
                        # intent (for example 巡解/巡检) as agreement even if
                        # their literal transcripts differ.
                        remote_intent_command = (
                            parse_command_text(remote_suffix, confidence if confidence is not None else 1.0,
                                                source="voice", request_id=request_id)
                            if remote_suffix else None
                        )
                        same_intent = bool(local_command and remote_intent_command
                                           and local_command.intent == remote_intent_command.intent)
                        conflict = bool(raw and remote_text and not texts_match and not same_intent)
                        # A conflict is diagnostic for read-only commands, but
                        # motion commands fail closed until a future executor
                        # can request explicit confirmation.
                        router_called = False
                        if conflict and local_command is not None and local_command.intent in MOTION_INTENTS:
                            LOG.warning("voice_failure reason_code=intent_conflict local=%r remote=%r intent=%s", raw, remote_text, local_command.intent)
                            dispatched = {"accepted": False, "reason": "intent_conflict"}
                        elif local_command is not None:
                            router_called = True
                            dispatched = router.dispatch(local_command)
                            LOG.info("vendor command local=%r aiui=%r confidence=%.3f match=%s dispatch=%s",
                                     raw, remote_text, confidence, texts_match, json.dumps(dispatched, ensure_ascii=False))
                        elif remote_text and isinstance(remote_result.get("confidence"), (int, float)):
                            # Preserve online autonomy when local ASR is empty;
                            # trust only a gateway-provided, bounded score.
                            remote_confidence = float(remote_result["confidence"])
                            if 0.0 <= remote_confidence <= 1.0:
                                dispatched_command = parse_command_text(remote_suffix, remote_confidence, source="voice", request_id=request_id)
                                if dispatched_command is not None:
                                    router_called = True
                                    dispatched = router.dispatch(dispatched_command)
                                else:
                                    dispatched = {"accepted": False, "reason": "cloud_intent_unparsed"}
                            else:
                                dispatched = {"accepted": False, "reason": "cloud_confidence_invalid"}
                        else:
                            LOG.warning("voice_failure reason_code=recognition_unavailable cloud_text=%r", remote_text)
                            dispatched = {"accepted": False, "reason": "recognition_unavailable"}
                        if dispatched.get("status") == CONFIRM_REQUIRED_STATUS:
                            LOG.info("vendor command awaiting confirmation intent=%s ttl=%ss",
                                     dispatched.get("intent"),
                                     (dispatched.get("details") or {}).get("expires_at"))
                        if not raw.strip() and endpoint == "max_duration":
                            # The session hit the hard ceiling without ever
                            # producing a transcript. Nothing was understood, so
                            # ask for a repeat instead of announcing a failed
                            # command.
                            LOG.warning("voice_failure reason_code=no_utterance_detected duration=%.2fs detector=%s",
                                        detector.elapsed, detector.detector_name)
                        feedback = _play_dispatch_feedback(
                            dispatched,
                            prompt_dir,
                            router_called=router_called,
                            speech_seconds=detector.speech_seconds,
                            unparsed_prompt_min_speech_seconds=args.unparsed_prompt_min_speech_seconds,
                            transcript=raw,
                        )
                        LOG.info(
                            "vendor command feedback=%s router_called=%s reason=%s speech=%.2fs",
                            feedback,
                            router_called,
                            dispatched.get("reason"),
                            detector.speech_seconds,
                        )
                        _flush_capture_after_prompt(capture, reason="command_prompt_flush")
                        frames = []
                        active_wake_probe_speech_seconds = 0.0
                        rearm_reason = None if args.rearm_after_wake_ack else "session_complete"
    finally:
        arrival_stop.set()
        if arrival_thread.is_alive():
            arrival_thread.join(timeout=2.0)
        reader.close()
        capture.close()
        control.close()
        aiui.close()
        asr.shutdown()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port-device", default=os.getenv("RVCLAW_M260C_DEVICE", "/dev/rvclaw-m260c-control"))
    parser.add_argument("--audio-port", default="/dev/ttyACM0")
    parser.add_argument("--capture-device", default=os.getenv("RVCLAW_AUDIO_CAPTURE_DEVICE", "hw:2,0"))
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--asr-model-dir", default="/home/firefly/.cache/models/asr/sensevoice")
    parser.add_argument("--asr-backend", default="sensevoice")
    parser.add_argument("--prompt-dir", default="/data/RVClaw/services/rvclaw_voice/prompts")
    parser.add_argument("--silence-peak", type=int, default=700)
    parser.add_argument("--silence-seconds", type=float, default=0.8)
    parser.add_argument("--minimum-seconds", type=float, default=0.35)
    parser.add_argument("--maximum-seconds", type=float, default=8.0)
    parser.add_argument("--no-speech-seconds", type=float, default=4.0)
    parser.add_argument("--vad-model-dir", default=os.path.expanduser("~/.cache/models/vad/silero"))
    parser.add_argument("--endpoint-detector", choices=("auto", "vad", "energy"), default="auto",
                        help="auto uses the board VAD and degrades to calibrated energy detection")
    parser.add_argument("--aiui-host", default=os.getenv("RVCLAW_AIUI_HOST", "127.0.0.1"))
    parser.add_argument("--aiui-port", type=int, default=int(os.getenv("RVCLAW_AIUI_PORT", "18765")))
    parser.add_argument("--aiui-result-timeout", type=float, default=10.0)
    parser.add_argument("--aiui-required", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--local-wake-fallback",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("RVCLAW_LOCAL_WAKE_FALLBACK", "0").strip().lower()
        in {"1", "true", "yes", "on"},
        help="recognize an exact wake-prefixed command from continuous XFM audio",
    )
    parser.add_argument(
        "--fallback-silence-seconds",
        type=float,
        default=float(os.getenv("RVCLAW_LOCAL_WAKE_SILENCE_SECONDS", "1.0")),
    )
    parser.add_argument(
        "--fallback-no-speech-seconds",
        type=float,
        default=float(os.getenv("RVCLAW_LOCAL_WAKE_NO_SPEECH_SECONDS", "2.0")),
    )
    parser.add_argument(
        "--fallback-maximum-seconds",
        type=float,
        default=float(os.getenv("RVCLAW_LOCAL_WAKE_MAXIMUM_SECONDS", "7.0")),
    )
    parser.add_argument(
        "--fallback-preroll-seconds",
        type=float,
        default=float(os.getenv("RVCLAW_LOCAL_WAKE_PREROLL_SECONDS", "0.75")),
    )
    parser.add_argument(
        "--fallback-min-confidence",
        type=float,
        default=float(os.getenv("RVCLAW_LOCAL_WAKE_MIN_CONFIDENCE", "0.78")),
    )
    parser.add_argument(
        "--cross-source-wake-dedup-seconds",
        type=float,
        default=float(os.getenv("RVCLAW_CROSS_SOURCE_WAKE_DEDUP_SECONDS", "3.0")),
        help="suppress one delayed vendor wake already handled by local ASR",
    )
    parser.add_argument(
        "--vendor-wake-replay-dedup-seconds",
        type=float,
        default=float(os.getenv("RVCLAW_VENDOR_WAKE_REPLAY_DEDUP_SECONDS", "8.0")),
        help="ignore an exact IVW event replayed after the serial wake channel reopens",
    )
    parser.add_argument(
        "--rearm-after-wake-ack",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("RVCLAW_REARM_AFTER_WAKE_ACK", "0").strip().lower()
        in {"1", "true", "yes", "on"},
        help="reopen the M260C wake channel immediately after each acknowledgement",
    )
    parser.add_argument(
        "--active-wake-probe-silence-seconds",
        type=float,
        default=float(os.getenv("RVCLAW_ACTIVE_WAKE_PROBE_SILENCE_SECONDS", "0")),
        help="early silence interval for detecting a repeated wake inside a command session; 0 disables it",
    )
    parser.add_argument(
        "--unparsed-prompt-min-speech-seconds",
        type=float,
        default=float(os.getenv("RVCLAW_UNPARSED_PROMPT_MIN_SPEECH_SECONDS", "0.0")),
        help="minimum detected speech before an unparsed command asks the operator to repeat",
    )
    parser.add_argument(
        "--idle-rearm-seconds",
        type=float,
        default=float(os.getenv("RVCLAW_M260C_IDLE_REARM_SECONDS", "300")),
        help="reopen the M260C serial channel after this much idle time; 0 disables it",
    )
    parser.add_argument(
        "--handshake-timeout-seconds",
        type=float,
        default=float(os.getenv("RVCLAW_M260C_HANDSHAKE_TIMEOUT_SECONDS", "75")),
        help="rearm the M260C channel when handshake traffic is absent; 0 disables it",
    )
    parser.add_argument(
        "--rearm-delay-seconds",
        type=float,
        default=float(os.getenv("RVCLAW_M260C_REARM_DELAY_SECONDS", "0.35")),
        help="time both M260C serial descriptors remain closed during a light rearm",
    )
    parser.add_argument(
        "--capture-retry-seconds",
        type=float,
        default=float(os.getenv("RVCLAW_AUDIO_CAPTURE_RETRY_SECONDS", "1.0")),
        help="delay before reopening ALSA capture after arecord exits",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
