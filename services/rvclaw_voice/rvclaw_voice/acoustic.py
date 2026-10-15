"""Acoustic wake detection and incremental recognition in an isolated process."""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import select
import struct
import subprocess
import time
import wave
from typing import Optional

LOG = logging.getLogger("rvclaw_voice")


@dataclass
class AcousticUtterance:
    pcm: bytes
    transcript: str
    raw_transcript: str
    wake: dict
    raw_pcm: bytes = b""


def save_diagnostic(directory: str, utterance: AcousticUtterance, limit: int = 30) -> None:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    base = root / ("voice-" + str(time.time_ns()))
    with wave.open(str(base.with_suffix(".wav")), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(utterance.raw_pcm or utterance.pcm)
    base.with_suffix(".json").write_text(json.dumps({
        "raw_transcript": utterance.raw_transcript,
        "command": utterance.transcript,
        "wake": utterance.wake,
    }, ensure_ascii=False), encoding="utf-8")
    for old in sorted(root.glob("voice-*.wav"))[:-limit]:
        old.unlink()
        old.with_suffix(".json").unlink(missing_ok=True)


def command_text(text: str) -> str:
    """Remove punctuation and the exact wake tail, never substitute words."""
    text = "".join(c for c in text if c.isalnum())
    for prefix in ("小微小微", "你好小车", "小车", "车"):
        if text.startswith(prefix):
            return text[len(prefix):].lstrip("啊呀")
    return text.lstrip("啊呀")


def acoustic_command_text(text: str) -> Optional[str]:
    """Only call after acoustic wake; retain the full command including negation.

    The observed 川 transcription is permitted at this exact leading position.
    It is not a textual wake alias and never changes command words.
    """
    compact = "".join(c for c in text if c.isalnum())
    for prefix in ("小微小微", "你好小车", "你好小川"):
        if compact.startswith(prefix):
            return compact[len(prefix):].lstrip("啊呀")
    return None


class AcousticGate:
    def __init__(self, continuation_seconds: float = 3.0):
        self.continuation_seconds = continuation_seconds
        self.pending_until = 0.0

    def reset(self) -> None:
        self.pending_until = 0.0

    def expire(self, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        if self.pending_until > 0 and now > self.pending_until:
            self.reset()
            return True
        return False

    def finish(self, result: dict, pcm: bytes, now: Optional[float] = None, started_at: Optional[float] = None) -> Optional[AcousticUtterance]:
        now = time.monotonic() if now is None else now
        started_at = now if started_at is None else started_at
        raw_pcm = pcm
        decoded = result.get("asr", {})
        raw = decoded.get("text", "")
        wake = result.get("wake")
        if wake and wake.get("continuous"):
            # Continuous keyword timestamps are local to hidden decoder resets.
            # Require the full prefix in the recovered audio, never guess an
            # absolute boundary or authorize a bare command from an old wake.
            text = acoustic_command_text(raw)
            if text is None:
                self.reset()
                LOG.warning("continuous wake rejected: full prefix unavailable text=%r", raw)
                return AcousticUtterance(pcm, "", raw, wake, raw_pcm)
            self.pending_until = now + self.continuation_seconds
        elif wake:
            times = wake.get("timestamps", [])
            tokens = decoded.get("tokens", [])
            token_times = decoded.get("timestamps", [])
            if not times or len(tokens) != len(token_times) or (raw and not tokens):
                self.reset()
                LOG.warning("acoustic wake rejected: token alignment unavailable")
                return None
            boundary = float(wake.get("start_time", 0.0)) + float(times[-1])
            beginning = float(wake.get("start_time", 0.0)) + float(times[0])
            early = "".join(c for token, stamp in zip(tokens, token_times)
                            if float(stamp) < beginning - 0.15 for c in token if c.isalnum())
            known_wakes = ("小微小微", "你好小车", "你好小川")
            if early and not any(prefix.startswith(early) for prefix in known_wakes):
                self.reset()
                LOG.warning("acoustic wake rejected: speech precedes wake phrase")
                return None
            text = "".join(token for token, stamp in zip(tokens, token_times) if float(stamp) > boundary)
            # Keep the final wake phoneme in the verifier audio to avoid clipping
            # the following word; command_text removes only that exact wake tail.
            pcm = pcm[max(0, int(boundary * 16000)) * 2:]
            self.pending_until = now + self.continuation_seconds
        elif self.pending_until > 0 and started_at <= self.pending_until:
            text = raw
            wake = {"continuation": True}
        else:
            return None
        text = command_text(text)
        if not text and not wake.get("continuation"):
            return None
        self.reset()
        return AcousticUtterance(pcm, text, raw, wake, raw_pcm)


class NativeSpeech:
    """Bounded synchronous protocol; native dependencies never enter Python."""
    def __init__(self, config_path: str):
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.config = config
        self.process = None
        self.log = None
        self._buffer = b""
        keys = ["executable", "encoder", "decoder", "joiner", "kws_tokens", "asr_model", "asr_tokens"]
        if "asr_decoder" in config:
            keys.extend(["asr_decoder", "asr_joiner"])
        for key in keys:
            if not Path(config[key]).is_file():
                raise FileNotFoundError(config[key])
        env = os.environ.copy()
        threshold = float(config.get("keywords_threshold", .25))
        if not .1 <= threshold <= .5:
            raise ValueError("keywords_threshold must be between .1 and .5")
        env["RVCLAW_KWS_THRESHOLD"] = str(threshold)
        env["LD_LIBRARY_PATH"] = config["library_dir"]
        env["SPACEMIT_EP_DISABLE_OP_TYPE_FILTER"] = "Conv"
        log_path = Path(config["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = log_path.open("ab")
        self.process = subprocess.Popen(
            [config[k] for k in keys],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, env=env, bufsize=0,
        )
        try:
            if not self._read(60).get("ready"):
                raise RuntimeError("native speech did not become ready")
        except BaseException:
            self.close()
            raise

    def _read(self, timeout: float = 5.0) -> dict:
        deadline = time.monotonic() + timeout
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise TimeoutError("native speech response timed out")
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError("native speech exited; inspect native log")
            self._buffer += chunk
            if len(self._buffer) > 1024 * 1024:
                raise RuntimeError("native speech response exceeds limit")
        line, self._buffer = self._buffer.split(b"\n", 1)
        return json.loads(line)

    def request(self, operation: str, pcm: bytes = b"") -> dict:
        if len(pcm) > 64000 or len(pcm) % 2:
            raise ValueError("invalid audio packet")
        packet = operation.encode("ascii") + struct.pack("<I", len(pcm)) + pcm
        self.process.stdin.write(packet)
        return self._read()

    def close(self) -> None:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
            self.process.stdin.close()
            self.process.stdout.close()
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None
