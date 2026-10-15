"""Bounded input level control for mono signed 16-bit speech frames."""
import math

import numpy as np


class InputLevelControl:
    """Slowly raise speech, quickly reduce loud input, never amplify silence."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.gain = 1.0
        self.envelope = 0.0

    def process(self, pcm: bytes) -> bytes:
        if not pcm:
            return pcm
        samples = np.frombuffer(pcm, dtype='<i2').astype(np.float32)
        rms = float(np.sqrt(np.mean(samples * samples))) / 32768.0
        peak = float(np.max(np.abs(samples))) / 32768.0
        duration = len(samples) / 16000.0
        self.envelope = max(rms, self.envelope * math.exp(-duration / .4))
        # Low-level background should not be raised towards speech loudness.
        if rms < .003:
            desired = min(self.gain, 1.0)
            response_seconds = .5
        else:
            desired = min(4.0, max(.25, .08 / max(self.envelope, .003)))
            response_seconds = .12 if desired > self.gain else .025
        alpha = 1.0 - math.exp(-duration / response_seconds)
        self.gain += alpha * (desired - self.gain)
        # A sudden transient must be bounded in its first frame, not after the
        # smoothing filter reacts. This cannot repair already clipped capture.
        if peak:
            self.gain = min(self.gain, .9 / peak)
        return np.rint(samples * self.gain).astype('<i2').tobytes()
