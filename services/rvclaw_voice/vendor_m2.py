"""Read-only新版 M2 serial protocol adapter."""
from __future__ import annotations

import json
import os
import select
import struct
import time
import errno
from pathlib import Path


SYNC = 0xA5
USER_ID = 0x01
WAKEUP = 0x04
AUDIO = 0x06
HEADER_SIZE = 7
MAX_PAYLOAD = 8192


class M2DeviceError(OSError):
    """A serial device failure with a stable reason code for supervision."""

    def __init__(self, reason_code: str, message: str, *, path: str = ""):
        super().__init__(message)
        self.reason_code = reason_code
        self.path = path

    def __str__(self) -> str:
        return f"reason_code={self.reason_code}: {self.args[0]}"


def resolve_m260c_device(path: str | None = None) -> str:
    """Resolve the M260C control port without relying on ttyACM numbering.

    The udev alias is the product default.  A caller may override it with
    ``RVCLAW_M260C_DEVICE`` for a lab fixture or an explicit by-id path.
    Numbered ``ttyACM`` nodes are intentionally accepted only when explicitly
    requested, so a USB re-enumeration cannot silently bind another device.
    """

    configured = path or os.getenv("RVCLAW_M260C_DEVICE") or "/dev/rvclaw-m260c-control"
    return os.path.expanduser(configured)


def parse_frame(frame: bytes):
    if len(frame) < HEADER_SIZE + 1 or frame[0] != SYNC or frame[1] != USER_ID:
        return None
    length = struct.unpack_from('<H', frame, 3)[0]
    total = HEADER_SIZE + length + 1
    if len(frame) != total or (sum(frame) & 0xFF) != 0:
        return None
    payload = frame[HEADER_SIZE:-1]
    sid = struct.unpack_from('<H', frame, 5)[0]
    if frame[2] == WAKEUP:
        try:
            return {"type": "wake", "sid": sid, "message": json.loads(payload.decode('utf-8'))}
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {"type": "wake", "sid": sid, "message": None}
    if frame[2] == AUDIO:
        return {"type": "audio", "pcm": payload}
    return {"type": "message", "message_type": frame[2], "sid": sid, "payload": payload}


class VendorM2Reader:
    def __init__(self, port: str | None = None, baud: int = 115200):
        self.port = resolve_m260c_device(port)
        self.baud = baud
        self.fd = None
        self.buffer = bytearray()
        self._identity = None

    def open(self):
        # termios is Linux-only; keep frame parsing portable for development tests.
        import termios
        try:
            self.fd = os.open(self.port, os.O_RDONLY | os.O_NOCTTY | os.O_NONBLOCK)
        except FileNotFoundError as exc:
            raise M2DeviceError("wake_device_missing", f"M260C device not found: {self.port}", path=self.port) from exc
        except PermissionError as exc:
            raise M2DeviceError("wake_device_permission", f"M260C device permission denied: {self.port}", path=self.port) from exc
        except OSError as exc:
            raise M2DeviceError("wake_device_open_failed", f"M260C device open failed: {self.port}: {exc}", path=self.port) from exc
        self._identity = self._stat_identity()
        try:
            attrs = termios.tcgetattr(self.fd)
        except OSError as exc:
            self.close()
            raise M2DeviceError("wake_device_config_failed", f"M260C serial setup failed: {self.port}: {exc}", path=self.port) from exc
        attrs[0] = 0
        attrs[1] = 0
        attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        attrs[3] = 0
        try:
            speed = getattr(termios, 'B' + str(self.baud))
        except AttributeError as exc:
            self.close()
            raise M2DeviceError("wake_device_baud_unsupported", f"unsupported serial baud: {self.baud}", path=self.port) from exc
        attrs[4] = speed
        attrs[5] = speed
        termios.tcsetattr(self.fd, termios.TCSANOW, attrs)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        self._identity = None

    def _stat_identity(self):
        """Return the target inode and real path used when opening the port."""
        try:
            stat = os.fstat(self.fd) if self.fd is not None else os.stat(self.port)
            return (int(stat.st_dev), int(stat.st_ino), os.path.realpath(self.port))
        except OSError as exc:
            raise M2DeviceError("wake_device_stat_failed", f"cannot inspect M260C device: {self.port}: {exc}", path=self.port) from exc

    def _ensure_identity(self) -> None:
        if self.fd is None or self._identity is None:
            return
        current = self._stat_identity()
        if current != self._identity:
            raise M2DeviceError(
                "wake_device_reenumerated",
                f"M260C device changed while open: {self.port}",
                path=self.port,
            )

    def read(self, timeout: float = 0.2):
        if self.fd is None:
            raise RuntimeError('reader is closed')
        self._ensure_identity()
        if not select.select([self.fd], [], [], timeout)[0]:
            return []
        try:
            self.buffer.extend(os.read(self.fd, 65536))
        except BlockingIOError:
            return []
        except OSError as exc:
            reason = "wake_device_disconnected" if exc.errno in {errno.ENODEV, errno.EIO, errno.ENXIO, errno.EBADF} else "wake_device_read_failed"
            raise M2DeviceError(reason, f"M260C device read failed: {self.port}: {exc}", path=self.port) from exc
        messages = []
        while len(self.buffer) >= HEADER_SIZE + 1:
            try:
                offset = self.buffer.index(SYNC)
            except ValueError:
                self.buffer.clear()
                break
            if offset:
                del self.buffer[:offset]
            # A truncated or misaligned header can advertise an absurd length.
            # Waiting for that length would permanently block later wake frames.
            if len(self.buffer) < 5:
                break
            length = struct.unpack_from('<H', self.buffer, 3)[0]
            if length > MAX_PAYLOAD:
                del self.buffer[0]
                continue
            total = HEADER_SIZE + length + 1
            if len(self.buffer) < total:
                break
            frame = bytes(self.buffer[:total])
            del self.buffer[:total]
            message = parse_frame(frame)
            if message is not None:
                messages.append(message)
        return messages
