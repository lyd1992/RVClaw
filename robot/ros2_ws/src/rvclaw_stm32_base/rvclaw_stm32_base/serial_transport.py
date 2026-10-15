"""Exclusive, zero-only serial transport primitives for staged testing."""

import array
from dataclasses import dataclass
import errno
import fcntl
import math
import os
import select
import stat
import termios
import time
from typing import List, Optional, Sequence

from .protocol import (
    FeedbackFrame,
    FeedbackParser,
    encode_auto_enable_handshake,
    encode_velocity_command,
)


ZERO_COMMAND_FRAME = encode_velocity_command(0.0, 0.0, 0.0)
AUTO_ENABLE_HANDSHAKE_FRAME = encode_auto_enable_handshake()
BAUD_CONSTANTS = {115200: termios.B115200}


class TransportError(RuntimeError):
    """Base transport failure."""


class TransportDisabledError(TransportError):
    """Raised when hardware output is not explicitly enabled."""


class TransportOwnerBusyError(TransportError):
    """Raised when another process already owns the serial endpoint."""


class NonZeroFrameBlockedError(TransportError):
    """Raised when a raw write differs from the ordinary zero frame."""


class UnsafeMotionCommandError(TransportError):
    """Raised when a typed motion command exceeds the reviewed boundary."""


@dataclass(frozen=True)
class GuardedMotionLimits:
    max_linear_x_m_s: float
    max_angular_z_rad_s: float

    def validate(self) -> None:
        values = (self.max_linear_x_m_s, self.max_angular_z_rad_s)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0.0
            for value in values
        ):
            raise ValueError("guarded motion limits must be finite and positive")


@dataclass(frozen=True)
class ZeroOnlyTransportConfig:
    device: str
    baud: int = 115200
    tx_enabled: bool = False
    zero_only: bool = True
    write_timeout_s: float = 0.05
    drain_timeout_s: float = 0.10
    zero_rate_hz: float = 20.0
    close_zero_repetitions: int = 3

    def validate(self) -> None:
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty path string")
        if "\x00" in self.device:
            raise ValueError("device path must not contain NUL")
        if (
            isinstance(self.baud, bool)
            or not isinstance(self.baud, int)
            or self.baud not in BAUD_CONSTANTS
        ):
            raise ValueError("only the confirmed 115200 baud rate is supported")
        if not isinstance(self.tx_enabled, bool):
            raise ValueError("tx_enabled must be a bool")
        if not isinstance(self.zero_only, bool):
            raise ValueError("zero_only must be a bool")
        if (
            isinstance(self.write_timeout_s, bool)
            or not isinstance(self.write_timeout_s, (int, float))
            or not math.isfinite(self.write_timeout_s)
            or self.write_timeout_s <= 0.0
        ):
            raise ValueError("write_timeout_s must be positive")
        if (
            isinstance(self.drain_timeout_s, bool)
            or not isinstance(self.drain_timeout_s, (int, float))
            or not math.isfinite(self.drain_timeout_s)
            or self.drain_timeout_s <= 0.0
        ):
            raise ValueError("drain_timeout_s must be positive")
        if (
            isinstance(self.zero_rate_hz, bool)
            or not isinstance(self.zero_rate_hz, (int, float))
            or not math.isfinite(self.zero_rate_hz)
            or self.zero_rate_hz < 20.0
        ):
            raise ValueError("zero_rate_hz must be at least 20 Hz")
        if (
            isinstance(self.close_zero_repetitions, bool)
            or not isinstance(self.close_zero_repetitions, int)
            or self.close_zero_repetitions < 1
        ):
            raise ValueError("close_zero_repetitions must be positive")


class ZeroOnlySerialTransport:
    """Own one O_RDWR descriptor while making nonzero output impossible."""

    def __init__(self, config: ZeroOnlyTransportConfig) -> None:
        config.validate()
        self.config = config
        self.fd: Optional[int] = None
        self.original_termios: Optional[List[object]] = None
        self.parser = FeedbackParser()
        self.tx_frames = 0
        self.tx_bytes = 0
        self.rx_bytes = 0
        self.write_failures = 0
        self.last_tx_monotonic: Optional[float] = None
        self.last_valid_rx_monotonic: Optional[float] = None
        self.close_zero_succeeded: Optional[bool] = None
        self.last_close_error: Optional[str] = None
        self.auto_enable_zero_sequences = 0
        self.auto_session_admitted = False

    @property
    def is_open(self) -> bool:
        return self.fd is not None

    @staticmethod
    def _external_owner_pids(endpoint_stat: os.stat_result) -> List[int]:
        """Best-effort discovery of processes with an already-open endpoint.

        ``flock`` and ``TIOCEXCL`` only cover cooperating processes or opens
        that happen after the exclusive ioctl.  On Linux, inspecting procfs
        closes the remaining pre-existing, non-cooperating-owner gap when the
        process descriptors are visible to us.  Permission and race errors
        are intentionally ignored because procfs visibility is not universal;
        the kernel locks below remain authoritative where available.
        """
        if not os.path.isdir("/proc"):
            return []

        current_pid = os.getpid()
        owners = set()
        try:
            processes = os.scandir("/proc")
        except OSError:
            return []
        with processes:
            for process in processes:
                if not process.name.isdigit() or int(process.name) == current_pid:
                    continue
                fd_root = os.path.join(process.path, "fd")
                try:
                    descriptors = os.scandir(fd_root)
                except OSError:
                    continue
                found = False
                with descriptors:
                    for descriptor in descriptors:
                        try:
                            descriptor_stat = os.stat(descriptor.path)
                        except OSError:
                            continue
                        if (
                            stat.S_ISCHR(descriptor_stat.st_mode)
                            and descriptor_stat.st_rdev == endpoint_stat.st_rdev
                        ):
                            owners.add(int(process.name))
                            found = True
                            break
                if found:
                    continue
        return sorted(owners)

    @classmethod
    def _assert_no_external_owner(cls, endpoint_stat: os.stat_result) -> None:
        owners = cls._external_owner_pids(endpoint_stat)
        if owners:
            joined = ", ".join(str(pid) for pid in owners)
            raise TransportOwnerBusyError(
                "serial endpoint is already open by process(es): %s" % joined
            )

    @staticmethod
    def _release_fd(
        fd: int,
        original_termios: Optional[Sequence[object]],
    ) -> List[str]:
        """Restore and release a descriptor, returning cleanup diagnostics."""
        errors: List[str] = []
        if original_termios is not None:
            try:
                termios.tcsetattr(fd, termios.TCSANOW, original_termios)
            except (OSError, termios.error) as exc:
                errors.append("restore termios failed: %s" % exc)
        if hasattr(termios, "TIOCNXCL"):
            try:
                fcntl.ioctl(fd, termios.TIOCNXCL)
            except OSError as exc:
                errors.append("clear exclusive mode failed: %s" % exc)
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError as exc:
            errors.append("unlock failed: %s" % exc)
        try:
            os.close(fd)
        except OSError as exc:
            errors.append("close failed: %s" % exc)
        return errors

    def open(self) -> None:
        if self.fd is not None:
            return
        if not self.config.tx_enabled:
            raise TransportDisabledError("tx_enabled=false; serial was not opened")
        if not self.config.zero_only:
            raise TransportDisabledError("zero_only must remain true in this stage")

        try:
            endpoint_stat = os.stat(self.config.device)
        except OSError as exc:
            raise TransportError("cannot stat serial endpoint: %s" % exc) from exc
        if not stat.S_ISCHR(endpoint_stat.st_mode):
            raise TransportError("configured serial endpoint is not a character device")
        self._assert_no_external_owner(endpoint_stat)

        try:
            fd = os.open(
                self.config.device,
                os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK,
            )
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EBUSY):
                raise TransportOwnerBusyError(str(exc)) from exc
            raise TransportError("cannot open serial endpoint: %s" % exc) from exc

        original: Optional[List[object]] = None
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if hasattr(termios, "TIOCEXCL"):
                fcntl.ioctl(fd, termios.TIOCEXCL)
            # Recheck after TIOCEXCL to close the race with an owner that
            # opened the endpoint between the preflight scan and our open.
            self._assert_no_external_owner(endpoint_stat)
            original = termios.tcgetattr(fd)
            configured = list(original)
            configured[0] = 0
            configured[1] = 0
            configured[2] = termios.CLOCAL | termios.CREAD | termios.CS8
            configured[3] = 0
            configured[4] = BAUD_CONSTANTS[self.config.baud]
            configured[5] = BAUD_CONSTANTS[self.config.baud]
            configured[6] = list(original[6])
            configured[6][termios.VMIN] = 0
            configured[6][termios.VTIME] = 0
            termios.tcsetattr(fd, termios.TCSANOW, configured)
        except Exception as exc:
            self._release_fd(fd, original)
            if isinstance(exc, TransportOwnerBusyError):
                raise
            if isinstance(exc, BlockingIOError) or getattr(exc, "errno", None) in (
                errno.EACCES,
                errno.EAGAIN,
                errno.EBUSY,
            ):
                raise TransportOwnerBusyError(str(exc)) from exc
            raise TransportError("cannot configure serial endpoint: %s" % exc) from exc

        self.fd = fd
        self.original_termios = original
        self.parser.buffer.clear()
        self.last_tx_monotonic = None
        self.last_valid_rx_monotonic = None
        self.close_zero_succeeded = None
        self.last_close_error = None
        self.auto_session_admitted = False

    def _write_exact_frame(self, frame: bytes, frame_name: str) -> None:
        if self.fd is None:
            raise TransportError("serial endpoint is not open")

        deadline = time.monotonic() + self.config.write_timeout_s
        offset = 0
        while offset < len(frame):
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                self.write_failures += 1
                raise TransportError("%s write timed out" % frame_name)
            try:
                written = os.write(self.fd, frame[offset:])
            except InterruptedError:
                continue
            except BlockingIOError:
                wait_remaining = deadline - time.monotonic()
                if wait_remaining <= 0.0:
                    self.write_failures += 1
                    raise TransportError("%s write timed out" % frame_name)
                try:
                    select.select([], [self.fd], [], wait_remaining)
                except InterruptedError:
                    continue
                except OSError as exc:
                    self.write_failures += 1
                    raise TransportError(
                        "%s write wait failed: %s" % (frame_name, exc)
                    ) from exc
                continue
            except OSError as exc:
                self.write_failures += 1
                raise TransportError(
                    "%s write failed: %s" % (frame_name, exc)
                ) from exc

            if written <= 0:
                self.write_failures += 1
                raise TransportError("%s write made no progress" % frame_name)
            offset += written
            # A successful syscall that completed after the deadline is still
            # a failed bounded write from the caller's perspective.
            if time.monotonic() > deadline:
                self.write_failures += 1
                raise TransportError("%s write timed out" % frame_name)

        self.tx_frames += 1
        self.tx_bytes += len(frame)
        self.last_tx_monotonic = time.monotonic()

    def _write_frame(self, frame: bytes) -> None:
        """Keep the generic raw-write boundary limited to ordinary zero."""
        if frame != ZERO_COMMAND_FRAME:
            raise NonZeroFrameBlockedError(
                "only the exact ordinary mode-0 zero frame is allowed"
            )
        self._write_exact_frame(frame, "zero-frame")

    def send_zero(self) -> None:
        if not self.config.tx_enabled or not self.config.zero_only:
            raise TransportDisabledError("zero output is not enabled")
        self._write_frame(ZERO_COMMAND_FRAME)

    def send_auto_enable_zero_sequence(self) -> None:
        """Admit AUTO using only the fixed handshake and a fresh zero frame.

        The handshake is deliberately unavailable through ``_write_frame``.
        If the second write fails, the STM32 remains in AUTO_WAIT_FIRST and
        its 250 ms timeout returns it to the latched stop state.
        """
        if not self.config.tx_enabled or not self.config.zero_only:
            raise TransportDisabledError("zero-only AUTO admission is not enabled")
        if self.fd is None:
            raise TransportError("serial endpoint is not open")

        self._write_exact_frame(
            AUTO_ENABLE_HANDSHAKE_FRAME,
            "zero-motion AUTO-enable handshake",
        )
        self._write_frame(ZERO_COMMAND_FRAME)
        self.auto_enable_zero_sequences += 1
        self.auto_session_admitted = True

    def revoke_auto_session(self) -> None:
        """Require a fresh explicit handshake before later motion output."""
        self.auto_session_admitted = False

    def poll_feedback(self, max_reads: int = 8) -> List[FeedbackFrame]:
        if self.fd is None:
            raise TransportError("serial endpoint is not open")
        if (
            isinstance(max_reads, bool)
            or not isinstance(max_reads, int)
            or max_reads < 1
        ):
            raise ValueError("max_reads must be a positive integer")
        frames: List[FeedbackFrame] = []
        for _ in range(max_reads):
            try:
                chunk = os.read(self.fd, 4096)
            except BlockingIOError:
                break
            except InterruptedError:
                continue
            except OSError as exc:
                raise TransportError("feedback read failed: %s" % exc) from exc
            if not chunk:
                break
            self.rx_bytes += len(chunk)
            decoded = self.parser.feed(chunk)
            if decoded:
                self.last_valid_rx_monotonic = time.monotonic()
                frames.extend(decoded)
            if len(chunk) < 4096:
                break
        return frames

    def _wait_output_drained(self) -> None:
        if self.fd is None:
            raise TransportError("serial endpoint is not open")
        if not hasattr(termios, "TIOCOUTQ"):
            raise TransportError("TIOCOUTQ is unavailable on this platform")

        deadline = time.monotonic() + self.config.drain_timeout_s
        pending = array.array("i", [0])
        while True:
            try:
                fcntl.ioctl(self.fd, termios.TIOCOUTQ, pending, True)
            except InterruptedError:
                if time.monotonic() >= deadline:
                    raise TransportError("zero-frame drain timed out")
                continue
            except OSError as exc:
                raise TransportError("zero-frame drain failed: %s" % exc) from exc
            if pending[0] <= 0:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TransportError("zero-frame drain timed out")
            time.sleep(min(remaining, 0.001))

    def run_zero_only(self, duration_s: float) -> None:
        """Send only zero frames at the configured cadence for a bounded time."""
        if (
            isinstance(duration_s, bool)
            or not isinstance(duration_s, (int, float))
            or not math.isfinite(duration_s)
            or duration_s <= 0.0
        ):
            raise ValueError("duration_s must be positive")
        if self.fd is None:
            raise TransportError("serial endpoint is not open")

        interval = 1.0 / self.config.zero_rate_hz
        end = time.monotonic() + duration_s
        next_send = time.monotonic()
        while True:
            now = time.monotonic()
            if now >= end:
                break
            if now >= next_send:
                self.send_zero()
                next_send += interval
                if next_send < now:
                    next_send = now + interval
            self.poll_feedback()
            sleep_s = min(max(next_send - time.monotonic(), 0.0), 0.005)
            if sleep_s > 0.0:
                time.sleep(sleep_s)

    def close(self) -> bool:
        if self.fd is None:
            return bool(self.close_zero_succeeded)
        fd = self.fd
        close_errors = []
        zero_writes = 0
        self.close_zero_succeeded = False

        if self.config.tx_enabled and self.config.zero_only:
            for _ in range(self.config.close_zero_repetitions):
                try:
                    self._write_frame(ZERO_COMMAND_FRAME)
                    zero_writes += 1
                except TransportError as exc:
                    close_errors.append(str(exc))
            try:
                self._wait_output_drained()
            except TransportError as exc:
                close_errors.append(str(exc))

        close_errors.extend(self._release_fd(fd, self.original_termios))
        self.original_termios = None
        self.fd = None
        self.last_tx_monotonic = None
        self.last_valid_rx_monotonic = None
        self.parser.buffer.clear()
        self.auto_session_admitted = False

        self.close_zero_succeeded = (
            zero_writes == self.config.close_zero_repetitions
            and not close_errors
        )
        self.last_close_error = "; ".join(close_errors) or None
        return self.close_zero_succeeded

    def __enter__(self) -> "ZeroOnlySerialTransport":
        self.open()
        try:
            self.send_zero()
        except BaseException as exc:
            # A failed startup zero must release the owner, but must not hide
            # the original failure or pretend that close zeros succeeded by
            # issuing a second batch during cleanup.
            fd = self.fd
            cleanup_errors: List[str] = []
            if fd is not None:
                cleanup_errors = self._release_fd(fd, self.original_termios)
            self.fd = None
            self.original_termios = None
            self.last_tx_monotonic = None
            self.last_valid_rx_monotonic = None
            self.parser.buffer.clear()
            self.close_zero_succeeded = False
            errors = ["startup zero failed: %s" % exc]
            errors.extend(cleanup_errors)
            self.last_close_error = "; ".join(errors)
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class GuardedSerialTransport(ZeroOnlySerialTransport):
    """Explicit typed motion output behind AUTO admission and fixed limits."""

    def __init__(
        self,
        config: ZeroOnlyTransportConfig,
        motion_limits: GuardedMotionLimits,
    ) -> None:
        super().__init__(config)
        motion_limits.validate()
        self.motion_limits = motion_limits
        self.motion_frames = 0

    def send_guarded_velocity(
        self,
        velocity_x_m_s: float,
        velocity_y_m_s: float,
        angular_z_rad_s: float,
    ) -> None:
        values = (velocity_x_m_s, velocity_y_m_s, angular_z_rad_s)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            raise UnsafeMotionCommandError("motion command must contain finite numbers")
        if velocity_y_m_s != 0.0:
            raise UnsafeMotionCommandError("V550 lateral velocity must remain exactly zero")
        if abs(velocity_x_m_s) > self.motion_limits.max_linear_x_m_s:
            raise UnsafeMotionCommandError("linear X command exceeds the reviewed limit")
        if abs(angular_z_rad_s) > self.motion_limits.max_angular_z_rad_s:
            raise UnsafeMotionCommandError("angular Z command exceeds the reviewed limit")
        if not self.auto_session_admitted:
            raise UnsafeMotionCommandError(
                "fresh zero-motion AUTO admission is required before motion"
            )

        frame = encode_velocity_command(
            velocity_x_m_s,
            0.0,
            angular_z_rad_s,
        )
        self._write_exact_frame(frame, "guarded-motion-frame")
        self.motion_frames += 1
