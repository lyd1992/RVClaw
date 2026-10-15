"""Parser for WHEELTEC 24-byte STM32 feedback frames."""

from dataclasses import dataclass
import math
from typing import List


FRAME_HEADER = 0x7B
FRAME_TAIL = 0x7D
FRAME_SIZE = 24
ACCEL_RATIO = 1671.84
GYRO_RATIO = 0.00026644
COMMAND_FRAME_SIZE = 11
COMMAND_SCALE = 1000.0
NORMAL_COMMAND_SECURITY = 0x00
AUTO_ENABLE_SECURITY = 0xA5


def signed_i16(high: int, low: int) -> int:
    return int.from_bytes(bytes((high, low)), byteorder="big", signed=True)


def frame_checksum(frame: bytes) -> int:
    checksum = 0
    for value in frame[:22]:
        checksum ^= value
    return checksum


def command_checksum(frame: bytes) -> int:
    checksum = 0
    for value in frame[:9]:
        checksum ^= value
    return checksum


def _encode_command_value(value: float) -> bytes:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("command values must be finite")
    scaled = int(value * COMMAND_SCALE)
    if scaled < -32768 or scaled > 32767:
        raise ValueError("command value exceeds signed 16-bit protocol range")
    return scaled.to_bytes(2, byteorder="big", signed=True)


def encode_velocity_command(
    velocity_x_m_s: float,
    velocity_y_m_s: float,
    angular_z_rad_s: float,
    auto_recharge: int = 0,
    security: int = NORMAL_COMMAND_SECURITY,
) -> bytes:
    """Encode an ordinary motion frame with the security byte held at zero."""
    if (
        isinstance(auto_recharge, bool)
        or not isinstance(auto_recharge, int)
        or not 0 <= auto_recharge <= 255
        or isinstance(security, bool)
        or not isinstance(security, int)
        or not 0 <= security <= 255
    ):
        raise ValueError("command flag bytes must be in the range 0..255")
    if security != NORMAL_COMMAND_SECURITY:
        raise ValueError(
            "ordinary motion frames require security=0; "
            "use encode_auto_enable_handshake() for AUTO enable"
        )

    frame = bytearray(COMMAND_FRAME_SIZE)
    frame[0] = FRAME_HEADER
    frame[1] = auto_recharge
    frame[2] = NORMAL_COMMAND_SECURITY
    frame[3:5] = _encode_command_value(velocity_x_m_s)
    frame[5:7] = _encode_command_value(velocity_y_m_s)
    frame[7:9] = _encode_command_value(angular_z_rad_s)
    frame[9] = command_checksum(frame)
    frame[10] = FRAME_TAIL
    return bytes(frame)


def encode_auto_enable_handshake() -> bytes:
    """Encode the unique all-zero AUTO-enable handshake frame."""
    frame = bytearray(COMMAND_FRAME_SIZE)
    frame[0] = FRAME_HEADER
    frame[1] = 0
    frame[2] = AUTO_ENABLE_SECURITY
    frame[3:9] = b"\x00" * 6
    frame[9] = command_checksum(frame)
    frame[10] = FRAME_TAIL
    return bytes(frame)


@dataclass(frozen=True)
class FeedbackFrame:
    software_disable_flag_raw: int
    velocity_x_m_s: float
    velocity_y_m_s: float
    angular_z_rad_s: float
    acceleration_x_m_s2: float
    acceleration_y_m_s2: float
    acceleration_z_m_s2: float
    angular_velocity_x_rad_s: float
    angular_velocity_y_rad_s: float
    angular_velocity_z_rad_s: float
    battery_voltage_v: float
    raw: bytes


def decode_frame(frame: bytes) -> FeedbackFrame:
    if len(frame) != FRAME_SIZE:
        raise ValueError("feedback frame must be exactly 24 bytes")
    if frame[0] != FRAME_HEADER or frame[-1] != FRAME_TAIL:
        raise ValueError("invalid feedback frame boundary")
    if frame_checksum(frame) != frame[22]:
        raise ValueError("invalid feedback frame checksum")

    return FeedbackFrame(
        software_disable_flag_raw=frame[1],
        velocity_x_m_s=signed_i16(frame[2], frame[3]) / 1000.0,
        velocity_y_m_s=signed_i16(frame[4], frame[5]) / 1000.0,
        angular_z_rad_s=signed_i16(frame[6], frame[7]) / 1000.0,
        acceleration_x_m_s2=signed_i16(frame[8], frame[9]) / ACCEL_RATIO,
        acceleration_y_m_s2=signed_i16(frame[10], frame[11]) / ACCEL_RATIO,
        acceleration_z_m_s2=signed_i16(frame[12], frame[13]) / ACCEL_RATIO,
        angular_velocity_x_rad_s=signed_i16(frame[14], frame[15]) * GYRO_RATIO,
        angular_velocity_y_rad_s=signed_i16(frame[16], frame[17]) * GYRO_RATIO,
        angular_velocity_z_rad_s=signed_i16(frame[18], frame[19]) * GYRO_RATIO,
        battery_voltage_v=int.from_bytes(frame[20:22], "big") / 1000.0,
        raw=frame,
    )


class FeedbackParser:
    """Incrementally reassemble and validate feedback frames."""

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.bytes_received = 0
        self.valid_frames = 0
        self.discarded_bytes = 0
        self.tail_failures = 0
        self.checksum_failures = 0

    def feed(self, data: bytes) -> List[FeedbackFrame]:
        self.bytes_received += len(data)
        self.buffer.extend(data)
        frames: List[FeedbackFrame] = []

        while True:
            try:
                header_index = self.buffer.index(FRAME_HEADER)
            except ValueError:
                self.discarded_bytes += len(self.buffer)
                self.buffer.clear()
                break

            if header_index:
                del self.buffer[:header_index]
                self.discarded_bytes += header_index

            if len(self.buffer) < FRAME_SIZE:
                break

            candidate = bytes(self.buffer[:FRAME_SIZE])
            if candidate[-1] != FRAME_TAIL:
                del self.buffer[0]
                self.tail_failures += 1
                continue
            if frame_checksum(candidate) != candidate[22]:
                del self.buffer[0]
                self.checksum_failures += 1
                continue

            frames.append(decode_frame(candidate))
            self.valid_frames += 1
            del self.buffer[:FRAME_SIZE]

        return frames
