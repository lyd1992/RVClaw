#!/usr/bin/env python3

import argparse
import fcntl
import json
import os
import select
import signal
import sys
import termios
import time
import tty


class ProbeInterrupted(Exception):
    pass


def handle_signal(signum, _frame):
    raise ProbeInterrupted(f"received signal {signum}")


def parse_args():
    parser = argparse.ArgumentParser(description="Probe raw M10 UART receive data safely")
    parser.add_argument("--device", default="/dev/wheeltec_lidar")
    parser.add_argument("--baud", type=int, choices=(230400, 460800, 500000, 921600), default=460800)
    parser.add_argument("--motor-action", choices=("cycle", "none", "stop"), default="cycle")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--max-bytes", type=int, default=65536)
    return parser.parse_args()


def motor_command(start):
    command = bytearray(188)
    command[0:3] = b"\xa5\x5a\x55"
    command[184] = 0x01 if start else 0x03
    command[185] = 0x01 if start else 0x00
    command[186:188] = b"\xfa\xfb"
    return command


def write_all(fd, data, timeout_seconds=2.0):
    view = memoryview(data)
    deadline = time.monotonic() + timeout_seconds
    total = 0
    while view and time.monotonic() < deadline:
        _, writable, _ = select.select([], [fd], [], 0.2)
        if not writable:
            continue
        written = os.write(fd, view)
        total += written
        view = view[written:]
    return total


def count_headers(data):
    return sum(1 for index in range(len(data) - 1) if data[index : index + 2] == b"\xa5\x5a")


def main():
    args = parse_args()
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    resolved_device = os.path.realpath(args.device)
    fd = None
    original_attributes = None
    start_bytes_written = 0
    stop_bytes_written = 0
    chunks = []
    error = None
    baud_constants = {
        230400: termios.B230400,
        460800: termios.B460800,
        500000: termios.B500000,
        921600: termios.B921600,
    }

    try:
        fd = os.open(resolved_device, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        fcntl.ioctl(fd, termios.TIOCEXCL)
        original_attributes = termios.tcgetattr(fd)
        tty.setraw(fd, termios.TCSANOW)
        attributes = termios.tcgetattr(fd)
        attributes[2] &= ~(termios.PARENB | termios.CSTOPB | termios.CSIZE)
        if hasattr(termios, "CRTSCTS"):
            attributes[2] &= ~termios.CRTSCTS
        attributes[2] |= termios.CLOCAL | termios.CREAD | termios.CS8
        attributes[4] = baud_constants[args.baud]
        attributes[5] = baud_constants[args.baud]
        attributes[6][termios.VMIN] = 0
        attributes[6][termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSANOW, attributes)
        termios.tcflush(fd, termios.TCIOFLUSH)

        if args.motor_action == "cycle":
            start_bytes_written = write_all(fd, motor_command(True))
            termios.tcdrain(fd)
        elif args.motor_action == "stop":
            stop_bytes_written = write_all(fd, motor_command(False))
            termios.tcdrain(fd)

        deadline = time.monotonic() + args.duration
        received = 0
        while time.monotonic() < deadline and received < args.max_bytes:
            readable, _, _ = select.select([fd], [], [], 0.25)
            if not readable:
                continue
            chunk = os.read(fd, min(4096, args.max_bytes - received))
            if chunk:
                chunks.append(chunk)
                received += len(chunk)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if fd is not None:
            try:
                if args.motor_action == "cycle":
                    stop_bytes_written = write_all(fd, motor_command(False))
                    termios.tcdrain(fd)
                    time.sleep(0.5)
            except Exception as exc:
                if error is None:
                    error = f"stop failed: {type(exc).__name__}: {exc}"
                else:
                    error += f"; stop failed: {type(exc).__name__}: {exc}"
            try:
                if original_attributes is not None:
                    termios.tcsetattr(fd, termios.TCSANOW, original_attributes)
            finally:
                os.close(fd)

    data = b"".join(chunks)
    result = {
        "device": resolved_device,
        "baud": args.baud,
        "motor_action": args.motor_action,
        "duration_seconds": args.duration,
        "start_bytes_written": start_bytes_written,
        "stop_bytes_written": stop_bytes_written,
        "bytes_received": len(data),
        "a5_5a_header_count": count_headers(data),
        "first_128_bytes_hex": data[:128].hex(" "),
        "error": error,
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    expected_start_bytes = 188 if args.motor_action == "cycle" else 0
    expected_stop_bytes = 188 if args.motor_action in ("cycle", "stop") else 0
    if (
        error is not None
        or start_bytes_written != expected_start_bytes
        or stop_bytes_written != expected_stop_bytes
    ):
        print("M10_SERIAL_PROBE_ERROR", file=sys.stderr)
        return 2
    if not data:
        print("M10_SERIAL_NO_RX_DATA", file=sys.stderr)
        return 1

    print("M10_SERIAL_RX_DATA_PRESENT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
