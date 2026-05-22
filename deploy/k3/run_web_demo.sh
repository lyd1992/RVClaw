#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

mkdir -p "$RVCLAW_LOG_DIR" "$RVCLAW_UPLOADS_DIR" "$(dirname "$RVCLAW_VISION_SOURCE")"

DEFAULT_VISION_SOURCE="$RVCLAW_DATA_DIR/cache/a03_normal.png"
if [ ! -f "$RVCLAW_VISION_SOURCE" ] || [ "$RVCLAW_VISION_SOURCE" = "$DEFAULT_VISION_SOURCE" ]; then
  python3 - "$RVCLAW_VISION_SOURCE" <<'PY'
import binascii
import struct
import sys
import zlib
from pathlib import Path


def chunk(tag: bytes, payload: bytes) -> bytes:
    crc = binascii.crc32(tag + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)


def make_sample_png(width: int = 640, height: int = 360) -> bytes:
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            r = 18 + (x * 18 // width)
            g = 28 + (y * 24 // height)
            b = 32 + (x * 20 // width)
            if 70 <= x <= 570 and 70 <= y <= 285:
                r, g, b = 28, 42, 46
            if 96 <= x <= 544 and 108 <= y <= 245:
                r, g, b = 37, 53, 57
            dx = x - 498
            dy = y - 176
            if dx * dx + dy * dy <= 34 * 34:
                r, g, b = 46, 216, 135
            if 138 <= x <= 368 and 146 <= y <= 169:
                r, g, b = 70, 94, 101
            if 138 <= x <= 320 and 194 <= y <= 217:
                r, g, b = 58, 82, 88
            rows.extend((r, g, b))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(rows), 9))
        + chunk(b"IEND", b"")
    )


Path(sys.argv[1]).write_bytes(make_sample_png())
PY
fi

if [ "${RVCLAW_DEVICE_BACKEND:-mock}" = "mock" ]; then
  export RVCLAW_DEVICE_BACKEND="cv_sample"
fi

cd "$RVCLAW_HOME"
python3 -m rvclaw serve \
  --host "$RVCLAW_WEB_HOST" \
  --port "$RVCLAW_WEB_PORT" \
  --planner "${RVCLAW_PLANNER:-llama_cpp}" \
  --runs-dir "$RVCLAW_RUNS_DIR" 2>&1 | tee "$RVCLAW_LOG_DIR/web.log"
