"""Fail-closed, read-only comparison of the selected map and live /map grid.

The file/catalog binding alone cannot prove which map server is actually
publishing.  This independent gate compares *all* occupied/free/unknown cells
plus the map frame and geometry before route preflight or motion.  It does not
prove localization or physical clearance.
"""

import math
from pathlib import Path

import numpy as np
import yaml


def _yaw_from_quaternion(q):
    values = tuple(float(getattr(q, name)) for name in ("x", "y", "z", "w"))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("live map quaternion is not finite")
    x, y, z, w = values
    norm = sum(value * value for value in values)
    if abs(norm - 1.0) > 1e-3 or abs(x) > 1e-4 or abs(y) > 1e-4:
        raise ValueError("live map origin quaternion is not a planar unit rotation")
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def compare_live_map(map_yaml, message, publisher_count):
    """Return PASS only on a single publisher and an exact occupancy match.

    Callers must obtain a fresh sample from a transient-local /map subscription.
    This validates the currently published grid, not a path printed at launch.
    """
    selected = Path(map_yaml).expanduser().resolve()
    metadata = yaml.safe_load(selected.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("mode", "trinary") != "trinary":
        raise ValueError("only trinary map-server images are supported by this exact check")
    # This check does not need the expensive obstacle-distance field used by
    # localization. Decode just the occupancy values to keep startup short.
    from PIL import Image

    try:
        image = (selected.parent / metadata["image"]).resolve()
        with Image.open(image) as raster:
            pixels = np.asarray(raster.convert("L"), dtype=np.float32)
        resolution = float(metadata["resolution"])
        origin_x, origin_y, origin_yaw = (float(value) for value in metadata["origin"])
        occupied = float(metadata["occupied_thresh"])
        free = float(metadata["free_thresh"])
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError("selected map metadata is incomplete or invalid") from exc
    if not all(math.isfinite(value) for value in (
        resolution, origin_x, origin_y, origin_yaw, occupied, free,
    )) or resolution <= 0 or not 0 <= free < occupied <= 1:
        raise ValueError("selected map geometry or occupancy thresholds are invalid")
    probability = pixels / 255.0 if int(metadata.get("negate", 0)) else (255.0 - pixels) / 255.0
    expected = np.full(pixels.shape, -1, dtype=np.int8)
    expected[probability >= occupied] = 100
    expected[probability <= free] = 0
    expected = np.flipud(expected)
    height, width = expected.shape
    if publisher_count != 1:
        raise ValueError("live /map must have exactly one publisher")
    if getattr(message.header, "frame_id", None) != "map":
        raise ValueError("live /map frame is not map")
    info = message.info
    if (int(info.width), int(info.height)) != (width, height):
        raise ValueError("live /map dimensions differ from selected map")
    if not math.isfinite(float(info.resolution)) or abs(
        float(info.resolution) - resolution
    ) > 1e-7:
        raise ValueError("live /map resolution differs from selected map")
    position = info.origin.position
    if any(not math.isfinite(float(value)) for value in (position.x, position.y, position.z)):
        raise ValueError("live /map origin is not finite")
    yaw = _yaw_from_quaternion(info.origin.orientation)
    if (abs(float(position.x) - origin_x) > 1e-5
            or abs(float(position.y) - origin_y) > 1e-5
            or abs(float(position.z)) > 1e-5
            or abs(math.atan2(math.sin(yaw - origin_yaw),
                               math.cos(yaw - origin_yaw))) > 1e-5):
        raise ValueError("live /map origin differs from selected map")
    cells = np.asarray(message.data, dtype=np.int16)
    if cells.size != width * height:
        raise ValueError("live /map cell count differs from selected map")
    mismatches = int(np.count_nonzero(cells != expected.ravel()))
    if mismatches:
        raise ValueError("live /map cell contents differ from selected map (%d cells)" % mismatches)
    return {
        "status": "PASS", "map_yaml": str(selected),
        "width": width, "height": height,
        "resolution_m": resolution, "cell_count": int(cells.size),
        "mismatch_count": 0, "publisher_count": 1,
    }
