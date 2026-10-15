import argparse
import math
from pathlib import Path

import yaml


class MeasurementError(ValueError):
    """Raised when a physical measurement document is incomplete or unsafe."""


def _mapping(value, path):
    if not isinstance(value, dict):
        raise MeasurementError(f"{path} must be a mapping")
    return value


def _number(value, path, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MeasurementError(f"{path} must be a number, not {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise MeasurementError(f"{path} must be finite")
    if number < minimum or number > maximum:
        raise MeasurementError(
            f"{path}={number:g} is outside [{minimum:g}, {maximum:g}]"
        )
    return number


def _required(mapping, key, path, minimum, maximum):
    if key not in mapping or mapping[key] is None:
        raise MeasurementError(f"{path}.{key} is required")
    return _number(mapping[key], f"{path}.{key}", minimum, maximum)


def _optional(mapping, key, path, minimum, maximum):
    value = mapping.get(key)
    if value is None:
        return None
    return _number(value, f"{path}.{key}", minimum, maximum)


def validate_measurements(document):
    root = _mapping(document, "document")
    if root.get("schema_version") != 1:
        raise MeasurementError("schema_version must be 1")
    if root.get("units") != "mm":
        raise MeasurementError("units must be exactly 'mm'")

    footprint = _mapping(root.get("footprint"), "footprint")
    checked_footprint = {
        "front": _required(footprint, "front", "footprint", 10, 2000),
        "rear": _required(footprint, "rear", "footprint", 10, 2000),
        "left": _required(footprint, "left", "footprint", 10, 2000),
        "right": _required(footprint, "right", "footprint", 10, 2000),
        "safety_margin": _required(
            footprint, "safety_margin", "footprint", 0, 500
        ),
    }

    laser = _mapping(root.get("laser"), "laser")
    checked_laser = {
        "x": _required(laser, "x", "laser", -2000, 2000),
        "y": _required(laser, "y", "laser", -2000, 2000),
        "z": _required(laser, "z", "laser", 0, 2000),
        "roll_deg": _required(laser, "roll_deg", "laser", -180, 180),
        "pitch_deg": _required(laser, "pitch_deg", "laser", -180, 180),
        "yaw_deg": _required(laser, "yaw_deg", "laser", -360, 360),
    }

    chassis = root.get("chassis", {})
    if chassis is None:
        chassis = {}
    chassis = _mapping(chassis, "chassis")
    checked_chassis = {
        "wheel_diameter": _optional(
            chassis, "wheel_diameter", "chassis", 10, 1000
        ),
        "front_axle_x": _optional(
            chassis, "front_axle_x", "chassis", -2000, 2000
        ),
        "rear_axle_x": _optional(
            chassis, "rear_axle_x", "chassis", -2000, 2000
        ),
        "left_wheel_y": _optional(
            chassis, "left_wheel_y", "chassis", -2000, 2000
        ),
        "right_wheel_y": _optional(
            chassis, "right_wheel_y", "chassis", -2000, 2000
        ),
    }
    return checked_footprint, checked_laser, checked_chassis


def _metres(value_mm):
    return round(value_mm / 1000.0, 9)


def _radians(value_deg):
    return round(math.radians(value_deg), 9)


def build_geometry_artifact(document, source_name="physical_measurements.yaml"):
    footprint, laser, chassis = validate_measurements(document)
    margin = footprint["safety_margin"]
    front = _metres(footprint["front"] + margin)
    rear = _metres(footprint["rear"] + margin)
    left = _metres(footprint["left"] + margin)
    right = _metres(footprint["right"] + margin)
    points = [
        [-rear, -right],
        [-rear, left],
        [front, left],
        [front, -right],
    ]
    nav2_parameter = "[" + ", ".join(str(point) for point in points) + "]"

    optional_chassis = {
        key: _metres(value)
        for key, value in chassis.items()
        if value is not None
    }
    return {
        "schema_version": 1,
        "review_required": True,
        "source": source_name,
        "frames": {
            "parent": "base_footprint",
            "laser_child": "laser",
        },
        "footprint": {
            "safety_margin_m": _metres(margin),
            "points_m": points,
            "nav2_parameter": nav2_parameter,
            "apply_identically_to": ["local_costmap", "global_costmap"],
        },
        "laser_transform": {
            "xyz_m": [
                _metres(laser["x"]),
                _metres(laser["y"]),
                _metres(laser["z"]),
            ],
            "rpy_rad": [
                _radians(laser["roll_deg"]),
                _radians(laser["pitch_deg"]),
                _radians(laser["yaw_deg"]),
            ],
            "launch_arguments": {
                "laser_x": _metres(laser["x"]),
                "laser_y": _metres(laser["y"]),
                "laser_z": _metres(laser["z"]),
                "laser_roll": _radians(laser["roll_deg"]),
                "laser_pitch": _radians(laser["pitch_deg"]),
                "laser_yaw": _radians(laser["yaw_deg"]),
            },
        },
        "optional_chassis_dimensions_m": optional_chassis,
    }


def generate_file(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    with input_path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    artifact = build_geometry_artifact(document, input_path.name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(artifact, stream, sort_keys=False, allow_unicode=False)
    return artifact


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate RVClaw measurements and generate review YAML."
    )
    parser.add_argument("input", help="physical_measurements.yaml path")
    parser.add_argument(
        "--output",
        required=True,
        help="new review artifact path; production Nav2 YAML is not edited",
    )
    args = parser.parse_args(argv)

    try:
        artifact = generate_file(args.input, args.output)
    except (OSError, yaml.YAMLError, MeasurementError) as error:
        parser.exit(2, f"measurement validation failed: {error}\n")

    print(f"generated review artifact: {args.output}")
    print(f"footprint: {artifact['footprint']['nav2_parameter']}")
    print("review_required: true")


if __name__ == "__main__":
    main()
