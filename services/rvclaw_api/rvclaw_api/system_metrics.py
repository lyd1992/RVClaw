import glob
import os
import platform
import shutil
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


_CPU_LOCK = threading.Lock()
_PREVIOUS_CPU_SAMPLE = None


def get_system_metrics():
    started = time.perf_counter()
    memory = _memory_metrics()
    temperatures = _temperature_metrics()
    power = _power_metrics()
    disk = shutil.disk_usage("/")

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": {
            "hostname": socket.gethostname(),
            "architecture": platform.machine(),
            "os": platform.system(),
            "kernel": platform.release(),
            "cpu_model": _cpu_model(),
            "cpu_count": os.cpu_count() or 0,
            "cpu_frequency_mhz": _cpu_frequency_mhz(),
            "uptime_seconds": _read_first_number("/proc/uptime"),
        },
        "usage": {
            "cpu_percent": _cpu_percent(),
            "memory_percent": memory["percent"],
            "memory_used_bytes": memory["used_bytes"],
            "memory_total_bytes": memory["total_bytes"],
            "disk_percent": round((disk.used / disk.total) * 100, 1) if disk.total else 0.0,
            "disk_used_bytes": disk.used,
            "disk_total_bytes": disk.total,
            "load_average": _load_average(),
        },
        "thermal": {
            "available": bool(temperatures),
            "primary_celsius": max(
                (sensor["celsius"] for sensor in temperatures), default=None
            ),
            "sensors": temperatures,
        },
        "power": power,
        "collection_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def _cpu_percent():
    global _PREVIOUS_CPU_SAMPLE
    sample = _read_cpu_times()
    if sample is None:
        return None

    with _CPU_LOCK:
        previous = _PREVIOUS_CPU_SAMPLE
        _PREVIOUS_CPU_SAMPLE = sample

    if previous is None:
        load = _load_average()[0]
        return round(min(100.0, load / max(os.cpu_count() or 1, 1) * 100), 1)

    total_delta = sample[0] - previous[0]
    idle_delta = sample[1] - previous[1]
    if total_delta <= 0:
        return 0.0
    return round(max(0.0, min(100.0, (1 - idle_delta / total_delta) * 100)), 1)


def _read_cpu_times():
    try:
        fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
        values = [int(value) for value in fields]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return sum(values), idle
    except (OSError, ValueError, IndexError):
        return None


def _memory_metrics():
    values = {}
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
        for line in lines:
            key, raw_value = line.split(":", 1)
            values[key] = int(raw_value.strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return {"percent": None, "used_bytes": None, "total_bytes": None}

    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", values.get("MemFree", 0))
    used = max(0, total - available)
    return {
        "percent": round(used / total * 100, 1) if total else None,
        "used_bytes": used,
        "total_bytes": total,
    }


def _temperature_metrics():
    sensors = []
    seen = set()
    for path in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
        zone = Path(path).parent
        label = _read_text(zone / "type") or zone.name
        _append_temperature(sensors, seen, label, path)
    for path in sorted(glob.glob("/sys/class/hwmon/hwmon*/temp*_input")):
        sensor = Path(path)
        chip = _read_text(sensor.parent / "name") or sensor.parent.name
        label_path = sensor.with_name(sensor.name.replace("_input", "_label"))
        label = _read_text(label_path) or sensor.stem.replace("_input", "")
        _append_temperature(sensors, seen, f"{chip} {label}", path)
    return sensors


def _append_temperature(sensors, seen, label, path):
    value = _read_first_number(path)
    if value is None:
        return
    celsius = value / 1000 if abs(value) >= 1000 else value
    key = (label, round(celsius, 2))
    if key in seen or not -40 <= celsius <= 200:
        return
    seen.add(key)
    sensors.append({"label": label, "celsius": round(celsius, 1)})


def _power_metrics():
    sensors = []
    for pattern, kind in (
        ("/sys/class/hwmon/hwmon*/power*_input", "instantaneous"),
        ("/sys/class/hwmon/hwmon*/power*_average", "average"),
    ):
        for path in sorted(glob.glob(pattern)):
            sensor = Path(path)
            value = _read_first_number(sensor)
            if value is None:
                continue
            chip = _read_text(sensor.parent / "name") or sensor.parent.name
            label_path = sensor.with_name(sensor.name.rsplit("_", 1)[0] + "_label")
            sensors.append({
                "label": _read_text(label_path) or chip,
                "watts": round(value / 1_000_000, 3),
                "kind": kind,
            })

    return {
        "available": bool(sensors),
        "watts": sensors[0]["watts"] if sensors else None,
        "sensors": sensors,
        "message": None if sensors else "Power sensor unavailable",
    }


def _cpu_model():
    try:
        models = []
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                model = line.split(":", 1)[1].strip()
                if model and model not in models:
                    models.append(model)
        return " / ".join(models) or platform.processor() or None
    except (OSError, IndexError):
        return platform.processor() or None


def _cpu_frequency_mhz():
    frequencies = []
    for path in glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq"):
        value = _read_first_number(path)
        if value is not None:
            frequencies.append(value / 1000)
    return round(sum(frequencies) / len(frequencies), 1) if frequencies else None


def _load_average():
    try:
        return [round(value, 2) for value in os.getloadavg()]
    except (AttributeError, OSError):
        return [0.0, 0.0, 0.0]


def _read_first_number(path):
    value = _read_text(path)
    if value is None:
        return None
    try:
        return float(value.split()[0])
    except (ValueError, IndexError):
        return None


def _read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None
