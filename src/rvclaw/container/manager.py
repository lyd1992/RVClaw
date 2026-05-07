from __future__ import annotations

import json
import platform
import shutil
import subprocess
from pathlib import Path

from rvclaw.install.manager import write_script


DEFAULT_GCC15_IMAGE = "rvclaw/sg2044-gcc15.1:openeuler-riscv64"
DEFAULT_MNN_REF = "master"
DEFAULT_BUILD_DIR = "build/mnn-sg2044-gcc15"
DEFAULT_SCRIPT_PATH = "deploy/sg2044/mnn_container_build.sh"


def collect_container_doctor(image: str = DEFAULT_GCC15_IMAGE) -> dict:
    docker = shutil.which("docker")
    report = {
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
            "python": platform.python_version(),
        },
        "docker": {
            "path": docker,
            "version": None,
            "image": image,
            "image_present": False,
        },
    }
    if docker:
        report["docker"]["version"] = _run_capture([docker, "--version"])
        inspect = subprocess.run(
            [docker, "image", "inspect", image],
            text=True,
            capture_output=True,
            check=False,
        )
        report["docker"]["image_present"] = inspect.returncode == 0
    return report


def format_container_doctor(report: dict) -> str:
    host = report["host"]
    docker = report["docker"]
    lines = [
        "RVClaw container doctor",
        "",
        f"Host: {host['system']} {host['release']} / {host['machine']} / Python {host['python']}",
        f"Docker: {docker['path'] or 'MISSING'}",
        f"Docker version: {docker['version'] or '-'}",
        f"GCC15 image: {docker['image']}",
        f"GCC15 image present: {docker['image_present']}",
    ]
    return "\n".join(lines)


def format_mnn_plan(
    image: str = DEFAULT_GCC15_IMAGE,
    mnn_ref: str = DEFAULT_MNN_REF,
    build_dir: str = DEFAULT_BUILD_DIR,
) -> str:
    lines = [
        "# RVClaw MNN container build plan",
        "",
        "# 1. Build or load the GCC 15.1 toolchain image on SG2044:",
        "bash docker/sg2044/gcc15.1/build_image.sh",
        "bash docker/sg2044/gcc15.1/verify_image.sh",
        "",
        "# 2. Build MNN inside the toolchain container:",
        f"RVCLAW_GCC15_IMAGE={image} \\",
        f"RVCLAW_MNN_REF={mnn_ref} \\",
        f"RVCLAW_MNN_BUILD_DIR={build_dir} \\",
        "bash docker/sg2044/mnn/run_container.sh bash docker/sg2044/mnn/build_mnn.sh",
        "",
        "# 3. Expected host artifacts:",
        "third_party/MNN/",
        f"{build_dir}/",
        "runs/env/<run_id>/container_manifest.json",
        "runs/env/<run_id>/mnn_build.log",
    ]
    return "\n".join(lines)


def build_mnn_script(
    image: str = DEFAULT_GCC15_IMAGE,
    mnn_ref: str = DEFAULT_MNN_REF,
    build_dir: str = DEFAULT_BUILD_DIR,
) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
            'RVCLAW_ROOT="${RVCLAW_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"',
            'cd "$RVCLAW_ROOT"',
            "",
            f'export RVCLAW_GCC15_IMAGE="${{RVCLAW_GCC15_IMAGE:-{image}}}"',
            f'export RVCLAW_MNN_REF="${{RVCLAW_MNN_REF:-{mnn_ref}}}"',
            f'export RVCLAW_MNN_BUILD_DIR="${{RVCLAW_MNN_BUILD_DIR:-{build_dir}}}"',
            "",
            'echo "[RVClaw] image: $RVCLAW_GCC15_IMAGE"',
            'echo "[RVClaw] MNN ref: $RVCLAW_MNN_REF"',
            'echo "[RVClaw] MNN build dir: $RVCLAW_MNN_BUILD_DIR"',
            "",
            'bash docker/sg2044/mnn/run_container.sh bash docker/sg2044/mnn/build_mnn.sh',
            "",
        ]
    )


def write_mnn_script(
    output: str | Path = DEFAULT_SCRIPT_PATH,
    image: str = DEFAULT_GCC15_IMAGE,
    mnn_ref: str = DEFAULT_MNN_REF,
    build_dir: str = DEFAULT_BUILD_DIR,
) -> Path:
    return write_script(output, build_mnn_script(image=image, mnn_ref=mnn_ref, build_dir=build_dir))


def run_mnn_container_build(
    image: str = DEFAULT_GCC15_IMAGE,
    mnn_ref: str = DEFAULT_MNN_REF,
    build_dir: str = DEFAULT_BUILD_DIR,
) -> int:
    script = build_mnn_script(image=image, mnn_ref=mnn_ref, build_dir=build_dir)
    bash = shutil.which("bash")
    if not bash:
        print("bash is required to run the MNN container build script.")
        return 127
    completed = subprocess.run([bash, "-lc", script], check=False)
    return completed.returncode


def _run_capture(command: list[str]) -> str | None:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or completed.stderr.strip() or None


def to_json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)
