from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DEPS_DIR = "third_party"


@dataclass(frozen=True, slots=True)
class BackendSpec:
    key: str
    display_name: str
    repo_url: str
    repo_dir: str
    default_ref: str
    status: str
    purpose: str
    official_docs: str
    system_packages: tuple[str, ...]
    configure_commands: tuple[str, ...]
    build_commands: tuple[str, ...]
    verify_commands: tuple[str, ...]
    notes: tuple[str, ...] = ()


BACKENDS: dict[str, BackendSpec] = {
    "llama_cpp": BackendSpec(
        key="llama_cpp",
        display_name="llama.cpp",
        repo_url="https://github.com/ggml-org/llama.cpp.git",
        repo_dir="llama.cpp",
        default_ref="master",
        status="p0-baseline",
        purpose="Local GGUF LLM and embedding baseline.",
        official_docs="https://www.mintlify.com/ggml-org/llama.cpp/development/building",
        system_packages=("git", "gcc", "gcc-c++", "make", "cmake", "ccache"),
        configure_commands=(
            "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release",
        ),
        build_commands=(
            "cmake --build build --config Release -j ${MAX_JOBS:-$(nproc)}",
        ),
        verify_commands=(
            "./build/bin/llama-cli --help >/dev/null 2>&1 || ./build/bin/main --help >/dev/null 2>&1 || true",
        ),
        notes=(
            "Use CFLAGS/CXXFLAGS externally when pinning SG2044-specific march/mtune flags.",
            "Model download is intentionally not part of the source build step.",
        ),
    ),
    "mnn": BackendSpec(
        key="mnn",
        display_name="MNN",
        repo_url="https://github.com/alibaba/MNN.git",
        repo_dir="MNN",
        default_ref="master",
        status="p1-plugin",
        purpose="Edge vision, small-model and future VLM runtime backend.",
        official_docs="https://github.com/alibaba/MNN/wiki/cmake",
        system_packages=("git", "gcc", "gcc-c++", "make", "cmake", "ccache"),
        configure_commands=(
            "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release "
            "-DMNN_BUILD_SHARED_LIBS=ON "
            "-DMNN_BUILD_TOOLS=ON "
            "-DMNN_BUILD_BENCHMARK=ON "
            "-DMNN_BUILD_TEST=OFF "
            "-DMNN_BUILD_CONVERTER=OFF",
        ),
        build_commands=(
            "cmake --build build --config Release -j ${MAX_JOBS:-$(nproc)}",
        ),
        verify_commands=(
            "test -d build && find build -maxdepth 3 -type f -name '*MNN*' | head -20",
        ),
        notes=(
            "Converter/protobuf and GPU backends are not enabled by default for the SG2044 baseline.",
            "Enable additional CMake flags only after the CPU baseline is reproducible.",
        ),
    ),
    "vllm": BackendSpec(
        key="vllm",
        display_name="vLLM",
        repo_url="https://github.com/vllm-project/vllm.git",
        repo_dir="vllm",
        default_ref="main",
        status="p1-p2-experimental",
        purpose="Optional service-style LLM backend after the Demo Claw API is stable.",
        official_docs="https://docs.vllm.ai/en/stable/getting_started/installation/",
        system_packages=("git", "gcc", "gcc-c++", "make", "cmake", "python3", "python3-pip", "ccache"),
        configure_commands=(
            "python3 -m pip install --upgrade pip",
        ),
        build_commands=(
            "python3 -m pip install -r requirements-build.txt || true",
            "VLLM_TARGET_DEVICE=empty python3 -m pip install -e .",
        ),
        verify_commands=(
            "python3 -c \"import vllm; print(getattr(vllm, '__version__', 'unknown'))\" || true",
        ),
        notes=(
            "vLLM does not currently provide a first-class RISC-V hardware backend in the stable docs.",
            "This installer prepares a source/Python development checkout, not an optimized RISC-V serving backend.",
            "Use a dedicated Python environment on SG2044 if you enable this target.",
        ),
    ),
}


def normalize_backend_key(name: str) -> str:
    normalized = name.lower().replace("-", "_").replace(".", "_")
    aliases = {
        "llama": "llama_cpp",
        "llamacpp": "llama_cpp",
        "llama_cpp": "llama_cpp",
        "mnn": "mnn",
        "vllm": "vllm",
    }
    if normalized not in aliases:
        raise KeyError(f"Unknown backend: {name}")
    return aliases[normalized]


def list_backend_specs() -> list[BackendSpec]:
    return [BACKENDS[key] for key in ("llama_cpp", "mnn", "vllm")]


def get_backend_spec(name: str) -> BackendSpec:
    return BACKENDS[normalize_backend_key(name)]


def plan_backend_install(
    backend_names: list[str] | None = None,
    deps_dir: str = DEFAULT_DEPS_DIR,
    refs: dict[str, str] | None = None,
) -> list[tuple[BackendSpec, list[str]]]:
    names = backend_names or [spec.key for spec in list_backend_specs()]
    refs = refs or {}
    return [(get_backend_spec(name), _commands_for_spec(get_backend_spec(name), deps_dir, refs)) for name in names]


def build_install_script(
    backend_names: list[str] | None = None,
    deps_dir: str = DEFAULT_DEPS_DIR,
    refs: dict[str, str] | None = None,
) -> str:
    plan = plan_backend_install(backend_names, deps_dir=deps_dir, refs=refs)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'RVCLAW_ROOT="${RVCLAW_ROOT:-$(pwd)}"',
        f'RVCLAW_DEPS_DIR="${{RVCLAW_DEPS_DIR:-$RVCLAW_ROOT/{deps_dir}}}"',
        'MAX_JOBS="${MAX_JOBS:-$(nproc)}"',
        "",
        'echo "[RVClaw] deps dir: $RVCLAW_DEPS_DIR"',
        "mkdir -p \"$RVCLAW_DEPS_DIR\"",
        "",
    ]
    packages = sorted({package for spec, _ in plan for package in spec.system_packages})
    lines.extend(
        [
            "# Optional openEuler dependency bootstrap:",
            f"# sudo dnf install -y {' '.join(packages)}",
            "",
        ]
    )
    for spec, commands in plan:
        lines.append(f'echo "[RVClaw] installing {spec.display_name} ({spec.status})"')
        for note in spec.notes:
            lines.append(f"# NOTE: {note}")
        lines.extend(commands)
        lines.append("")
    lines.extend(
        [
            'echo "[RVClaw] backend installation plan finished"',
            "",
        ]
    )
    return "\n".join(lines)


def run_backend_install(
    backend_names: list[str] | None = None,
    deps_dir: str = DEFAULT_DEPS_DIR,
    refs: dict[str, str] | None = None,
) -> int:
    script = build_install_script(backend_names, deps_dir=deps_dir, refs=refs)
    bash = shutil.which("bash")
    if bash:
        completed = subprocess.run([bash, "-lc", script], check=False)
    else:
        completed = subprocess.run(script, shell=True, check=False)
    return completed.returncode


def collect_doctor() -> dict:
    return {
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
            "python": platform.python_version(),
        },
        "commands": {name: shutil.which(name) for name in ["git", "python3", "pip3", "gcc", "g++", "cmake", "make", "nproc"]},
        "environment": {
            "RVCLAW_DEPS_DIR": os.environ.get("RVCLAW_DEPS_DIR"),
            "MAX_JOBS": os.environ.get("MAX_JOBS"),
            "CFLAGS": os.environ.get("CFLAGS"),
            "CXXFLAGS": os.environ.get("CXXFLAGS"),
        },
    }


def format_doctor_report(report: dict) -> str:
    lines = ["RVClaw backend environment doctor", ""]
    platform_info = report["platform"]
    lines.append(
        f"Platform: {platform_info['system']} {platform_info['release']} / {platform_info['machine']} / Python {platform_info['python']}"
    )
    lines.append("")
    lines.append("Commands:")
    for name, path in report["commands"].items():
        lines.append(f"  {name:8} {'OK ' + path if path else 'MISSING'}")
    lines.append("")
    lines.append("Environment:")
    for name, value in report["environment"].items():
        lines.append(f"  {name:16} {value or '-'}")
    lines.append("")
    lines.append("Backends:")
    for spec in list_backend_specs():
        lines.append(f"  {spec.key:10} {spec.status:18} {spec.purpose}")
    return "\n".join(lines)


def _commands_for_spec(spec: BackendSpec, deps_dir: str, refs: dict[str, str]) -> list[str]:
    ref = refs.get(spec.key, spec.default_ref)
    target = f'"$RVCLAW_DEPS_DIR/{spec.repo_dir}"'
    commands = [
        f'if [ ! -d {target}/.git ]; then git clone {spec.repo_url} {target}; fi',
        f"cd {target}",
        "git fetch --tags --all --prune",
        f"git checkout {ref}",
    ]
    commands.extend(spec.configure_commands)
    commands.extend(spec.build_commands)
    commands.extend(spec.verify_commands)
    commands.append('cd "$RVCLAW_ROOT"')
    return commands


def write_script(path: str | Path, script: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script + "\n", encoding="utf-8", newline="\n")
    return path
