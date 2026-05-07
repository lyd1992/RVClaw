from __future__ import annotations

import argparse
import json
from pathlib import Path

from rvclaw.api import run_demo
from rvclaw.container.manager import (
    DEFAULT_BUILD_DIR,
    DEFAULT_GCC15_IMAGE,
    DEFAULT_MNN_REF,
    build_mnn_script,
    collect_container_doctor,
    format_container_doctor,
    format_mnn_plan,
    run_mnn_container_build,
)
from rvclaw.install.manager import (
    DEFAULT_DEPS_DIR,
    build_install_script,
    collect_doctor,
    format_doctor_report,
    list_backend_specs,
    normalize_backend_key,
    plan_backend_install,
    run_backend_install,
    write_script,
)


DEFAULT_GOAL = "检查 A-03 区域设备状态并生成报告"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rvclaw", description="RVClaw Demo Claw v0.1 runtime CLI")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="run a Demo Claw inspection task")
    run_parser.add_argument("goal", nargs="?", default=DEFAULT_GOAL, help="natural-language task goal")
    run_parser.add_argument("--planner", default="auto", choices=["auto", "mock", "claude"], help="planner backend")
    run_parser.add_argument("--runs-dir", default="runs", help="directory for run artifacts")
    run_parser.add_argument("--memory-db", default=None, help="SQLite memory database path")
    run_parser.add_argument("--json", action="store_true", help="print machine-readable summary")

    replay_parser = subparsers.add_parser("replay", help="print a saved trace.jsonl")
    replay_parser.add_argument("trace", help="path to trace.jsonl")

    doctor_parser = subparsers.add_parser("doctor", help="inspect backend build prerequisites")
    doctor_parser.add_argument("--json", action="store_true", help="print machine-readable diagnostics")

    install_parser = subparsers.add_parser("install", help="manage optional backend source installs")
    install_subparsers = install_parser.add_subparsers(dest="install_command")

    install_subparsers.add_parser("list", help="list known optional backends")

    plan_parser = install_subparsers.add_parser("plan", help="print backend install commands")
    _add_install_args(plan_parser)

    script_parser = install_subparsers.add_parser("script", help="write a backend install shell script")
    _add_install_args(script_parser)
    script_parser.add_argument("--output", default="deploy/sg2044/install_backends.sh", help="script output path")

    run_install_parser = install_subparsers.add_parser("run", help="execute backend install commands")
    _add_install_args(run_install_parser)
    run_install_parser.add_argument("--yes", action="store_true", help="confirm execution of network/build commands")

    container_parser = subparsers.add_parser("container", help="manage Docker-based SG2044 build workflows")
    container_subparsers = container_parser.add_subparsers(dest="container_command")

    container_doctor = container_subparsers.add_parser("doctor", help="inspect Docker and RVClaw images")
    container_doctor.add_argument("--image", default=DEFAULT_GCC15_IMAGE, help="GCC 15.1 toolchain image")
    container_doctor.add_argument("--json", action="store_true", help="print machine-readable diagnostics")

    mnn_parser = container_subparsers.add_parser("mnn", help="manage MNN container build workflow")
    mnn_subparsers = mnn_parser.add_subparsers(dest="mnn_command")

    mnn_plan = mnn_subparsers.add_parser("plan", help="print the MNN container build plan")
    _add_mnn_container_args(mnn_plan)

    mnn_script = mnn_subparsers.add_parser("script", help="write an MNN container build script")
    _add_mnn_container_args(mnn_script)
    mnn_script.add_argument("--output", default="deploy/sg2044/mnn_container_build.sh", help="script output path")

    mnn_run = mnn_subparsers.add_parser("run", help="execute the MNN container build workflow")
    _add_mnn_container_args(mnn_run)
    mnn_run.add_argument("--yes", action="store_true", help="confirm Docker execution")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in {None, "run"}:
        summary = run_demo(
            goal=getattr(args, "goal", DEFAULT_GOAL),
            runs_dir=getattr(args, "runs_dir", "runs"),
            planner_name=getattr(args, "planner", "auto"),
            memory_db=getattr(args, "memory_db", None),
        )
        if getattr(args, "json", False):
            print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(f"RVClaw run {summary.run_id} finished: {summary.status}")
            print(f"  run_dir: {summary.run_dir}")
            print(f"  report:  {summary.report_path}")
            print(f"  metrics: {summary.metrics_path}")
            print(f"  trace:   {summary.trace_path}")
        return 0 if summary.status == "completed" else 1

    if args.command == "replay":
        print(Path(args.trace).read_text(encoding="utf-8"))
        return 0

    if args.command == "doctor":
        report = collect_doctor()
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(format_doctor_report(report))
        return 0

    if args.command == "install":
        return _handle_install(args, parser)

    if args.command == "container":
        return _handle_container(args, parser)

    parser.print_help()
    return 1


def _add_install_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "backends",
        nargs="*",
        default=None,
        help="backend names: llama_cpp, mnn, vllm; defaults to all",
    )
    parser.add_argument("--deps-dir", default=DEFAULT_DEPS_DIR, help="source checkout/build directory")
    parser.add_argument(
        "--ref",
        action="append",
        default=[],
        metavar="BACKEND=REF",
        help="pin a backend git ref, for example llama_cpp=b1234 or mnn=3.5.0",
    )


def _add_mnn_container_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image", default=DEFAULT_GCC15_IMAGE, help="GCC 15.1 toolchain image")
    parser.add_argument("--mnn-ref", default=DEFAULT_MNN_REF, help="MNN git ref to build")
    parser.add_argument("--build-dir", default=DEFAULT_BUILD_DIR, help="MNN CMake build directory")


def _handle_install(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    command = getattr(args, "install_command", None)
    if command is None:
        print("Missing install subcommand. Use: rvclaw install list|plan|script|run")
        return 1

    if command == "list":
        for spec in list_backend_specs():
            print(f"{spec.key:10} {spec.status:18} {spec.display_name} - {spec.purpose}")
        return 0

    refs = _parse_refs(getattr(args, "ref", []))
    backends = getattr(args, "backends", None) or None
    deps_dir = getattr(args, "deps_dir", DEFAULT_DEPS_DIR)

    if command == "plan":
        for spec, commands in plan_backend_install(backends, deps_dir=deps_dir, refs=refs):
            print(f"# {spec.display_name} ({spec.status})")
            print(f"# docs: {spec.official_docs}")
            for note in spec.notes:
                print(f"# note: {note}")
            for command_line in commands:
                print(command_line)
            print()
        return 0

    if command == "script":
        script = build_install_script(backends, deps_dir=deps_dir, refs=refs)
        output = write_script(args.output, script)
        print(output)
        return 0

    if command == "run":
        if not args.yes:
            print("Refusing to execute install commands without --yes. Use 'rvclaw install plan' first.")
            return 2
        return run_backend_install(backends, deps_dir=deps_dir, refs=refs)

    parser.print_help()
    return 1


def _handle_container(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    command = getattr(args, "container_command", None)
    if command is None:
        print("Missing container subcommand. Use: rvclaw container doctor|mnn")
        return 1

    if command == "doctor":
        report = collect_container_doctor(image=args.image)
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(format_container_doctor(report))
        return 0

    if command == "mnn":
        mnn_command = getattr(args, "mnn_command", None)
        if mnn_command is None:
            print("Missing MNN subcommand. Use: rvclaw container mnn plan|script|run")
            return 1

        if mnn_command == "plan":
            print(format_mnn_plan(image=args.image, mnn_ref=args.mnn_ref, build_dir=args.build_dir))
            return 0

        if mnn_command == "script":
            script = build_mnn_script(image=args.image, mnn_ref=args.mnn_ref, build_dir=args.build_dir)
            output = write_script(args.output, script)
            print(output)
            return 0

        if mnn_command == "run":
            if not args.yes:
                print("Refusing to execute Docker workflow without --yes. Use 'rvclaw container mnn plan' first.")
                return 2
            return run_mnn_container_build(image=args.image, mnn_ref=args.mnn_ref, build_dir=args.build_dir)

    parser.print_help()
    return 1


def _parse_refs(values: list[str]) -> dict[str, str]:
    refs: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--ref must use BACKEND=REF format, got: {value}")
        key, ref = value.split("=", 1)
        refs[normalize_backend_key(key)] = ref
    return refs
