"""Client and task coordinator for the local RVClaw vision API."""

from __future__ import annotations

import json
import logging
import math
import os
import signal
import subprocess
import tempfile
from threading import Event, Lock, Thread
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen
from uuid import uuid4

LOG = logging.getLogger("rvclaw_voice")


def _external_script_pid(script_path: str) -> Optional[int]:
    """Return a chassis wrapper process launched outside this voice service."""

    try:
        result = subprocess.run(
            ["pgrep", "-f", script_path],
            capture_output=True,
            check=False,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for value in result.stdout.splitlines():
        try:
            pid = int(value.strip())
        except ValueError:
            continue
        if pid != os.getpid():
            return pid
    return None


class ChassisMotionClient:
    """Run one bounded chassis motion through the reviewed ROS safety wrapper."""

    ARM_TOKEN = "WHEELS_GROUND_PS2_OFF_NAV2_SPIN_LEFT_90"
    CONFIRMATION_TOKEN = "CONFIRM_NAV2_SPIN_LEFT_90_ARMING"

    def __init__(
        self,
        script_path: Optional[str] = None,
        run_root: Optional[str] = None,
        timeout: float = 180.0,
    ):
        self.script_path = script_path or os.getenv(
            "RVCLAW_LEFT_TURN_SCRIPT",
            "/data/rvclaw-k3-lab/scripts/nav2_heading_spin_left_90.sh",
        )
        self.run_root = Path(
            run_root
            or os.getenv("RVCLAW_MOTION_RUN_ROOT", "/data/rvclaw-k3-lab/runs/voice-motion")
        )
        self.timeout = timeout
        self._lock = Lock()
        self._process: Optional[subprocess.Popen] = None
        self._run: Optional[Dict[str, str]] = None

    def status(self) -> Dict[str, Any]:
        with self._lock:
            process = self._process
            run = dict(self._run or {})
        if process is not None:
            return {
                **run,
                "state": "running" if process.poll() is None else "finishing",
                "pid": process.pid,
            }
        external_pid = _external_script_pid(self.script_path)
        if external_pid is not None:
            return {**run, "state": "running", "pid": external_pid, "source": "external"}
        return {"state": "idle", **run}

    def left_turn(self, command: Any) -> Dict[str, Any]:
        angle_deg = int((getattr(command, "slots", {}) or {}).get("angle_deg", 90))
        if angle_deg != 90:
            raise RuntimeError("only the reviewed 90 degree left-turn profile is enabled")
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("another chassis motion is already running")
            self.run_root.mkdir(parents=True, exist_ok=True)
            run_dir = Path(tempfile.mkdtemp(prefix="left-turn-", dir=str(self.run_root)))
            trace_path = run_dir / "trace.json"
            log_path = run_dir / "command.log"
            environment = os.environ.copy()
            environment["RVCLAW_AUTO_RVIZ"] = "off"
            process = subprocess.Popen(
                [
                    "bash",
                    self.script_path,
                    "--trace-file",
                    str(trace_path),
                    "--arm-token",
                    self.ARM_TOKEN,
                ],
                cwd="/data/rvclaw-k3-lab",
                stdout=log_path.open("w", encoding="utf-8"),
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
                start_new_session=True,
                env=environment,
            )
            # The voice command is the operator's explicit arming action.  The
            # wrapper still blocks until every navigation preflight passes
            # before it reads this token, so failed safety checks cannot arm.
            if process.stdin is not None:
                process.stdin.write((self.CONFIRMATION_TOKEN + "\n").encode())
                process.stdin.flush()
                process.stdin.close()
            self._process = process
            self._run = {
                "run_dir": str(run_dir),
                "trace_file": str(trace_path),
                "log_file": str(log_path),
            }
        try:
            return_code = process.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            self.stop()
            raise RuntimeError("left-turn motion timed out") from exc
        finally:
            with self._lock:
                if self._process is process:
                    self._process = None
        if return_code != 0:
            raise RuntimeError("left-turn motion failed; see %s" % self._run["log_file"])
        return {"state": "completed", "motion": "turn_left", "angle_deg": angle_deg, **(self._run or {})}

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            process = self._process
            run = dict(self._run or {})
        if process is None or process.poll() is not None:
            return {"state": "idle", **run}
        try:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=8.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=4.0)
        return {"state": "stopped", **run}


class ChassisRouteClient:
    """Supervise the reviewed route runner, including its own localization."""

    ARM_TOKEN = "WHEELS_GROUND_PS2_OFF_NAV2_WAYPOINT_ROUTE"
    CONFIRMATION_TOKEN = "CONFIRM_NAV2_WAYPOINT_ROUTE_ARMING"
    ARRIVAL_EVENT_DIR = "/data/RVClaw/runtime/waypoint-arrivals"
    CHECKPOINT_PATH = "/data/RVClaw/runtime/inspection-route-checkpoint.json"
    CHECKPOINT_MAX_AGE_S = 86400.0
    DEFAULT_ROUTE = ("A", "B", "C", "ORIGIN")

    def __init__(
        self,
        script_path: Optional[str] = None,
        run_root: Optional[str] = None,
        timeout: float = 660.0,
        task_recorder: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        checkpoint_path: Optional[str] = None,
        checkpoint_max_age_s: Optional[float] = None,
        wall_clock=time.time,
    ):
        self.script_path = script_path or os.getenv(
            "RVCLAW_ROUTE_SCRIPT",
            "/data/rvclaw-k3-lab/scripts/nav2_persistent_route.sh",
        )
        self.run_root = Path(
            run_root
            or os.getenv("RVCLAW_ROUTE_RUN_ROOT", "/data/rvclaw-k3-lab/runs/voice-inspection")
        )
        self.timeout = timeout
        self.checkpoint_path = Path(
            checkpoint_path
            or os.getenv("RVCLAW_ROUTE_CHECKPOINT_FILE", self.CHECKPOINT_PATH)
        )
        self.checkpoint_max_age_s = float(
            checkpoint_max_age_s
            if checkpoint_max_age_s is not None
            else os.getenv(
                "RVCLAW_ROUTE_CHECKPOINT_MAX_AGE_S",
                str(self.CHECKPOINT_MAX_AGE_S),
            )
        )
        if not math.isfinite(self.checkpoint_max_age_s) or self.checkpoint_max_age_s <= 0:
            raise ValueError("checkpoint maximum age must be positive")
        self._wall_clock = wall_clock
        # Terminal-state sink.  The router supplies a recorder bound to the
        # task gateway so a finished route closes its own task row instead of
        # leaving it running forever.
        self.task_recorder = task_recorder
        self._lock = Lock()
        self._termination_lock = Lock()
        self._process: Optional[subprocess.Popen] = None
        self._run: Optional[Dict[str, str]] = None
        self._started_at: Optional[float] = None
        self._stop_requested = Event()
        self._monitor: Optional[Thread] = None

    def status(self) -> Dict[str, Any]:
        with self._lock:
            process = self._process
            run = dict(self._run or {})
            started_at = self._started_at
        if process is not None:
            return {
                **run,
                "state": "running" if process.poll() is None else "finishing",
                "pid": process.pid,
                "elapsed_s": None if started_at is None else max(0.0, time.monotonic() - started_at),
            }
        external_pid = _external_script_pid(self.script_path)
        if external_pid is not None:
            return {
                **run,
                "state": "running",
                "pid": external_pid,
                "source": "external",
            }
        return {"state": "idle", **run}

    def start(
        self,
        _command: Any = None,
        *,
        route_names=None,
        resume_context: Optional[Path] = None,
        force_relocalization: bool = False,
        clear_checkpoint: bool = True,
    ) -> Dict[str, Any]:
        route_names = tuple(route_names or self.DEFAULT_ROUTE)
        if not route_names or any(name not in self.DEFAULT_ROUTE for name in route_names):
            raise RuntimeError("invalid inspection resume route")
        with self._lock:
            if self._process is not None:
                raise RuntimeError("inspection route is already running")
            self._stop_requested.clear()
            self.run_root.mkdir(parents=True, exist_ok=True)
            run_dir = Path(tempfile.mkdtemp(prefix="inspection-", dir=str(self.run_root)))
            trace_path = run_dir / "trace.json"
            fast_localization_trace_path = run_dir / "fast-localization.json"
            log_path = run_dir / "command.log"
            environment = os.environ.copy()
            environment["RVCLAW_AUTO_RVIZ"] = "off"
            environment["RVCLAW_NAV2_FIXED_PROFILE"] = "route"
            environment["RVCLAW_NAV2_WAYPOINT_ROUTE"] = ",".join(route_names)
            environment["RVCLAW_ROUTE_TRACE_FILE"] = str(trace_path)
            environment["RVCLAW_FAST_LOCALIZATION_TRACE_FILE"] = str(
                fast_localization_trace_path
            )
            environment["RVCLAW_ROUTE_CHECKPOINT_FILE"] = str(self.checkpoint_path)
            if resume_context is not None:
                environment["RVCLAW_ROUTE_RESUME_CONTEXT"] = str(resume_context)
            else:
                environment.pop("RVCLAW_ROUTE_RESUME_CONTEXT", None)
            if force_relocalization:
                environment["RVCLAW_FORCE_RELOCALIZATION"] = "1"
            else:
                environment.pop("RVCLAW_FORCE_RELOCALIZATION", None)
            # The current route runner resolves a fresh pose from its own
            # stationary scan and rejects all externally supplied initial poses.
            for variable in (
                "RVCLAW_NAV2_INITIAL_POSE",
                "RVCLAW_NAV2_INITIAL_X",
                "RVCLAW_NAV2_INITIAL_Y",
                "RVCLAW_NAV2_INITIAL_YAW_DEG",
            ):
                environment.pop(variable, None)
            if clear_checkpoint:
                self.checkpoint_path.unlink(missing_ok=True)
            route_process = subprocess.Popen(
                [
                    "bash",
                    self.script_path,
                    # The reviewed wrapper has its own argparse contract: it
                    # accepts exactly one of --describe/--preflight/--run and
                    # derives the arm token, waypoints and trace paths itself.
                    # Passing --trace-file/--arm-token made it print its usage
                    # line and exit 2, so the route never started.
                    "--run",
                ],
                cwd=str(Path(self.script_path).resolve().parent.parent),
                stdout=log_path.open("w", encoding="utf-8"),
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
                start_new_session=True,
                env=environment,
            )
            if route_process.stdin is not None:
                route_process.stdin.write((self.CONFIRMATION_TOKEN + "\n").encode())
                route_process.stdin.flush()
                route_process.stdin.close()
            self._process = route_process
            self._started_at = time.monotonic()
            self._run = {
                "run_dir": str(run_dir),
                "trace_file": str(trace_path),
                "fast_localization_trace_file": str(fast_localization_trace_path),
                "log_file": str(log_path),
                "route": ",".join(route_names),
                "resumed": resume_context is not None,
                "force_relocalization": bool(force_relocalization),
                "state": "running",
            }
            self._monitor = Thread(
                target=self._watch_route_process,
                args=(route_process,),
                name="rvclaw-inspection-watch",
                daemon=True,
            )
            self._monitor.start()
        return {"state": "running", "route": ",".join(route_names), **(self._run or {})}

    def resume(self, _command: Any = None, *, force_relocalization: bool = False) -> Dict[str, Any]:
        try:
            checkpoint = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RuntimeError("no paused inspection is available") from exc
        except (OSError, ValueError) as exc:
            raise RuntimeError("inspection checkpoint is unreadable") from exc
        if int(checkpoint.get("schema_version", 0)) != 1:
            raise RuntimeError("inspection checkpoint schema is unsupported")
        try:
            timestamp = float(checkpoint["timestamp"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("inspection checkpoint timestamp is invalid") from exc
        if not math.isfinite(timestamp):
            raise RuntimeError("inspection checkpoint timestamp is invalid")
        age_s = float(self._wall_clock()) - timestamp
        if age_s < -60.0 or age_s > self.checkpoint_max_age_s:
            raise RuntimeError("inspection checkpoint is stale")
        route_names = tuple(
            str(name).strip().upper()
            for name in checkpoint.get("remaining_waypoints", ())
        )
        suffix_start = len(self.DEFAULT_ROUTE) - len(route_names)
        if (
            not route_names
            or suffix_start < 0
            or self.DEFAULT_ROUTE[suffix_start:] != route_names
        ):
            raise RuntimeError("inspection checkpoint has no valid remaining route")
        completed = tuple(
            str(name).strip().upper()
            for name in checkpoint.get("completed_waypoints", ())
        )
        requested = tuple(
            str(name).strip().upper()
            for name in checkpoint.get("route_requested", ())
        )
        failed_waypoint = str(checkpoint.get("failed_waypoint", "")).strip().upper()
        if completed != self.DEFAULT_ROUTE[:suffix_start]:
            raise RuntimeError("inspection checkpoint progress is inconsistent")
        if requested != self.DEFAULT_ROUTE:
            raise RuntimeError("inspection checkpoint route does not match the mission")
        if failed_waypoint != route_names[0]:
            raise RuntimeError("inspection checkpoint failed waypoint is inconsistent")
        checkpoint_requires_relocalization = bool(
            checkpoint.get("requires_global_localization")
        )
        return self.start(
            _command,
            route_names=route_names,
            resume_context=self.checkpoint_path,
            force_relocalization=(
                force_relocalization or checkpoint_requires_relocalization
            ),
            clear_checkpoint=False,
        )

    def _terminate_route_process(self, process: subprocess.Popen) -> int:
        """Allow the wrapper to disarm and clean up before bounded escalation."""
        with self._termination_lock:
            for signum, grace in ((signal.SIGINT, 60.0), (signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
                if process.poll() is not None:
                    return int(process.wait())
                try:
                    os.killpg(process.pid, signum)
                except ProcessLookupError:
                    pass
                try:
                    return int(process.wait(timeout=grace))
                except subprocess.TimeoutExpired:
                    continue
            raise RuntimeError("inspection process did not exit after forced termination")

    def _watch_route_process(self, process: subprocess.Popen) -> None:
        """Bound the whole route, including startup and navigation preflights."""
        timed_out = False
        cleanup_error = None
        try:
            return_code = int(process.wait(timeout=self.timeout))
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                self._terminate_route_process(process)
            except Exception as exc:
                cleanup_error = str(exc)
            return_code = 124
        except Exception as exc:
            cleanup_error = str(exc)
            return_code = 1
        with self._lock:
            stopped_by_command = self._stop_requested.is_set()
            if self._process is process:
                # Keep ownership if cleanup failed while the process is alive.
                if cleanup_error is None or process.poll() is not None:
                    self._process = None
                run = dict(self._run or {})
                run.update({
                    "state": "failed" if timed_out else "stopped" if stopped_by_command else "completed" if return_code == 0 else "failed",
                    "return_code": return_code,
                })
                if timed_out:
                    run["timed_out"] = True
                if cleanup_error is not None:
                    run["cleanup_error"] = cleanup_error
                self._run = run
        if timed_out or (return_code != 0 and not stopped_by_command):
            self._publish_failure_event(return_code, run)
        # A successful inspection route publishes its detailed summary before
        # exiting. Do not append a generic "小车执行成功" event here: it can
        # contradict a valid "completed with inspection failures" summary and
        # used to mask summary-playback timeouts in the field.
        # Report the terminal state to the task gateway.  This is the only
        # place that observes the route's real outcome, so without it the task
        # row stays "running" forever and no TASK_COMPLETED event is produced.
        self._report_terminal_state()

    def _report_terminal_state(self) -> None:
        """Hand the recorded terminal outcome to the gateway exactly once.

        Runs on the monitor thread after the ``with self._lock`` block, and
        never raises: a gateway problem must not kill the route supervisor,
        and must not turn a finished route into a failed one.
        """

        recorder = self.task_recorder
        if recorder is None:
            return
        with self._lock:
            run = dict(self._run or {})
        state = str(run.get("state", "")).lower()
        if state not in {"completed", "failed", "stopped"}:
            return
        try:
            recorder(state, run)
        except Exception:
            LOG.exception("failed to report route terminal state state=%s", state)

    @classmethod
    def _publish_terminal_event(
        cls,
        event_name: str,
        status: str,
        return_code: int,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Atomically publish one terminal route outcome for the speech watcher."""

        target_dir = Path(os.getenv("RVCLAW_ARRIVAL_EVENT_DIR", cls.ARRIVAL_EVENT_DIR))
        sequence = "900" if status == "PASS" else "500"
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "event_id": str(uuid4()),
                "event": event_name,
                "status": status,
                "timestamp": time.time(),
                "return_code": int(return_code),
            }
            if details:
                payload.update(details)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=target_dir,
                prefix=".%s-" % event_name,
                suffix=".tmp",
                delete=False,
            ) as stream:
                json.dump(payload, stream, ensure_ascii=False)
                stream.flush()
                temp_path = Path(stream.name)
            os.replace(
                temp_path,
                target_dir / ("%s-%s-%s.json" % (sequence, event_name, payload["event_id"])),
            )
        except OSError:
            # The process result remains available in the run log even if the
            # optional speech event directory cannot be written.
            pass

    @classmethod
    def _resume_failure_details(cls, run: Dict[str, Any]) -> Dict[str, Any]:
        # Once the route exists, its failed stage is authoritative. A PASS
        # startup-localization trace says nothing about a later heading or
        # obstacle failure, and must not turn that failure into "定位失败".
        route_path = Path(str(run.get("trace_file", "")))
        if route_path.is_file():
            try:
                route_trace = json.loads(route_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                route_trace = {}
            route_rows = route_trace.get("route", []) if isinstance(route_trace, dict) else []
            if isinstance(route_rows, list):
                failed = next((row for row in reversed(route_rows)
                               if isinstance(row, dict) and row.get("status") == "FAIL"), None)
                decision = failed.get("navigation_failure") if failed else None
                if isinstance(decision, dict) and decision.get("paused_speech_key"):
                    ack = failed.get("navigation_paused_speech_ack")
                    return {
                        "speech_key": decision["paused_speech_key"],
                        "failure_code": decision.get("code"),
                        "diagnostic_status": "FAIL",
                        "diagnostic_error": failed.get("error"),
                        "route_failure_already_announced": (
                            isinstance(ack, dict) and ack.get("played") is True
                            and ack.get("status") == "PASS"
                        ),
                    }
        trace = {}
        trace_path = Path(str(run.get("fast_localization_trace_file", "")))
        if trace_path.is_file():
            try:
                document = json.loads(trace_path.read_text(encoding="utf-8"))
                if isinstance(document, dict):
                    trace = document
            except (OSError, ValueError):
                trace = {}
        error = str(trace.get("error", "")).strip()
        status = str(trace.get("status", "")).upper()
        diagnostic = (status + " " + error).lower()
        if not diagnostic.strip():
            log_path = Path(str(run.get("log_file", "")))
            try:
                diagnostic = log_path.read_text(encoding="utf-8", errors="replace")[-32768:].lower()
            except OSError:
                diagnostic = ""
        if (
            status == "BLOCKED"
            or "temporary obstacle" in diagnostic
            or "hazard to clear" in diagnostic
        ):
            speech_key = "navigation_paused_obstacle"
            failure_code = "TEMPORARY_OBSTRUCTION"
        elif "lidar" in diagnostic or "fresh independent lidar scans" in diagnostic:
            speech_key = "navigation_paused_lidar"
            failure_code = "LIDAR_UNAVAILABLE"
        elif (
            "hardware" in diagnostic
            or "stm32" in diagnostic
            or "controller must remain explicitly disarmed" in diagnostic
        ):
            speech_key = "navigation_paused_chassis"
            failure_code = "CHASSIS_UNAVAILABLE"
        else:
            speech_key = "navigation_paused_localization"
            failure_code = "LOCALIZATION_UNCONFIRMED"
        return {
            "speech_key": speech_key,
            "failure_code": failure_code,
            "diagnostic_status": status or None,
            "diagnostic_error": error or None,
        }

    @classmethod
    def _publish_failure_event(
        cls,
        return_code: int,
        run: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Atomically publish a route failure for the speech event watcher."""

        if run and bool(run.get("resumed")):
            details = cls._resume_failure_details(run)
            if details.get("route_failure_already_announced") is True:
                LOG.info("resumed route failure already announced code=%s",
                         details.get("failure_code"))
                return
            cls._publish_terminal_event(
                "navigation_resume_failed",
                "FAIL",
                return_code,
                details,
            )
            return
        if run and run.get("fast_localization_trace_file") and not Path(
            str(run.get("trace_file", ""))
        ).is_file():
            details = cls._resume_failure_details(run)
            # Start is asynchronous: the immediate ACK cannot announce a
            # preflight failure several seconds later. Give this phase its
            # own event; the legacy generic failure is intentionally silent.
            details["speech_key"] = {
                "navigation_paused_obstacle": "navigation_start_failed_obstacle",
                "navigation_paused_lidar": "navigation_start_failed_lidar",
                "navigation_paused_chassis": "navigation_start_failed_chassis",
            }.get(details["speech_key"], "navigation_start_failed_localization")
            if details["diagnostic_status"] == "PASS":
                details["speech_key"] = "navigation_start_failed_system"
                details["failure_code"] = "NAVIGATION_START_FAILED"
            cls._publish_terminal_event(
                "navigation_start_failed", "FAIL", return_code, details
            )
            return
        cls._publish_terminal_event("inspection_failed", "FAIL", return_code)

    @classmethod
    def _publish_success_event(cls, return_code: int) -> None:
        """Atomically publish a clean route finish for the speech event watcher."""

        cls._publish_terminal_event("inspection_finished", "PASS", return_code)

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            self._stop_requested.set()
            process = self._process
            run = dict(self._run or {})
            monitor = self._monitor
        if process is None:
            return {**run, "state": "idle"}
        self._terminate_route_process(process)
        if monitor is not None and monitor.is_alive():
            monitor.join(timeout=1.0)
        with self._lock:
            run = dict(self._run or {})
        return {**run, "state": "stopped"}


class ManualMappingClient:
    """Own one PS2-driven SLAM session controlled by atomic voice commands.

    The child script is the hardware safety boundary: it starts read-only STM32
    telemetry, the LiDAR and slam_toolbox, but never creates a ROS velocity
    publisher. This client only starts that reviewed wrapper and writes SAVE or
    CANCEL to its atomic control file.
    """

    ARM_TOKEN = "VOICE_WAKE_PS2_MAPPING_SUPERVISED"
    ACTIVE_RAW_STATES = frozenset({"STARTING", "READY", "VERIFYING_STOP", "SAVING"})
    TERMINAL_RAW_STATES = frozenset({"COMPLETED", "FAILED", "CANCELLED"})
    EVENT_DIR = "/data/RVClaw/runtime/waypoint-arrivals"

    def __init__(
        self,
        script_path: Optional[str] = None,
        run_root: Optional[str] = None,
        map_root: Optional[str] = None,
        state_root: Optional[str] = None,
        latest_file: Optional[str] = None,
        nav_service: Optional[str] = None,
        start_timeout: Optional[float] = None,
        save_timeout: Optional[float] = None,
        nav_stop_timeout: Optional[float] = None,
        nav_restore_timeout: Optional[float] = None,
        proc_root: Optional[str] = None,
        task_recorder: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        command_runner=None,
        popen_factory=None,
        monotonic=time.monotonic,
        wall_clock=time.time,
    ):
        lab_root = os.getenv("RVCLAW_LAB_ROOT", "/data/rvclaw-k3-lab")
        self.lab_root = Path(lab_root)
        self.base_root = Path(os.getenv("RVCLAW_BASE_ROOT", "/data/rvclaw-k3-lab"))
        self.script_path = script_path or os.getenv(
            "RVCLAW_MAPPING_SCRIPT",
            str(self.lab_root / "scripts" / "slam_mapping_session.sh"),
        )
        self.run_root = Path(
            run_root
            or os.getenv("RVCLAW_MAPPING_RUN_ROOT", str(self.lab_root / "runs" / "voice-mapping"))
        )
        self.map_root = Path(
            map_root or os.getenv("RVCLAW_MAPPING_MAP_ROOT", str(self.lab_root / "maps"))
        )
        self.state_root = Path(
            state_root
            or os.getenv("RVCLAW_MAPPING_STATE_ROOT", "/data/RVClaw/runtime/manual-mapping")
        )
        self.state_file = self.state_root / "session-state.json"
        self.control_file = self.state_root / "session-control.txt"
        self.latest_file = Path(
            latest_file
            or os.getenv(
                "RVCLAW_MAPPING_LATEST_FILE",
                str(self.lab_root / "state" / "manual-mapping-latest.json"),
            )
        )
        self.nav_service = nav_service or os.getenv(
            "RVCLAW_NAV_SERVICE", "rvclaw-nav2-persistent.service"
        )
        self.start_timeout = float(
            start_timeout
            if start_timeout is not None
            else os.getenv("RVCLAW_MAPPING_START_TIMEOUT_S", "90")
        )
        self.save_timeout = float(
            save_timeout
            if save_timeout is not None
            else os.getenv("RVCLAW_MAPPING_SAVE_TIMEOUT_S", "75")
        )
        self.nav_stop_timeout = float(
            nav_stop_timeout
            if nav_stop_timeout is not None
            else os.getenv("RVCLAW_NAV_STOP_TIMEOUT_S", "45")
        )
        self.nav_restore_timeout = float(
            nav_restore_timeout
            if nav_restore_timeout is not None
            else os.getenv("RVCLAW_NAV_RESTORE_TIMEOUT_S", "60")
        )
        if (
            self.start_timeout <= 0
            or self.save_timeout <= 0
            or self.nav_stop_timeout <= 0
            or self.nav_restore_timeout <= 0
        ):
            raise ValueError("mapping timeouts must be positive")
        self.proc_root = Path(proc_root or "/proc")
        self.task_recorder = task_recorder
        self._command_runner = command_runner or subprocess.run
        self._popen_factory = popen_factory or subprocess.Popen
        self._monotonic = monotonic
        self._wall_clock = wall_clock
        self._lock = Lock()
        self._process: Optional[subprocess.Popen] = None
        self._monitor: Optional[Thread] = None
        self._run: Optional[Dict[str, Any]] = None
        self._log_stream = None
        self._ready_seen = False
        self._nav_was_active = False
        self._not_stationary_generations: set[int] = set()

    @staticmethod
    def _public_state(raw_state: str) -> str:
        return {
            "STARTING": "starting",
            "READY": "running",
            "VERIFYING_STOP": "finishing",
            "SAVING": "finishing",
            "COMPLETED": "completed",
            "FAILED": "failed",
            "CANCELLED": "stopped",
        }.get(raw_state, "idle")

    def _read_state(self) -> Dict[str, Any]:
        try:
            value = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _pid_alive(pid: Any) -> bool:
        try:
            number = int(pid)
        except (TypeError, ValueError):
            return False
        if number <= 1:
            return False
        if os.name == "nt":
            # On Windows os.kill(pid, 0) is not a harmless existence probe: it
            # may terminate the target process. OpenProcess provides the same
            # read-only liveness check without delivering a signal.
            try:
                import ctypes

                handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, number)
                if not handle:
                    return False
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            except (AttributeError, OSError):
                return False
        try:
            os.kill(number, 0)
        except (AttributeError, OSError, ProcessLookupError):
            return False
        return True

    def _process_alive(self, snapshot: Dict[str, Any]) -> bool:
        with self._lock:
            process = self._process
        if process is not None:
            return process.poll() is None
        return self._pid_alive(snapshot.get("pid"))

    def status(self) -> Dict[str, Any]:
        snapshot = self._read_state()
        raw_state = str(snapshot.get("state", "")).upper()
        alive = self._process_alive(snapshot)
        with self._lock:
            run = dict(self._run or {})
        if raw_state in self.ACTIVE_RAW_STATES and not alive:
            return {
                **run,
                **snapshot,
                "state": "failed",
                "raw_state": raw_state,
                "error": snapshot.get("error") or "mapping_process_not_running",
                "process_alive": False,
            }
        if raw_state:
            return {
                **run,
                **snapshot,
                "state": self._public_state(raw_state),
                "raw_state": raw_state,
                "process_alive": alive,
            }
        return {"state": "idle", **run, "process_alive": False}

    def _systemctl(
        self,
        action: str,
        *,
        check: bool = False,
        timeout: float = 20.0,
        no_block: bool = False,
    ):
        command = ["systemctl", "--user", action]
        if no_block:
            command.append("--no-block")
        command.append(self.nav_service)
        completed = self._command_runner(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout,
        )
        if check and int(getattr(completed, "returncode", 1)) != 0:
            raise RuntimeError("failed to %s persistent navigation service" % action)
        return completed

    def _persistent_navigation_state(self) -> str:
        completed = self._systemctl("is-active", timeout=5.0)
        state = str(getattr(completed, "stdout", "") or "").strip().lower()
        if state:
            return state.splitlines()[0]
        return "active" if int(getattr(completed, "returncode", 1)) == 0 else "unknown"

    def _stop_persistent_navigation(self) -> bool:
        """Stop Nav2 completely before mapping can claim LiDAR and STM32.

        ``systemctl is-active`` returns a non-zero status for ``deactivating``.
        Treating that as already stopped allowed a second voice command to
        launch SLAM while the old navigation process still owned the devices.
        Queue the stop job, then wait for a terminal unit state instead.
        """

        running_states = {"active", "activating", "reloading", "deactivating"}
        stopped_states = {"inactive", "failed"}
        initial_state = self._persistent_navigation_state()
        if initial_state in stopped_states:
            return False
        if initial_state not in running_states:
            raise RuntimeError(
                "persistent navigation service state is not safe for mapping: %s"
                % initial_state
            )

        self._systemctl("stop", check=True, timeout=10.0, no_block=True)
        deadline = self._monotonic() + self.nav_stop_timeout
        last_state = initial_state
        while self._monotonic() < deadline:
            last_state = self._persistent_navigation_state()
            if last_state in stopped_states:
                return True
            if last_state not in running_states:
                raise RuntimeError(
                    "persistent navigation entered unexpected state while stopping: %s"
                    % last_state
                )
            time.sleep(0.25)
        raise RuntimeError(
            "persistent navigation did not stop within %.1fs state=%s"
            % (self.nav_stop_timeout, last_state)
        )

    def _restore_persistent_navigation(self) -> bool:
        if not self._nav_was_active:
            return True
        try:
            # The deployed unit deliberately has a 20 second ExecStartPre delay
            # for USB enumeration.  A synchronous `systemctl start` therefore
            # races the old fixed 20 second subprocess timeout even though the
            # service subsequently starts normally. Queue the job without
            # blocking, then observe the real unit state with a wider deadline.
            self._systemctl("start", check=True, timeout=10.0, no_block=True)
            deadline = self._monotonic() + self.nav_restore_timeout
            last_state = "unknown"
            while self._monotonic() < deadline:
                completed = self._systemctl("is-active", timeout=5.0)
                last_state = str(getattr(completed, "stdout", "") or "").strip()
                if int(getattr(completed, "returncode", 1)) == 0:
                    self._nav_was_active = False
                    return True
                if last_state == "failed":
                    LOG.error(
                        "persistent navigation entered failed state after mapping startup failure"
                    )
                    return False
                time.sleep(0.25)
            LOG.error(
                "persistent navigation did not become active within %.1fs state=%s",
                self.nav_restore_timeout,
                last_state or "unknown",
            )
        except Exception:
            LOG.exception("failed to restore persistent navigation after mapping startup failure")
        return False

    def _verified_external_mapping_pid(self, snapshot: Dict[str, Any]) -> Optional[int]:
        """Return an attached mapping PID only after verifying its exact identity.

        A restarted voice service has no ``Popen`` object for the still-running
        mapping wrapper. The state file provides its PID, but a PID alone is
        not safe enough to signal because it may have been reused. Require the
        expected script, voice admission mode, arm token, and a session-leading
        process group before treating it as ours.
        """

        try:
            pid = int(snapshot.get("pid"))
        except (TypeError, ValueError):
            return None
        if pid <= 1 or not self._pid_alive(pid):
            return None
        try:
            raw_cmdline = (self.proc_root / str(pid) / "cmdline").read_bytes()
            arguments = [
                value.decode("utf-8", errors="surrogateescape")
                for value in raw_cmdline.split(b"\0")
                if value
            ]
        except OSError:
            return None
        expected_script = os.path.realpath(self.script_path)
        if not any(os.path.realpath(value) == expected_script for value in arguments):
            return None
        if "--voice-run" not in arguments or self.ARM_TOKEN not in arguments:
            return None
        try:
            if os.getpgid(pid) != pid:
                return None
        except (AttributeError, OSError, ProcessLookupError):
            return None
        return pid

    def _wait_pid_exit(self, pid: int, timeout: float) -> bool:
        deadline = self._monotonic() + timeout
        while self._monotonic() < deadline:
            if not self._pid_alive(pid):
                return True
            time.sleep(0.1)
        return not self._pid_alive(pid)

    def _terminate_process(
        self,
        process,
        snapshot: Optional[Dict[str, Any]] = None,
    ) -> bool:
        external = process is None
        if process is not None:
            if process.poll() is not None:
                return True
            pid = process.pid
        else:
            pid = self._verified_external_mapping_pid(snapshot or {})
            if pid is None:
                LOG.error("refusing to terminate an unverified external mapping process")
                return False
        try:
            os.killpg(pid, signal.SIGINT)
        except (AttributeError, OSError, ProcessLookupError):
            if external:
                return not self._pid_alive(pid)
            try:
                process.terminate()
            except OSError:
                return process.poll() is not None
        if external:
            if self._wait_pid_exit(pid, 8.0):
                return True
            try:
                os.killpg(pid, signal.SIGKILL)
            except (AttributeError, OSError, ProcessLookupError):
                return not self._pid_alive(pid)
            return self._wait_pid_exit(pid, 3.0)
        try:
            process.wait(timeout=8.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(pid, signal.SIGKILL)
            except (AttributeError, OSError, ProcessLookupError):
                process.kill()
            process.wait(timeout=3.0)
        return process.poll() is not None

    def _close_log(self) -> None:
        with self._lock:
            stream = self._log_stream
            self._log_stream = None
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass

    def start(self, _command: Any = None) -> Dict[str, Any]:
        current = self.status()
        if current.get("state") in {"starting", "running", "finishing"}:
            raise RuntimeError("manual mapping is already running")

        suffix = time.strftime("%Y%m%d-%H%M%S", time.localtime(self._wall_clock()))
        suffix += "-" + uuid4().hex[:8]
        run_dir = self.run_root / ("mapping-" + suffix)
        map_dir = self.map_root / ("voice-slam-" + suffix)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.map_root.mkdir(parents=True, exist_ok=True)
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.latest_file.parent.mkdir(parents=True, exist_ok=True)
        self.control_file.unlink(missing_ok=True)
        self.state_file.unlink(missing_ok=True)

        self._nav_was_active = self._stop_persistent_navigation()
        log_path = run_dir / "voice-client.log"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_stream = log_path.open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment.update(
            {
                "RVCLAW_LAB_ROOT": str(self.lab_root),
                "RVCLAW_BASE_ROOT": str(self.base_root),
                "RVCLAW_AUTO_RVIZ": "off",
                "RVCLAW_MAPPING_CONTROL_FILE": str(self.control_file),
                "RVCLAW_MAPPING_STATE_FILE": str(self.state_file),
                "RVCLAW_MAPPING_LATEST_FILE": str(self.latest_file),
                "RVCLAW_MAPPING_RUN_DIR": str(run_dir),
                "RVCLAW_MAPPING_MAP_DIR": str(map_dir),
            }
        )
        try:
            process = self._popen_factory(
                ["bash", self.script_path, "--voice-run", "--arm-token", self.ARM_TOKEN],
                cwd=str(self.lab_root),
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env=environment,
            )
        except Exception:
            log_stream.close()
            self._restore_persistent_navigation()
            raise
        with self._lock:
            self._process = process
            self._log_stream = log_stream
            self._run = {
                "run_dir": str(run_dir),
                "map_dir": str(map_dir),
                "state_file": str(self.state_file),
                "control_file": str(self.control_file),
                "log_path": str(log_path),
                "pid": process.pid,
                "started_at": self._wall_clock(),
            }
            self._ready_seen = False
            self._not_stationary_generations.clear()

        deadline = self._monotonic() + self.start_timeout
        try:
            while self._monotonic() < deadline:
                snapshot = self._read_state()
                raw_state = str(snapshot.get("state", "")).upper()
                if raw_state == "READY":
                    with self._lock:
                        self._ready_seen = True
                        run = {**(self._run or {}), **snapshot, "state": "running"}
                        self._run = run
                    self.start_monitor_if_active()
                    return run
                if raw_state in self.TERMINAL_RAW_STATES:
                    raise RuntimeError(snapshot.get("error") or "mapping startup failed")
                return_code = process.poll()
                if return_code is not None:
                    raise RuntimeError("mapping process exited before readiness: %s" % return_code)
                time.sleep(0.2)
            raise RuntimeError("mapping startup timed out")
        except Exception:
            self._terminate_process(process)
            self._close_log()
            with self._lock:
                self._process = None
                self._run = {**(self._run or {}), "state": "failed"}
            self._restore_persistent_navigation()
            raise

    def _write_control(self, command: str) -> None:
        self.state_root.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".mapping-control-", suffix=".tmp", dir=str(self.state_root)
        )
        try:
            with os.fdopen(descriptor, "w", encoding="ascii") as stream:
                stream.write(command + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self.control_file)
        except Exception:
            try:
                os.close(descriptor)
            except OSError:
                pass
            Path(temporary_name).unlink(missing_ok=True)
            raise

    def finish(self, _command: Any = None) -> Dict[str, Any]:
        current = self.status()
        if current.get("state") == "finishing":
            return {**current, "follow_up_text": "正在保存地图，请保持小车静止"}
        if current.get("state") != "running":
            return {
                **current,
                "state": "idle",
                "follow_up_text": "当前没有正在进行的建图任务",
            }
        self._write_control("SAVE")
        return {
            **current,
            "state": "finishing",
            "follow_up_text": "正在保存地图，请保持小车静止",
        }

    def cancel(self) -> Dict[str, Any]:
        current = self.status()
        if current.get("state") not in {"starting", "running", "finishing"}:
            return {**current, "state": "idle"}
        self._write_control("CANCEL")
        return {
            **current,
            "state": "stopping",
            "follow_up_text": "建图已取消，地图没有保存",
        }

    def stop(self) -> Dict[str, Any]:
        return self.cancel()

    def start_monitor_if_active(self) -> bool:
        current = self.status()
        if current.get("state") not in {"starting", "running", "finishing"}:
            return False
        with self._lock:
            if self._run is None:
                self._run = dict(current)
            if self._monitor is not None and self._monitor.is_alive():
                return True
            self._monitor = Thread(
                target=self._monitor_loop,
                name="rvclaw-mapping-watch",
                daemon=True,
            )
            self._monitor.start()
        return True

    def _monitor_loop(self) -> None:
        deadline_after_exit: Optional[float] = None
        save_deadline: Optional[float] = None
        while True:
            snapshot = self._read_state()
            raw_state = str(snapshot.get("state", "")).upper()
            generation = int(snapshot.get("generation", 0) or 0)
            if raw_state in {"VERIFYING_STOP", "SAVING"}:
                if save_deadline is None:
                    save_deadline = self._monotonic() + self.save_timeout
                elif self._monotonic() >= save_deadline:
                    with self._lock:
                        process = self._process
                    termination_confirmed = self._terminate_process(process, snapshot)
                    failed = {
                        **snapshot,
                        "state": "FAILED",
                        "error": "mapping_save_timeout",
                        "termination_confirmed": termination_confirmed,
                    }
                    self._record_terminal("FAILED", failed)
                    return
            elif raw_state == "READY":
                save_deadline = None
            if (
                raw_state == "READY"
                and snapshot.get("error") == "not_stationary"
                and generation not in self._not_stationary_generations
            ):
                self._not_stationary_generations.add(generation)
                self._publish_event("mapping_not_stationary", "RETRY", snapshot)
            if raw_state in self.TERMINAL_RAW_STATES:
                self._record_terminal(raw_state, snapshot)
                return
            if not self._process_alive(snapshot):
                if deadline_after_exit is None:
                    deadline_after_exit = self._monotonic() + 3.0
                elif self._monotonic() >= deadline_after_exit:
                    failed = {
                        **snapshot,
                        "state": "FAILED",
                        "error": snapshot.get("error") or "mapping_process_exited_without_terminal_state",
                    }
                    self._record_terminal("FAILED", failed)
                    return
            else:
                deadline_after_exit = None
            time.sleep(0.2)

    def _record_terminal(self, raw_state: str, snapshot: Dict[str, Any]) -> None:
        state = self._public_state(raw_state)
        with self._lock:
            process = self._process
            return_code = process.poll() if process is not None else snapshot.get("result_code")
            run = {
                **(self._run or {}),
                **snapshot,
                "state": state,
                "raw_state": raw_state,
                "return_code": return_code,
                "finished_at": self._wall_clock(),
            }
            self._run = run
            self._process = None
        self._close_log()
        if raw_state == "COMPLETED":
            self._publish_event("mapping_completed", "PASS", run)
            recorder_state = "completed"
        elif raw_state == "CANCELLED":
            self._publish_event("mapping_cancelled", "CANCELLED", run)
            recorder_state = "stopped"
        else:
            self._publish_event("mapping_failed", "FAIL", run)
            recorder_state = "failed"
        if self.task_recorder is not None:
            try:
                self.task_recorder(recorder_state, run)
            except Exception:
                LOG.exception("failed to report mapping terminal state state=%s", raw_state)

    @classmethod
    def _publish_event(cls, event_name: str, status: str, details: Dict[str, Any]) -> None:
        target_dir = Path(os.getenv("RVCLAW_ARRIVAL_EVENT_DIR", cls.EVENT_DIR))
        sequence = {
            "mapping_not_stationary": "520",
            "mapping_failed": "530",
            "mapping_cancelled": "540",
            "mapping_completed": "920",
        }.get(event_name, "550")
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "event_id": str(uuid4()),
                "event": event_name,
                "status": status,
                "timestamp": time.time(),
                "error": details.get("error"),
                "result_code": details.get("result_code"),
                "run_dir": details.get("run_dir"),
                "map_dir": details.get("map_dir"),
            }
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=target_dir,
                prefix=".%s-" % event_name,
                suffix=".tmp",
                delete=False,
            ) as stream:
                json.dump(payload, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
                temporary = Path(stream.name)
            os.replace(
                temporary,
                target_dir / ("%s-%s-%s.json" % (sequence, event_name, payload["event_id"])),
            )
        except OSError:
            LOG.exception("failed to publish mapping speech event event=%s", event_name)


class VisionRobotClient:
    def __init__(self, base_url: Optional[str] = None, timeout: float = 5.0):
        self.base_url = (base_url or os.getenv("RVCLAW_VISION_API", "http://127.0.0.1:8017")).rstrip("/")
        self.timeout = timeout

    def _request(self, path: str, method: str = "GET", payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"} if body else {},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, URLError, ValueError) as exc:
            raise RuntimeError(f"vision API unavailable: {exc}") from exc

    def status(self) -> Dict[str, Any]:
        return self._request("/api/cameras")

    def meter(self) -> Dict[str, Any]:
        return self._request("/api/meter-reading")

    def switch(self, mode: str) -> Dict[str, Any]:
        return self._request("/api/camera-mode", "POST", {"mode": mode})

    def stop(self) -> Dict[str, Any]:
        return self._request("/api/camera-stop", "POST", {})

    def wait_ready(self, mode: str, timeout: float = 12.0) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            last = self.status()
            camera = (last.get("cameras") or [{}])[0]
            inference = camera.get("inference") or {}
            if (
                last.get("active_mode") == mode
                and camera.get("active")
                and camera.get("online")
                and inference.get("ready")
            ):
                return last
            time.sleep(0.25)
        raise RuntimeError(f"vision mode {mode} did not become ready: {last}")


class InspectionCoordinator:
    """Serializes voice-triggered tasks onto the API's single camera state machine."""

    MODES = {"check_ppe": "astra_hardhat", "inspect_meter": "astra_meter"}

    def __init__(self, client: Optional[VisionRobotClient] = None):
        self.client = client or VisionRobotClient()
        self._lock = Lock()
        self._task: Optional[Dict[str, Any]] = None
        self._last_result: Optional[Dict[str, Any]] = None
        self._stop_event = Event()
        self._monitor: Optional[Thread] = None

    def start(self, intent: str) -> Dict[str, Any]:
        mode = self.MODES[intent]
        switched = self.client.switch(mode)
        status = self.client.wait_ready(mode)
        old_monitor = None
        with self._lock:
            task = {
                "task_id": str(uuid4()),
                "intent": intent,
                "mode": mode,
                "state": "running",
                "started_at": time.time(),
            }
            self._task = task
            self._stop_event.set()
            old_monitor = self._monitor
            self._stop_event = Event()
            self._monitor = Thread(target=self._monitor_loop, args=(mode, self._stop_event), daemon=True)
            self._monitor.start()
        if old_monitor and old_monitor.is_alive():
            old_monitor.join(timeout=1)
        return {"task_id": task["task_id"], "state": "running", "mode": mode, "vision": status, "switch": switched}

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            result = self.client.stop()
            self._stop_event.set()
            if self._task:
                self._task = {**self._task, "state": "stopped", "stopped_at": time.time()}
            return {"state": "stopped", "vision": result}

    def status(self) -> Dict[str, Any]:
        try:
            vision = self.client.status()
        except RuntimeError:
            with self._lock:
                task = dict(self._task) if self._task else None
            state = "running" if task and task.get("state") == "running" else "degraded"
            return {
                "state": state,
                "vision": {
                    "available": False,
                    "reason": "vision_service_unavailable",
                },
                "task": task,
            }
        with self._lock:
            return {"state": self._task_state(vision), "vision": vision, "task": self._task}

    def last_result(self) -> Dict[str, Any]:
        with self._lock:
            if self._last_result is None:
                raise RuntimeError("no completed inspection result")
            return dict(self._last_result)

    def _task_state(self, vision: Dict[str, Any]) -> str:
        if self._task and self._task.get("state") == "running":
            return "running"
        camera = (vision.get("cameras") or [{}])[0]
        return "idle" if not camera.get("active") else "ready"

    def _monitor_loop(self, mode: str, stop_event: Event) -> None:
        while not stop_event.is_set():
            try:
                vision = self.client.status()
                camera = (vision.get("cameras") or [{}])[0]
                inference = camera.get("inference") or {}
                if vision.get("active_mode") != mode:
                    return
                if mode == "astra_hardhat":
                    result = {
                        "type": "helmet_compliance",
                        "mode": mode,
                        "state": "checking" if inference.get("checking", 0) else "detected",
                        "persons": inference.get("persons", 0),
                        "helmet": inference.get("helmet", 0),
                        "no_helmet": inference.get("no_helmet", 0),
                        "checking": inference.get("checking", 0),
                        "timestamp": time.time(),
                    }
                else:
                    meter = self.client.meter()
                    result = {
                        "type": "meter_reading",
                        "mode": mode,
                        "state": meter.get("status", "checking"),
                        "value": meter.get("value"),
                        "unit": meter.get("unit"),
                        "confidence": meter.get("confidence"),
                        "raw_text": meter.get("raw_text", ""),
                        "timestamp": time.time(),
                    }
                with self._lock:
                    self._last_result = result
            except RuntimeError:
                pass
            stop_event.wait(1.0)
