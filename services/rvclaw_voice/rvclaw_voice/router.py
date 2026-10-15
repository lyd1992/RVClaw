"""Command dispatch boundary shared by voice, timers, and the HTTP API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
import time
from typing import Any, Callable, Dict, Optional

from .voice_command import (
    CONFIRMATION_INTENTS,
    CONFIRM_INTENTS,
    VoiceCommand,
    evaluate_policy,
    parse_transcript,
)
from .task_gateway import TaskGateway


Handler = Callable[[VoiceCommand], Dict[str, Any]]


ACK_TEXT = "小车，收到"
FAILURE_TEXT = "小车执行失败"
SUCCESS_TEXT = "小车执行成功"
CONFIRM_TEXT = "请确认开始巡检"
ROUTE_TASK_INTENTS = frozenset({
    "start_inspection",
    "start_mapping",
    "resume_inspection",
    "resume_inspection_relocalize",
})
DIRECT_CONTROL_INTENTS = frozenset({"finish_mapping", "stop"})


class CommandHandlerError(RuntimeError):
    """A rejected command with a fixed, pre-generated spoken explanation."""

    def __init__(self, message: str, response_text: str):
        super().__init__(message)
        self.response_text = response_text


def _is_active_state(value: Any) -> bool:
    return str(value or "").lower() in {"running", "localizing", "finishing"}


def _status_response(
    vision: Dict[str, Any],
    motion: Dict[str, Any],
    route: Dict[str, Any],
    mapping: Optional[Dict[str, Any]] = None,
) -> str:
    """Return the short spoken status required by the voice contract."""

    mapping = mapping or {}
    if str(mapping.get("state", "")).lower() in {
        "starting", "running", "finishing"
    }:
        return "当前正在建图"
    if _is_active_state(route.get("state")):
        return "当前正在执行巡检任务"
    if _is_active_state(motion.get("state")):
        return "当前正在执行左转任务"
    task = vision.get("task") or {}
    if str(task.get("state", "")).lower() == "running":
        task_names = {
            "inspect_meter": "仪表检测",
            "check_ppe": "安全帽检测",
            "start_inspection": "巡检",
        }
        return "当前正在执行%s任务" % task_names.get(task.get("intent"), "检测")
    return "当前没有任务"


def _last_result_response(result: Dict[str, Any]) -> str:
    """Turn the latest completed detection into a short spoken confirmation."""

    result_type = str(result.get("type", ""))
    if result_type == "meter_reading":
        value = result.get("value")
        unit = result.get("unit") or ""
        if value is not None:
            return "上次仪表检测结果为%s%s" % (value, unit)
        return "上次仪表检测已完成"
    if result_type == "helmet_compliance":
        persons = result.get("persons")
        helmet = result.get("helmet")
        no_helmet = result.get("no_helmet")
        if persons is not None and helmet is not None and no_helmet is not None:
            return "上次安全帽检测共%s人，佩戴%s人，未佩戴%s人" % (
                persons,
                helmet,
                no_helmet,
            )
        return "上次安全帽检测已完成"
    return "上次检测结果已获取"


@dataclass
class CommandResult:
    request_id: str
    accepted: bool
    status: str
    reason: str
    intent: Optional[str] = None
    task_id: Optional[str] = None
    state: Optional[str] = None
    response_text: Optional[str] = None
    follow_up_text: Optional[str] = None
    details: Dict[str, Any] = None

    def as_dict(self) -> Dict[str, Any]:
        result = {
            "request_id": self.request_id,
            "accepted": self.accepted,
            "status": self.status,
            "reason": self.reason,
            "intent": self.intent,
            "task_id": self.task_id,
            "state": self.state,
            "response_text": self.response_text,
            "follow_up_text": self.follow_up_text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if self.details:
            result["details"] = self.details
        return result


class CommandRouter:
    """Idempotent, policy-gated dispatcher with explicit handler registration."""

    def __init__(self, *, min_confidence: float = 0.72, gateway: Optional[TaskGateway] = None,
                 confirmation_ttl: float = 10.0, clock=time.time):
        self.min_confidence = min_confidence
        self._handlers: Dict[str, Handler] = {}
        self._results: Dict[str, Dict[str, Any]] = {}
        self._lock = RLock()
        self.gateway = gateway
        self.confirmation_ttl = float(confirmation_ttl)
        self._clock = clock
        self._pending_confirmation: Optional[tuple[VoiceCommand, float]] = None
        # Task id of the asynchronously executing chassis job, if any.  The
        # chassis is mutually exclusive, so a single slot is sufficient and it
        # lets the executor close its own task when the route really ends.
        self._active_task_id: Optional[str] = None

    def register(self, intent: str, handler: Handler) -> None:
        self._handlers[intent] = handler

    def health(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "ready": True,
                "min_confidence": self.min_confidence,
                "handlers": sorted(self._handlers),
                "cached_requests": len(self._results),
            }

    def adopt_active_task(self, task_id: Optional[str]) -> None:
        """Reconnect a surviving external executor to its durable task row."""

        if not task_id:
            return
        with self._lock:
            if self._active_task_id not in (None, task_id):
                raise RuntimeError("another durable chassis task is already active")
            self._active_task_id = str(task_id)

    def dispatch_transcript(self, transcript: str, confidence: float, *, source: str = "voice", request_id: Optional[str] = None) -> Dict[str, Any]:
        command = parse_transcript(transcript, confidence, source=source, request_id=request_id)
        if command is None:
            return CommandResult(
                request_id=request_id or "unparsed",
                accepted=False,
                status="rejected",
                reason="ambiguous_or_invalid_transcript",
                response_text=FAILURE_TEXT,
            ).as_dict()
        return self.dispatch(command)

    def dispatch(self, command: VoiceCommand, *, _confirmed: bool = False) -> Dict[str, Any]:
        if not _confirmed:
            with self._lock:
                cached = self._results.get(command.request_id)
                if cached is not None:
                    return {**cached, "replayed": True}
        decision = evaluate_policy(command, min_confidence=self.min_confidence)
        # The wake phrase plus the explicit "start inspection" command is the
        # user's authorization.  Keep confirmation for pause/return commands,
        # but dispatch a new inspection immediately as requested by operators.
        needs_confirmation = command.intent in CONFIRMATION_INTENTS
        if not _confirmed and needs_confirmation and (
            decision.accepted or decision.requires_confirmation
        ):
            expires_at = self._clock() + self.confirmation_ttl
            confirm_text = CONFIRM_TEXT
            with self._lock:
                self._pending_confirmation = (command, expires_at)
            result = CommandResult(
                request_id=command.request_id,
                accepted=False,
                status="confirmation_required",
                reason="motion_requires_confirmation",
                intent=command.intent,
                response_text=confirm_text,
                follow_up_text=confirm_text,
                details={"expires_at": expires_at},
            ).as_dict()
            with self._lock:
                self._results[command.request_id] = result
            return result
        if not decision.accepted and not (
            _confirmed and command.intent in CONFIRMATION_INTENTS
        ):
            result = CommandResult(
                request_id=command.request_id,
                accepted=False,
                status=decision.status,
                reason=decision.reason,
                intent=command.intent,
                response_text=FAILURE_TEXT,
            ).as_dict()
            with self._lock:
                self._results[command.request_id] = result
            return result
        if command.intent in CONFIRM_INTENTS:
            with self._lock:
                pending = self._pending_confirmation
                self._pending_confirmation = None
            if pending is None:
                result = CommandResult(command.request_id, False, "rejected", "confirmation_not_pending", command.intent, response_text=FAILURE_TEXT).as_dict()
                with self._lock:
                    self._results[command.request_id] = result
                return result
            original, expires_at = pending
            if self._clock() > expires_at:
                result = CommandResult(command.request_id, False, "rejected", "confirmation_expired", command.intent, response_text=FAILURE_TEXT).as_dict()
                with self._lock:
                    self._results[command.request_id] = result
                return result
            result = self.dispatch(original, _confirmed=True)
            result["confirmation_request_id"] = command.request_id
            # Starting a route only proves dispatch.  Completion feedback is
            # emitted by the executor's terminal event subscriber.
            if result.get("intent") in ROUTE_TASK_INTENTS:
                result["follow_up_text"] = None
            with self._lock:
                self._results[command.request_id] = result
            return result
        if command.intent == "stop":
            with self._lock:
                self._pending_confirmation = None
        # Read-only queries are answered by the registered status/result
        # provider so callers receive the current robot state immediately.
        # They do not create executable tasks in the gateway.
        if self.gateway is not None and command.intent not in {
            "query_status",
            "repeat_last_result",
            *DIRECT_CONTROL_INTENTS,
        }:
            # Use the gateway clock so deterministic replay and a clock source
            # shared with the executor cannot fail an otherwise fresh proposal.
            now = float(self.gateway.clock())
            proposal = {
                "schema_version": 1,
                "proposal_id": command.request_id,
                "request_id": command.request_id,
                "source": "local" if command.source == "voice" else command.source,
                "intent": command.intent,
                "parameters": dict(command.slots),
                "confidence": command.confidence,
                "created_at": now,
                "expires_at": now + 30,
                "owner_epoch": self.gateway.owner_epoch,
                "evidence": {"transcript": command.transcript},
            }
            accepted = self.gateway.submit(proposal)
            if accepted.get("status") == "accepted" and command.intent in ROUTE_TASK_INTENTS:
                # The local executor owns the reviewed A,B,C,ORIGIN route.
                # Submit is durable first; only then launch the route runner.
                handler = self._handlers.get(command.intent)
                try:
                    # Unit and cloud-only deployments may register no local
                    # executor; in that case retain the durable acceptance.
                    details = handler(command) if handler is not None else {}
                    if handler is not None:
                        self.gateway.record_event(accepted["task_id"], "TASK_STARTED", details)
                        # Remember which task owns the running chassis job so
                        # the executor can close it on the real terminal event.
                        self._active_task_id = accepted["task_id"]
                        accepted = {**accepted, **details, "status": "running", "reason": "task_started"}
                except Exception as exc:
                    self.gateway.fail_task(accepted["task_id"], {"error_type": type(exc).__name__})
                    accepted = {
                        **accepted,
                        "status": "rejected",
                        "reason": "task_start_failed",
                        "response_text": getattr(exc, "response_text", FAILURE_TEXT),
                    }
            result = CommandResult(
                request_id=command.request_id,
                accepted=accepted.get("status") in {"accepted", "running"},
                status=accepted.get("status", "rejected"),
                reason=accepted.get("reason", "task_rejected"),
                intent=command.intent,
                task_id=accepted.get("task_id"),
                # A local chassis handler advances the status to "running"
                # before this result is built, so comparing against "accepted"
                # alone made a successfully started inspection announce
                # "小车执行失败".
                response_text=(
                    ACK_TEXT
                    if accepted.get("status") in {"accepted", "running"}
                    else accepted.get("response_text", FAILURE_TEXT)
                ),
                # The handler's follow-up is launch progress (for example,
                # "巡检恢复任务已启动"), not a success claim. Completion or
                # failure is still announced only by the terminal-event path.
                follow_up_text=(
                    accepted.get("follow_up_text")
                    if accepted.get("status") == "running"
                    else None
                ),
                details=accepted,
            ).as_dict()
            with self._lock:
                self._results[command.request_id] = result
            return result
        handler = self._handlers.get(command.intent)
        if handler is None:
            result = CommandResult(
                request_id=command.request_id,
                accepted=False,
                status="rejected",
                reason="handler_not_configured",
                intent=command.intent,
                response_text=FAILURE_TEXT,
            ).as_dict()
        else:
            try:
                details = handler(command) or {}
                result = CommandResult(
                    request_id=command.request_id,
                    accepted=True,
                    status="accepted",
                    reason="command_dispatched",
                    intent=command.intent,
                    task_id=details.get("task_id"),
                    state=details.get("state"),
                    response_text=ACK_TEXT,
                    follow_up_text=details.get("follow_up_text") or SUCCESS_TEXT,
                    details=details,
                ).as_dict()
            except Exception as exc:  # boundary must fail closed
                result = CommandResult(
                    request_id=command.request_id,
                    accepted=False,
                    status="error",
                    reason="handler_error",
                    intent=command.intent,
                    response_text=getattr(exc, "response_text", FAILURE_TEXT),
                    details={"error_type": type(exc).__name__},
                ).as_dict()
        with self._lock:
            self._results[command.request_id] = result
        return result

    def record_terminal_state(self, state: str, run: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Close the active task when the chassis executor reports its outcome.

        The route runner is asynchronous, so the dispatch call cannot know how
        the job ends.  The executor calls this on its monitor thread instead of
        the router polling process state, which keeps task ownership in one
        place.  A missing gateway or task id is not an error: read-only and
        cloud-only deployments legitimately have nothing to close.
        """

        event_type = {
            "completed": "TASK_COMPLETED",
            "failed": "TASK_FAILED",
            "stopped": "TASK_CANCELLED",
        }.get(str(state).lower())
        if event_type is None:
            return None
        with self._lock:
            task_id = self._active_task_id
            self._active_task_id = None
        if self.gateway is None or not task_id:
            return None
        return self.gateway.record_event(task_id, event_type, dict(run))

    @staticmethod
    def event_response(event: Dict[str, Any]) -> Optional[str]:
        """Map durable executor events to user-facing completion feedback.

        Acceptance is intentionally absent here: the submit response already
        says “收到”.  Only terminal or meaningful lifecycle events may be
        announced as progress, preventing an accepted task from being called
        successful before the executor reports completion.
        """
        event_type = event.get("type") or event.get("event")
        return {
            "TASK_STARTED": "任务已开始",
            "TASK_COMPLETED": "小车执行完成",
            "TASK_FAILED": "小车执行失败",
            "TASK_CANCELLED": "任务已取消",
            "TASK_PAUSED": "任务已暂停",
        }.get(event_type)

    @staticmethod
    def _response_for(value: str) -> str:
        return {
            "query_status": ACK_TEXT,
            "inspect_meter": ACK_TEXT,
            "check_ppe": ACK_TEXT,
            "repeat_last_result": ACK_TEXT,
            "start_inspection": ACK_TEXT,
            "start_mapping": ACK_TEXT,
            "finish_mapping": ACK_TEXT,
            "allowed_voice_command": ACK_TEXT,
            "low_confidence": FAILURE_TEXT,
            "motion_requires_confirmation": FAILURE_TEXT,
            "voice_forbidden_safety_action": FAILURE_TEXT,
        }.get(value, FAILURE_TEXT)
