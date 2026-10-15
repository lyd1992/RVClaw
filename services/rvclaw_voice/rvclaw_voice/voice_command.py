"""Voice command parsing and safety policy for RVClaw.

The module deliberately contains no audio or ROS dependencies.  It is safe to
use from the HTTP API, timer callbacks, and the on-device speech process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from typing import Any, Dict, Optional
from uuid import uuid4


LOW_RISK_INTENTS = frozenset({"query_status", "inspect_meter", "check_ppe", "repeat_last_result"})
# Inspection drives the chassis over a recorded route, so it needs the same
# recognition-confidence floor as an explicit turn.
MOTION_INTENTS = frozenset({
    "start_inspection",
    "start_mapping",
    "finish_mapping",
    "turn_left",
    "resume_inspection",
    "resume_inspection_relocalize",
})
MOTION_MIN_CONFIDENCE = 0.78
NEGATION_MARKERS = ("不要", "别", "不许", "不能")
CONFIRMATION_INTENTS = frozenset({
    "pause_inspection",
    "return_home",
})
CONFIRM_INTENTS = frozenset({"confirm_inspection"})
STOP_INTENTS = frozenset({"stop"})
NEVER_VOICE_INTENTS = frozenset({"clear_estop", "release_manual_override"})
# SenseVoice occasionally substitutes the final character or drops the
# pronunciation of "好" on the XFM microphone. Keep this list closed and
# limited to variants observed on the board so a command still needs an exact
# registered intent before it can be dispatched.
WAKE_PHRASES = (
    "小微小微",
    "小薇小薇",
    "小伟小伟",
    "小威小威",
    "小鱼小鱼",
    "蔷薇蔷薇",
    "蔷威强威",
    "笑的笑的",
    "小微,小微",
    "小微，小微",
    "你好小车",
    "你好,小车",
    "你好，小车",
    "你好小蛇",
    "你好,小蛇",
    "你好，小蛇",
    "你号小车",
    "你号,小车",
    "你号，小车",
)


@dataclass(frozen=True)
class VoiceCommand:
    request_id: str
    source: str
    intent: str
    transcript: str
    confidence: float
    timestamp: str
    slots: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "source": self.source,
            "intent": self.intent,
            "transcript": self.transcript,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "slots": dict(self.slots),
        }


@dataclass(frozen=True)
class PolicyDecision:
    accepted: bool
    status: str
    reason: str
    requires_confirmation: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "status": self.status,
            "reason": self.reason,
            "requires_confirmation": self.requires_confirmation,
        }


_PATTERNS = (
    # Do not accept the bare word "状态": ASR(Automatic Speech Recognition)
    # substitutions such as "交钱状态" would otherwise become an action.
    ("query_status", ("当前状态", "目前状态", "现在状态", "机器人状态", "什么状态", "状态怎么样", "当前情况", "目前任务")),
    ("inspect_meter", ("检查仪表", "识别仪表", "仪表识别", "读仪表", "看看仪表")),
    ("check_ppe", ("安全帽检测", "检测安全帽", "安全帽识别", "检查安全帽", "ppe检测", "ppe识别", "检查防护用品")),
    ("repeat_last_result", ("上次结果", "重复结果", "再播报一次", "播报结果")),
    # SenseVoice occasionally renders 检 as 解 on the M2 microphone. Keep the
    # observed homophone variants closed and require the complete phrase.
    ("start_inspection", ("开始巡检", "启动巡检", "执行巡检", "开始巡解", "启动巡解", "执行巡解")),
    ("start_mapping", ("开始建图", "启动建图")),
    ("finish_mapping", ("建图结束", "结束建图", "保存地图")),
    ("confirm_inspection", ("确认", "确认巡检", "确认开始巡检", "确认执行巡检")),
    ("pause_inspection", ("暂停巡检", "暂停任务")),
    # Closed field-ASR substitutions observed for "继续巡检". Keep these exact;
    # fuzzy matching here could turn unrelated speech into a motion command.
    # The resume executor still requires a valid saved inspection checkpoint,
    # so this cannot start a new route from idle state.
    # The short alias uses the same wake/confidence policy and resume executor.
    # That executor validates a saved unfinished route before starting motion.
    ("resume_inspection", ("继续巡检", "恢复巡检", "继续巡转", "继续巡找", "继续旋转", "继续")),
    ("resume_inspection_relocalize", ("重新定位并继续巡检", "重新定位继续巡检")),
    ("return_home", ("返航", "回充电桩", "回到起点")),
    ("turn_left", ("左转", "向左转", "往左转", "左拐", "向左拐")),
    ("stop", ("停止", "立刻停", "马上停止", "停下来")),
    ("clear_estop", ("解除急停", "取消急停")),
    ("release_manual_override", ("解除人工接管", "释放人工接管")),
)


def _normalise(text: str) -> str:
    return re.sub(r"[^\w]+", "", text.strip().lower())


def is_complete_command(text: str) -> bool:
    """Require a whole registered command for acoustic wake sessions."""
    text = _normalise(text)
    if text.startswith("请"):
        text = text[1:]
    return any(text == _normalise(phrase) for _, phrases in _PATTERNS for phrase in phrases)


def _consume_normalised_prefix(text: str, prefix: str) -> Optional[int]:
    """Return the original-text length consumed by a punctuation-free prefix."""

    consumed = 0
    normalised = []
    while consumed < len(text) and len(normalised) < len(prefix):
        character = text[consumed].lower()
        if character.isalnum() or character == "_":
            normalised.append(character)
        consumed += 1
    return consumed if "".join(normalised) == prefix else None


def strip_wake_phrase(transcript: str) -> Optional[str]:
    """Return the command portion only when the required wake phrase is present."""

    if not isinstance(transcript, str):
        return None
    text = transcript.strip()
    normalised = _normalise(text)
    for phrase in WAKE_PHRASES:
        marker = _normalise(phrase)
        if normalised.startswith(marker):
            consumed = _consume_normalised_prefix(text, marker)
            if consumed is not None:
                return text[consumed:].lstrip(" ，,。.!！")
    return None


def parse_transcript(
    transcript: str,
    confidence: float,
    *,
    source: str = "voice",
    request_id: Optional[str] = None,
) -> Optional[VoiceCommand]:
    """Map a transcript to a closed-set command; return None for ambiguity."""

    if not isinstance(transcript, str) or not transcript.strip():
        return None
    try:
        score = float(confidence)
    except (TypeError, ValueError):
        return None
    if not 0.0 <= score <= 1.0:
        return None
    command_text = strip_wake_phrase(transcript)
    if command_text is None:
        return None
    return _parse_command_text(command_text, score, source=source, request_id=request_id,
                                transcript=transcript)


def _parse_command_text(
    command_text: str,
    score: float,
    *,
    source: str,
    request_id: Optional[str],
    transcript: Optional[str] = None,
) -> Optional[VoiceCommand]:
    """Parse a command after an already trusted hardware wake event."""
    text = _normalise(command_text)
    if text.startswith("请"):
        text = text[1:]
    # Closed-set parsing: a phrase must consume the complete command after the
    # wake word. Substring matching turns ASR substitutions into actions.
    matches = [intent for intent, phrases in _PATTERNS if any(text == _normalise(p) for p in phrases)]
    if len(matches) != 1:
        return None
    if matches[0] in MOTION_INTENTS and any(marker in text for marker in NEGATION_MARKERS):
        return None
    return VoiceCommand(
        request_id=request_id or str(uuid4()),
        source=source,
        intent=matches[0],
        transcript=(transcript if transcript is not None else command_text).strip(),
        confidence=score,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def parse_command_text(
    command_text: str,
    confidence: float,
    *,
    source: str = "voice",
    request_id: Optional[str] = None,
) -> Optional[VoiceCommand]:
    """Parse a closed-set command suffix after hardware wake authorization.

    This entry point deliberately accepts no wake phrase because the caller's
    M2 event is the wake authority. It still requires an exact registered
    command and applies the same negation and confidence checks downstream.
    """
    try:
        score = float(confidence)
    except (TypeError, ValueError):
        return None
    if not 0.0 <= score <= 1.0 or not isinstance(command_text, str):
        return None
    return _parse_command_text(command_text, score, source=source, request_id=request_id)


def evaluate_policy(command: VoiceCommand, *, min_confidence: float = 0.72) -> PolicyDecision:
    """Apply conservative policy before a command reaches a robot subsystem."""

    if command.source != "voice":
        return PolicyDecision(False, "rejected", "invalid_source")
    required_confidence = max(
        min_confidence,
        MOTION_MIN_CONFIDENCE if command.intent in MOTION_INTENTS else 0.0,
    )
    if command.confidence < required_confidence:
        return PolicyDecision(False, "rejected", "low_confidence")
    if command.intent in NEVER_VOICE_INTENTS:
        return PolicyDecision(False, "rejected", "voice_forbidden_safety_action")
    if command.intent in CONFIRMATION_INTENTS:
        return PolicyDecision(False, "confirmation_required", "motion_requires_confirmation", True)
    if command.intent in CONFIRM_INTENTS:
        return PolicyDecision(True, "accepted", "confirmation_received")
    if command.intent in LOW_RISK_INTENTS or command.intent in STOP_INTENTS or command.intent in MOTION_INTENTS:
        return PolicyDecision(True, "accepted", "allowed_voice_command")
    return PolicyDecision(False, "rejected", "unknown_intent")
