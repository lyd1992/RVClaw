from __future__ import annotations

from pathlib import Path

from rvclaw.adapters.mock_device import MockDevice
from rvclaw.agent.core import AgentCore
from rvclaw.agent.planner import planner_from_name
from rvclaw.agent.safety_guard import SafetyGuard, SkillRegistry
from rvclaw.agent.tool_router import ToolRouter
from rvclaw.memory.memory_api import MemoryManager
from rvclaw.models import RunSummary
from rvclaw.observability import RunRecorder
from rvclaw.skills.builtin import build_builtin_skills
from rvclaw.utils import ensure_dir, make_run_id


def run_demo(
    goal: str,
    runs_dir: str | Path = "runs",
    planner_name: str = "auto",
    memory_db: str | Path | None = None,
    run_id: str | None = None,
) -> RunSummary:
    runs_dir = ensure_dir(runs_dir)
    run_id = run_id or make_run_id()
    run_dir = runs_dir / run_id
    recorder = RunRecorder(run_id=run_id, run_dir=run_dir)

    memory_path = Path(memory_db) if memory_db else runs_dir / "memory.sqlite3"
    memory = MemoryManager.from_path(memory_path)
    device = MockDevice(artifact_dir=run_dir / "artifacts")
    registry = SkillRegistry.from_default()
    guard = SafetyGuard(registry)
    skills = build_builtin_skills(device=device, memory=memory, run_id=run_id)
    router = ToolRouter(skills=skills, safety_guard=guard, recorder=recorder)
    planner = planner_from_name(planner_name)

    core = AgentCore(planner=planner, router=router, memory=memory, recorder=recorder)
    return core.run(goal=goal)
