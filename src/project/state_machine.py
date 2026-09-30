from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from project.models import StepHistory, get_session_factory


@dataclass
class StepDef:
    id: str
    requires: list[str]
    hint: str
    spatial: str | None = None
    pin: str | None = None
    action: str | None = None
    sketch: str | None = None


@dataclass
class ProjectDef:
    id: str
    title: str
    steps: list[StepDef]


def load_project(path: Path) -> ProjectDef:
    data = json.loads(path.read_text(encoding="utf-8"))
    steps = [
        StepDef(
            id=s["id"],
            requires=s.get("requires", []),
            hint=s.get("hint", ""),
            spatial=s.get("spatial"),
            pin=s.get("pin"),
            action=s.get("action"),
            sketch=s.get("sketch"),
        )
        for s in data["steps"]
    ]
    return ProjectDef(id=data["id"], title=data.get("title", data["id"]), steps=steps)


class ProjectStateMachine:
    def __init__(self, project: ProjectDef, session_id: int, step_index: int = 0) -> None:
        self.project = project
        self.session_id = session_id
        self._index = max(0, min(step_index, len(project.steps) - 1))

    @property
    def current_step(self) -> StepDef:
        return self.project.steps[self._index]

    def current_step_id(self) -> str:
        return self.current_step.id

    def next_step(self) -> StepDef | None:
        if self._index + 1 >= len(self.project.steps):
            return None
        return self.project.steps[self._index + 1]

    def advance(self) -> StepDef | None:
        nxt = self.next_step()
        if nxt is None:
            return None
        self._index += 1
        self._log("advanced", self.current_step.id)
        return self.current_step

    def verify_step(self, scene_summary: dict[str, Any]) -> tuple[bool, str]:
        step = self.current_step
        if step.action == "upload_sketch":
            return True, "准备烧录（请确认接线后执行 upload）"

        counts = scene_summary.get("summary", {})
        missing = [r for r in step.requires if counts.get(r, 0) < 1]
        if missing:
            return False, f"未检测到: {', '.join(missing)}（置信度可能较低，可语音确认）"
        return True, "步骤条件已满足（启发式判断）"

    def progress(self) -> dict[str, Any]:
        completed = [s.id for s in self.project.steps[: self._index]]
        return {
            "project_id": self.project.id,
            "title": self.project.title,
            "current_step": self.current_step.id,
            "current_hint": self.current_step.hint,
            "completed_steps": completed,
            "step_index": self._index,
            "total_steps": len(self.project.steps),
        }

    def _log(self, status: str, step_id: str, detail: str = "") -> None:
        factory = get_session_factory()
        with factory() as db:
            db.add(
                StepHistory(
                    session_id=self.session_id,
                    step_id=step_id,
                    status=status,
                    detail=detail,
                )
            )
            db.commit()
