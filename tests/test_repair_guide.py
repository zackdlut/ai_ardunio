"""A step-by-step AR guide: start, confirm, correct, jump, rebase and end."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import repair.models  # noqa: E402,F401
from agent.repair_context import build_repair_extra_system  # noqa: E402
from agent.repair_tools import RepairContext, build_repair_handlers, build_repair_tools  # noqa: E402
from project.models import Base  # noqa: E402
from repair.case_manager import CaseManager  # noqa: E402
from repair.documents import DocumentLibrary  # noqa: E402


@pytest.fixture()
def ctx(tmp_path: Path) -> RepairContext:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    cases = CaseManager(sessionmaker(bind=engine, expire_on_commit=False))
    return RepairContext(cases=cases, library=DocumentLibrary(tmp_path / "docs"), image_size=(1280, 720))


def _start(tools) -> dict:
    return json.loads(
        tools["start_guide"](
            title="拆后盖",
            board_box=[100, 100, 900, 900],
            steps=[
                {
                    "title": "拧下螺丝",
                    "instruction": "十字螺丝刀拧下右上角螺丝",
                    "verify": "右上角螺丝孔是空的",
                    "shapes": [
                        {"type": "circle", "label": "螺丝1", "box": [800, 100, 860, 160]},
                        {"type": "arrow", "from": [700, 300], "to": [820, 140]},
                    ],
                },
                {
                    "title": "撬开卡扣",
                    "instruction": "从缺口向上撬",
                    "verify": "后盖翘起",
                    "shapes": [{"type": "arrow", "label": "向上撬", "from": [500, 900], "to": [500, 700]}],
                },
                {
                    "title": "测电池电压",
                    "instruction": "直流 20V 档",
                    "shapes": [{"type": "probe", "red": [300, 500], "black": [400, 500], "expected": "3.7V"}],
                },
            ],
        )
    )


def test_drew_this_turn_tracks_drawing_tools(ctx: RepairContext) -> None:
    tools = build_repair_handlers(ctx)
    ctx.begin_turn(None, (1280, 720))
    assert not ctx.drew_this_turn
    tools["annotate"](items=[{"type": "text", "at": [50, 50], "text": "先断电"}])
    assert ctx.drew_this_turn
    ctx.begin_turn(None, (1280, 720))
    assert not ctx.drew_this_turn
    _start(tools)
    assert ctx.drew_this_turn
    ctx.begin_turn(None, (1280, 720))
    ctx.advance("done")
    assert ctx.drew_this_turn


def test_guide_tools_are_registered() -> None:
    names = {t["function"]["name"] for t in build_repair_tools()}
    assert {"start_guide", "set_guide_step", "update_guide_step", "confirm_step", "end_guide"} <= names


def test_only_current_step_shapes_are_shown(ctx: RepairContext) -> None:
    tools = build_repair_handlers(ctx)
    tools["annotate"](items=[{"type": "text", "at": [50, 50], "text": "先断电"}])
    started = _start(tools)
    assert [s["shapes"] for s in started["steps"]] == [2, 1, 1]
    types = [s["type"] for s in ctx.current_shapes()]
    assert types == ["text", "circle", "arrow"]
    assert ctx.board_box == [0.1, 0.1, 0.9, 0.9]
    assert ctx.current_step()[0] == 0

    ctx.advance(mark="skipped")
    assert [s["type"] for s in ctx.current_shapes()] == ["text", "arrow"]
    assert ctx.guide["steps"][0]["status"] == "skipped"
    ctx.back()
    assert ctx.current_step()[0] == 0


def test_confirm_pass_fail_and_finish(ctx: RepairContext) -> None:
    tools = build_repair_handlers(ctx)
    _start(tools)

    failed = json.loads(
        tools["confirm_step"](
            passed=False,
            feedback="螺丝还在",
            shapes=[{"type": "point", "label": "这里", "at": [830, 130]}],
        )
    )
    assert failed["passed"] is False
    step = ctx.current_step()
    assert step[0] == 0 and step[1]["status"] == "failed" and step[1]["feedback"] == "螺丝还在"
    assert [s["type"] for s in step[1]["shapes"]] == ["circle", "arrow", "point"]

    ok = json.loads(tools["confirm_step"](passed=True, feedback="螺丝已拆下"))
    assert ok["passed"] and ok["next_step"] == {"step": 2, "title": "撬开卡扣"}
    assert ctx.guide["steps"][0]["status"] == "done"

    json.loads(tools["set_guide_step"](step=3))
    done = json.loads(tools["confirm_step"](passed=True, feedback="3.8V"))
    assert done["finished"] is True and ctx.guide is None
    assert ctx.current_shapes() == []
    assert "当前没有" in tools["end_guide"]()


def test_update_step_and_persistence(ctx: RepairContext) -> None:
    tools = build_repair_handlers(ctx)
    started = _start(tools)
    json.loads(
        tools["update_guide_step"](
            step=2, shapes=[{"type": "line", "from": [100, 100], "to": [200, 200]}], verify="能看到卡扣"
        )
    )
    stored = ctx.cases.get_guide(started["guide_id"])
    assert stored["steps"][1]["verify"] == "能看到卡扣"
    assert [s["type"] for s in stored["steps"][1]["shapes"]] == ["line"]

    # A fresh context picks the active guide back up from the database.
    again = RepairContext(cases=ctx.cases, library=ctx.library, case_id=ctx.case_id)
    assert again.load_guide()["id"] == started["guide_id"]
    assert again.current_step()[1]["title"] == "拧下螺丝"


def test_rebase_moves_annotations_steps_and_board(ctx: RepairContext) -> None:
    tools = build_repair_handlers(ctx)
    tools["annotate"](items=[{"type": "point", "at": [500, 500]}])
    started = _start(tools)
    H = np.array([[1, 0, 0.1], [0, 1, -0.05], [0, 0, 1]], dtype=np.float64)
    ctx.rebase(H)
    assert ctx.annotations[0]["pts"] == [[0.6, 0.45]]
    assert ctx.board_box == [0.2, 0.05, 1.0, 0.85]
    stored = ctx.cases.get_guide(started["guide_id"])
    assert stored["steps"][1]["shapes"][0]["pts"] == [[0.6, 0.85], [0.6, 0.65]]
    ctx.rebase(None)
    assert ctx.annotations[0]["pts"] == [[0.6, 0.45]]


def test_extra_system_describes_guide_and_tracking(ctx: RepairContext) -> None:
    tools = build_repair_handlers(ctx)
    _start(tools)
    tools["confirm_step"](passed=False, feedback="螺丝还在")
    extra = build_repair_extra_system(
        ctx.case(),
        [],
        ctx.annotations,
        [],
        False,
        has_image=True,
        guide=ctx.guide,
        track_status="tracking",
        image_size=ctx.image_size,
    )
    assert "进行中的指引" in extra and "← 当前步" in extra
    assert "右上角螺丝孔是空的" in extra and "反馈: 螺丝还在" in extra
    assert "[800, 100]" in extra
    assert "贴在板子上" in extra
