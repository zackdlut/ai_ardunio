"""zoom_view crops the reference frame; shapes measured on the crop map back to the full image."""

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
from agent.repair_tools import RepairContext, build_repair_handlers  # noqa: E402
from overlay.engine import OverlayEngine  # noqa: E402
from project.models import Base  # noqa: E402
from repair.case_manager import CaseManager  # noqa: E402
from repair.debug_log import RepairDebugLog  # noqa: E402
from repair.documents import DocumentLibrary  # noqa: E402


@pytest.fixture()
def ctx(tmp_path: Path) -> RepairContext:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    cases = CaseManager(sessionmaker(bind=engine, expire_on_commit=False))
    ctx = RepairContext(cases=cases, library=DocumentLibrary(tmp_path / "docs"))
    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[360:720, 640:1280] = (0, 0, 255)
    ctx.begin_turn(frame, (1280, 720))
    return ctx


def test_zoom_crops_and_enlarges(ctx: RepairContext) -> None:
    tools = build_repair_handlers(ctx)
    reply = tools["zoom_view"](box=[500, 500, 1000, 1000], reason="看清右下角")
    assert "zoom_id=1" in reply
    images = ctx.drain_images()
    assert len(images) == 1 and images[0][1].startswith("data:image/jpeg;base64,")
    info = ctx.zoom_info(1)
    x1, y1, x2, y2 = info["region"]
    assert x1 == pytest.approx(0.47, abs=0.01) and y1 == pytest.approx(0.47, abs=0.01)
    assert x2 == pytest.approx(1.0) and y2 == pytest.approx(1.0)
    assert max(info["size"]) == 1280


def test_annotate_with_zoom_maps_back(ctx: RepairContext) -> None:
    tools = build_repair_handlers(ctx)
    tools["zoom_view"](box=[500, 500, 1000, 1000])
    x1, y1, x2, y2 = ctx.zoom_info(1)["region"]
    tools["annotate"](
        items=[
            {"type": "rect", "label": "C3", "box": [0, 0, 500, 500]},
            {"type": "arrow", "from": [1000, 1000], "to": [500, 500]},
        ],
        zoom_id=1,
    )
    rect, arrow = ctx.current_shapes()
    mx, my = x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2
    assert rect["pts"][0] == pytest.approx([x1, y1], abs=1e-3)
    assert rect["pts"][2] == pytest.approx([mx, my], abs=1e-3)
    assert arrow["pts"][0] == pytest.approx([x2, y2], abs=1e-3)
    assert arrow["pts"][1] == pytest.approx([mx, my], abs=1e-3)


def test_unknown_zoom_id_is_an_error(ctx: RepairContext) -> None:
    tools = build_repair_handlers(ctx)
    with pytest.raises(ValueError, match="zoom_id"):
        tools["annotate"](items=[{"type": "rect", "box": [0, 0, 100, 100]}], zoom_id=3)
    assert ctx.current_shapes() == []


def test_new_turn_forgets_zooms(ctx: RepairContext) -> None:
    tools = build_repair_handlers(ctx)
    tools["zoom_view"](box=[0, 0, 500, 500])
    ctx.begin_turn(ctx.anchor_frame, (1280, 720))
    with pytest.raises(ValueError):
        ctx.zoom_info(1)


def test_zoom_without_frame(ctx: RepairContext) -> None:
    ctx.begin_turn(None, None)
    with pytest.raises(ValueError, match="没有附画面"):
        build_repair_handlers(ctx)["zoom_view"](box=[0, 0, 500, 500])


def test_render_marks_leaves_shapes_off(ctx: RepairContext) -> None:
    frame = np.full((240, 320, 3), 40, np.uint8)
    clean = OverlayEngine().render_marks(frame, [])
    assert np.array_equal(clean, frame) and clean is not frame
    marked = OverlayEngine().render_marks(frame, [{"nx": 0.5, "ny": 0.5}])
    assert not np.array_equal(marked, frame)


def test_debug_log_records_a_turn(ctx: RepairContext, tmp_path: Path) -> None:
    log = RepairDebugLog(tmp_path / "debug")
    ctx.debug = log
    tools = build_repair_handlers(ctx)
    frame = ctx.anchor_frame
    folder = log.begin(7, "框出 C3", "extra", frame[::2, ::2], frame, {"box_format": "norm1000"})
    tools["zoom_view"](box=[500, 500, 1000, 1000])
    tools["annotate"](items=[{"type": "rect", "label": "C3", "box": [0, 0, 500, 500]}], zoom_id=1)
    with pytest.raises(ValueError):
        tools["annotate"](items=[{"type": "rect", "box": [0, 0, 1, 1]}], zoom_id=9)
    assert log.finish(7, "好了", ctx.current_shapes(), frame) == folder

    names = {p.name for p in folder.iterdir()}
    assert {"sent.jpg", "raw.jpg", "meta.json", "tools.jsonl", "img_1.jpg", "images.txt",
            "reply.txt", "shapes.json", "result.jpg"} <= names
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["sent_size"] == [640, 360] and meta["raw_size"] == [1280, 720]
    calls = [json.loads(line) for line in (folder / "tools.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [c["tool"] for c in calls] == ["zoom_view", "annotate", "annotate"]
    assert calls[1]["args"]["zoom_id"] == 1
    assert calls[2]["result"].startswith("ERROR ValueError")
    assert json.loads((folder / "shapes.json").read_text(encoding="utf-8"))[0]["label"] == "C3"
