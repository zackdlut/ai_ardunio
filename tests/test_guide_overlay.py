"""AR guide, gesture latch, and click-to-mark helpers."""

from __future__ import annotations

import math
import sys
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from agent.tutor_context import build_extra_system  # noqa: E402
from gesture.debounce import GestureDebouncer  # noqa: E402
from gesture.mediapipe_ctrl import GestureCommand, GestureHandler  # noqa: E402
from overlay.engine import OverlayEngine  # noqa: E402
from overlay.text_sprite import TextSpriteCache, blit_bgra  # noqa: E402
from project.state_machine import ProjectDef, ProjectStateMachine, StepDef  # noqa: E402
from ui.camera_widget import (  # noqa: E402
    CameraWidget,
    effective_mode,
    map_viewport_click,
    normalize_stroke,
    stroke_kind,
)
from vision.detector import Detection, scale_detection_boxes  # noqa: E402
from vision.marks import components_in_mark, make_region_mark, nearest_component  # noqa: E402


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_gesture_fires_once_until_release() -> None:
    debouncer = GestureDebouncer(stable_frames=3, release_frames=2)
    assert debouncer.push(GestureCommand.NEXT) is None
    assert debouncer.push(GestureCommand.NEXT) is None
    assert debouncer.push(GestureCommand.NEXT) is GestureCommand.NEXT
    assert debouncer.push(GestureCommand.NEXT) is None
    assert debouncer.push(GestureCommand.NONE) is None
    assert debouncer.push(GestureCommand.NEXT) is None
    assert debouncer.push(GestureCommand.NONE) is None
    assert debouncer.push(GestureCommand.NONE) is None
    assert debouncer.push(GestureCommand.PAUSE) is None
    assert debouncer.push(GestureCommand.PAUSE) is None
    assert debouncer.push(GestureCommand.PAUSE) is GestureCommand.PAUSE


def test_click_maps_through_letterbox() -> None:
    assert map_viewport_click(200, 100, 100, 50, 50, 25) == (0.0, 0.0)
    assert map_viewport_click(200, 100, 100, 50, 149, 74) == (0.99, 0.98)
    assert map_viewport_click(200, 100, 100, 50, 10, 10) is None


def _circle(cx: float, cy: float, r: float, n: int = 40) -> list[tuple[float, float]]:
    return [(cx + r * math.cos(i * math.tau / n), cy + r * math.sin(i * math.tau / n)) for i in range(n + 1)]


def test_stroke_kind_tells_click_box_and_lasso() -> None:
    assert stroke_kind([(10, 10)]) == "point"
    assert stroke_kind([(10, 10), (12, 13)]) == "point"
    assert stroke_kind([(10, 10), (40, 35), (80, 70), (120, 100)]) == "box"
    assert stroke_kind([(120, 100), (60, 50), (10, 10)]) == "box"
    assert stroke_kind([(10, 10), (60, 12), (120, 11)]) == "point"
    assert stroke_kind(_circle(100, 100, 40)) == "lasso"
    assert stroke_kind(_circle(100, 100, 40), mode="box") == "point"
    assert stroke_kind([(10, 10), (100, 100), (60, 60)]) == "box"


def test_forced_modes_skip_the_guess() -> None:
    wobbly_box = [(10, 10), (120, 100), (40, 90)]
    assert stroke_kind(wobbly_box) == "lasso"
    assert stroke_kind(wobbly_box, mode="box") == "box"
    open_arc = _circle(100, 100, 40)[:12]
    assert stroke_kind(open_arc) == "box"
    assert stroke_kind(open_arc, mode="lasso") == "lasso"
    assert stroke_kind(_circle(100, 100, 40), mode="point") == "point"
    assert stroke_kind([(10, 10), (120, 100)], mode="lasso") == "point"

    shift = Qt.KeyboardModifier.ShiftModifier
    ctrl = Qt.KeyboardModifier.ControlModifier
    none = Qt.KeyboardModifier.NoModifier
    assert effective_mode("lasso", shift) == "box"
    assert effective_mode("point", ctrl) == "lasso"
    assert effective_mode("point", none) == "point"


def test_mode_bar_switches_mark_mode() -> None:
    _app()
    widget = CameraWidget()
    assert widget.mark_mode() == "auto"
    widget.set_mark_mode("lasso")
    assert widget.mark_mode() == "lasso"
    widget.set_mark_mode("nope")
    assert widget.mark_mode() == "lasso"


def test_stroke_clamps_to_letterboxed_picture() -> None:
    assert normalize_stroke(200, 100, 100, 50, [(50, 25), (300, -5)]) == [(0.0, 0.0), (1.0, 0.0)]


def test_region_marks_find_parts_inside() -> None:
    scene = {
        "frame": {"width": 100, "height": 100},
        "nodes": {
            "led_1": {"type": "LED", "bbox": [10, 10, 20, 20]},
            "res_1": {"type": "Resistor", "bbox": [70, 70, 90, 90]},
        },
    }
    box = make_region_mark({"kind": "box", "box": [0.3, 0.3, 0.0, 0.0]}, "标记 1")
    assert box == {"label": "标记 1", "kind": "box", "nx": 0.15, "ny": 0.15, "box": [0.0, 0.0, 0.3, 0.3]}
    assert components_in_mark(scene, box) == ["LED"]

    loop = make_region_mark({"kind": "lasso", "points": [(0.6, 0.6), (1.2, 0.6), (0.6, 0.9)]}, "标记 2")
    assert loop is not None
    assert loop["box"] == [0.6, 0.6, 1.0, 0.9]
    assert components_in_mark(scene, loop) == []
    loop["points"] = [[0.6, 0.6], [1.0, 0.6], [1.0, 1.0], [0.6, 1.0]]
    assert components_in_mark(scene, loop) == ["电阻"]

    assert make_region_mark({"kind": "box", "box": [0.2, 0.2, 0.2, 0.5]}, "x") is None
    assert make_region_mark({"kind": "lasso", "points": [(0.1, 0.1)]}, "x") is None


def test_scale_boxes_back_to_camera_frame() -> None:
    det = Detection(type="LED", bbox=[10, 20, 30, 40], confidence=0.9)
    scale_detection_boxes([det], 2.0, 2.0)
    assert det.bbox == [20, 40, 60, 80]


def test_nearest_mark_uses_frame_size() -> None:
    scene = {
        "frame": {"width": 100, "height": 100},
        "nodes": {"led_1": {"type": "LED", "bbox": [40, 40, 60, 60]}},
    }
    assert nearest_component(scene, 0.5, 0.5) == "LED"
    assert nearest_component(scene, 0.0, 0.0, max_px=40) is None


def test_missing_part_is_chinese() -> None:
    project = ProjectDef(
        id="blink",
        title="Blink LED",
        steps=[StepDef(id="insert_resistor", requires=["Resistor"], hint="串联电阻")],
    )
    machine = ProjectStateMachine(project, session_id=0)
    ok, msg = machine.verify_step({"summary": {}})
    assert ok is False
    assert "电阻" in msg


def test_tutor_sees_marks() -> None:
    text = build_extra_system(
        {"title": "Blink", "project_id": "blink", "step_index": 0, "total_steps": 5, "current_step": "a", "current_hint": "b", "completed_steps": []},
        False,
        "还没看到：LED",
        {
            "summary": {},
            "user_marks": [
                {"label": "标记 1", "nx": 0.2, "ny": 0.4},
                {"label": "标记 2", "kind": "lasso", "nx": 0.5, "ny": 0.5, "box": [0.4, 0.4, 0.6, 0.6], "points": [[0.4, 0.4]]},
            ],
        },
        False,
    )
    assert "标记 1" in text
    assert "标记 2" in text
    assert "points" not in text


def test_chinese_sprite_and_hud() -> None:
    _app()
    cache = TextSpriteCache()
    sprite = cache.get("将 LED 插入面包板", (244, 250, 246), px=20, max_width=280)
    assert sprite is not None
    assert sprite.shape[2] == 4
    assert int(sprite[:, :, 3].max()) > 0

    seen: dict[str, np.ndarray] = {}

    def render() -> None:
        seen["sprite"] = TextSpriteCache().get("电阻", (255, 255, 255), px=18)

    worker = threading.Thread(target=render)
    worker.start()
    worker.join()
    assert seen["sprite"] is not None
    assert int(seen["sprite"][:, :, 3].max()) > 0

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    blit_bgra(frame, sprite, 8, 8)
    assert int(frame.max()) > 0

    guided = OverlayEngine().render(
        frame,
        [Detection(type="LED", bbox=[80, 60, 140, 130], confidence=0.91)],
        hint={"hint": "将 LED 插入面包板", "requires": ["LED"], "step_label": "第 1/5 步"},
        marks=[{"nx": 0.5, "ny": 0.5, "label": "标记 1"}],
        pulse=0.2,
    )
    assert guided.shape == frame.shape
    assert int(guided.max()) > int(frame.max())

    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    regions = OverlayEngine().render(
        blank,
        [],
        marks=[
            {"label": "标记 2", "kind": "box", "nx": 0.5, "ny": 0.5, "box": [0.4, 0.4, 0.6, 0.6]},
            {"label": "标记 3", "kind": "lasso", "nx": 0.2, "ny": 0.8, "box": [0.1, 0.7, 0.3, 0.9],
             "points": [[0.1, 0.7], [0.3, 0.7], [0.2, 0.9]]},
        ],
    )
    assert int(regions[96, 160:192].max()) > 0
    assert int(regions[167, 40:90].max()) > 0


class _Pt:
    def __init__(self, y: float) -> None:
        self.y = y


def _hand(extended: list[bool], thumb_up: bool) -> list[_Pt]:
    pts = [_Pt(0.5) for _ in range(21)]
    pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
    for flag, (tip, pip) in zip(extended, pairs):
        pts[tip] = _Pt(0.2 if flag else 0.7)
        pts[pip] = _Pt(0.45)
    pts[4] = _Pt(0.2 if thumb_up else 0.7)
    pts[3] = _Pt(0.45)
    return pts


def test_v_sign_is_not_an_open_palm() -> None:
    handler = GestureHandler()
    assert handler._classify(_hand([True, True, False, False], False)) is GestureCommand.RESCAN
    assert handler._classify(_hand([True, True, True, True], False)) is GestureCommand.NONE
    assert handler._classify(_hand([False, False, False, False], True)) is GestureCommand.NEXT
    assert handler._classify(_hand([False, False, False, False], False)) is GestureCommand.PAUSE
