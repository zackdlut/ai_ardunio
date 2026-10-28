"""AR shapes: parsing from tool calls, moving with a homography, and drawing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from overlay.engine import OverlayEngine  # noqa: E402
from repair.annotations import as_shape, make_shape, make_shapes  # noqa: E402
from repair.geometry import tracking_roi, warp_points, warp_shape  # noqa: E402


def _shift(dx: float, dy: float) -> np.ndarray:
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)


def test_every_shape_type_parses() -> None:
    items = [
        {"type": "rect", "label": "U3", "box": [100, 100, 300, 200]},
        {"type": "circle", "label": "C1", "box": [400, 400, 500, 500]},
        {"type": "point", "at": [500, 500]},
        {"type": "text", "at": [100, 900], "text": "先断电"},
        {"type": "arrow", "label": "向上撬", "from": [500, 800], "to": [500, 600]},
        {"type": "line", "from": [100, 100], "to": [900, 900], "kind": "danger"},
        {"type": "probe", "red": [200, 300], "black": [800, 300], "meter": "DCV 20V", "expected": "3.3V"},
    ]
    shapes, skipped = make_shapes(items)
    assert skipped == 0
    by_type = {s["type"]: s for s in shapes}
    assert by_type["rect"]["pts"] == [[0.1, 0.1], [0.3, 0.1], [0.3, 0.2], [0.1, 0.2]]
    assert by_type["arrow"]["pts"] == [[0.5, 0.8], [0.5, 0.6]]
    assert by_type["probe"]["pts"] == [[0.2, 0.3], [0.8, 0.3]]
    assert by_type["probe"]["meter"] == "DCV 20V" and by_type["probe"]["expected"] == "3.3V"
    assert by_type["text"]["text"] == "先断电"
    assert by_type["line"]["kind"] == "danger"
    assert len({s["id"] for s in shapes}) == len(shapes)


def test_bad_shapes_are_skipped_and_types_guessed() -> None:
    shapes, skipped = make_shapes(
        [
            {"type": "arrow", "from": [500, 500], "to": [502, 501]},
            {"type": "probe", "black": [1, 2]},
            {"type": "text", "at": [1, 2]},
            "nonsense",
            {"red": [100, 100]},
            {"from": [0, 0], "to": [500, 500]},
        ]
    )
    assert skipped == 4
    assert [s["type"] for s in shapes] == ["probe", "arrow"]
    assert make_shape({"type": "point", "at": [0.25, 0.75]})["pts"] == [[0.25, 0.75]]


def test_old_box_annotations_become_rects() -> None:
    shape = as_shape({"label": "U3", "box": [0.1, 0.2, 0.3, 0.4]})
    assert shape["type"] == "rect" and "box" not in shape
    assert shape["pts"][2] == [0.3, 0.4]


def test_warp_moves_all_points() -> None:
    shape = make_shape({"type": "arrow", "from": [100, 100], "to": [300, 100]})
    moved = warp_shape(shape, _shift(0.2, 0.1))
    assert moved["pts"] == [[0.3, 0.2], [0.5, 0.2]]
    assert shape["pts"] == [[0.1, 0.1], [0.3, 0.1]]
    assert warp_points(None, [[0.5, 0.5]]) == [[0.5, 0.5]]


def test_tracking_roi_prefers_board_and_grows() -> None:
    assert tracking_roi([], [0.1, 0.1, 0.9, 0.8]) == [0.1, 0.1, 0.9, 0.8]
    assert tracking_roi([]) is None
    small = make_shape({"type": "rect", "box": [450, 450, 550, 550]})
    x1, y1, x2, y2 = tracking_roi([small])
    assert x1 < 0.45 and x2 > 0.55 and (x2 - x1) * (y2 - y1) >= 0.4


def test_shapes_follow_the_transform_when_drawn() -> None:
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    engine = OverlayEngine()
    shapes, _ = make_shapes(
        [
            {"type": "rect", "label": "", "box": [100, 300, 200, 400]},
            {"type": "arrow", "from": [500, 900], "to": [500, 700]},
            {"type": "probe", "red": [800, 300], "black": [900, 300], "meter": "DCV"},
            {"type": "circle", "box": [700, 700, 800, 800]},
            {"type": "line", "from": [100, 100], "to": [300, 100]},
            {"type": "point", "at": [300, 900]},
        ]
    )
    still = engine.render(frame, [], mode="repair", annotations=shapes, pulse=0.3)
    # Rect edge at x = 0.1 * 599 ~ 60, y between 120 and 160.
    assert int(still[130:150, 58:62].max()) > 0
    # Probe red dot at (479, 120).
    assert int(still[118:122, 477:481, 2].max()) > 150

    moved = engine.render(
        frame, [], mode="repair", annotations=shapes, pulse=0.3, transform=_shift(0.2, 0.0)
    )
    assert int(moved[130:150, 58:62].max()) == 0
    assert int(moved[130:150, 178:182].max()) > 0

    faded = engine.render(frame, [], mode="repair", annotations=shapes, shape_alpha=0.0)
    assert int(faded[130:150, 58:62].max()) == 0


def test_hud_shows_tracking_badge_and_lost_hint() -> None:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    shapes, _ = make_shapes([{"type": "point", "at": [500, 500]}])
    engine = OverlayEngine()
    lost = engine.render(frame, [], mode="repair", annotations=shapes, track_status="lost")
    tracking = engine.render(frame, [], mode="repair", annotations=shapes, track_status="tracking")
    # The lost hint sits on the line under the top bar.
    assert int(lost[50:70, 12:300].max()) > 0
    assert int(tracking[50:70, 12:300].max()) == 0
