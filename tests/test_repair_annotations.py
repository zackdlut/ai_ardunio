"""Boxes from the vision model land on the right part of the camera frame."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from overlay.engine import OverlayEngine  # noqa: E402
from repair.annotations import (  # noqa: E402
    make_annotation,
    merge_annotations,
    normalize_box,
    sent_image_size,
)
from vision.detector import Detection  # noqa: E402


def test_norm1000_box_becomes_fraction() -> None:
    assert normalize_box([100, 200, 300, 400]) == [0.1, 0.2, 0.3, 0.4]


def test_pixel_box_uses_sent_image_size() -> None:
    assert normalize_box([128, 72, 640, 360], "pixel", (1280, 720)) == [0.1, 0.1, 0.5, 0.5]


def test_pixel_without_size_falls_back_to_norm1000() -> None:
    assert normalize_box([100, 100, 500, 500], "pixel", None) == [0.1, 0.1, 0.5, 0.5]


def test_fraction_box_is_recognised_whatever_the_format() -> None:
    assert normalize_box([0.1, 0.2, 0.3, 0.4], "norm1000") == [0.1, 0.2, 0.3, 0.4]


def test_swapped_and_out_of_range_boxes_are_fixed() -> None:
    assert normalize_box([300, 400, 100, 200]) == [0.1, 0.2, 0.3, 0.4]
    assert normalize_box([-50, 900, 1200, 1100]) == [0.0, 0.9, 1.0, 1.0]
    assert normalize_box("[100, 200, 300, 400]") == [0.1, 0.2, 0.3, 0.4]


def test_unusable_boxes_are_dropped() -> None:
    assert normalize_box(None) is None
    assert normalize_box([1, 2, 3]) is None
    assert normalize_box(["a", 2, 3, 4]) is None
    assert normalize_box([500, 500, 501, 501]) is None


def test_annotation_defaults_and_kind() -> None:
    ann = make_annotation({"box": [0, 0, 500, 500], "kind": "FAULT", "check_id": "3"})
    ann.pop("id")
    assert ann == {
        "type": "rect",
        "label": "检查点",
        "kind": "fault",
        "pts": [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]],
        "note": "",
        "check_id": 3,
    }
    assert make_annotation({"box": [0, 0, 500, 500], "kind": "weird"})["kind"] == "check"


def test_sent_image_size_matches_encoder() -> None:
    assert sent_image_size(1280, 720, 768) == (768, 432)
    assert sent_image_size(640, 480, 1280) == (640, 480)


def test_merge_replaces_same_label_or_check() -> None:
    old = [
        {"label": "U3", "box": [0, 0, 0.1, 0.1]},
        {"label": "1. 保险丝", "check_id": 1, "box": [0, 0, 0.1, 0.1]},
    ]
    new = [
        {"label": "U3", "box": [0.5, 0.5, 0.6, 0.6]},
        {"label": "保险丝 F1", "check_id": 1, "box": [0.2, 0.2, 0.3, 0.3]},
    ]
    merged = merge_annotations(old, new)
    assert [a["label"] for a in merged] == ["U3", "保险丝 F1"]
    assert merged[0]["box"] == [0.5, 0.5, 0.6, 0.6]
    assert len(merge_annotations([], [{"label": str(i)} for i in range(20)], limit=5)) == 5


def test_repair_overlay_draws_annotations_and_hides_yolo() -> None:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    yolo = [Detection(type="LED", bbox=[20, 250, 120, 330], confidence=0.9)]
    out = OverlayEngine().render(
        frame,
        yolo,
        hint={"hint": "测 3.3V", "step_label": "检查 1/3", "check_id": 7},
        mode="repair",
        annotations=[
            {"label": "U3", "kind": "fault", "box": [0.5, 0.5, 0.7, 0.7], "check_id": 7},
            {"label": "高压区", "kind": "danger", "box": [0.05, 0.2, 0.25, 0.4]},
        ],
        frozen=True,
    )
    assert int(out[180:252, 320].max()) > 0
    assert int(out[100:140, 40:150].max()) > 0
    assert int(out[260:320, 30:110].max()) == 0

    tutor = OverlayEngine().render(frame, yolo, hint={"hint": "插 LED"})
    assert int(tutor[248:252, 20:40].max()) > 0
