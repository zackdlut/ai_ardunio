"""Turn boxes, points and shapes from the vision model into 0..1 frame coordinates.

A shape is ``{"id", "type", "label", "kind", "note", "pts", ...}``. ``pts``
holds every point the shape needs so a homography can move it as a whole:

- rect / circle: the four corners TL, TR, BR, BL (circle = inscribed ellipse)
- point / text: ``[at]``
- arrow / line: ``[from, to]``
- probe: ``[red]`` or ``[red, black]``, plus ``meter`` and ``expected``
"""

from __future__ import annotations

import itertools
from typing import Any, Iterable

BOX_FORMATS = ("norm1000", "norm1", "pixel")
ANNOTATION_KINDS = ("check", "fault", "ok", "danger")
SHAPE_TYPES = ("rect", "circle", "point", "text", "arrow", "line", "probe")
MAX_ANNOTATIONS = 12
_MIN_SIDE = 0.004
_MIN_LENGTH = 0.01
_ids = itertools.count(1)


def sent_image_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    """Size of the frame after ``encode_frame_data_url`` shrinks it."""
    longest = max(width, height)
    if max_side <= 0 or longest <= max_side:
        return width, height
    scale = max_side / longest
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def normalize_box(
    box: Any,
    box_format: str = "norm1000",
    image_size: tuple[int, int] | None = None,
) -> list[float] | None:
    """Return ``[x1, y1, x2, y2]`` within 0..1, or None if the box is unusable."""
    values = _numbers(box, 4)
    if values is None:
        return None
    sx, sy = _scale(values, box_format, image_size)
    x1, y1, x2, y2 = values
    x1, x2 = sorted((x1 / sx, x2 / sx))
    y1, y2 = sorted((y1 / sy, y2 / sy))
    x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in (x1, y1, x2, y2))
    if x2 - x1 < _MIN_SIDE or y2 - y1 < _MIN_SIDE:
        return None
    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]


def normalize_point(
    point: Any,
    box_format: str = "norm1000",
    image_size: tuple[int, int] | None = None,
) -> list[float] | None:
    values = _numbers(point, 2)
    if values is None:
        return None
    sx, sy = _scale(values, box_format, image_size)
    x, y = values[0] / sx, values[1] / sy
    return [round(min(max(x, 0.0), 1.0), 4), round(min(max(y, 0.0), 1.0), 4)]


def box_corners(box: list[float]) -> list[list[float]]:
    x1, y1, x2, y2 = box
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def make_shape(
    item: dict[str, Any],
    box_format: str = "norm1000",
    image_size: tuple[int, int] | None = None,
) -> dict[str, Any] | None:
    """One shape from a tool call, or None if its coordinates are unusable."""
    shape_type = str(item.get("type") or "").strip().lower()
    if shape_type not in SHAPE_TYPES:
        shape_type = _guess_type(item)
    if shape_type is None:
        return None

    def pt(*keys: str) -> list[float] | None:
        for key in keys:
            if item.get(key) is not None:
                return normalize_point(item.get(key), box_format, image_size)
        return None

    extra: dict[str, Any] = {}
    if shape_type in ("rect", "circle"):
        box = normalize_box(item.get("box"), box_format, image_size)
        if box is None:
            return None
        pts = box_corners(box)
    elif shape_type in ("point", "text"):
        at = pt("at", "point", "center")
        if at is None:
            return None
        pts = [at]
        if shape_type == "text":
            extra["text"] = str(item.get("text") or item.get("label") or "").strip()[:40]
            if not extra["text"]:
                return None
    elif shape_type in ("arrow", "line"):
        start, end = pt("from", "start"), pt("to", "end")
        if start is None or end is None:
            return None
        if abs(start[0] - end[0]) + abs(start[1] - end[1]) < _MIN_LENGTH:
            return None
        pts = [start, end]
    else:  # probe
        red, black = pt("red"), pt("black")
        if red is None:
            return None
        pts = [red] if black is None else [red, black]
        extra["meter"] = str(item.get("meter") or "").strip()[:16]
        extra["expected"] = str(item.get("expected") or "").strip()[:24]

    kind = str(item.get("kind") or "check").strip().lower()
    out: dict[str, Any] = {
        "id": str(item.get("id") or f"s{next(_ids)}"),
        "type": shape_type,
        "label": str(item.get("label") or "").strip()[:24],
        "kind": kind if kind in ANNOTATION_KINDS else "check",
        "note": str(item.get("note") or "").strip()[:60],
        "pts": pts,
        **extra,
    }
    if shape_type == "rect" and not out["label"]:
        out["label"] = "检查点"
    check_id = item.get("check_id")
    if check_id not in (None, ""):
        try:
            out["check_id"] = int(check_id)
        except (TypeError, ValueError):
            pass
    return out


def make_shapes(
    items: Iterable[Any],
    box_format: str = "norm1000",
    image_size: tuple[int, int] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """``(shapes, skipped)`` for a list of tool-call items."""
    made, skipped = [], 0
    for item in items or []:
        shape = make_shape(item, box_format, image_size) if isinstance(item, dict) else None
        if shape is None:
            skipped += 1
        else:
            made.append(shape)
    return made, skipped


def make_annotation(
    item: dict[str, Any],
    box_format: str = "norm1000",
    image_size: tuple[int, int] | None = None,
) -> dict[str, Any] | None:
    """A rect shape (the pre-AR annotation)."""
    return make_shape({**item, "type": "rect"}, box_format, image_size)


def as_shape(item: dict[str, Any]) -> dict[str, Any]:
    """Accept old ``{"box": [...]}`` annotations next to shapes."""
    if item.get("pts") or not item.get("box"):
        return item
    try:
        box = [float(v) for v in item["box"]]
    except (TypeError, ValueError):
        return item
    out = {k: v for k, v in item.items() if k != "box"}
    out.setdefault("type", "rect")
    out["pts"] = box_corners(box)
    return out


def shape_summary(shape: dict[str, Any]) -> dict[str, Any]:
    """What the model gets back about a shape (0..1, rounded)."""
    out = {k: shape[k] for k in ("id", "type", "label", "kind") if shape.get(k)}
    pts = shape.get("pts") or []
    if pts:
        out["pts"] = [[round(float(x), 3), round(float(y), 3)] for x, y in pts]
    for key in ("check_id", "text", "meter", "expected"):
        if shape.get(key):
            out[key] = shape[key]
    return out


def merge_annotations(
    current: Iterable[dict[str, Any]],
    new: Iterable[dict[str, Any]],
    limit: int = MAX_ANNOTATIONS,
) -> list[dict[str, Any]]:
    """Append ``new``; an item with the same id, label or check_id replaces the old one."""
    merged = [dict(item) for item in current]
    for item in new:
        label = item.get("label")
        merged = [
            old
            for old in merged
            if not (label and old.get("label") == label)
            and not (item.get("id") and old.get("id") == item.get("id"))
            and not (item.get("check_id") is not None and old.get("check_id") == item.get("check_id"))
        ]
        merged.append(dict(item))
    return merged[-limit:]


def _guess_type(item: dict[str, Any]) -> str | None:
    if item.get("red") is not None:
        return "probe"
    if item.get("from") is not None or item.get("start") is not None:
        return "arrow"
    if item.get("box") is not None:
        return "rect"
    if item.get("at") is not None:
        return "text" if item.get("text") else "point"
    return None


def _scale(
    values: list[float], box_format: str, image_size: tuple[int, int] | None
) -> tuple[float, float]:
    fmt = box_format if box_format in BOX_FORMATS else "norm1000"
    if fmt == "pixel" and not image_size:
        fmt = "norm1000"
    if fmt != "norm1" and max(values) <= 1.0:
        # Models often fall back to 0..1 whatever the prompt asked for.
        fmt = "norm1"
    if fmt == "norm1000":
        return 1000.0, 1000.0
    if fmt == "pixel":
        return float(image_size[0]), float(image_size[1])  # type: ignore[index]
    return 1.0, 1.0


def _numbers(value: Any, count: int) -> list[float] | None:
    if isinstance(value, dict) and "x" in value and "y" in value:
        value = [value["x"], value["y"]]
    if isinstance(value, str):
        value = value.replace("[", " ").replace("]", " ").replace(",", " ").split()
    if not isinstance(value, (list, tuple)) or len(value) != count:
        return None
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError):
        return None
