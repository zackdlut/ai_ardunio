"""Turn boxes from the vision model into 0..1 frame coordinates for the overlay."""

from __future__ import annotations

from typing import Any, Iterable

BOX_FORMATS = ("norm1000", "norm1", "pixel")
ANNOTATION_KINDS = ("check", "fault", "ok", "danger")
MAX_ANNOTATIONS = 12
_MIN_SIDE = 0.004


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
    values = _four_numbers(box)
    if values is None:
        return None
    fmt = box_format if box_format in BOX_FORMATS else "norm1000"
    if fmt == "pixel" and not image_size:
        fmt = "norm1000"
    if fmt != "norm1" and max(values) <= 1.0:
        # Models often fall back to 0..1 whatever the prompt asked for.
        fmt = "norm1"

    if fmt == "norm1000":
        sx = sy = 1000.0
    elif fmt == "pixel":
        sx, sy = float(image_size[0]), float(image_size[1])  # type: ignore[index]
    else:
        sx = sy = 1.0

    x1, y1, x2, y2 = values
    x1, x2 = sorted((x1 / sx, x2 / sx))
    y1, y2 = sorted((y1 / sy, y2 / sy))
    x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in (x1, y1, x2, y2))
    if x2 - x1 < _MIN_SIDE or y2 - y1 < _MIN_SIDE:
        return None
    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]


def make_annotation(
    item: dict[str, Any],
    box_format: str = "norm1000",
    image_size: tuple[int, int] | None = None,
) -> dict[str, Any] | None:
    box = normalize_box(item.get("box"), box_format, image_size)
    if box is None:
        return None
    kind = str(item.get("kind") or "check").strip().lower()
    out: dict[str, Any] = {
        "label": str(item.get("label") or "").strip()[:24] or "检查点",
        "kind": kind if kind in ANNOTATION_KINDS else "check",
        "box": box,
        "note": str(item.get("note") or "").strip()[:60],
    }
    check_id = item.get("check_id")
    if check_id not in (None, ""):
        try:
            out["check_id"] = int(check_id)
        except (TypeError, ValueError):
            pass
    return out


def merge_annotations(
    current: Iterable[dict[str, Any]],
    new: Iterable[dict[str, Any]],
    limit: int = MAX_ANNOTATIONS,
) -> list[dict[str, Any]]:
    """Append ``new``; an item with the same label or check_id replaces the old one."""
    merged = [dict(item) for item in current]
    for item in new:
        merged = [
            old
            for old in merged
            if old.get("label") != item.get("label")
            and not (item.get("check_id") is not None and old.get("check_id") == item.get("check_id"))
        ]
        merged.append(dict(item))
    return merged[-limit:]


def _four_numbers(box: Any) -> list[float] | None:
    if isinstance(box, str):
        box = [part for part in box.replace("[", " ").replace("]", " ").replace(",", " ").split()]
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        return [float(v) for v in box]
    except (TypeError, ValueError):
        return None
