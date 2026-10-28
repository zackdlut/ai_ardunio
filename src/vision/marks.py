from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from vision.component_db import display_name


def nearest_component(scene: dict[str, Any], nx: float, ny: float, max_px: float = 180) -> str | None:
    """Return the Chinese name of the detection closest to a normalized click."""
    frame = scene.get("frame") or {}
    width = float(frame.get("width") or 0)
    height = float(frame.get("height") or 0)
    if width <= 1 or height <= 1:
        return None
    px = nx * width
    py = ny * height
    best_name: str | None = None
    best_dist = max_px
    for node in (scene.get("nodes") or {}).values():
        bbox = node.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        dist = math.hypot(cx - px, cy - py)
        if dist < best_dist:
            best_dist = dist
            best_name = display_name(str(node.get("type") or ""))
    return best_name


def make_region_mark(shape: dict[str, Any], label: str) -> dict[str, Any] | None:
    """Build a ``box`` or ``lasso`` mark (normalized coords) from a widget stroke."""
    kind = shape.get("kind")
    try:
        if kind == "box":
            x1, y1, x2, y2 = (_unit(v) for v in shape.get("box") or ())
            box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            extra: dict[str, Any] = {}
        elif kind == "lasso":
            pts = [(_unit(x), _unit(y)) for x, y in shape.get("points") or ()]
            if len(pts) < 3:
                return None
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            box = [min(xs), min(ys), max(xs), max(ys)]
            extra = {"points": [[round(x, 3), round(y, 3)] for x, y in pts]}
        else:
            return None
    except (TypeError, ValueError):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return {
        "label": label,
        "kind": kind,
        "nx": round((box[0] + box[2]) / 2, 3),
        "ny": round((box[1] + box[3]) / 2, 3),
        "box": [round(v, 3) for v in box],
        **extra,
    }


def components_in_mark(scene: dict[str, Any], mark: dict[str, Any]) -> list[str]:
    """Chinese names of detections whose box center falls inside a box/lasso mark."""
    frame = scene.get("frame") or {}
    width = float(frame.get("width") or 0)
    height = float(frame.get("height") or 0)
    box = mark.get("box")
    if width <= 1 or height <= 1 or not box or len(box) != 4:
        return []
    polygon = mark.get("points") if mark.get("kind") == "lasso" else None
    names: list[str] = []
    for node in (scene.get("nodes") or {}).values():
        bbox = node.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        cx = (bbox[0] + bbox[2]) / 2 / width
        cy = (bbox[1] + bbox[3]) / 2 / height
        if not (box[0] <= cx <= box[2] and box[1] <= cy <= box[3]):
            continue
        if polygon and not _inside_polygon(cx, cy, polygon):
            continue
        name = display_name(str(node.get("type") or ""))
        if name not in names:
            names.append(name)
    return names


def _unit(value: Any) -> float:
    return min(max(float(value), 0.0), 1.0)


def _inside_polygon(x: float, y: float, polygon: Sequence[Sequence[float]]) -> bool:
    inside = False
    n = len(polygon)
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside
