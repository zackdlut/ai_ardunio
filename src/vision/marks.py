from __future__ import annotations

import math
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
