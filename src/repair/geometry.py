"""Homographies between 0..1 frame coordinates and the shapes they move."""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

UNIT_SQUARE = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])


def identity() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def warp_points(H: np.ndarray | None, pts: Iterable[Sequence[float]]) -> list[list[float]]:
    arr = np.asarray([[float(p[0]), float(p[1])] for p in pts], dtype=np.float64)
    if arr.size == 0:
        return []
    if H is None:
        return arr.tolist()
    ones = np.ones((arr.shape[0], 1))
    mapped = np.hstack([arr, ones]) @ np.asarray(H, dtype=np.float64).T
    w = mapped[:, 2:3]
    w[np.abs(w) < 1e-9] = 1e-9
    return (mapped[:, :2] / w).tolist()


def warp_shape(shape: dict[str, Any], H: np.ndarray | None) -> dict[str, Any]:
    out = dict(shape)
    pts = shape.get("pts") or []
    if H is not None and pts:
        out["pts"] = [[round(x, 5), round(y, 5)] for x, y in warp_points(H, pts)]
    return out


def warp_shapes(shapes: Iterable[dict[str, Any]], H: np.ndarray | None) -> list[dict[str, Any]]:
    return [warp_shape(item, H) for item in shapes]


def quad_homography(quad: Sequence[Sequence[float]]) -> np.ndarray:
    """Homography taking the unit square onto ``quad`` (TL, TR, BR, BL)."""
    dst = np.float32([[float(p[0]), float(p[1])] for p in quad])
    return cv2.getPerspectiveTransform(UNIT_SQUARE, dst).astype(np.float64)


def circle_polygon(quad: Sequence[Sequence[float]], samples: int = 40) -> list[list[float]]:
    """The ellipse inscribed in the unit square, mapped into ``quad``.

    A conic stays a conic under a homography, so this stays exact when the
    board tilts instead of drifting like a fitted ellipse would.
    """
    ring = [
        [0.5 + 0.5 * math.cos(math.tau * i / samples), 0.5 + 0.5 * math.sin(math.tau * i / samples)]
        for i in range(samples)
    ]
    return warp_points(quad_homography(quad), ring)


def shape_bounds(shape: dict[str, Any]) -> list[float] | None:
    pts = shape.get("pts") or []
    if not pts:
        return None
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def union_bounds(shapes: Iterable[dict[str, Any]]) -> list[float] | None:
    boxes = [b for b in (shape_bounds(s) for s in shapes) if b]
    if not boxes:
        return None
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def tracking_roi(
    shapes: Iterable[dict[str, Any]],
    board_box: Sequence[float] | None = None,
    min_area: float = 0.4,
) -> list[float] | None:
    """Where to look for features on the reference frame.

    ``board_box`` wins. Otherwise the shapes' extent grown by its own size on
    every side, then grown further until it covers ``min_area`` of the frame.
    None means the whole frame.
    """
    if board_box:
        box = [float(v) for v in board_box]
    else:
        box = union_bounds(shapes)
        if box is None:
            return None
        w, h = box[2] - box[0], box[3] - box[1]
        box = [box[0] - w, box[1] - h, box[2] + w, box[3] + h]
    x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in box)
    for _ in range(12):
        if (x2 - x1) * (y2 - y1) >= min_area:
            break
        gx = max(0.05, (x2 - x1) * 0.25)
        gy = max(0.05, (y2 - y1) * 0.25)
        x1, y1 = max(0.0, x1 - gx), max(0.0, y1 - gy)
        x2, y2 = min(1.0, x2 + gx), min(1.0, y2 + gy)
    if (x2 - x1) * (y2 - y1) >= 0.95:
        return None
    return [x1, y1, x2, y2]
