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


def invert(H: np.ndarray | None) -> np.ndarray | None:
    if H is None:
        return None
    try:
        return np.linalg.inv(np.asarray(H, dtype=np.float64))
    except np.linalg.LinAlgError:
        return None


def _rounded(pts: Iterable[Sequence[float]]) -> list[list[float]]:
    return [[round(x, 5), round(y, 5)] for x, y in pts]


def mark_quad(mark: dict[str, Any]) -> list[list[float]] | None:
    """A box mark's corners (TL, TR, BR, BL); None for point and lasso marks."""
    if mark.get("quad"):
        return [[float(p[0]), float(p[1])] for p in mark["quad"]]
    box = mark.get("box")
    if mark.get("kind") != "box" or not box:
        return None
    x1, y1, x2, y2 = (float(v) for v in box)
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def warp_mark(mark: dict[str, Any], H: np.ndarray | None) -> dict[str, Any]:
    """Move a user mark (point, box or lasso, 0..1 coordinates) by ``H``.

    A box stops being axis-aligned under a homography, so its corners are kept
    as ``quad`` and ``box`` becomes their bounds.
    """
    out = dict(mark)
    if H is None:
        return out
    if "nx" in mark and "ny" in mark:
        (nx, ny), = warp_points(H, [[mark["nx"], mark["ny"]]])
        out["nx"], out["ny"] = round(nx, 5), round(ny, 5)
    if mark.get("points"):
        out["points"] = _rounded(warp_points(H, mark["points"]))
    quad = mark_quad(mark)
    if quad:
        out["quad"] = _rounded(warp_points(H, quad))
        xs = [p[0] for p in out["quad"]]
        ys = [p[1] for p in out["quad"]]
        out["box"] = [min(xs), min(ys), max(xs), max(ys)]
    return out


def warp_marks(marks: Iterable[dict[str, Any]], H: np.ndarray | None) -> list[dict[str, Any]]:
    return [warp_mark(item, H) for item in marks]


def mark_bounds(mark: dict[str, Any]) -> list[float] | None:
    pts = mark.get("quad") or mark.get("points")
    if not pts and mark.get("box"):
        return [float(v) for v in mark["box"]]
    if not pts and "nx" in mark and "ny" in mark:
        pts = [[mark["nx"], mark["ny"]]]
    if not pts:
        return None
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


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


def union_bounds(
    shapes: Iterable[dict[str, Any]], marks: Iterable[dict[str, Any]] = ()
) -> list[float] | None:
    boxes = [b for b in (shape_bounds(s) for s in shapes) if b]
    boxes += [b for b in (mark_bounds(m) for m in marks) if b]
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
    marks: Iterable[dict[str, Any]] = (),
    min_area: float = 0.08,
) -> list[float] | None:
    """Where to look for features on the reference frame.

    ``board_box`` wins and only gets a 5% margin: features outside the object
    are the static background, and RANSAC would lock onto them when the object
    moves. Otherwise the extent of shapes and marks grown by half its size on
    every side, then until it covers ``min_area``. None means the whole frame.
    """
    if board_box:
        x1, y1, x2, y2 = (float(v) for v in board_box)
        gx, gy = (x2 - x1) * 0.05, (y2 - y1) * 0.05
        box = [x1 - gx, y1 - gy, x2 + gx, y2 + gy]
        min_area = 0.0
    else:
        box = union_bounds(shapes, marks)
        if box is None:
            return None
        w, h = (box[2] - box[0]) * 0.5, (box[3] - box[1]) * 0.5
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
