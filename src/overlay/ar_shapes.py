"""Draw repair shapes (rect, circle, arrow, line, text, probe) through a homography."""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from repair.annotations import as_shape
from repair.geometry import circle_polygon, warp_points

Color = tuple[int, int, int]
LabelFn = Callable[..., None]

INK: Color = (244, 250, 246)
AMBER: Color = (70, 180, 255)
GREEN: Color = (106, 168, 72)
RED: Color = (60, 64, 235)
KIND_COLORS: dict[str, Color] = {
    "check": AMBER,
    "fault": RED,
    "ok": GREEN,
    "danger": (40, 40, 220),
}
PROBE_RED: Color = (40, 40, 230)
PROBE_BLACK: Color = (34, 34, 34)
_DIM: Color = (200, 200, 200)
BACKING: Color = (22, 32, 28)


def draw_shapes(
    frame: np.ndarray,
    shapes: Sequence[dict[str, Any]],
    label: LabelFn,
    H: np.ndarray | None = None,
    pulse: float = 0.0,
    focus_check: Any = None,
    alpha: float = 1.0,
) -> None:
    """Draw ``shapes`` (reference 0..1 coordinates) onto ``frame`` in place.

    ``label(frame, text, x, y, bgr, px=..., max_width=..., bg=...)`` renders text.
    ``alpha`` < 1 fades the whole layer (used while tracking is lost).
    """
    if not shapes or alpha <= 0.02:
        return
    target = frame if alpha >= 0.99 else frame.copy()
    height, width = frame.shape[:2]
    wave = 0.5 + 0.5 * math.sin(pulse * math.tau)

    def px(points: Sequence[Sequence[float]], warp: bool = True) -> list[tuple[int, int]]:
        out = []
        for x, y in warp_points(H if warp else None, points):
            x = min(max(x, -4.0), 5.0)
            y = min(max(y, -4.0), 5.0)
            out.append((int(round(x * (width - 1))), int(round(y * (height - 1)))))
        return out

    for raw in shapes:
        shape = as_shape(raw)
        pts = shape.get("pts") or []
        if not pts:
            continue
        kind = str(shape.get("kind") or "check")
        color = KIND_COLORS.get(kind, AMBER)
        focused = focus_check is not None and shape.get("check_id") == focus_check
        kind_type = shape.get("type") or "rect"
        try:
            if kind_type in ("rect", "circle") and len(pts) == 4:
                if kind_type == "rect":
                    poly = px(pts)
                else:
                    poly = px(circle_polygon(warp_points(H, pts)), warp=False)
                _region(target, poly, shape, color, kind, focused, wave, label, rect=kind_type == "rect")
            elif kind_type == "arrow" and len(pts) >= 2:
                a, b = px(pts[:2])
                _arrow(target, a, b, color, pulse)
                _caption(target, shape, a, color, label, offset=(-8, -30) if a[1] > 40 else (8, 8))
            elif kind_type == "line" and len(pts) >= 2:
                a, b = px(pts[:2])
                _dashed(target, a, b, color, 2, pulse)
                for p in (a, b):
                    cv2.circle(target, p, 6, color, -1, cv2.LINE_AA)
                    cv2.circle(target, p, 6, INK, 1, cv2.LINE_AA)
                mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
                _caption(target, shape, mid, color, label, offset=(8, -28))
            elif kind_type == "probe":
                _probe(target, px(pts), shape, wave, pulse, label)
            elif kind_type == "text":
                (p,) = px(pts[:1])
                cv2.circle(target, p, 4, color, -1, cv2.LINE_AA)
                text = str(shape.get("text") or shape.get("label") or "")
                label(target, text, p[0] + 10, p[1] - 12, color, px=18, max_width=max(160, width // 2), bg=BACKING)
            else:  # point
                (p,) = px(pts[:1])
                ring = 10 + int(5 * wave)
                cv2.circle(target, p, ring, color, 2, cv2.LINE_AA)
                cv2.circle(target, p, 4, color, -1, cv2.LINE_AA)
                _caption(target, shape, p, color, label, offset=(14, -12))
        except (TypeError, ValueError, cv2.error):
            continue

    if target is not frame:
        cv2.addWeighted(target, alpha, frame, 1.0 - alpha, 0, frame)


def _region(
    frame: np.ndarray,
    poly: list[tuple[int, int]],
    shape: dict[str, Any],
    color: Color,
    kind: str,
    focused: bool,
    wave: float,
    label: LabelFn,
    rect: bool,
) -> None:
    arr = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    if max(xs) - min(xs) < 2 or max(ys) - min(ys) < 2:
        return
    if kind == "danger":
        _fill_poly(frame, arr, color, 0.35, hatch=True)
    elif kind == "fault" or focused:
        _fill_poly(frame, arr, color, 0.12 + 0.18 * wave)
    if kind == "fault":
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        grow = 1.0 + (0.03 + 0.05 * wave)
        outer = np.array(
            [(int(cx + (x - cx) * grow), int(cy + (y - cy) * grow)) for x, y in poly], dtype=np.int32
        ).reshape(-1, 1, 2)
        cv2.polylines(frame, [outer], True, color, 1, cv2.LINE_AA)
    thickness = 3 if focused or kind == "fault" else 2
    cv2.polylines(frame, [arr], True, color, thickness, cv2.LINE_AA)
    if rect:
        _quad_corners(frame, poly, color)
    text = str(shape.get("label") or "")
    note = str(shape.get("note") or "")
    if note and (focused or kind in ("fault", "danger")):
        text = f"{text} · {note}" if text else note
    if text:
        top = min(ys)
        left = min(xs)
        height = frame.shape[0]
        ly = top - 26 if top >= 30 else min(max(ys) + 4, height - 24)
        label(frame, text, max(0, left), max(4, ly), color, px=18, max_width=max(120, frame.shape[1] // 2), bg=BACKING)


def _quad_corners(frame: np.ndarray, poly: list[tuple[int, int]], color: Color) -> None:
    n = len(poly)
    for i, p in enumerate(poly):
        for q in (poly[(i + 1) % n], poly[(i - 1) % n]):
            dx, dy = q[0] - p[0], q[1] - p[1]
            length = math.hypot(dx, dy)
            if length < 1:
                continue
            k = min(18.0, length / 3) / length
            cv2.line(frame, p, (int(p[0] + dx * k), int(p[1] + dy * k)), color, 4, cv2.LINE_AA)


def _fill_poly(frame: np.ndarray, poly: np.ndarray, color: Color, alpha: float, hatch: bool = False) -> None:
    x, y, w, h = cv2.boundingRect(poly)
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    roi = frame[y0:y1, x0:x1]
    mask = np.zeros(roi.shape[:2], np.uint8)
    cv2.fillPoly(mask, [poly - np.array([[x0, y0]], dtype=np.int32)], 255)
    layer = roi.copy()
    if hatch:
        rh, rw = roi.shape[:2]
        for offset in range(-rh, rw, 14):
            cv2.line(layer, (offset, rh), (offset + rh, 0), color, 2, cv2.LINE_AA)
    else:
        layer[:] = color
    blended = cv2.addWeighted(layer, alpha, roi, 1 - alpha, 0)
    roi[mask > 0] = blended[mask > 0]


def _arrow(frame: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: Color, pulse: float) -> None:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < 4:
        return
    ux, uy = dx / length, dy / length
    head = min(26.0, max(12.0, length * 0.28))
    base = (int(b[0] - ux * head), int(b[1] - uy * head))
    cv2.line(frame, a, base, (20, 20, 20), 8, cv2.LINE_AA)
    cv2.line(frame, a, base, color, 5, cv2.LINE_AA)
    nx, ny = -uy, ux
    tri = np.array(
        [
            b,
            (int(base[0] + nx * head * 0.6), int(base[1] + ny * head * 0.6)),
            (int(base[0] - nx * head * 0.6), int(base[1] - ny * head * 0.6)),
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(frame, tri, color, cv2.LINE_AA)
    cv2.polylines(frame, [tri.reshape(-1, 1, 2)], True, (20, 20, 20), 1, cv2.LINE_AA)
    # Chevrons flowing toward the head show the direction of the action.
    spacing = 26.0
    run = length - head
    offset = (pulse * 52.0) % spacing
    s = offset
    while s < run - 6:
        cx, cy = a[0] + ux * s, a[1] + uy * s
        wing = 5.0
        tip = (int(cx + ux * 4), int(cy + uy * 4))
        left = (int(cx - ux * 3 + nx * wing), int(cy - uy * 3 + ny * wing))
        right = (int(cx - ux * 3 - nx * wing), int(cy - uy * 3 - ny * wing))
        cv2.line(frame, left, tip, INK, 2, cv2.LINE_AA)
        cv2.line(frame, right, tip, INK, 2, cv2.LINE_AA)
        s += spacing
    cv2.circle(frame, a, 5, color, -1, cv2.LINE_AA)


def _dashed(
    frame: np.ndarray,
    a: tuple[int, int],
    b: tuple[int, int],
    color: Color,
    thickness: int,
    pulse: float,
    dash: float = 12.0,
    gap: float = 8.0,
) -> None:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < 2:
        return
    ux, uy = dx / length, dy / length
    period = dash + gap
    s = -((pulse * 40.0) % period)
    while s < length:
        s0, s1 = max(0.0, s), min(length, s + dash)
        if s1 > s0:
            p0 = (int(a[0] + ux * s0), int(a[1] + uy * s0))
            p1 = (int(a[0] + ux * s1), int(a[1] + uy * s1))
            cv2.line(frame, p0, p1, color, thickness, cv2.LINE_AA)
        s += period


def _probe(
    frame: np.ndarray,
    pts: list[tuple[int, int]],
    shape: dict[str, Any],
    wave: float,
    pulse: float,
    label: LabelFn,
) -> None:
    red = pts[0]
    black = pts[1] if len(pts) > 1 else None
    if black is not None:
        _dashed(frame, red, black, _DIM, 2, pulse)
    ring = 13 + int(4 * wave)
    for point, fill, tag in ((red, PROBE_RED, "红+"), (black, PROBE_BLACK, "黑-")):
        if point is None:
            continue
        cv2.circle(frame, point, ring, fill if fill != PROBE_BLACK else INK, 2, cv2.LINE_AA)
        cv2.circle(frame, point, 8, fill, -1, cv2.LINE_AA)
        cv2.circle(frame, point, 8, INK, 1, cv2.LINE_AA)
        label(frame, tag, point[0] + ring + 2, point[1] - 11, PROBE_RED if fill == PROBE_RED else INK, px=16, bg=BACKING)
    if black is not None:
        anchor = ((red[0] + black[0]) // 2, (red[1] + black[1]) // 2)
    else:
        anchor = (red[0], red[1] + ring + 6)
    lines = []
    if shape.get("label"):
        lines.append(str(shape["label"]))
    meter = str(shape.get("meter") or "")
    expected = str(shape.get("expected") or "")
    if meter or expected:
        parts = []
        if meter:
            parts.append(f"档位 {meter}")
        if expected:
            parts.append(f"预期 {expected}")
        lines.append(" / ".join(parts))
    if shape.get("note"):
        lines.append(str(shape["note"]))
    y = anchor[1] + 10
    for text in lines:
        label(frame, text, max(0, anchor[0] - 40), y, AMBER, px=16, max_width=max(160, frame.shape[1] // 2), bg=BACKING)
        y += 22


def _caption(
    frame: np.ndarray,
    shape: dict[str, Any],
    at: tuple[int, int],
    color: Color,
    label: LabelFn,
    offset: tuple[int, int] = (8, -28),
) -> None:
    text = str(shape.get("label") or "")
    note = str(shape.get("note") or "")
    if note:
        text = f"{text} · {note}" if text else note
    if not text:
        return
    h, w = frame.shape[:2]
    x = min(max(0, at[0] + offset[0]), max(0, w - 80))
    y = min(max(4, at[1] + offset[1]), max(4, h - 24))
    label(frame, text, x, y, color, px=18, max_width=max(140, w // 2), bg=BACKING)
