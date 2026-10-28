from __future__ import annotations

import math
from typing import Any, Tuple

import cv2
import numpy as np

from overlay.text_sprite import TextSpriteCache, blit_bgra
from vision.component_db import display_name
from vision.detector import Detection

_ACCENT = (106, 168, 72)  # BGR, close to the app green
_INK = (244, 250, 246)
_AMBER = (70, 180, 255)
_BOX_COLORS = {
    "LED": (90, 200, 120),
    "Resistor": (60, 160, 255),
    "JumperWire": (80, 220, 240),
    "ArduinoUNO": (220, 160, 70),
    "Breadboard": (200, 210, 200),
    "Button": (180, 120, 255),
}
_HAND_LINKS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)
_RED = (60, 64, 235)
_ANNOTATION_COLORS = {
    "check": _AMBER,
    "fault": _RED,
    "ok": _ACCENT,
    "danger": (40, 40, 220),
}
_PIN_ZONES = {
    "D13": (0.72, 0.15, 0.82, 0.28),
    "D12": (0.72, 0.28, 0.82, 0.40),
    "D11": (0.72, 0.40, 0.82, 0.52),
    "GND": (0.05, 0.55, 0.15, 0.68),
    "5V": (0.05, 0.18, 0.15, 0.32),
    "3V3": (0.05, 0.06, 0.15, 0.16),
}


class OverlayEngine:
    def __init__(self) -> None:
        self._text = TextSpriteCache()

    def show_bbox(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        out = frame.copy()
        self._draw_detections(out, detections, fallback=color)
        return out

    def draw_arrow(
        self,
        frame: np.ndarray,
        start: Tuple[int, int],
        end: Tuple[int, int],
        label: str = "",
        color: Tuple[int, int, int] = (0, 200, 255),
    ) -> np.ndarray:
        out = frame.copy()
        self._arrow(out, start, end, label, color)
        return out

    def highlight_target(
        self,
        frame: np.ndarray,
        bbox: list[int] | None,
        alpha: float = 0.35,
    ) -> np.ndarray:
        if not bbox:
            return frame
        out = frame.copy()
        self._tint_box(out, bbox, _AMBER, alpha)
        return out

    def uno_pin_zone(self, uno_bbox: list[int], pin: str) -> list[int]:
        """Approximate pin pocket inside a detected UNO box."""
        x1, y1, x2, y2 = uno_bbox
        w, h = x2 - x1, y2 - y1
        rel = _PIN_ZONES.get(pin.upper())
        if rel is None:
            return uno_bbox
        rx1, ry1, rx2, ry2 = rel
        return [
            int(x1 + rx1 * w),
            int(y1 + ry1 * h),
            int(x1 + rx2 * w),
            int(y1 + ry2 * h),
        ]

    def render_step_guide(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        hint: str,
        target_types: list[str] | None = None,
        pin: str | None = None,
    ) -> np.ndarray:
        return self.render(
            frame,
            detections,
            hint={"hint": hint, "requires": target_types or [], "pin": pin},
        )

    def render(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        hint: dict[str, Any] | None = None,
        marks: list[dict[str, Any]] | None = None,
        paused: bool = False,
        pulse: float = 0.0,
        hand: list[tuple[float, float]] | None = None,
        mode: str = "tutor",
        annotations: list[dict[str, Any]] | None = None,
        frozen: bool = False,
    ) -> np.ndarray:
        out = frame.copy()
        hint = hint or {}
        repair = mode == "repair"
        if repair:
            self._draw_annotations(out, annotations or [], hint.get("check_id"), pulse)
        else:
            self._draw_detections(out, detections)
            self._draw_guide(out, detections, hint, pulse)
        self._draw_hand(out, hand)
        self._draw_marks(out, marks or [], pulse)
        self._draw_hud(out, hint, paused, bool(marks), repair=repair, frozen=frozen)
        return out

    def _draw_annotations(
        self,
        frame: np.ndarray,
        annotations: list[dict[str, Any]],
        focus_check: Any,
        pulse: float,
    ) -> None:
        height, width = frame.shape[:2]
        wave = 0.5 + 0.5 * math.sin(pulse * math.tau)
        for item in annotations:
            try:
                bx1, by1, bx2, by2 = (float(v) for v in item.get("box") or ())
            except (TypeError, ValueError):
                continue
            x1, y1 = int(bx1 * (width - 1)), int(by1 * (height - 1))
            x2, y2 = int(bx2 * (width - 1)), int(by2 * (height - 1))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            kind = str(item.get("kind") or "check")
            color = _ANNOTATION_COLORS.get(kind, _AMBER)
            focused = focus_check is not None and item.get("check_id") == focus_check
            thickness = 3 if focused or kind == "fault" else 2
            if kind == "danger":
                _hatch(frame, x1, y1, x2, y2, color, 0.35)
            elif kind == "fault" or focused:
                self._tint_box(frame, [x1, y1, x2, y2], color, 0.12 + 0.18 * wave)
            if kind == "fault":
                pad = int(2 + 4 * wave)
                cv2.rectangle(
                    frame, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), color, 1, cv2.LINE_AA
                )
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
            label = str(item.get("label") or "")
            note = str(item.get("note") or "")
            if note and (focused or kind in ("fault", "danger")):
                label = f"{label} · {note}" if label else note
            if label:
                ly = y1 - 26 if y1 >= 30 else min(y2 + 4, height - 24)
                self._label(frame, label, x1, max(4, ly), color, max_width=max(120, width // 2))

    def _draw_detections(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        fallback: tuple[int, int, int] = _ACCENT,
    ) -> None:
        for det in detections:
            color = _BOX_COLORS.get(det.type, fallback)
            x1, y1, x2, y2 = _clip_box(det.bbox, frame.shape[1], frame.shape[0])
            if x2 <= x1 or y2 <= y1:
                continue
            _corners(frame, x1, y1, x2, y2, color)
            label = f"{display_name(det.type)} {det.confidence:.0%}"
            self._label(frame, label, x1, max(4, y1 - 28), color)

    def _draw_guide(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        hint: dict[str, Any],
        pulse: float,
    ) -> None:
        target_types = list(hint.get("requires") or [])
        pin = hint.get("pin") or None
        targets = [d for d in detections if d.type in target_types]
        alpha = 0.22 + 0.16 * (0.5 + 0.5 * math.sin(pulse * math.tau))

        if pin and any(d.type == "ArduinoUNO" for d in detections):
            uno = next(d for d in detections if d.type == "ArduinoUNO")
            zone = self.uno_pin_zone(uno.bbox, str(pin))
            self._tint_box(frame, zone, _AMBER, alpha)
            cx, cy = (zone[0] + zone[2]) // 2, (zone[1] + zone[3]) // 2
            self._label(frame, str(pin), zone[0], max(4, zone[1] - 26), _AMBER)
            if targets:
                tx, ty = _bbox_center(targets[0].bbox)
                self._arrow(frame, (tx, ty), (cx, cy), "", _AMBER)
            else:
                self._arrow(frame, (cx, cy - 72), (cx, cy), "", _AMBER)
            return

        if targets:
            box = targets[0].bbox
            self._tint_box(frame, box, _ACCENT, alpha)
            x1, y1, x2, _y2 = box
            self._arrow(frame, ((x1 + x2) // 2, y1 - 56), ((x1 + x2) // 2, y1), "", _ACCENT)

    def _draw_hand(self, frame: np.ndarray, hand: list[tuple[float, float]] | None) -> None:
        if not hand or len(hand) < 21:
            return
        height, width = frame.shape[:2]
        pts = []
        for nx, ny in hand[:21]:
            x = min(max(int(round(float(nx) * (width - 1))), 0), width - 1)
            y = min(max(int(round(float(ny) * (height - 1))), 0), height - 1)
            pts.append((x, y))
        for a, b in _HAND_LINKS:
            cv2.line(frame, pts[a], pts[b], _ACCENT, 2, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(frame, pt, 3, _INK, -1, cv2.LINE_AA)

    def _draw_marks(self, frame: np.ndarray, marks: list[dict[str, Any]], pulse: float) -> None:
        height, width = frame.shape[:2]
        ring = 11 + int(3 * (0.5 + 0.5 * math.sin(pulse * math.tau)))
        for mark in marks:
            if mark.get("kind") in ("box", "lasso") and self._draw_region_mark(frame, mark):
                continue
            try:
                nx = float(mark.get("nx", 0))
                ny = float(mark.get("ny", 0))
            except (TypeError, ValueError):
                continue
            x = int(round(nx * (width - 1)))
            y = int(round(ny * (height - 1)))
            x = min(max(x, 0), width - 1)
            y = min(max(y, 0), height - 1)
            cv2.circle(frame, (x, y), ring, _AMBER, 2, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 4, _INK, -1, cv2.LINE_AA)
            cv2.line(frame, (x - 14, y), (x - 6, y), _INK, 1, cv2.LINE_AA)
            cv2.line(frame, (x + 6, y), (x + 14, y), _INK, 1, cv2.LINE_AA)
            label = str(mark.get("label") or "标记")
            self._label(frame, label, x + 16, y - 14, _INK)

    def _draw_region_mark(self, frame: np.ndarray, mark: dict[str, Any]) -> bool:
        height, width = frame.shape[:2]

        def to_px(nx: Any, ny: Any) -> tuple[int, int]:
            x = int(round(min(max(float(nx), 0.0), 1.0) * (width - 1)))
            y = int(round(min(max(float(ny), 0.0), 1.0) * (height - 1)))
            return x, y

        try:
            if mark.get("kind") == "lasso":
                pts = [to_px(nx, ny) for nx, ny in mark.get("points") or ()]
                if len(pts) < 3:
                    return False
                poly = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [poly], True, _AMBER, 2, cv2.LINE_AA)
                x1 = min(p[0] for p in pts)
                y1 = min(p[1] for p in pts)
            else:
                bx1, by1, bx2, by2 = mark.get("box") or ()
                x1, y1 = to_px(bx1, by1)
                x2, y2 = to_px(bx2, by2)
                if x2 <= x1 or y2 <= y1:
                    return False
                cv2.rectangle(frame, (x1, y1), (x2, y2), _AMBER, 2, cv2.LINE_AA)
        except (TypeError, ValueError):
            return False
        label = str(mark.get("label") or "标记")
        self._label(frame, label, x1, max(4, y1 - 26), _AMBER)
        return True

    def _draw_hud(
        self,
        frame: np.ndarray,
        hint: dict[str, Any],
        paused: bool,
        has_marks: bool,
        repair: bool = False,
        frozen: bool = False,
    ) -> None:
        height, width = frame.shape[:2]
        if height < 48 or width < 80:
            return
        bar_h = 44 if height >= 180 else 32
        _fill_bar(frame, 0, bar_h, (22, 32, 28), 0.62)
        if paused and not repair:
            headline = "识别已暂停 · 竖起拇指，或在视图菜单里继续"
        else:
            step = str(hint.get("step_label") or "").strip()
            fallback = (
                "说说设备型号和故障现象，我来列检查清单"
                if repair
                else "把电路放进画面，我会标出这一步"
            )
            guide = str(hint.get("hint") or fallback).strip()
            headline = f"{step}  ·  {guide}" if step else guide
        badge_w = 0
        if frozen:
            badge = "已定格"
            badge_w = 84
            self._label(frame, badge, width - badge_w, 10, _AMBER, px=18)
        self._label(frame, headline, 12, 8, _INK, px=20, max_width=width - 24 - badge_w)

        status = str(hint.get("status_text") or "").strip()
        if status and not paused and hint.get("status") == "missing" and height >= 180:
            self._label(frame, status, 12, bar_h + 6, _AMBER, px=16, max_width=width - 24)

        if not has_marks and height >= 220:
            footer = (
                "单击、框选或圈出可疑位置，然后问「这里是什么、该怎么测」"
                if repair
                else "单击、拖出矩形或画圈标记，然后直接问「这里该怎么接」"
            )
            foot_h = 28
            _fill_bar(frame, height - foot_h, height, (22, 32, 28), 0.5)
            self._label(frame, footer, 12, height - foot_h + 2, (210, 230, 216), px=15, max_width=width - 24)

    def _label(
        self,
        frame: np.ndarray,
        text: str,
        x: int,
        y: int,
        bgr: tuple[int, int, int],
        px: int = 18,
        max_width: int = 0,
    ) -> None:
        sprite = self._text.get(text, bgr, px=px, max_width=max_width)
        if sprite is None:
            cv2.putText(
                frame,
                text,
                (x, y + px),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                bgr,
                1,
                cv2.LINE_AA,
            )
            return
        blit_bgra(frame, sprite, x, y)

    def _arrow(
        self,
        frame: np.ndarray,
        start: tuple[int, int],
        end: tuple[int, int],
        label: str,
        color: tuple[int, int, int],
    ) -> None:
        cv2.arrowedLine(frame, start, end, color, 3, cv2.LINE_AA, tipLength=0.22)
        if label:
            self._label(frame, label, end[0] + 8, end[1] - 12, color, px=18, max_width=frame.shape[1] // 2)

    def _tint_box(
        self,
        frame: np.ndarray,
        bbox: list[int],
        color: tuple[int, int, int],
        alpha: float,
    ) -> None:
        x1, y1, x2, y2 = _clip_box(bbox, frame.shape[1], frame.shape[0])
        if x2 - x1 < 2 or y2 - y1 < 2:
            return
        roi = frame[y1:y2, x1:x2]
        tint = np.empty_like(roi)
        tint[:] = color
        cv2.addWeighted(tint, alpha, roi, 1 - alpha, 0, roi)
        _corners(frame, x1, y1, x2, y2, color, length=max(12, (x2 - x1) // 5), thickness=3)


def _bbox_center(bbox: list[int]) -> tuple[int, int]:
    return (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2


def _clip_box(bbox: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1 = min(max(x1, 0), width - 1)
    y1 = min(max(y1, 0), height - 1)
    x2 = min(max(x2, 0), width - 1)
    y2 = min(max(y2, 0), height - 1)
    return x1, y1, x2, y2


def _corners(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    length: int = 16,
    thickness: int = 2,
) -> None:
    length = max(6, min(length, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness, cv2.LINE_AA)


def _hatch(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    x1, y1, x2, y2 = _clip_box([x1, y1, x2, y2], frame.shape[1], frame.shape[0])
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = frame[y1:y2, x1:x2]
    lines = roi.copy()
    h, w = roi.shape[:2]
    for offset in range(-h, w, 14):
        cv2.line(lines, (offset, h), (offset + h, 0), color, 2, cv2.LINE_AA)
    cv2.addWeighted(lines, alpha, roi, 1 - alpha, 0, roi)


def _fill_bar(
    frame: np.ndarray,
    y0: int,
    y1: int,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    height = frame.shape[0]
    y0 = min(max(y0, 0), height)
    y1 = min(max(y1, 0), height)
    if y1 <= y0:
        return
    roi = frame[y0:y1]
    tint = np.empty_like(roi)
    tint[:] = color
    cv2.addWeighted(tint, alpha, roi, 1 - alpha, 0, roi)
