from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from vision.detector import Detection


class OverlayEngine:
    def show_bbox(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{det.type} {det.confidence:.2f}"
            cv2.putText(
                out,
                label,
                (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
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
        cv2.arrowedLine(out, start, end, color, 3, tipLength=0.2)
        if label:
            cv2.putText(
                out,
                label,
                (end[0] + 4, end[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
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
        overlay = out.copy()
        x1, y1, x2, y2 = bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), -1)
        cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 3)
        return out

    def uno_pin_zone(self, uno_bbox: list[int], pin: str) -> list[int]:
        """Fixed relative zones inside UNO detection (MVP)."""
        x1, y1, x2, y2 = uno_bbox
        w, h = x2 - x1, y2 - y1
        if pin.upper() == "D13":
            return [int(x1 + 0.72 * w), int(y1 + 0.15 * h), int(x1 + 0.82 * w), int(y1 + 0.28 * h)]
        if pin.upper() == "GND":
            return [int(x1 + 0.05 * w), int(y1 + 0.55 * h), int(x1 + 0.15 * w), int(y1 + 0.68 * h)]
        return uno_bbox

    def render_step_guide(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        hint: str,
        target_types: list[str] | None = None,
        pin: str | None = None,
    ) -> np.ndarray:
        out = self.show_bbox(frame, detections)
        target_types = target_types or []
        targets = [d for d in detections if d.type in target_types]

        if pin and any(d.type == "ArduinoUNO" for d in detections):
            uno = next(d for d in detections if d.type == "ArduinoUNO")
            zone = self.uno_pin_zone(uno.bbox, pin)
            out = self.highlight_target(out, zone)
            cx, cy = (zone[0] + zone[2]) // 2, (zone[1] + zone[3]) // 2
            if targets:
                tx, ty = _bbox_center(targets[0].bbox)
                out = self.draw_arrow(out, (tx, ty), (cx, cy), hint or f"Connect to {pin}")
            else:
                out = self.draw_arrow(
                    out,
                    (cx, cy - 80),
                    (cx, cy),
                    hint or f"Connect to {pin}",
                )
        elif targets:
            out = self.highlight_target(out, targets[0].bbox)
            x1, y1, x2, y2 = targets[0].bbox
            out = self.draw_arrow(
                out,
                ((x1 + x2) // 2, y1 - 40),
                ((x1 + x2) // 2, y1),
                hint,
            )
        elif hint:
            cv2.putText(
                out,
                hint,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )
        return out


def _bbox_center(bbox: list[int]) -> tuple[int, int]:
    return (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
