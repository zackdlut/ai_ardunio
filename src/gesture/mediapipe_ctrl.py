from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class GestureCommand(str, Enum):
    NONE = "none"
    NEXT = "next"
    PAUSE = "pause"
    RESCAN = "rescan"


@dataclass
class GestureResult:
    command: GestureCommand
    label: str = ""
    points: list[tuple[float, float]] | None = None


class GestureHandler:
    """MediaPipe hand landmarks -> simple gesture mapping."""

    def __init__(self) -> None:
        self._hands = None
        self._mp_hands = None
        self._mp_draw = None

    def _ensure(self) -> bool:
        if self._hands is not None:
            return True
        try:
            import mediapipe as mp

            self._mp_hands = mp.solutions.hands
            self._mp_draw = mp.solutions.drawing_utils
            self._hands = self._mp_hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5,
            )
            return True
        except Exception:  # noqa: BLE001
            return False

    def warmup(self) -> bool:
        return self._ensure()

    def process(self, frame_bgr: np.ndarray) -> GestureResult:
        if not self._ensure():
            return GestureResult(GestureCommand.NONE)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return GestureResult(GestureCommand.NONE)

        lm = result.multi_hand_landmarks[0].landmark
        cmd = self._classify(lm)
        points = [(float(p.x), float(p.y)) for p in lm]
        return GestureResult(command=cmd, label=cmd.value, points=points)

    def _classify(self, landmarks) -> GestureCommand:
        # Finger extended heuristic (tip above pip in image y)
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        extended = []
        for tip, pip in zip(tips, pips):
            extended.append(landmarks[tip].y < landmarks[pip].y)

        thumb_up = landmarks[4].y < landmarks[3].y
        fingers_up = sum(extended)

        if thumb_up and fingers_up == 0:
            return GestureCommand.NEXT
        if fingers_up == 0 and not thumb_up:
            return GestureCommand.PAUSE
        # V: index + middle only. An open palm must not retrigger a rescan.
        index, middle, ring, pinky = extended
        if index and middle and not ring and not pinky:
            return GestureCommand.RESCAN
        return GestureCommand.NONE
