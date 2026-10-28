"""Keep repair shapes stuck to the board: ORB features + RANSAC homography.

All homographies map 0..1 coordinates of the reference frame to 0..1
coordinates of the current frame, so they do not care about resolution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from repair.geometry import UNIT_SQUARE

_WORK_WIDTH = 640
_RATIO = 0.75
_REPROJ_PX = 4.0
_JUMP_RESET = 0.12


@dataclass
class TrackState:
    status: str = "idle"  # idle | tracking | weak | lost
    H: np.ndarray | None = None
    inliers: int = 0
    lost_for: float = 0.0


class PlaneTracker:
    def __init__(
        self,
        features: int = 1000,
        min_inliers: int = 18,
        lost_after_s: float = 0.7,
        smoothing: float = 0.55,
    ) -> None:
        self.features = max(200, int(features))
        self.min_inliers = max(6, int(min_inliers))
        self.lost_after_s = lost_after_s
        self.smoothing = min(max(smoothing, 0.05), 1.0)
        self._orb = cv2.ORB_create(nfeatures=self.features, fastThreshold=10)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._ref_pts: np.ndarray | None = None
        self._ref_des: np.ndarray | None = None
        self._ref_size = (1, 1)
        self._corners: np.ndarray | None = None
        self._last_ok = 0.0
        self.state = TrackState()

    @property
    def has_reference(self) -> bool:
        return self._ref_des is not None

    def clear(self) -> None:
        self._ref_pts = self._ref_des = None
        self._corners = None
        self.state = TrackState()

    def set_reference(
        self,
        frame: np.ndarray,
        roi: Sequence[float] | None = None,
        now: float | None = None,
        reset: bool = True,
    ) -> bool:
        """Use ``frame`` as the reference. Returns False if it has too little texture.

        ``reset=False`` only re-picks features (say, for a new ``roi``) on the
        same reference frame and keeps the current pose.
        """
        gray = _work_gray(frame)
        h, w = gray.shape[:2]
        mask = None
        if roi:
            x1, y1, x2, y2 = (float(v) for v in roi)
            mask = np.zeros_like(gray)
            mask[int(y1 * h) : max(int(y1 * h) + 1, int(y2 * h)), int(x1 * w) : max(int(x1 * w) + 1, int(x2 * w))] = 255
        kps, des = self._orb.detectAndCompute(gray, mask)
        if not reset and des is not None and len(kps) >= self.min_inliers:
            self._ref_size = (w, h)
            self._ref_pts = np.float32([kp.pt for kp in kps])
            self._ref_des = des
            return True
        self._corners = UNIT_SQUARE.astype(np.float64).copy()
        self._last_ok = time.monotonic() if now is None else now
        if des is None or len(kps) < self.min_inliers:
            self._ref_pts = self._ref_des = None
            self.state = TrackState(status="lost", H=np.eye(3), inliers=0)
            return False
        self._ref_size = (w, h)
        self._ref_pts = np.float32([kp.pt for kp in kps])
        self._ref_des = des
        self.state = TrackState(status="tracking", H=np.eye(3), inliers=len(kps))
        return True

    def update(self, frame: np.ndarray, now: float | None = None) -> TrackState:
        now = time.monotonic() if now is None else now
        if self._ref_des is None:
            if self.state.status != "idle":
                self.state = TrackState(
                    status="lost", H=self.state.H, lost_for=now - self._last_ok
                )
            return self.state
        measured = self._measure(frame)
        if measured is None:
            lost_for = now - self._last_ok
            status = "lost" if lost_for > self.lost_after_s else "weak"
            self.state = TrackState(status=status, H=self.state.H, inliers=0, lost_for=lost_for)
            return self.state
        corners, inliers = measured
        if self._corners is None or float(np.abs(corners - self._corners).max()) > _JUMP_RESET:
            self._corners = corners
        else:
            a = self.smoothing
            self._corners = a * corners + (1 - a) * self._corners
        self._last_ok = now
        H = cv2.getPerspectiveTransform(
            UNIT_SQUARE, self._corners.astype(np.float32)
        ).astype(np.float64)
        status = "tracking" if inliers >= self.min_inliers * 2 else "weak"
        self.state = TrackState(status=status, H=H, inliers=inliers)
        return self.state

    def _measure(self, frame: np.ndarray) -> tuple[np.ndarray, int] | None:
        gray = _work_gray(frame)
        h, w = gray.shape[:2]
        kps, des = self._orb.detectAndCompute(gray, None)
        if des is None or len(kps) < self.min_inliers:
            return None
        pairs = self._matcher.knnMatch(self._ref_des, des, k=2)
        good = [m[0] for m in pairs if len(m) == 2 and m[0].distance < _RATIO * m[1].distance]
        if len(good) < self.min_inliers:
            return None
        src = np.float32([self._ref_pts[m.queryIdx] for m in good])
        dst = np.float32([kps[m.trainIdx].pt for m in good])
        H_px, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, _REPROJ_PX)
        if H_px is None or inlier_mask is None:
            return None
        inliers = int(inlier_mask.sum())
        if inliers < self.min_inliers:
            return None
        rw, rh = self._ref_size
        S_ref = np.diag([float(rw), float(rh), 1.0])
        S_cur_inv = np.diag([1.0 / w, 1.0 / h, 1.0])
        H = S_cur_inv @ H_px @ S_ref
        corners = _project(H, UNIT_SQUARE)
        if corners is None or not _plausible(corners):
            return None
        return corners, inliers


def _work_gray(frame: np.ndarray) -> np.ndarray:
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if width > _WORK_WIDTH:
        scale = _WORK_WIDTH / width
        gray = cv2.resize(gray, (_WORK_WIDTH, max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)
    return gray


def _project(H: np.ndarray, pts: np.ndarray) -> np.ndarray | None:
    homo = np.hstack([pts.astype(np.float64), np.ones((len(pts), 1))]) @ H.T
    if np.any(homo[:, 2] <= 1e-6):
        return None
    return homo[:, :2] / homo[:, 2:3]


def _plausible(corners: np.ndarray) -> bool:
    """Reject flipped, folded, exploded or collapsed boards."""
    if not cv2.isContourConvex(corners.astype(np.float32).reshape(-1, 1, 2)):
        return False
    area = abs(cv2.contourArea(corners.astype(np.float32)))
    if not 0.03 <= area <= 30.0:
        return False
    a, b, _c, d = corners
    cross = (b[0] - a[0]) * (d[1] - a[1]) - (b[1] - a[1]) * (d[0] - a[0])
    return cross > 0
