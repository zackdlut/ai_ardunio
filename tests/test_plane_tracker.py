"""The plane tracker finds how the board moved since the reference frame."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from repair.geometry import warp_points  # noqa: E402
from vision.plane_tracker import PlaneTracker  # noqa: E402


def _board(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((480, 640, 3), 40, np.uint8)
    for _ in range(160):
        x, y = int(rng.integers(20, 620)), int(rng.integers(20, 460))
        w, h = int(rng.integers(8, 50)), int(rng.integers(8, 50))
        color = tuple(int(c) for c in rng.integers(60, 255, 3))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        else:
            cv2.circle(img, (x, y), w // 2, color, 2)
    for _ in range(40):
        x, y = int(rng.integers(20, 600)), int(rng.integers(20, 450))
        cv2.putText(img, f"U{int(rng.integers(1, 99))}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    return img


def _warp(img: np.ndarray, angle: float, scale: float, tx: float, ty: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    out = cv2.warpAffine(img, M, (w, h), borderValue=(40, 40, 40))
    H_px = np.vstack([M, [0, 0, 1]])
    S = np.diag([w, h, 1.0])
    return out, np.linalg.inv(S) @ H_px @ S


def test_tracker_recovers_known_motion():
    ref = _board()
    tracker = PlaneTracker(features=1500, smoothing=1.0)
    assert tracker.set_reference(ref, now=0.0)
    probes = [[0.3, 0.3], [0.7, 0.4], [0.5, 0.8]]
    for i, (angle, scale, tx, ty) in enumerate([(0, 1.0, 40, 20), (12, 1.0, 0, 0), (-8, 0.85, -30, 15)]):
        cur, truth = _warp(ref, angle, scale, tx, ty)
        state = tracker.update(cur, now=0.1 * (i + 1))
        assert state.status in ("tracking", "weak")
        got = np.array(warp_points(state.H, probes))
        want = np.array(warp_points(truth, probes))
        assert np.abs(got - want).max() < 0.02


def test_tracker_goes_lost_on_unrelated_frames():
    tracker = PlaneTracker(lost_after_s=0.5)
    assert tracker.set_reference(_board(), now=0.0)
    noise = np.random.default_rng(9).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    assert tracker.update(noise, now=0.2).status == "weak"
    state = tracker.update(noise, now=0.8)
    assert state.status == "lost"
    assert state.H is not None


def test_flat_reference_is_rejected():
    tracker = PlaneTracker()
    assert not tracker.set_reference(np.full((480, 640, 3), 128, np.uint8))
    assert tracker.state.status == "lost"
