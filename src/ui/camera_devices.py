from __future__ import annotations

import sys

import cv2


def _capture_backend() -> int:
    if sys.platform == "win32":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def list_cameras(max_devices: int = 8) -> list[tuple[int, str]]:
    """Probe camera indices; returns (index, label) for devices that yield a frame."""
    backend = _capture_backend()
    devices: list[tuple[int, str]] = []
    for index in range(max_devices):
        cap = cv2.VideoCapture(index, backend)
        try:
            if not cap.isOpened():
                continue
            ok, _ = cap.read()
            if not ok:
                continue
            devices.append((index, f"摄像头 {index}"))
        finally:
            cap.release()
    return devices
