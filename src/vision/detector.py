from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from config.settings import get_settings
from vision.component_db import normalize_class_name


def _configure_inference_runtime() -> None:
    """Keep OpenMP from pinning every core and spinning while idle."""
    threads = os.environ.get("OMP_NUM_THREADS")
    if not threads:
        count = os.cpu_count() or 2
        threads = str(max(1, min(4, count - 2)))
        os.environ["OMP_NUM_THREADS"] = threads
    os.environ.setdefault("MKL_NUM_THREADS", threads)
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")


def _apply_torch_thread_cap() -> None:
    try:
        import torch
    except ImportError:
        return
    try:
        torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    except (RuntimeError, ValueError):
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


@dataclass
class Detection:
    type: str
    bbox: list[int]
    confidence: float
    track_id: int | None = None

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "track_id": self.track_id,
        }


class YOLODetector:
    def __init__(self, model_path: Path | None = None) -> None:
        settings = get_settings()
        self._path = Path(model_path or settings.yolo_model_path)
        self._model = None
        self._enabled = False
        self._load_error: str | None = None
        self._load_attempted = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def load(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        self._try_load()

    def _try_load(self) -> None:
        _configure_inference_runtime()
        try:
            from ultralytics import YOLO
            from ultralytics.utils import events as yolo_events
        except ImportError:
            self._load_error = "ultralytics not installed"
            return

        # Telemetry starts a thread from the predict callback and can stall the
        # camera thread on Thread.start() while that post holds the GIL.
        yolo_events.events.enabled = False
        _apply_torch_thread_cap()

        if self._path.is_file():
            self._model = YOLO(str(self._path))
            self._enabled = True
            return

        # Fallback: tiny COCO model so pipeline runs without custom weights
        try:
            self._model = YOLO("yolov8n.pt")
            self._enabled = True
            self._load_error = (
                f"Custom model not found at {self._path}; using yolov8n (COCO) placeholder"
            )
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)

    def detect(self, frame_bgr: np.ndarray, conf: float = 0.35) -> List[Detection]:
        if not self._enabled or self._model is None:
            return []

        results = self._model.predict(frame_bgr, conf=conf, verbose=False)
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            names = result.names or {}
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                cls_id = int(box.cls[0])
                raw_name = names.get(cls_id, str(cls_id))
                detections.append(
                    Detection(
                        type=normalize_class_name(raw_name),
                        bbox=xyxy,
                        confidence=float(box.conf[0]),
                    )
                )
        return detections


def scale_detection_boxes(detections: List[Detection], scale_x: float, scale_y: float) -> None:
    """Map boxes from a resized inference frame back to the camera frame."""
    if scale_x == 1 and scale_y == 1:
        return
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        det.bbox = [
            int(round(x1 * scale_x)),
            int(round(y1 * scale_y)),
            int(round(x2 * scale_x)),
            int(round(y2 * scale_y)),
        ]
