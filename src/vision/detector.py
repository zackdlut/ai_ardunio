from __future__ import annotations

import os
from collections.abc import Callable
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


def select_yolo_runtime(preference: str, npu_available: bool) -> str:
    """Pick ``npu`` only when requested and an Intel NPU is present."""
    pref = (preference or "auto").strip().lower()
    if pref in {"auto", "npu"} and npu_available:
        return "npu"
    return "cpu"


def intel_npu_available() -> bool:
    try:
        import openvino as ov
    except ImportError:
        return False
    try:
        devices = ov.Core().available_devices
    except Exception:
        return False
    return any(device == "NPU" or device.startswith("NPU.") for device in devices)


def openvino_export_dir(weights: Path) -> Path:
    return weights.with_name(f"{weights.stem}_openvino_model")


def is_openvino_model(path: Path) -> bool:
    if path.is_dir():
        return any(path.glob("*.xml"))
    return path.suffix.lower() == ".xml" and path.is_file()


def openvino_export_is_fresh(weights: Path, export_dir: Path) -> bool:
    xmls = list(export_dir.glob("*.xml")) if export_dir.is_dir() else []
    if not xmls:
        return False
    if not weights.is_file():
        return True
    return xmls[0].stat().st_mtime >= weights.stat().st_mtime


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
    def __init__(self, model_path: Path | None = None, device: str | None = None) -> None:
        settings = get_settings()
        self._path = Path(model_path or settings.yolo_model_path)
        self._device_pref = (device or settings.yolo_device).strip().lower()
        self._model = None
        self._enabled = False
        self._runtime = "cpu"
        self._predict_device: str | None = None
        self._load_error: str | None = None
        self._load_attempted = False
        self._status: Callable[[str], None] | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def runtime_label(self) -> str:
        if self._runtime == "npu":
            return "Intel NPU"
        return "CPU"

    def load(self, status: Callable[[str], None] | None = None) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        self._status = status
        self._try_load()

    def _notify(self, text: str) -> None:
        if self._status is not None:
            self._status(text)

    def _import_yolo(self):
        from ultralytics import YOLO
        from ultralytics.utils import events as yolo_events

        # Telemetry starts a thread from the predict callback and can stall the
        # camera thread on Thread.start() while that post holds the GIL.
        yolo_events.events.enabled = False
        return YOLO

    def _try_load(self) -> None:
        _configure_inference_runtime()
        try:
            yolo_cls = self._import_yolo()
        except ImportError:
            self._load_error = "ultralytics not installed"
            return

        _apply_torch_thread_cap()
        source, placeholder = self._resolve_source()
        if select_yolo_runtime(self._device_pref, intel_npu_available()) == "npu":
            try:
                self._load_npu(yolo_cls, source)
            except Exception as exc:  # noqa: BLE001
                self._model = None
                self._enabled = False
                self._runtime = "cpu"
                self._predict_device = None
                self._load_error = f"Intel NPU 不可用，已改用 CPU：{exc}"
                self._notify(self._load_error)
            else:
                if placeholder:
                    self._load_error = (
                        f"Custom model not found at {self._path}; using yolov8n (COCO) placeholder"
                    )
                return

        if self._device_pref == "npu" and self._load_error is None:
            self._load_error = "已设置 YOLO_DEVICE=npu，但未检测到 Intel NPU，已改用 CPU"

        try:
            self._load_cpu(yolo_cls, source, placeholder)
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)

    def _resolve_source(self) -> tuple[Path, bool]:
        if self._path.is_file() or is_openvino_model(self._path):
            return self._path, False
        return Path("yolov8n.pt"), True

    def _load_npu(self, yolo_cls, source: Path) -> None:
        if is_openvino_model(source):
            model_ref = str(source)
        else:
            model_ref = self._ensure_openvino_export(yolo_cls, source)
        self._notify("正在编译 NPU 模型（首次会较慢）…")
        self._model = yolo_cls(model_ref)
        self._runtime = "npu"
        self._predict_device = "intel:npu"
        self._enabled = True
        self.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    def _ensure_openvino_export(self, yolo_cls, weights: Path) -> str:
        if weights.is_file():
            cached = openvino_export_dir(weights)
            if openvino_export_is_fresh(weights, cached):
                return str(cached)
        self._notify("正在把 YOLO 导出为 OpenVINO（首次会较慢）…")
        pytorch = yolo_cls(str(weights))
        ckpt = Path(getattr(pytorch, "ckpt_path", None) or weights)
        export_dir = openvino_export_dir(ckpt)
        if openvino_export_is_fresh(ckpt, export_dir):
            return str(export_dir)
        export_kwargs = {
            "format": "openvino",
            "imgsz": 640,
            "dynamic": False,
            "batch": 1,
        }
        try:
            exported = pytorch.export(**export_kwargs, quantize=16)
        except TypeError:
            exported = pytorch.export(**export_kwargs, half=True)
        return str(exported)

    def _load_cpu(self, yolo_cls, source: Path, placeholder: bool) -> None:
        if is_openvino_model(source):
            self._model = yolo_cls(str(source))
            self._predict_device = "intel:cpu"
        elif placeholder:
            self._model = yolo_cls("yolov8n.pt")
            note = f"Custom model not found at {self._path}; using yolov8n (COCO) placeholder"
            self._load_error = f"{self._load_error}；{note}" if self._load_error else note
        else:
            self._model = yolo_cls(str(source))
        self._runtime = "cpu"
        self._enabled = True

    def detect(self, frame_bgr: np.ndarray, conf: float = 0.35) -> List[Detection]:
        if not self._enabled or self._model is None:
            return []

        predict_kwargs: dict = {"conf": conf, "verbose": False}
        if self._predict_device:
            predict_kwargs["device"] = self._predict_device
        results = self._model.predict(frame_bgr, **predict_kwargs)
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
