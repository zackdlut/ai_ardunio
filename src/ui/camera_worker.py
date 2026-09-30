from __future__ import annotations

import threading
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QMutex, QThread, Signal

from gesture.mediapipe_ctrl import GestureCommand, GestureHandler
from overlay.engine import OverlayEngine
from vision.detector import YOLODetector
from vision.scene_parser import SceneGraphBuilder
from vision.tracker import SimpleTracker
from ui.camera_devices import _capture_backend


class CameraWorker(QThread):
    frame_ready = Signal(np.ndarray)
    scene_updated = Signal(dict)
    gesture_detected = Signal(str)
    status_message = Signal(str)

    def __init__(self, camera_index: int = 0) -> None:
        super().__init__()
        self._running = True
        self._paused = False
        self._mutex = QMutex()
        self._camera_index = camera_index
        self._reopen_camera = False
        self._overlay_hint: dict[str, Any] = {}
        self._force_rescan = False
        self._models_ready = False

        self._detector = YOLODetector()
        self._tracker = SimpleTracker()
        self._scene_builder = SceneGraphBuilder()
        self._overlay = OverlayEngine()
        self._gesture = GestureHandler()

    def set_overlay_hint(self, hint: dict[str, Any]) -> None:
        self._mutex.lock()
        self._overlay_hint = hint or {}
        self._mutex.unlock()

    def set_paused(self, paused: bool) -> None:
        self._mutex.lock()
        self._paused = paused
        self._mutex.unlock()

    def request_rescan(self) -> None:
        self._mutex.lock()
        self._force_rescan = True
        self._mutex.unlock()

    def set_camera_index(self, index: int) -> None:
        self._mutex.lock()
        if index != self._camera_index:
            self._camera_index = index
            self._reopen_camera = True
        self._mutex.unlock()

    def stop(self) -> None:
        self._running = False

    def _open_capture(self, index: int) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(index, _capture_backend())
        return cap

    def _load_models(self) -> None:
        if not self._running:
            return
        self.status_message.emit("正在加载检测模型…")
        try:
            self._detector.load()
            if self._running and self._detector.enabled:
                dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                self._detector.detect(dummy)
            if self._running:
                self._gesture.warmup()
        except Exception as exc:  # noqa: BLE001
            if self._running:
                self.status_message.emit(f"检测模型加载失败: {exc}")
        if not self._running:
            return
        self._mutex.lock()
        self._models_ready = True
        self._mutex.unlock()
        if self._detector.load_error:
            self.status_message.emit(self._detector.load_error)
        if self._detector.enabled:
            self.status_message.emit("检测模型已就绪")
        else:
            self.status_message.emit("检测模型未能加载")

    def run(self) -> None:
        cap = self._open_capture(self._camera_index)
        if not cap.isOpened():
            self.status_message.emit(f"无法打开摄像头 index={self._camera_index}")
            return

        loader = threading.Thread(target=self._load_models, name="vision-load", daemon=True)
        loader.start()

        frame_idx = 0
        last_scene: dict = {"nodes": {}, "summary": {}}
        last_detections = []

        while self._running:
            self._mutex.lock()
            if self._reopen_camera:
                new_index = self._camera_index
                self._reopen_camera = False
                self._mutex.unlock()
                cap.release()
                cap = self._open_capture(new_index)
                if not cap.isOpened():
                    self.status_message.emit(f"无法打开摄像头 index={new_index}")
                    self.msleep(500)
                    continue
                frame_idx = 0
                last_scene = {"nodes": {}, "summary": {}}
                last_detections = []
                self.status_message.emit(f"已切换到摄像头 {new_index}")
            else:
                self._mutex.unlock()

            ok, frame = cap.read()
            if not ok:
                self.msleep(30)
                continue

            self._mutex.lock()
            paused = self._paused
            hint = dict(self._overlay_hint)
            force = self._force_rescan
            models_ready = self._models_ready
            if force and models_ready and not paused:
                self._force_rescan = False
            self._mutex.unlock()

            if not paused and models_ready:
                if frame_idx % 2 == 0 or force:
                    dets = self._detector.detect(frame)
                    last_detections = self._tracker.update(dets)
                    last_scene = self._scene_builder.build(last_detections)
                    self.scene_updated.emit(last_scene)

                g = self._gesture.process(frame)
                if g.command != GestureCommand.NONE:
                    self.gesture_detected.emit(g.command.value)

            out = frame
            if last_detections:
                out = self._overlay.show_bbox(out, last_detections)
            if hint:
                out = self._overlay.render_step_guide(
                    out,
                    last_detections,
                    hint.get("hint", ""),
                    target_types=hint.get("requires"),
                    pin=hint.get("pin"),
                )

            self.frame_ready.emit(out)
            frame_idx += 1
            self.msleep(15)

        cap.release()
