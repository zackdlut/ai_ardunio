from __future__ import annotations

import threading
import time
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QMutex, QThread, Signal

from gesture.debounce import GestureDebouncer
from gesture.mediapipe_ctrl import GestureCommand, GestureHandler
from overlay.engine import OverlayEngine
from repair.geometry import tracking_roi, warp_shapes
from ui.camera_devices import _capture_backend
from vision.detector import YOLODetector, scale_detection_boxes
from vision.plane_tracker import PlaneTracker
from vision.scene_parser import SceneGraphBuilder
from vision.tracker import SimpleTracker

_INFER_MAX_WIDTH = 640
_FRAME_PERIOD = 1.0 / 30.0
_LOST_FADE_S = 1.5


class CameraWorker(QThread):
    frame_ready = Signal(np.ndarray)
    scene_updated = Signal(dict)
    gesture_detected = Signal(str)
    status_message = Signal(str)
    tracking_changed = Signal(str)

    def __init__(self, camera_index: int = 0) -> None:
        super().__init__()
        self._running = True
        self._paused = False
        self._mutex = QMutex()
        self._camera_index = camera_index
        self._reopen_camera = False
        self._overlay_hint: dict[str, Any] = {}
        self._user_marks: list[dict[str, Any]] = []
        self._mode = "tutor"
        self._shapes: list[dict[str, Any]] = []
        self._board_box: list[float] | None = None
        self._frozen = False
        self._frozen_frame: np.ndarray | None = None
        self._last_raw: np.ndarray | None = None
        self._last_shown: np.ndarray | None = None
        # Plane tracking. Shapes live in the reference frame's 0..1 coordinates;
        # _track_H maps them onto the live frame (None means identity).
        self._tracking_enabled = True
        self._track_features = 1000
        self._track_min_inliers = 18
        self._tracker_dirty = False
        self._plane = PlaneTracker()
        self._ref_gen = 0
        self._ref_frame: np.ndarray | None = None
        self._pending_ref: np.ndarray | None = None
        self._roi_dirty = False
        self._track_H: np.ndarray | None = None
        self._track_status = "idle"
        self._lost_at = 0.0
        self._emitted_status = ""
        self._force_rescan = False
        self._models_ready = False
        self._infer_busy = False
        self._infer_frame: np.ndarray | None = None
        self._last_detections: list = []
        self._infer_event = threading.Event()
        self._last_infer_error = ""

        self._detector = YOLODetector()
        self._tracker = SimpleTracker()
        self._scene_builder = SceneGraphBuilder()
        self._overlay = OverlayEngine()
        self._gesture = GestureHandler()
        self._gesture_debounce = GestureDebouncer()
        self._last_gesture_at = 0.0
        self._hand_points: list[tuple[float, float]] | None = None
        self._hand_seen_at = 0.0

    def set_overlay_hint(self, hint: dict[str, Any]) -> None:
        self._mutex.lock()
        self._overlay_hint = hint or {}
        self._mutex.unlock()

    def set_user_marks(self, marks: list[dict[str, Any]]) -> None:
        self._mutex.lock()
        self._user_marks = [dict(item) for item in marks]
        self._mutex.unlock()

    def set_mode(self, mode: str) -> None:
        """``repair`` skips YOLO (it only knows Arduino parts); gestures keep running."""
        self._mutex.lock()
        self._mode = mode
        if mode == "repair":
            self._last_detections = []
        self._mutex.unlock()

    def set_shapes(
        self, shapes: list[dict[str, Any]], board_box: list[float] | None = None
    ) -> None:
        """Shapes to draw, in the current reference frame's 0..1 coordinates."""
        self._mutex.lock()
        self._shapes = [dict(item) for item in shapes]
        if board_box != self._board_box:
            self._board_box = list(board_box) if board_box else None
        self._roi_dirty = True
        self._mutex.unlock()

    def configure_tracking(self, enabled: bool, features: int, min_inliers: int) -> None:
        self._mutex.lock()
        self._tracking_enabled = bool(enabled)
        self._track_features = int(features)
        self._track_min_inliers = int(min_inliers)
        self._tracker_dirty = True
        if not enabled:
            self._track_H = None
            self._track_status = "off"
        self._mutex.unlock()

    def tracking_status(self) -> str:
        self._mutex.lock()
        status = "frozen" if self._frozen else self._track_status
        self._mutex.unlock()
        return status

    def anchor_snapshot(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
        """Make the latest frame the tracking reference.

        Returns ``(raw, shown, H)``: the clean frame, the frame as displayed
        (what the model gets to see), and the homography that carried the
        existing shapes into the new reference, or None if they were kept as is
        (frozen, tracking off or lost).
        """
        self._mutex.lock()
        try:
            raw = self._frozen_frame if self._frozen else self._last_raw
            shown = self._last_shown
            if raw is None or shown is None:
                return None
            H = self._rebase_locked(raw)
            return raw.copy(), shown.copy(), H
        finally:
            self._mutex.unlock()

    def set_frozen(self, frozen: bool) -> np.ndarray | None:
        """Freezing makes the frozen frame the reference; returns the rebase homography."""
        self._mutex.lock()
        try:
            self._frozen = frozen
            if not frozen:
                self._frozen_frame = None
                return None
            if self._last_raw is None:
                return None
            self._frozen_frame = self._last_raw.copy()
            return self._rebase_locked(self._frozen_frame)
        finally:
            self._mutex.unlock()

    def _rebase_locked(self, raw: np.ndarray) -> np.ndarray | None:
        # While lost this is the last good pose: shapes stay where they were last seen.
        H = self._track_H
        if H is not None:
            self._shapes = warp_shapes(self._shapes, H)
        self._ref_gen += 1
        self._ref_frame = raw
        self._track_H = None
        if self._tracking_enabled:
            self._pending_ref = raw
            self._track_status = "tracking"
        return H

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
        self._infer_event.set()

    def _open_capture(self, index: int) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(index, _capture_backend())
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap

    def _load_models(self) -> None:
        if not self._running:
            return
        self.status_message.emit("正在加载检测模型…")
        try:
            self._detector.load(status=self.status_message.emit)
            if self._running and self._detector.enabled and self._detector.runtime == "cpu":
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
        if self._detector.enabled:
            ready = f"检测模型已就绪（{self._detector.runtime_label}）"
            if self._detector.load_error:
                ready = f"{ready}。{self._detector.load_error}"
            self.status_message.emit(ready)
        elif self._detector.load_error:
            self.status_message.emit(f"检测模型未能加载：{self._detector.load_error}")
        else:
            self.status_message.emit("检测模型未能加载")

    def _submit_infer(self, frame: np.ndarray) -> None:
        self._mutex.lock()
        if not self._models_ready or self._infer_busy:
            self._mutex.unlock()
            return
        self._infer_busy = True
        self._infer_frame = frame.copy()
        self._mutex.unlock()
        self._infer_event.set()

    def _detect_fast(self, frame: np.ndarray):
        height, width = frame.shape[:2]
        if width <= _INFER_MAX_WIDTH:
            return self._detector.detect(frame)
        scale = _INFER_MAX_WIDTH / width
        small_h = max(1, int(round(height * scale)))
        small = cv2.resize(frame, (_INFER_MAX_WIDTH, small_h), interpolation=cv2.INTER_LINEAR)
        dets = self._detector.detect(small)
        scale_detection_boxes(dets, width / _INFER_MAX_WIDTH, height / small_h)
        return dets

    def _gesture_frame(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if width <= _INFER_MAX_WIDTH:
            return frame
        scale = _INFER_MAX_WIDTH / width
        return cv2.resize(
            frame,
            (_INFER_MAX_WIDTH, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_LINEAR,
        )

    def _read_gesture(self, frame: np.ndarray) -> None:
        now = time.monotonic()
        if now - self._last_gesture_at < 0.12:
            return
        self._last_gesture_at = now
        result = self._gesture.process(self._gesture_frame(frame))
        now = time.monotonic()
        self._mutex.lock()
        if result.points:
            self._hand_points = result.points
            self._hand_seen_at = now
        elif now - self._hand_seen_at > 0.4:
            self._hand_points = None
        self._mutex.unlock()
        fired = self._gesture_debounce.push(result.command)
        if fired is not None and fired != GestureCommand.NONE:
            self.gesture_detected.emit(fired.value)

    def _report_infer_error(self, exc: Exception) -> None:
        text = f"识别暂时失败: {exc}"
        if text == self._last_infer_error:
            return
        self._last_infer_error = text
        self.status_message.emit(text)

    def _track(self, frame: np.ndarray) -> None:
        self._mutex.lock()
        enabled = self._tracking_enabled
        if self._tracker_dirty:
            self._tracker_dirty = False
            self._plane = PlaneTracker(self._track_features, self._track_min_inliers)
            if enabled and self._ref_frame is not None:
                self._pending_ref = self._ref_frame
        gen = self._ref_gen
        pending, self._pending_ref = self._pending_ref, None
        refresh, self._roi_dirty = self._roi_dirty, False
        roi = tracking_roi(self._shapes, self._board_box)
        ref_frame = self._ref_frame
        frozen = self._frozen
        has_shapes = bool(self._shapes)
        self._mutex.unlock()
        if not enabled:
            self._publish_status("off")
            return

        plane = self._plane
        if pending is not None:
            plane.set_reference(pending, roi)
        elif refresh and ref_frame is not None:
            plane.set_reference(ref_frame, roi, reset=False)
        elif not plane.has_reference and (plane.state.status == "idle" or not has_shapes):
            # Nothing anchored yet: anchor to whatever is in view now.
            self._mutex.lock()
            if gen == self._ref_gen:
                self._ref_gen += 1
                gen = self._ref_gen
                self._ref_frame = frame
            self._mutex.unlock()
            plane.set_reference(frame, roi)
        if frozen:
            self._publish_status("frozen")
            return
        state = plane.update(frame) if pending is None else plane.state
        self._mutex.lock()
        if gen == self._ref_gen:
            if state.status == "lost" and self._track_status != "lost":
                self._lost_at = time.monotonic()
            self._track_status = state.status
            self._track_H = state.H
        status = self._track_status
        self._mutex.unlock()
        self._publish_status(status)

    def _publish_status(self, status: str) -> None:
        if status != self._emitted_status:
            self._emitted_status = status
            self.tracking_changed.emit(status)

    def _infer_loop(self) -> None:
        self._load_models()
        while self._running:
            if not self._infer_event.wait(0.3):
                continue
            self._infer_event.clear()
            self._mutex.lock()
            frame = self._infer_frame
            self._infer_frame = None
            paused = self._paused
            force = self._force_rescan
            if force:
                self._force_rescan = False
            repair = self._mode == "repair"
            self._mutex.unlock()
            try:
                if frame is None:
                    continue
                self._read_gesture(frame)
                if repair:
                    self._track(frame)
                    continue
                if paused and not force:
                    continue
                dets = self._detect_fast(frame)
                tracked = self._tracker.update(dets)
                scene = self._scene_builder.build(tracked)
                scene["frame"] = {
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                }
                self._mutex.lock()
                self._last_detections = tracked
                self._mutex.unlock()
                self._last_infer_error = ""
                self.scene_updated.emit(scene)
            except Exception as exc:  # noqa: BLE001
                self._report_infer_error(exc)
            finally:
                self._mutex.lock()
                self._infer_busy = False
                self._mutex.unlock()

    def _capture_loop(self, cap: cv2.VideoCapture) -> None:
        next_tick = time.perf_counter()
        while self._running:
            self._mutex.lock()
            reopen = self._reopen_camera
            if reopen:
                new_index = self._camera_index
                self._reopen_camera = False
            self._mutex.unlock()
            if reopen:
                cap.release()
                cap = self._open_capture(new_index)
                if not cap.isOpened():
                    self.status_message.emit(f"无法打开摄像头 index={new_index}")
                    self.msleep(500)
                    continue
                self._mutex.lock()
                self._last_detections = []
                self._mutex.unlock()
                self.status_message.emit(f"已切换到摄像头 {new_index}")

            ok, frame = cap.read()
            if not ok:
                self.msleep(30)
                continue

            self._mutex.lock()
            paused = self._paused
            ready = self._models_ready
            repair = self._mode == "repair"
            detections = [] if repair else list(self._last_detections)
            hint = dict(self._overlay_hint)
            marks = [dict(item) for item in self._user_marks]
            shapes = list(self._shapes)
            hand = list(self._hand_points) if self._hand_points else None
            frozen = self._frozen
            if frozen and self._frozen_frame is None:
                self._frozen_frame = frame.copy()
            base = self._frozen_frame if frozen else frame
            transform = None if frozen else self._track_H
            track_status = "frozen" if frozen else self._track_status
            lost_at = self._lost_at
            self._mutex.unlock()
            if ready:
                self._submit_infer(frame)

            now = time.monotonic()
            shape_alpha = 1.0
            if track_status == "lost":
                shape_alpha = max(0.0, 1.0 - (now - lost_at) / _LOST_FADE_S)
            shown = self._overlay.render(
                base,
                detections,
                hint=hint,
                marks=marks,
                paused=paused,
                pulse=now,
                hand=None if frozen else hand,
                mode="repair" if repair else "tutor",
                annotations=shapes,
                frozen=frozen,
                transform=transform,
                shape_alpha=shape_alpha,
                track_status=track_status,
            )
            self._mutex.lock()
            self._last_raw = base
            self._last_shown = shown
            self._mutex.unlock()
            self.frame_ready.emit(shown)

            next_tick += _FRAME_PERIOD
            delay = next_tick - time.perf_counter()
            if delay > 0:
                self.msleep(max(1, int(delay * 1000)))
            else:
                next_tick = time.perf_counter()

    def run(self) -> None:
        cv2.setNumThreads(2)
        cap = self._open_capture(self._camera_index)
        if not cap.isOpened():
            self.status_message.emit(f"无法打开摄像头 index={self._camera_index}")
            return

        infer = threading.Thread(target=self._infer_loop, name="vision-infer", daemon=True)
        infer.start()
        try:
            self._capture_loop(cap)
        finally:
            self._running = False
            self._infer_event.set()
            infer.join(timeout=2.0)
            cap.release()
