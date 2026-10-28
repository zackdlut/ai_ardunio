from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy


def map_viewport_click(
    widget_w: int,
    widget_h: int,
    pix_w: int,
    pix_h: int,
    x: float,
    y: float,
) -> tuple[float, float] | None:
    """Map a click on a centered letterboxed pixmap to 0–1 frame coordinates."""
    if widget_w <= 0 or widget_h <= 0 or pix_w <= 0 or pix_h <= 0:
        return None
    origin_x = (widget_w - pix_w) / 2
    origin_y = (widget_h - pix_h) / 2
    if x < origin_x or y < origin_y or x >= origin_x + pix_w or y >= origin_y + pix_h:
        return None
    return (x - origin_x) / pix_w, (y - origin_y) / pix_h


class CameraWidget(QLabel):
    marked = Signal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("cameraViewport")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setText("正在打开摄像头…")
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setToolTip("单击画面添加标记。竖起拇指进入下一步，握拳暂停，比 V 重新识别。")
        self._latest: np.ndarray | None = None
        self._flush_scheduled = False
        self._pixmap_size = (0, 0)

    def latest_frame(self) -> np.ndarray | None:
        return self._latest

    def show_frame(self, frame_bgr: np.ndarray) -> None:
        self._latest = frame_bgr
        if self._flush_scheduled:
            return
        self._flush_scheduled = True
        QTimer.singleShot(0, self._present)

    def _present(self) -> None:
        self._flush_scheduled = False
        frame_bgr = self._latest
        if frame_bgr is None:
            return
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(image)
        target = self.size()
        if pix.width() != target.width() or pix.height() != target.height():
            pix = pix.scaled(
                target,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self._pixmap_size = (pix.width(), pix.height())
        self.setPixmap(pix)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self.pixmap() is not None:
            pix_w, pix_h = self._pixmap_size
            mapped = map_viewport_click(
                self.width(),
                self.height(),
                pix_w,
                pix_h,
                event.position().x(),
                event.position().y(),
            )
            if mapped is not None:
                self.marked.emit(mapped[0], mapped[1])
                event.accept()
                return
        super().mousePressEvent(event)
