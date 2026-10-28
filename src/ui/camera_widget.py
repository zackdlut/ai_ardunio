from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QButtonGroup, QFrame, QHBoxLayout, QLabel, QSizePolicy, QToolButton

_DRAG_START_PX = 6
_MIN_BOX_PX = 10
_STROKE_STEP_PX = 3
_LASSO_CLOSURE = 0.75
_LASSO_FILL = 0.2
_LASSO_MAX_POINTS = 64
_PREVIEW_RGB = (255, 180, 70)

Point = tuple[float, float]


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


def normalize_stroke(
    widget_w: int,
    widget_h: int,
    pix_w: int,
    pix_h: int,
    points: Sequence[Point],
) -> list[Point]:
    """Map widget-space stroke points to 0–1 frame coordinates, clamped to the picture."""
    if widget_w <= 0 or widget_h <= 0 or pix_w <= 0 or pix_h <= 0:
        return []
    origin_x = (widget_w - pix_w) / 2
    origin_y = (widget_h - pix_h) / 2
    out: list[Point] = []
    for x, y in points:
        nx = min(max((x - origin_x) / pix_w, 0.0), 1.0)
        ny = min(max((y - origin_y) / pix_h, 0.0), 1.0)
        out.append((nx, ny))
    return out


def effective_mode(mode: str, modifiers: Qt.KeyboardModifier) -> str:
    """Shift forces a rectangle and Ctrl forces a lasso, whatever the toolbar says."""
    if modifiers & Qt.KeyboardModifier.ShiftModifier:
        return "box"
    if modifiers & Qt.KeyboardModifier.ControlModifier:
        return "lasso"
    return mode


def stroke_kind(points: Sequence[Point], mode: str = "auto") -> str:
    """Classify a mouse stroke as ``point``, ``box`` (diagonal drag) or ``lasso`` (free loop).

    ``mode`` is ``auto`` (guess from the path shape) or one of the kinds to force it.
    """
    if mode == "point" or len(points) < 2:
        return "point"
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    span_w = max(xs) - min(xs)
    span_h = max(ys) - min(ys)
    if max(span_w, span_h) < _DRAG_START_PX:
        return "point"
    area = _polygon_area(points) if len(points) >= 3 else 0.0
    if mode == "lasso":
        return "lasso" if area >= _MIN_BOX_PX**2 else "point"
    if mode == "auto" and area > 0:
        path = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
        chord = math.dist(points[0], points[-1])
        if chord / path < _LASSO_CLOSURE and area >= max(_MIN_BOX_PX**2, _LASSO_FILL * span_w * span_h):
            return "lasso"
    (x0, y0), (x1, y1) = points[0], points[-1]
    if abs(x1 - x0) >= _MIN_BOX_PX and abs(y1 - y0) >= _MIN_BOX_PX:
        return "box"
    return "point"


def simplify_stroke(points: Sequence[Point], max_points: int = _LASSO_MAX_POINTS) -> list[Point]:
    if len(points) <= max_points:
        return list(points)
    step = len(points) / max_points
    return [points[int(i * step)] for i in range(max_points)]


def _polygon_area(points: Sequence[Point]) -> float:
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points, [*points[1:], points[0]]):
        total += x0 * y1 - x1 * y0
    return abs(total) / 2


MARK_MODES = (
    ("auto", "自动", "单击加点；拖动时按轨迹自动判断矩形或圈选"),
    ("point", "点", "只加点标记，拖动也按单击处理"),
    ("box", "框", "按住左键沿对角线拖出矩形框（也可在任意模式下按住 Shift）"),
    ("lasso", "圈", "按住左键自由画圈（也可在任意模式下按住 Ctrl）"),
)


class CameraWidget(QLabel):
    marked = Signal(float, float)
    region_marked = Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("cameraViewport")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setText("正在打开摄像头…")
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setToolTip(
            "单击画面加点标记，拖动画矩形框或圈选；右下角可切换「自动 / 点 / 框 / 圈」。"
            "按住 Shift 拖动只画矩形，按住 Ctrl 只画圈。竖起拇指进入下一步，握拳暂停，比 V 重新识别。"
        )
        self._latest: np.ndarray | None = None
        self._flush_scheduled = False
        self._pixmap_size = (0, 0)
        self._stroke: list[Point] = []
        self._stroke_mode = "auto"
        self._mark_mode = "auto"
        self._mode_buttons: dict[str, QToolButton] = {}
        self._mode_bar = self._build_mode_bar()

    def mark_mode(self) -> str:
        return self._mark_mode

    def set_mark_mode(self, mode: str) -> None:
        if mode not in self._mode_buttons:
            return
        self._mark_mode = mode
        self._mode_buttons[mode].setChecked(True)
        if self._stroke:
            self._stroke = []
            self.update()

    def _build_mode_bar(self) -> QFrame:
        bar = QFrame(self)
        bar.setObjectName("markModeBar")
        bar.setCursor(Qt.CursorShape.ArrowCursor)
        row = QHBoxLayout(bar)
        row.setContentsMargins(3, 3, 3, 3)
        row.setSpacing(2)
        group = QButtonGroup(bar)
        group.setExclusive(True)
        for mode, text, tip in MARK_MODES:
            button = QToolButton(bar)
            button.setObjectName("markModeButton")
            button.setText(text)
            button.setToolTip(tip)
            button.setCheckable(True)
            button.setChecked(mode == self._mark_mode)
            button.clicked.connect(lambda _checked=False, m=mode: self.set_mark_mode(m))
            group.addButton(button)
            row.addWidget(button)
            self._mode_buttons[mode] = button
        bar.hide()
        return bar

    def _place_mode_bar(self) -> None:
        pix_w, pix_h = self._pixmap_size
        if pix_w <= 0 or pix_h <= 0:
            self._mode_bar.hide()
            return
        rect = self._pixmap_rect()
        self._mode_bar.adjustSize()
        x = int(rect.right()) - self._mode_bar.width() - 10
        y = int(rect.bottom()) - self._mode_bar.height() - 10
        self._mode_bar.move(max(0, x), max(0, y))
        self._mode_bar.show()
        self._mode_bar.raise_()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._place_mode_bar()

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
        size = (pix.width(), pix.height())
        moved = size != self._pixmap_size
        self._pixmap_size = size
        self.setPixmap(pix)
        if moved or self._mode_bar.isHidden():
            self._place_mode_bar()

    def _pixmap_rect(self) -> QRectF:
        pix_w, pix_h = self._pixmap_size
        return QRectF((self.width() - pix_w) / 2, (self.height() - pix_h) / 2, pix_w, pix_h)

    def _clamped(self, pos: QPointF) -> Point:
        rect = self._pixmap_rect()
        x = min(max(pos.x(), rect.left()), rect.right())
        y = min(max(pos.y(), rect.top()), rect.bottom())
        return (x, y)

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
                self._stroke = [self._clamped(event.position())]
                self._stroke_mode = effective_mode(self._mark_mode, event.modifiers())
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._stroke and event.buttons() & Qt.MouseButton.LeftButton:
            mode = effective_mode(self._mark_mode, event.modifiers())
            point = self._clamped(event.position())
            if mode != self._stroke_mode or math.dist(point, self._stroke[-1]) >= _STROKE_STEP_PX:
                self._stroke_mode = mode
                if point != self._stroke[-1]:
                    self._stroke.append(point)
                self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._stroke:
            point = self._clamped(event.position())
            if point != self._stroke[-1]:
                self._stroke.append(point)
            stroke = self._stroke
            mode = effective_mode(self._mark_mode, event.modifiers())
            self._stroke = []
            self.update()
            self._finish_stroke(stroke, mode)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _finish_stroke(self, stroke: list[Point], mode: str) -> None:
        pix_w, pix_h = self._pixmap_size
        norm = normalize_stroke(self.width(), self.height(), pix_w, pix_h, stroke)
        if not norm:
            return
        kind = stroke_kind(stroke, mode)
        if kind == "box":
            (x0, y0), (x1, y1) = norm[0], norm[-1]
            self.region_marked.emit(
                {"kind": "box", "box": [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]}
            )
        elif kind == "lasso":
            self.region_marked.emit({"kind": "lasso", "points": simplify_stroke(norm)})
        else:
            self.marked.emit(norm[0][0], norm[0][1])

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        if len(self._stroke) < 2 or self._stroke_mode == "point":
            return
        kind = stroke_kind(self._stroke, self._stroke_mode)
        color = QColor(*_PREVIEW_RGB)
        fill = QColor(*_PREVIEW_RGB, 40)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if kind == "box":
            (x0, y0), (x1, y1) = self._stroke[0], self._stroke[-1]
            painter.setPen(QPen(color, 2))
            painter.setBrush(fill)
            painter.drawRect(QRectF(QPointF(x0, y0), QPointF(x1, y1)).normalized())
        else:
            polygon = QPolygonF([QPointF(x, y) for x, y in self._stroke])
            if kind == "lasso":
                painter.setPen(QPen(color, 2))
                painter.setBrush(fill)
                painter.drawPolygon(polygon)
            else:
                painter.setPen(QPen(color, 2, Qt.PenStyle.DashLine))
                painter.drawPolyline(polygon)
        painter.end()
