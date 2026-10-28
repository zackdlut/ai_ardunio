"""Stroke icons in the same ink as the menu text, plus themed editor menus."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QAction, QColor, QGuiApplication, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLineEdit, QMenu, QTextEdit, QWidget

from ui.chrome import ThemedMenu
from ui.theme import COLOR_TEXT, COLOR_TEXT_MUTED

_SIZE = 16
_Draw = Callable[[QPainter], None]


def _stroke_icon(draw: _Draw) -> QIcon:
    icon = QIcon()
    for mode, color in (
        (QIcon.Mode.Normal, COLOR_TEXT),
        (QIcon.Mode.Disabled, COLOR_TEXT_MUTED),
        (QIcon.Mode.Active, COLOR_TEXT),
    ):
        pixmap = QPixmap(_SIZE, _SIZE)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(color))
        pen.setWidthF(1.35)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        draw(painter)
        painter.end()
        icon.addPixmap(pixmap, mode, QIcon.State.Off)
        icon.addPixmap(pixmap, mode, QIcon.State.On)
    return icon


def with_label_gap(icon: QIcon, gap: int = 8) -> QIcon:
    """Widen a 16px icon with clear space on the right, so a label can sit apart from it."""
    gapped = QIcon()
    for mode in (QIcon.Mode.Normal, QIcon.Mode.Disabled, QIcon.Mode.Active, QIcon.Mode.Selected):
        source = icon.pixmap(QSize(_SIZE, _SIZE), mode)
        ratio = source.devicePixelRatio() or 1.0
        canvas = QPixmap(int((_SIZE + gap) * ratio), int(_SIZE * ratio))
        canvas.setDevicePixelRatio(ratio)
        canvas.fill(Qt.GlobalColor.transparent)
        painter = QPainter(canvas)
        painter.drawPixmap(0, 0, source)
        painter.end()
        gapped.addPixmap(canvas, mode, QIcon.State.Off)
        gapped.addPixmap(canvas, mode, QIcon.State.On)
    return gapped


def icon_project() -> QIcon:
    def draw(p: QPainter) -> None:
        p.drawLine(3, 4, 8, 3)
        p.drawLine(8, 3, 13, 4)
        p.drawLine(3, 4, 3, 13)
        p.drawLine(13, 4, 13, 13)
        p.drawLine(3, 13, 8, 12)
        p.drawLine(8, 12, 13, 13)
        p.drawLine(8, 3, 8, 12)

    return _stroke_icon(draw)


def icon_hardware() -> QIcon:
    def draw(p: QPainter) -> None:
        p.drawRoundedRect(QRectF(4.5, 4.5, 7, 7), 1.4, 1.4)
        for y in (6.5, 8, 9.5):
            p.drawLine(2, int(y), 4, int(y))
            p.drawLine(12, int(y), 14, int(y))

    return _stroke_icon(draw)


def icon_camera() -> QIcon:
    def draw(p: QPainter) -> None:
        p.drawRoundedRect(QRectF(2, 5, 12, 8), 1.6, 1.6)
        p.drawEllipse(QRectF(6.2, 6.6, 3.6, 3.6))
        p.drawLine(5, 5, 6.5, 3.2)
        p.drawLine(6.5, 3.2, 9, 3.2)

    return _stroke_icon(draw)


def icon_voice() -> QIcon:
    def draw(p: QPainter) -> None:
        p.drawRoundedRect(QRectF(6.2, 2, 3.6, 6.4), 1.6, 1.6)
        p.drawArc(QRectF(4, 4.5, 8, 6.5), 200 * 16, 140 * 16)
        p.drawLine(8, 11.2, 8, 13.2)
        p.drawLine(5.5, 13.2, 10.5, 13.2)

    return _stroke_icon(draw)


def icon_view() -> QIcon:
    def draw(p: QPainter) -> None:
        p.drawArc(QRectF(2, 4, 12, 8), 20 * 16, 140 * 16)
        p.drawArc(QRectF(2, 4, 12, 8), 200 * 16, 140 * 16)
        p.drawEllipse(QRectF(6.3, 6.3, 3.4, 3.4))

    return _stroke_icon(draw)


def icon_settings() -> QIcon:
    def draw(p: QPainter) -> None:
        p.drawEllipse(QRectF(4.5, 4.5, 7, 7))
        p.drawEllipse(QRectF(6.8, 6.8, 2.4, 2.4))
        for x1, y1, x2, y2 in (
            (8, 2, 8, 4.5), (8, 11.5, 8, 14), (2, 8, 4.5, 8), (11.5, 8, 14, 8),
            (3.8, 3.8, 5.5, 5.5), (10.5, 10.5, 12.2, 12.2),
            (3.8, 12.2, 5.5, 10.5), (10.5, 5.5, 12.2, 3.8),
        ):
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

    return _stroke_icon(draw)


def icon_next() -> QIcon:
    def draw(p: QPainter) -> None:
        p.drawLine(3, 8, 12, 8)
        p.drawLine(8.5, 4.2, 12.2, 8)
        p.drawLine(8.5, 11.8, 12.2, 8)

    return _stroke_icon(draw)


def icon_clear() -> QIcon:
    def draw(p: QPainter) -> None:
        p.drawLine(4, 4.5, 12, 4.5)
        p.drawLine(6, 4.5, 6.4, 3)
        p.drawLine(6.4, 3, 9.6, 3)
        p.drawLine(9.6, 3, 10, 4.5)
        p.drawLine(5, 4.5, 5.6, 13)
        p.drawLine(11, 4.5, 10.4, 13)
        p.drawLine(5.6, 13, 10.4, 13)
        p.drawLine(8, 7, 8, 11)

    return _stroke_icon(draw)


def icon_log() -> QIcon:
    def draw(p: QPainter) -> None:
        p.drawRoundedRect(QRectF(2.5, 3, 11, 10), 1.5, 1.5)
        p.drawLine(5, 6.5, 11, 6.5)
        p.drawLine(5, 9, 9, 9)

    return _stroke_icon(draw)


def icon_refresh() -> QIcon:
    def draw(p: QPainter) -> None:
        p.drawArc(QRectF(3, 3, 10, 10), 40 * 16, 280 * 16)
        p.drawLine(11.2, 3.2, 11.2, 6.2)
        p.drawLine(11.2, 3.2, 8.4, 3.4)

    return _stroke_icon(draw)


def icon_upload() -> QIcon:
    def draw(p: QPainter) -> None:
        p.drawLine(8, 12.5, 8, 4)
        p.drawLine(4.8, 7, 8, 3.6)
        p.drawLine(11.2, 7, 8, 3.6)
        p.drawLine(4, 13.2, 12, 13.2)

    return _stroke_icon(draw)


def icon_local() -> QIcon:
    def draw(p: QPainter) -> None:
        p.drawRoundedRect(QRectF(3, 3.5, 10, 7), 1.4, 1.4)
        p.drawLine(6, 12.5, 10, 12.5)
        p.drawLine(8, 10.5, 8, 12.5)

    return _stroke_icon(draw)


def icon_remote() -> QIcon:
    def draw(p: QPainter) -> None:
        p.drawArc(QRectF(3, 5, 5, 5), 0, 180 * 16)
        p.drawArc(QRectF(6.5, 4, 6.5, 6), 0, 180 * 16)
        p.drawLine(3.2, 8, 12.8, 8)
        p.drawLine(4, 8, 4, 11)
        p.drawLine(12, 8, 12, 11)
        p.drawLine(4, 11, 12, 11)

    return _stroke_icon(draw)


def icon_speaker() -> QIcon:
    def draw(p: QPainter) -> None:
        p.drawLine(3, 6.5, 5.5, 6.5)
        p.drawLine(5.5, 6.5, 9, 4)
        p.drawLine(9, 4, 9, 12)
        p.drawLine(5.5, 9.5, 9, 12)
        p.drawLine(3, 6.5, 3, 9.5)
        p.drawLine(3, 9.5, 5.5, 9.5)
        p.drawArc(QRectF(8, 6, 4, 4), -40 * 16, 80 * 16)

    return _stroke_icon(draw)


def icon_stop() -> QIcon:
    def draw(p: QPainter) -> None:
        p.drawRoundedRect(QRectF(4, 4, 8, 8), 1.4, 1.4)

    return _stroke_icon(draw)


def icon_copy() -> QIcon:
    def draw(p: QPainter) -> None:
        p.drawRoundedRect(QRectF(5.5, 3, 7, 8.5), 1.2, 1.2)
        p.drawLine(3.5, 6, 3.5, 13)
        p.drawLine(3.5, 13, 10, 13)

    return _stroke_icon(draw)


def icon_cut() -> QIcon:
    def draw(p: QPainter) -> None:
        p.drawEllipse(QRectF(2.5, 2.5, 4, 4))
        p.drawEllipse(QRectF(2.5, 9.5, 4, 4))
        p.drawLine(6.2, 5.2, 13, 12)
        p.drawLine(6.2, 10.8, 13, 4)

    return _stroke_icon(draw)


def icon_paste() -> QIcon:
    def draw(p: QPainter) -> None:
        p.drawRoundedRect(QRectF(4, 4.5, 8, 9), 1.3, 1.3)
        p.drawRoundedRect(QRectF(6, 2.5, 4, 3), 0.8, 0.8)
        p.drawLine(6.2, 8, 9.8, 8)
        p.drawLine(6.2, 10.2, 9.8, 10.2)

    return _stroke_icon(draw)


def icon_select_all() -> QIcon:
    def draw(p: QPainter) -> None:
        p.drawLine(3, 4, 13, 4)
        p.drawLine(3, 8, 13, 8)
        p.drawLine(3, 12, 10, 12)

    return _stroke_icon(draw)


def _clone_action(menu: QMenu, source: QAction) -> None:
    action = QAction(source.icon(), source.text(), menu)
    action.setEnabled(source.isEnabled())
    action.setToolTip(source.toolTip())
    if source.isCheckable():
        action.setCheckable(True)
        action.setChecked(source.isChecked())
        action.triggered.connect(lambda checked, src=source: src.setChecked(bool(checked)))
    else:
        action.triggered.connect(source.trigger)
    menu.addAction(action)


def build_editor_menu(
    widget: QWidget, extra: list[QAction] | None = None
) -> QMenu:
    """Right-click menu for a text field: edit commands, then any shared actions."""
    menu = ThemedMenu(widget)
    readonly = bool(getattr(widget, "isReadOnly", lambda: False)())
    if isinstance(widget, QTextEdit):
        has_selection = widget.textCursor().hasSelection()
        has_text = bool(widget.toPlainText())
    elif isinstance(widget, QLineEdit):
        has_selection = widget.hasSelectedText()
        has_text = bool(widget.text())
    else:
        has_selection = False
        has_text = False
    can_paste = bool(QGuiApplication.clipboard().text()) and not readonly

    if not readonly:
        cut_action = menu.addAction(icon_cut(), "剪切")
        cut_action.setEnabled(has_selection)
        cut_action.triggered.connect(widget.cut)  # type: ignore[attr-defined]
    copy_action = menu.addAction(icon_copy(), "复制")
    copy_action.setEnabled(has_selection)
    copy_action.triggered.connect(widget.copy)  # type: ignore[attr-defined]
    if not readonly:
        paste_action = menu.addAction(icon_paste(), "粘贴")
        paste_action.setEnabled(can_paste)
        paste_action.triggered.connect(widget.paste)  # type: ignore[attr-defined]
    select_action = menu.addAction(icon_select_all(), "全选")
    select_action.setEnabled(has_text)
    select_action.triggered.connect(widget.selectAll)  # type: ignore[attr-defined]

    if extra:
        menu.addSeparator()
        for source in extra:
            _clone_action(menu, source)
    return menu


def show_editor_menu(widget: QWidget, global_pos, extra: list[QAction] | None = None) -> None:
    build_editor_menu(widget, extra).exec(global_pos)


def bind_editor_menu(widget: QWidget, extra_provider: Callable[[], list[QAction]] | None = None) -> None:
    widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

    def _open(pos) -> None:
        extra = extra_provider() if extra_provider else None
        show_editor_menu(widget, widget.mapToGlobal(pos), extra)

    widget.customContextMenuRequested.connect(_open)
