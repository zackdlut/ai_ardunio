"""Popup menus and rounded cards whose corners are not filled squares."""

from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QRegion
from PySide6.QtWidgets import QFrame, QLabel, QMenu, QProxyStyle, QStyle, QStyleOptionMenuItem, QWidgetAction

from ui.theme import COLOR_ACCENT, COLOR_ACCENT_SOFT, COLOR_BORDER, COLOR_SURFACE, COLOR_TEXT, COLOR_TEXT_MUTED

_ICON = 16
_ICON_GAP = 12
_MENU_LEFT = 14
_MENU_RIGHT = 16
_menu_style: QProxyStyle | None = None


def _menu_gap_style() -> QProxyStyle:
    global _menu_style
    if _menu_style is None:
        _menu_style = _MenuGapStyle("Fusion")
    return _menu_style


class _MenuGapStyle(QProxyStyle):
    """Menu rows with a fixed gap between the stroke icon and the label."""

    def pixelMetric(self, metric, option=None, widget=None):  # noqa: N802
        if metric in (
            QStyle.PixelMetric.PM_MenuHMargin,
            QStyle.PixelMetric.PM_MenuVMargin,
        ):
            return 6
        if metric == QStyle.PixelMetric.PM_MenuPanelWidth:
            return 0
        return super().pixelMetric(metric, option, widget)

    def drawPrimitive(self, element, option, painter, widget=None):  # noqa: N802
        if element == QStyle.PrimitiveElement.PE_PanelMenu:
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QPen(QColor(COLOR_BORDER)))
            painter.setBrush(QColor(COLOR_SURFACE))
            rect = QRectF(option.rect).adjusted(0.5, 0.5, -0.5, -0.5)
            painter.drawRoundedRect(rect, 12, 12)
            painter.restore()
            return
        super().drawPrimitive(element, option, painter, widget)

    def sizeFromContents(self, contents_type, option, size, widget=None):  # noqa: N802
        if contents_type == QStyle.ContentsType.CT_MenuItem and isinstance(option, QStyleOptionMenuItem):
            if option.menuItemType == QStyleOptionMenuItem.MenuItemType.Separator:
                return QSize(8, 9)
            text_width = option.fontMetrics.horizontalAdvance(option.text or "")
            icon_width = 0 if option.icon.isNull() else _ICON
            gap = _ICON_GAP if icon_width else 0
            width = _MENU_LEFT + icon_width + gap + text_width + _MENU_RIGHT
            return QSize(width, 34)
        return super().sizeFromContents(contents_type, option, size, widget)

    def drawControl(self, element, option, painter, widget=None):  # noqa: N802
        if element == QStyle.ControlElement.CE_MenuItem and isinstance(option, QStyleOptionMenuItem):
            if option.menuItemType == QStyleOptionMenuItem.MenuItemType.Separator:
                self._draw_separator(option, painter)
                return
            if option.menuItemType in (
                QStyleOptionMenuItem.MenuItemType.Scroller,
                QStyleOptionMenuItem.MenuItemType.TearOff,
            ):
                super().drawControl(element, option, painter, widget)
                return
            self._draw_item(option, painter)
            return
        super().drawControl(element, option, painter, widget)

    def _draw_separator(self, option: QStyleOptionMenuItem, painter: QPainter) -> None:
        painter.save()
        painter.setPen(QPen(QColor(COLOR_BORDER)))
        rect = option.rect
        y = rect.center().y()
        painter.drawLine(rect.left() + 12, y, rect.right() - 12, y)
        painter.restore()

    def _draw_item(self, option: QStyleOptionMenuItem, painter: QPainter) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        enabled = bool(option.state & QStyle.StateFlag.State_Enabled)
        selected = bool(option.state & QStyle.StateFlag.State_Selected) and enabled
        rect = option.rect
        if (selected or option.checked) and enabled:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(COLOR_ACCENT_SOFT))
            painter.drawRoundedRect(rect.adjusted(4, 2, -4, -2), 8, 8)
        if option.checked and enabled:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(COLOR_ACCENT))
            bar = QRectF(rect.left() + 7, rect.center().y() - 7, 2.5, 14)
            painter.drawRoundedRect(bar, 1.2, 1.2)
        icon_right = rect.left() + _MENU_LEFT
        if not option.icon.isNull():
            mode = QIcon.Mode.Normal if enabled else QIcon.Mode.Disabled
            pixmap = option.icon.pixmap(QSize(_ICON, _ICON), mode)
            icon_top = rect.center().y() - _ICON // 2
            painter.drawPixmap(rect.left() + _MENU_LEFT, icon_top, pixmap)
            icon_right = rect.left() + _MENU_LEFT + _ICON + _ICON_GAP
        text_rect = rect.adjusted(0, 0, -_MENU_RIGHT, 0)
        text_rect.setLeft(icon_right)
        painter.setFont(option.font)
        painter.setPen(QColor(COLOR_TEXT if enabled else COLOR_TEXT_MUTED))
        label = option.fontMetrics.elidedText(
            option.text or "",
            Qt.TextElideMode.ElideRight,
            max(0, text_rect.width()),
        )
        painter.drawText(
            text_rect,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            label,
        )
        painter.restore()


class ThemedMenu(QMenu):
    """Rounded popup. The square corners are clipped so the window behind shows through."""

    _radius = 12.0

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        flags = self.windowFlags()
        flags |= Qt.WindowType.FramelessWindowHint
        flags |= Qt.WindowType.NoDropShadowWindowHint
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        font = QFont("Microsoft YaHei UI")
        font.setPixelSize(13)
        font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        self.setFont(font)
        self.setStyle(_menu_gap_style())
        self.setObjectName("appMenu")

    def add_section(self, text: str) -> QWidgetAction:
        """Small muted heading over a group of items; not hoverable or clickable."""
        if self.actions():
            self.addSeparator()
        label = QLabel(text)
        label.setObjectName("menuSection")
        action = QWidgetAction(self)
        action.setDefaultWidget(label)
        action.setEnabled(False)
        self.addAction(action)
        return action

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._clip_corners()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._clip_corners()

    def _clip_corners(self) -> None:
        if self.width() < 4 or self.height() < 4:
            return
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), self._radius, self._radius)
        self.setMask(QRegion(path.toFillPolygon().toPolygon()))


class RoundedCard(QFrame):
    """Paints only a rounded rectangle, so the corners stay clear of a square fill."""

    def __init__(self, radius: float = 16, parent=None) -> None:
        super().__init__(parent)
        self._radius = radius
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setAutoFillBackground(False)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(COLOR_BORDER)))
        painter.setBrush(QColor(COLOR_SURFACE))
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.drawRoundedRect(rect, self._radius, self._radius)


class RailCard(QFrame):
    """Rounded card with a colored rail down its left edge; the rail color carries state."""

    def __init__(
        self,
        rail: str = COLOR_ACCENT,
        fill: str = COLOR_SURFACE,
        radius: float = 10,
        rail_width: float = 3,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._rail = QColor(rail)
        self._fill = QColor(fill)
        self._radius = radius
        self._rail_width = rail_width
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setAutoFillBackground(False)

    def set_tone(self, rail: str, fill: str | None = None) -> None:
        self._rail = QColor(rail)
        if fill is not None:
            self._fill = QColor(fill)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(rect, self._radius, self._radius)
        painter.setPen(QPen(QColor(COLOR_BORDER)))
        painter.setBrush(self._fill)
        painter.drawPath(path)
        painter.setClipPath(path)
        painter.fillRect(QRectF(rect.left(), rect.top(), self._rail_width, rect.height()), self._rail)
        painter.end()
