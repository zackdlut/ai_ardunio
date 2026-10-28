from __future__ import annotations

import re
import time
from collections.abc import Callable
from datetime import datetime

from PySide6.QtCore import QEvent, QRectF, Qt, QTimer, Signal, QSize
from PySide6.QtGui import QAction, QActionGroup, QColor, QFont, QGuiApplication, QIcon, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ui.chrome import RailCard, RoundedCard, ThemedMenu
from ui.icons import bind_editor_menu, icon_copy, icon_project, icon_repair, with_label_gap
from ui.sidebar_controls import style_sidebar_chip
from ui.markdown_render import code_blocks, render_markdown
from ui.theme import (
    COLOR_ACCENT,
    COLOR_ACCENT_INK,
    COLOR_CAUTION,
    COLOR_CAUTION_SOFT,
    COLOR_FAULT,
    COLOR_FAULT_SOFT,
    COLOR_SURFACE,
    COLOR_TEXT_MUTED,
    COLOR_TRACK_EMPTY,
    COLOR_USER_BUBBLE,
)

# tone -> (rail, fill, eyebrow ink)
EVENT_TONES = {
    "ask": (COLOR_ACCENT, COLOR_SURFACE, COLOR_ACCENT_INK),
    "pass": (COLOR_ACCENT_INK, "#F2FAF6", COLOR_ACCENT_INK),
    "fail": (COLOR_FAULT, COLOR_FAULT_SOFT, COLOR_FAULT),
    "skip": (COLOR_CAUTION, COLOR_CAUTION_SOFT, COLOR_CAUTION),
    "info": ("#B7C4BC", COLOR_SURFACE, COLOR_TEXT_MUTED),
}
_STAMP_SPACE = 76
_MARK_SIZE = 18
_USER_SHARE = 0.85
_LOG_INSET = 12
_LOG_GUTTER = 46
_TURN_GAP = 10
_LOG_ALERT = re.compile(r"失败|警告|错误|不可用|未检测到")
_LONG_RUN = re.compile(r"\S{24,}")
_SOFT_BREAK = re.compile(r"([\\/_.:@?&=])")
_EMPTY_HINTS = {
    "tutor": "跟着当前步骤动手，卡住了就在下面问我",
    "repair": "先说说故障现象，或者把板子对准摄像头",
}

_Radii = tuple[float, float, float, float]


def _clock() -> str:
    return datetime.now().strftime("%H:%M")


def _soft_wrap(text: str) -> str:
    """Let long paths and URLs break after separators instead of widening the transcript."""
    return _LONG_RUN.sub(lambda run: _SOFT_BREAK.sub(lambda m: m.group(1) + "\u200b", run.group(0)), text)


def _clamp_radii(rect: QRectF, radii: _Radii) -> _Radii:
    top_left, top_right, bottom_right, bottom_left = radii
    scale = 1.0
    width = rect.width()
    height = rect.height()
    if top_left + top_right > width and width > 0:
        scale = min(scale, width / (top_left + top_right))
    if bottom_left + bottom_right > width and width > 0:
        scale = min(scale, width / (bottom_left + bottom_right))
    if top_left + bottom_left > height and height > 0:
        scale = min(scale, height / (top_left + bottom_left))
    if top_right + bottom_right > height and height > 0:
        scale = min(scale, height / (top_right + bottom_right))
    if scale < 1:
        return (
            top_left * scale,
            top_right * scale,
            bottom_right * scale,
            bottom_left * scale,
        )
    return radii


def _round_path(rect: QRectF, radii: _Radii) -> QPainterPath:
    top_left, top_right, bottom_right, bottom_left = _clamp_radii(rect, radii)
    path = QPainterPath()
    x, y, width, height = rect.x(), rect.y(), rect.width(), rect.height()
    path.moveTo(x + top_left, y)
    path.lineTo(x + width - top_right, y)
    path.arcTo(QRectF(x + width - 2 * top_right, y, 2 * top_right, 2 * top_right), 90, -90)
    path.lineTo(x + width, y + height - bottom_right)
    path.arcTo(
        QRectF(x + width - 2 * bottom_right, y + height - 2 * bottom_right, 2 * bottom_right, 2 * bottom_right),
        0,
        -90,
    )
    path.lineTo(x + bottom_left, y + height)
    path.arcTo(QRectF(x, y + height - 2 * bottom_left, 2 * bottom_left, 2 * bottom_left), 270, -90)
    path.lineTo(x, y + top_left)
    path.arcTo(QRectF(x, y, 2 * top_left, 2 * top_left), 180, -90)
    path.closeSubpath()
    return path


class _Bubble(QFrame):
    def __init__(self, fill: str, radii: _Radii) -> None:
        super().__init__()
        self._fill = QColor(fill)
        self._radii = radii
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setAutoFillBackground(False)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._fill)
        painter.drawPath(_round_path(rect, self._radii))


class _SpeakerMark(QWidget):
    """Small green square carrying the first character of the assistant's name, like the app mark."""

    def __init__(self, glyph: str) -> None:
        super().__init__()
        self._glyph = glyph[:1]
        self.setFixedSize(_MARK_SIZE, _MARK_SIZE)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(COLOR_ACCENT))
        painter.drawRoundedRect(rect, 5, 5)
        font = self.font()
        font.setPixelSize(11)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor("#FFFFFF"))
        painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), self._glyph)
        painter.end()


class _Speaker(QWidget):
    """Who is talking and since when; shown once at the top of an assistant turn."""

    def __init__(self, name: str) -> None:
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        label = QLabel(name)
        label.setObjectName("speakerName")
        stamp = QLabel(_clock())
        stamp.setObjectName("speakerTime")
        row.addWidget(_SpeakerMark(name), alignment=Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(label, alignment=Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(stamp, alignment=Qt.AlignmentFlag.AlignVCenter)
        row.addStretch(1)


class _ThinkingNote(QWidget):
    """Scratchpad that stays open while tokens arrive, then folds to one line with its duration."""

    assistant_side = True

    def __init__(self, speaker: str) -> None:
        super().__init__()
        self._text = ""
        self._live = True
        self._expanded = True
        self._started = time.monotonic()
        self._elapsed = 0.0
        self.setAutoFillBackground(False)
        self._layout = layout = QVBoxLayout(self)
        layout.setContentsMargins(0, _TURN_GAP, 0, 0)
        layout.setSpacing(4)
        self._speaker = _Speaker(speaker)
        self._toggle = QPushButton()
        self._toggle.setObjectName("thinkToggle")
        self._toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle.setAutoDefault(False)
        self._toggle.setDefault(False)
        self._toggle.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._toggle.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self._toggle.clicked.connect(self._on_toggle)
        self._body = QLabel("")
        self._body.setObjectName("thinkBody")
        self._body.setWordWrap(True)
        self._body.setTextFormat(Qt.TextFormat.PlainText)
        self._body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._body.setVisible(False)
        layout.addWidget(self._speaker)
        layout.addWidget(self._toggle, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._body)
        self._ticker = QTimer(self)
        self._ticker.setInterval(1000)
        self._ticker.timeout.connect(self._sync_label)
        self._ticker.start()
        self._sync_label()

    def text(self) -> str:
        return self._text

    def is_live(self) -> bool:
        return self._live

    def set_speaker_visible(self, visible: bool) -> None:
        self._speaker.setVisible(visible)
        self._layout.setContentsMargins(0, _TURN_GAP if visible else 0, 0, 0)

    def append_text(self, piece: str) -> None:
        self._text += piece
        self._body.setText(self._text)
        self.mark_live()

    def mark_live(self) -> None:
        self._live = True
        self._expanded = True
        self._body.setVisible(bool(self._text))
        if not self._ticker.isActive():
            self._ticker.start()
        self._sync_label()

    def mark_done(self) -> None:
        if not self._live and not self._expanded:
            return
        self._live = False
        self._expanded = False
        self._elapsed = time.monotonic() - self._started
        self._ticker.stop()
        self._body.setVisible(False)
        self._sync_label()

    def set_max_width(self, width: int) -> None:
        self._body.setMaximumWidth(max(120, width))

    def _on_toggle(self) -> None:
        if self._live:
            return
        self._expanded = not self._expanded
        self._body.setVisible(self._expanded and bool(self._text.strip()))
        self._sync_label()

    def _seconds(self) -> int:
        spent = self._elapsed if not self._live else time.monotonic() - self._started
        return max(1, round(spent))

    def _sync_label(self) -> None:
        if self._live:
            self._toggle.setText(f"▾  思考中 · {self._seconds()} 秒")
            self._toggle.setToolTip("思考还在继续")
            return
        arrow = "▾" if self._expanded else "▸"
        self._toggle.setText(f"{arrow}  思考了 {self._seconds()} 秒")
        self._toggle.setToolTip("收起思考" if self._expanded else "展开思考")


class _EventCard(RailCard):
    """A logged bench action (asked to verify, check result, step skipped) with an optional stamp."""

    def __init__(self, tone: str, stamp: str) -> None:
        rail, fill, ink = EVENT_TONES.get(tone, EVENT_TONES["info"])
        super().__init__(rail=rail, fill=fill, radius=10)
        self._ink = QColor(ink)
        self._stamp = stamp

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        if not self._stamp:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = self.font()
        font.setPixelSize(13)
        font.setBold(True)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2)
        painter.setFont(font)
        width = painter.fontMetrics().horizontalAdvance(self._stamp) + 18
        cx = self.width() - _STAMP_SPACE / 2 - 2
        cy = min(self.height() / 2, 30)
        painter.translate(cx, cy)
        painter.rotate(-9)
        box = QRectF(-width / 2, -13, width, 26)
        ink = QColor(self._ink)
        ink.setAlpha(215)
        painter.setPen(QPen(ink, 1.8))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(box, 4, 4)
        ink.setAlpha(110)
        painter.setPen(QPen(ink, 0.9))
        painter.drawRoundedRect(box.adjusted(2.5, 2.5, -2.5, -2.5), 2.5, 2.5)
        ink.setAlpha(225)
        painter.setPen(ink)
        painter.drawText(box, int(Qt.AlignmentFlag.AlignCenter), self._stamp)
        painter.end()


class _EventRow(QWidget):
    """Full-width log entry in the transcript; not a chat bubble, so nobody 'said' it."""

    def __init__(self, eyebrow: str, title: str, detail: str, tone: str, stamp: str) -> None:
        super().__init__()
        self._text = "\n".join(t for t in (eyebrow, title, detail) if t)
        self.setAutoFillBackground(False)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._open_menu)
        card = _EventCard(tone, stamp)
        inner = QVBoxLayout(card)
        inner.setContentsMargins(14, 8, 12 + (_STAMP_SPACE if stamp else 0), 9)
        inner.setSpacing(2)
        ink = EVENT_TONES.get(tone, EVENT_TONES["info"])[2]
        for name, value in (("eventEyebrow", eyebrow), ("eventTitle", title), ("eventDetail", detail)):
            if not value:
                continue
            label = QLabel(_soft_wrap(value))
            label.setObjectName(name)
            label.setWordWrap(True)
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            if name == "eventEyebrow":
                label.setStyleSheet(f"color: {ink};")
            inner.addWidget(label)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 4, 0, 4)
        outer.addWidget(card)

    def text(self) -> str:
        return self._text

    def set_max_width(self, width: int) -> None:
        del width

    def _open_menu(self, pos) -> None:
        menu = ThemedMenu(self)
        menu.addAction(icon_copy(), "复制").triggered.connect(
            lambda: QGuiApplication.clipboard().setText(self._text)
        )
        menu.exec(self.mapToGlobal(pos))


class _LogLine(QWidget):
    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text
        self._stamp = _clock()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        stamp = QLabel(self._stamp)
        stamp.setObjectName("logTime")
        body = QLabel(_soft_wrap(text))
        body.setObjectName("logText")
        body.setWordWrap(True)
        body.setTextFormat(Qt.TextFormat.PlainText)
        body.setProperty("alert", bool(_LOG_ALERT.search(text)))
        row.addWidget(stamp, alignment=Qt.AlignmentFlag.AlignTop)
        row.addWidget(body, stretch=1)
        self._body = body

    def text(self) -> str:
        return f"{self._stamp}  {self._text}"

    def set_max_width(self, width: int) -> None:
        self._body.setMaximumWidth(max(80, width - _LOG_GUTTER))


class _LogStrip(QWidget):
    """Back-to-back status lines folded into one tape; only the newest stays open."""

    def __init__(self) -> None:
        super().__init__()
        self._lines: list[_LogLine] = []
        self._expanded = False
        self._width = 0
        self.setAutoFillBackground(False)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._open_menu)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(_LOG_INSET, 3, 0, 3)
        self._layout.setSpacing(3)
        self._toggle = QPushButton()
        self._toggle.setObjectName("logToggle")
        self._toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle.setAutoDefault(False)
        self._toggle.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self._toggle.clicked.connect(self._on_toggle)
        self._toggle.setVisible(False)
        self._layout.addWidget(self._toggle, alignment=Qt.AlignmentFlag.AlignLeft)

    def add_line(self, text: str) -> None:
        line = _LogLine(text)
        if self._width:
            line.set_max_width(self._width)
        self._lines.append(line)
        self._layout.addWidget(line)
        self._sync()

    def text(self) -> str:
        return "\n".join(line.text() for line in self._lines)

    def set_max_width(self, width: int) -> None:
        self._width = max(120, width - _LOG_INSET)
        for line in self._lines:
            line.set_max_width(self._width)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(COLOR_TRACK_EMPTY))
        painter.drawRoundedRect(QRectF(1, 4, 2, max(0, self.height() - 8)), 1, 1)
        painter.end()

    def _on_toggle(self) -> None:
        self._expanded = not self._expanded
        self._sync()

    def _sync(self) -> None:
        earlier = len(self._lines) - 1
        self._toggle.setVisible(earlier > 0)
        self._toggle.setText("▾  收起" if self._expanded else f"▸  更早 {earlier} 条")
        self._toggle.setToolTip("" if self._expanded else "展开更早的状态记录")
        for index, line in enumerate(self._lines):
            line.setVisible(self._expanded or index == earlier)

    def _line_at(self, pos) -> _LogLine | None:
        widget = self.childAt(pos)
        while widget is not None and widget is not self:
            if isinstance(widget, _LogLine):
                return widget
            widget = widget.parentWidget()
        return None

    def _open_menu(self, pos) -> None:
        menu = ThemedMenu(self)
        line = self._line_at(pos)
        if line is not None:
            menu.addAction(icon_copy(), "复制这条").triggered.connect(
                lambda: QGuiApplication.clipboard().setText(line.text())
            )
        if len(self._lines) > 1 or line is None:
            menu.addAction(icon_copy(), "复制全部").triggered.connect(
                lambda: QGuiApplication.clipboard().setText(self.text())
            )
        menu.exec(self.mapToGlobal(pos))


class _MessageRow(QWidget):
    """User: compact mint bubble on the right. Assistant: full-width notes under a speaker line."""

    def __init__(self, role: str, text: str, extras, assistant_name: str = "助教") -> None:
        super().__init__()
        self._role = role
        self._text = text
        self._extras = extras
        self.assistant_side = role == "assistant"
        self.setAutoFillBackground(False)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._open_menu)

        body = QLabel()
        body.setObjectName("bubbleText")
        body.setWordWrap(True)
        body.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        body.setMinimumHeight(18)
        self._body = body
        self._speaker: _Speaker | None = None
        self._bubble: _Bubble | None = None

        if role == "user":
            body.setTextFormat(Qt.TextFormat.PlainText)
            body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            bubble = _Bubble(COLOR_USER_BUBBLE, (14, 14, 4, 14))
            inner = QVBoxLayout(bubble)
            inner.setContentsMargins(12, 8, 12, 8)
            inner.addWidget(body)
            self._bubble = bubble
            self._layout = QHBoxLayout(self)
            self._layout.setContentsMargins(0, _TURN_GAP, 0, 2)
            self._layout.addStretch(1)
            self._layout.addWidget(bubble)
        else:
            body.setTextFormat(Qt.TextFormat.RichText)
            body.setOpenExternalLinks(True)
            body.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.LinksAccessibleByMouse
            )
            self._speaker = _Speaker(assistant_name)
            self._layout = QVBoxLayout(self)
            self._layout.setContentsMargins(0, _TURN_GAP, 0, 2)
            self._layout.setSpacing(6)
            self._layout.addWidget(self._speaker)
            self._layout.addWidget(body)
        self._show(text)

    def text(self) -> str:
        return self._text

    def set_text(self, text: str) -> None:
        self._text = text
        self._show(text)

    def set_speaker_visible(self, visible: bool) -> None:
        if self._speaker is None:
            return
        self._speaker.setVisible(visible)
        self._layout.setContentsMargins(0, _TURN_GAP if visible else 0, 0, 2)

    def _show(self, text: str) -> None:
        self._body.setText(render_markdown(text) if self._role == "assistant" else _soft_wrap(text))

    def set_max_width(self, width: int) -> None:
        if self._bubble is None:
            self._body.setMaximumWidth(max(120, width))
            return
        capped = max(120, int(width * _USER_SHARE))
        self._bubble.setMaximumWidth(capped)
        lane = max(80, capped - 24)
        self._body.setMaximumWidth(lane)
        self._body.ensurePolished()
        metrics = self._body.fontMetrics()
        natural = max((metrics.horizontalAdvance(line) for line in self._text.splitlines()), default=0) + 2
        self._body.setMinimumWidth(min(natural, lane))

    def _open_menu(self, pos) -> None:
        menu = ThemedMenu(self)
        copy_action = menu.addAction(icon_copy(), "复制")
        copy_action.triggered.connect(self._copy)
        snippets = code_blocks(self._text) if self._role == "assistant" else []
        if snippets:
            code_action = menu.addAction(icon_copy(), "复制代码")
            code_action.triggered.connect(
                lambda: QGuiApplication.clipboard().setText("\n\n".join(snippets))
            )
        extra = self._extras()
        if extra:
            menu.addSeparator()
            for source in extra:
                action = QAction(source.icon(), source.text(), menu)
                action.setEnabled(source.isEnabled())
                if source.isCheckable():
                    action.setCheckable(True)
                    action.setChecked(source.isChecked())
                    action.triggered.connect(lambda checked, src=source: src.setChecked(bool(checked)))
                else:
                    action.triggered.connect(source.trigger)
                menu.addAction(action)
        menu.exec(self.mapToGlobal(pos))

    def _copy(self) -> None:
        selected = self._body.selectedText().replace("\u200b", "")
        QGuiApplication.clipboard().setText(selected or self._text)


class _ElidedLabel(QLabel):
    """One-line label that shrinks with a trailing ellipsis instead of widening its row."""

    def __init__(self) -> None:
        super().__init__()
        self._full = ""
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    def set_full_text(self, text: str) -> None:
        self._full = text
        self.setToolTip(text)
        self._elide()

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(self.fontMetrics().horizontalAdvance(self._full) + 4, super().sizeHint().height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(0, super().minimumSizeHint().height())

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        self.setText(self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideRight, self.width()))


_MODE_SPECS: tuple[tuple[str, Callable[[], QIcon], str, str], ...] = (
    ("tutor", icon_project, "学习", "跟着步骤搭建 Arduino 实验"),
    ("repair", icon_repair, "维修", "诊断故障并在画面上标出检查点"),
)


class ChatWidget(QWidget):
    send_requested = Signal(str)
    voice_session_toggled = Signal(bool)
    mode_change_requested = Signal(str)
    thinking_toggled = Signal(bool)

    def __init__(self, model_name: str = "") -> None:
        super().__init__()
        self._stream_open = False
        self._stream_row: _MessageRow | None = None
        self._rows: list[QWidget] = []
        self._thinking: _ThinkingNote | None = None
        self._notes: list[_ThinkingNote] = []
        self._context_actions: list[QAction] = []
        self._assistant_name = "助教"

        self.setObjectName("chatSidebar")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(8)

        self._scroll = QScrollArea()
        self._scroll.setObjectName("chatScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.viewport().setAutoFillBackground(False)

        self._transcript = QWidget()
        self._transcript.setObjectName("chatTranscript")
        self._transcript.setAutoFillBackground(False)
        self._transcript.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._transcript.customContextMenuRequested.connect(self._open_area_menu)
        self._list = QVBoxLayout(self._transcript)
        self._list.setContentsMargins(0, 0, 6, 4)
        self._list.setSpacing(4)
        self._placeholder = QLabel(_EMPTY_HINTS["tutor"])
        self._placeholder.setObjectName("chatPlaceholder")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._list.addWidget(self._placeholder)
        self._list.addStretch(1)
        self._scroll.setWidget(self._transcript)
        layout.addWidget(self._scroll, stretch=1)

        composer = RoundedCard(14)
        composer.setObjectName("composer")
        composer_layout = QVBoxLayout(composer)
        composer_layout.setContentsMargins(12, 10, 12, 8)
        composer_layout.setSpacing(8)

        self._input = QLineEdit()
        self._input.setObjectName("composerInput")
        self._input.setPlaceholderText("问问当前这一步…")
        self._input.returnPressed.connect(self._on_send)
        bind_editor_menu(self._input, self._context_actions_now)
        composer_layout.addWidget(self._input)

        row = QHBoxLayout()
        row.setSpacing(6)

        self._mode_actions: dict[str, QAction] = {}
        mode_menu = ThemedMenu(self)
        mode_menu.setObjectName("appMenu")
        self._mode_group = QActionGroup(self)
        self._mode_group.setExclusive(True)
        for mode, icon_fn, label, tip in _MODE_SPECS:
            action = mode_menu.addAction(icon_fn(), label)
            action.setCheckable(True)
            action.setData(mode)
            action.setToolTip(tip)
            self._mode_group.addAction(action)
            self._mode_actions[mode] = action
        self._mode_actions["tutor"].setChecked(True)
        self._mode_group.triggered.connect(self._on_mode_picked)

        self._mode_btn = QToolButton()
        style_sidebar_chip(self._mode_btn)
        self._mode_btn.setMenu(mode_menu)
        self._mode_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._mode_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._mode_btn.setIconSize(QSize(20, 16))
        self._sync_mode_button("tutor")

        self._think_btn = QPushButton("思考")
        style_sidebar_chip(self._think_btn)
        self._think_btn.setCheckable(True)
        self._think_btn.setChecked(True)
        self._think_btn.setToolTip("开启后模型会先推理再回答；关闭可加快回复")
        self._think_btn.toggled.connect(self.thinking_toggled.emit)

        self._voice_status = QLabel("")
        self._voice_status.setObjectName("voiceStatus")
        self._voice_status.setVisible(False)
        self._ptt = QPushButton("语音")
        style_sidebar_chip(self._ptt)
        self._ptt.setCheckable(True)
        self._ptt.setToolTip("开启语音辅导；再次点击结束。说话时可打断助手。")
        self._ptt.toggled.connect(self._on_voice_toggled)
        self._model_label = model = _ElidedLabel()
        model.setObjectName("modelLabel")
        self.set_model_name(model_name)
        self._send_btn = send_btn = QPushButton("↑")
        style_sidebar_chip(send_btn, compact=True)
        send_btn.setObjectName("chatSend")
        send_btn.setToolTip("发送")
        send_btn.setEnabled(False)
        send_btn.clicked.connect(self._on_send)
        self._input.textChanged.connect(lambda text: send_btn.setEnabled(bool(text.strip())))
        row.addWidget(self._mode_btn)
        row.addWidget(self._think_btn)
        row.addWidget(self._voice_status)
        row.addWidget(self._ptt)
        row.addWidget(model, stretch=1)
        row.addWidget(send_btn)
        composer_layout.addLayout(row)
        layout.addWidget(composer)
        self._scroll.viewport().installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if watched is self._scroll.viewport() and event.type() == QEvent.Type.Resize:
            self._sync_widths()
        return super().eventFilter(watched, event)

    def set_context_actions(self, actions: list[QAction]) -> None:
        self._context_actions = list(actions)

    def _context_actions_now(self) -> list[QAction]:
        return self._context_actions

    def set_input_placeholder(self, text: str) -> None:
        self._input.setPlaceholderText(text)

    def set_mode(self, mode: str) -> None:
        if mode not in self._mode_actions:
            return
        self._mode_group.blockSignals(True)
        self._mode_actions[mode].setChecked(True)
        self._mode_group.blockSignals(False)
        self._sync_mode_button(mode)

    def _sync_mode_button(self, mode: str) -> None:
        spec = next(item for item in _MODE_SPECS if item[0] == mode)
        _, icon_fn, label, tip = spec
        self._mode_btn.setText(f"{label} ▾")
        self._mode_btn.setIcon(with_label_gap(icon_fn(), 4))
        self._mode_btn.setToolTip(tip)
        self._placeholder.setText(_EMPTY_HINTS.get(mode, ""))

    def _on_mode_picked(self, action: QAction) -> None:
        mode = str(action.data())
        self._sync_mode_button(mode)
        self.mode_change_requested.emit(mode)

    def set_thinking_enabled(self, enabled: bool) -> None:
        self._think_btn.blockSignals(True)
        self._think_btn.setChecked(enabled)
        self._think_btn.blockSignals(False)

    def thinking_enabled(self) -> bool:
        return self._think_btn.isChecked()

    def set_model_name(self, name: str) -> None:
        self._model_label.set_full_text(name)
        self._model_label.setVisible(bool(name))

    def set_voice_checked(self, checked: bool) -> None:
        self._ptt.blockSignals(True)
        self._ptt.setChecked(checked)
        self._ptt.setText("结束" if checked else "语音")
        self._ptt.blockSignals(False)

    def set_voice_status(self, text: str) -> None:
        self._voice_status.setText(text)
        self._voice_status.setVisible(bool(text))

    def _on_voice_toggled(self, checked: bool) -> None:
        self._ptt.setText("结束" if checked else "语音")
        self.voice_session_toggled.emit(checked)

    def _on_send(self) -> None:
        text = self._input.text().strip()
        if text:
            self.append_user(text)
            self.send_requested.emit(text)
            self._input.clear()

    def append_user(self, text: str) -> None:
        self.end_assistant_stream()
        self._append_row("user", text)

    def append_assistant(self, text: str) -> None:
        self.end_assistant_stream()
        self._append_row("assistant", text)

    def append_system(self, text: str) -> None:
        index = self._insert_index()
        before = self._list.itemAt(index - 1).widget() if index > 0 else None
        if isinstance(before, _LogStrip):
            before.add_line(text)
            self._after_insert()
            return
        strip = _LogStrip()
        strip.set_max_width(self._lane_width())
        strip.add_line(text)
        self._place_before_stream(strip)

    def append_event(
        self, eyebrow: str, title: str, detail: str = "", tone: str = "info", stamp: str = ""
    ) -> None:
        """A card for something that happened on the bench (tone: ask/pass/fail/skip/info)."""
        self._place_before_stream(_EventRow(eyebrow, title, detail, tone, stamp))

    def set_assistant_name(self, name: str) -> None:
        self._assistant_name = name

    def _insert_index(self) -> int:
        if self._stream_open and self._stream_row is not None:
            return self._list.indexOf(self._stream_row)
        return self._list.count() - 1

    def _place_before_stream(self, row: QWidget) -> None:
        self._list.insertWidget(self._insert_index(), row)
        self._rows.append(row)
        self._after_insert()

    def begin_thinking(self) -> None:
        self.finish_thinking()
        self._insert_thinking()

    def append_thinking(self, text: str) -> None:
        if not text:
            return
        if self._thinking is None:
            self._insert_thinking()
        note = self._thinking
        if note is None:
            return
        note.append_text(text)
        self._scroll_bottom()

    def finish_thinking(self) -> None:
        note = self._thinking
        if note is None:
            return
        if not note.text().strip():
            self._remove_note(note)
            return
        if note.is_live():
            note.mark_done()

    def reopen_thinking(self) -> None:
        if self._thinking is None:
            self._insert_thinking()
            return
        self._thinking.mark_live()

    def discard_thinking(self) -> None:
        if self._thinking is not None:
            self._remove_note(self._thinking)

    def thinking_text(self) -> str:
        if self._thinking is None:
            return ""
        return self._thinking.text()

    def set_stream_text(self, text: str) -> None:
        if self._stream_row is not None:
            self._stream_row.set_text(text)

    def begin_assistant_stream(self) -> None:
        self.end_assistant_stream()
        self._stream_row = self._append_row("assistant", "")
        self._stream_open = True

    def append_assistant_chunk(self, text: str) -> None:
        if not text:
            return
        if not self._stream_open or self._stream_row is None:
            self._stream_row = self._append_row("assistant", text)
            self._stream_open = True
            return
        self._stream_row.set_text(self._stream_row.text() + text)
        self._scroll_bottom()

    def discard_assistant_stream(self) -> None:
        if not self._stream_open:
            return
        self._remove_stream_row()
        self._stream_open = False

    def end_assistant_stream(self) -> None:
        self.finish_thinking()
        if not self._stream_open:
            return
        self._stream_open = False
        if self._stream_row is not None and not self._stream_row.text():
            self._remove_stream_row()
        self._stream_row = None

    def _append_row(self, role: str, text: str) -> _MessageRow:
        row = _MessageRow(role, text, self._context_actions_now, self._assistant_name)
        row.set_max_width(self._lane_width())
        self._place(row)
        self._rows.append(row)
        self._after_insert()
        return row

    def _place(self, row: QWidget) -> None:
        self._list.insertWidget(self._list.count() - 1, row)

    def _insert_thinking(self) -> None:
        note = _ThinkingNote(self._assistant_name)
        note.set_max_width(self._lane_width())
        self._place(note)
        self._notes.append(note)
        self._thinking = note
        self._after_insert()

    def _remove_note(self, note: _ThinkingNote) -> None:
        if note in self._notes:
            self._notes.remove(note)
        if self._thinking is note:
            self._thinking = None
        self._list.removeWidget(note)
        note.deleteLater()
        self._after_remove()

    def _after_insert(self) -> None:
        self._placeholder.hide()
        self._relabel()
        self._scroll_bottom()

    def _after_remove(self) -> None:
        self._relabel()
        if not self._rows and not self._notes:
            self._placeholder.show()

    def _relabel(self) -> None:
        """Show the speaker line only where the assistant starts talking after someone or something else."""
        after_assistant = False
        for index in range(self._list.count()):
            widget = self._list.itemAt(index).widget()
            if widget is None or widget is self._placeholder:
                continue
            assistant_side = getattr(widget, "assistant_side", False)
            if assistant_side:
                widget.set_speaker_visible(not after_assistant)
            after_assistant = assistant_side

    def _remove_stream_row(self) -> None:
        row = self._stream_row
        self._stream_row = None
        if row is None:
            return
        if row in self._rows:
            self._rows.remove(row)
        self._list.removeWidget(row)
        row.deleteLater()
        self._after_remove()

    def _lane_width(self) -> int:
        margins = self._list.contentsMargins()
        return max(160, self._scroll.viewport().width() - margins.left() - margins.right())

    def _sync_widths(self) -> None:
        width = self._lane_width()
        for row in self._rows:
            row.set_max_width(width)
        for note in self._notes:
            note.set_max_width(width)

    def _scroll_bottom(self) -> None:
        QTimer.singleShot(0, self._snap_bottom)

    def _snap_bottom(self) -> None:
        bar = self._scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _open_area_menu(self, pos) -> None:
        menu = ThemedMenu(self._transcript)
        extra = self._context_actions
        if not extra:
            return
        for source in extra:
            action = QAction(source.icon(), source.text(), menu)
            action.setEnabled(source.isEnabled())
            if source.isCheckable():
                action.setCheckable(True)
                action.setChecked(source.isChecked())
                action.triggered.connect(lambda checked, src=source: src.setChecked(bool(checked)))
            else:
                action.triggered.connect(source.trigger)
            menu.addAction(action)
        menu.exec(self._transcript.mapToGlobal(pos))
