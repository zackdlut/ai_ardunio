from __future__ import annotations

from PySide6.QtCore import QEvent, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QGuiApplication, QPainter, QPainterPath, QPalette, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.chrome import RoundedCard, ThemedMenu
from ui.icons import bind_editor_menu, icon_copy
from ui.theme import COLOR_ACCENT, COLOR_BORDER, COLOR_SURFACE, COLOR_USER_BUBBLE

_Radii = tuple[float, float, float, float]


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
    def __init__(
        self,
        fill: str,
        border: str | None,
        radii: _Radii,
        rail: bool = False,
    ) -> None:
        super().__init__()
        self._fill = QColor(fill)
        self._border = QColor(border) if border else None
        self._radii = radii
        self._rail = rail
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setAutoFillBackground(False)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = _round_path(rect, self._radii)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._fill)
        painter.drawPath(path)
        if self._rail:
            painter.setClipPath(path)
            painter.fillRect(QRectF(rect.left(), rect.top(), 3.0, rect.height()), QColor(COLOR_ACCENT))
            painter.setClipping(False)
        if self._border is not None:
            painter.setPen(QPen(self._border))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)


class _ThinkingNote(QWidget):
    """Scratchpad that stays open while tokens arrive, then folds to one line."""

    def __init__(self) -> None:
        super().__init__()
        self._text = ""
        self._live = True
        self._expanded = True
        self.setAutoFillBackground(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 8, 2)
        layout.setSpacing(4)
        self._toggle = QPushButton("▾  思考中")
        self._toggle.setObjectName("thinkToggle")
        self._toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle.setFlat(True)
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
        layout.addWidget(self._toggle, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._body)
        self._sync_label()

    def text(self) -> str:
        return self._text

    def is_live(self) -> bool:
        return self._live

    def append_text(self, piece: str) -> None:
        self._text += piece
        self._body.setText(self._text)
        self._live = True
        self._expanded = True
        self._body.setVisible(True)
        self._sync_label()

    def mark_live(self) -> None:
        self._live = True
        self._expanded = True
        self._body.setVisible(bool(self._text))
        self._sync_label()

    def mark_done(self) -> None:
        if not self._live and not self._expanded:
            return
        self._live = False
        self._expanded = False
        self._body.setVisible(False)
        self._sync_label()

    def set_max_width(self, width: int) -> None:
        capped = max(160, width)
        self.setMaximumWidth(capped)
        self._body.setMaximumWidth(max(120, capped - 28))

    def _on_toggle(self) -> None:
        if self._live:
            return
        self._expanded = not self._expanded
        self._body.setVisible(self._expanded and bool(self._text.strip()))
        self._sync_label()

    def _sync_label(self) -> None:
        if self._live:
            self._toggle.setText("▾  思考中")
            self._toggle.setToolTip("思考还在继续")
            return
        if self._expanded:
            self._toggle.setText("▾  思考")
            self._toggle.setToolTip("收起思考")
            return
        self._toggle.setText("▸  思考")
        self._toggle.setToolTip("展开思考")


class _MessageRow(QWidget):
    def __init__(self, role: str, text: str, extras) -> None:
        super().__init__()
        self._role = role
        self._text = text
        self._extras = extras
        self.setAutoFillBackground(False)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._open_menu)

        if role == "system":
            row = QHBoxLayout(self)
            row.setContentsMargins(2, 2, 2, 2)
            row.setSpacing(6)
            dot = QLabel("●")
            dot.setObjectName("systemDot")
            body = QLabel(text)
            body.setObjectName("systemText")
            body.setWordWrap(True)
            body.setTextFormat(Qt.TextFormat.PlainText)
            body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            row.addWidget(dot, alignment=Qt.AlignmentFlag.AlignTop)
            row.addWidget(body, stretch=1)
            self._body = body
            self._bubble = None
            return

        if role == "user":
            bubble = _Bubble(COLOR_USER_BUBBLE, None, (16, 16, 6, 16))
            margins = (14, 10, 14, 10)
        else:
            bubble = _Bubble(COLOR_SURFACE, COLOR_BORDER, (16, 16, 16, 16), rail=True)
            margins = (16, 11, 14, 11)
        inner = QVBoxLayout(bubble)
        inner.setContentsMargins(*margins)
        body = QLabel(text)
        body.setObjectName("bubbleText")
        body.setWordWrap(True)
        body.setTextFormat(Qt.TextFormat.PlainText)
        body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        body.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        body.setMinimumHeight(18)
        inner.addWidget(body)
        self._body = body
        self._bubble = bubble

        role_label = QLabel("你" if role == "user" else "助教")
        role_label.setObjectName("bubbleRole")
        column = QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)
        column.addWidget(role_label, alignment=Qt.AlignmentFlag.AlignRight if role == "user" else Qt.AlignmentFlag.AlignLeft)
        column.addWidget(bubble)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 2, 0, 2)
        outer.setSpacing(0)
        if role == "user":
            outer.addStretch(1)
            outer.addLayout(column)
        else:
            outer.addLayout(column)
            outer.addStretch(1)

    def text(self) -> str:
        return self._text

    def set_text(self, text: str) -> None:
        self._text = text
        self._body.setText(text)

    def set_max_width(self, width: int) -> None:
        if self._bubble is None:
            return
        capped = max(120, width)
        self._bubble.setMaximumWidth(capped)
        inset = 32 if self._role == "assistant" else 28
        self._body.setMaximumWidth(max(80, capped - inset))

    def _open_menu(self, pos) -> None:
        menu = ThemedMenu(self)
        copy_action = menu.addAction(icon_copy(), "复制")
        copy_action.triggered.connect(self._copy)
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
        selected = self._body.selectedText()
        QGuiApplication.clipboard().setText(selected or self._text)


class ChatWidget(QWidget):
    send_requested = Signal(str)
    voice_session_toggled = Signal(bool)

    def __init__(self, model_name: str = "") -> None:
        super().__init__()
        self._stream_open = False
        self._stream_row: _MessageRow | None = None
        self._rows: list[_MessageRow] = []
        self._thinking: _ThinkingNote | None = None
        self._notes: list[_ThinkingNote] = []
        self._context_actions: list[QAction] = []

        self.setObjectName("chatSidebar")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(8)

        self._scroll = QScrollArea()
        self._scroll.setObjectName("chatScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        viewport = self._scroll.viewport()
        viewport.setAutoFillBackground(True)
        palette = viewport.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(COLOR_SURFACE))
        viewport.setPalette(palette)

        self._transcript = QWidget()
        self._transcript.setAutoFillBackground(False)
        self._transcript.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._transcript.customContextMenuRequested.connect(self._open_area_menu)
        self._list = QVBoxLayout(self._transcript)
        self._list.setContentsMargins(0, 0, 4, 0)
        self._list.setSpacing(8)
        self._placeholder = QLabel("提问会显示在这里")
        self._placeholder.setObjectName("chatPlaceholder")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._list.addWidget(self._placeholder)
        self._list.addStretch(1)
        self._scroll.setWidget(self._transcript)
        layout.addWidget(self._scroll, stretch=1)

        composer = RoundedCard(14)
        composer.setObjectName("composer")
        composer_layout = QVBoxLayout(composer)
        composer_layout.setContentsMargins(12, 10, 12, 10)
        composer_layout.setSpacing(6)

        self._input = QLineEdit()
        self._input.setObjectName("composerInput")
        self._input.setPlaceholderText("问问当前这一步…")
        self._input.returnPressed.connect(self._on_send)
        bind_editor_menu(self._input, self._context_actions_now)
        composer_layout.addWidget(self._input)

        row = QHBoxLayout()
        row.setSpacing(8)
        self._voice_status = QLabel("")
        self._voice_status.setObjectName("voiceStatus")
        self._voice_status.setVisible(False)
        self._ptt = QPushButton("语音辅导")
        self._ptt.setObjectName("voiceButton")
        self._ptt.setCheckable(True)
        self._ptt.setToolTip("点一下开始语音辅导，再说一次结束。开口即可打断老师。")
        self._ptt.toggled.connect(self._on_voice_toggled)
        model = QLabel(model_name)
        model.setObjectName("modelLabel")
        model.setVisible(bool(model_name))
        send_btn = QPushButton("↑")
        send_btn.setObjectName("sendButton")
        send_btn.setToolTip("发送")
        send_btn.setFixedSize(32, 32)
        send_btn.clicked.connect(self._on_send)
        row.addWidget(self._voice_status)
        row.addWidget(self._ptt)
        row.addStretch(1)
        row.addWidget(model)
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

    def set_voice_checked(self, checked: bool) -> None:
        self._ptt.blockSignals(True)
        self._ptt.setChecked(checked)
        self._ptt.setText("结束辅导" if checked else "语音辅导")
        self._ptt.blockSignals(False)

    def set_voice_status(self, text: str) -> None:
        self._voice_status.setText(text)
        self._voice_status.setVisible(bool(text))

    def _on_voice_toggled(self, checked: bool) -> None:
        self._ptt.setText("结束辅导" if checked else "语音辅导")
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
        row = self._make_row("system", text)
        if self._stream_open and self._stream_row is not None:
            index = self._list.indexOf(self._stream_row)
            self._list.insertWidget(index, row)
        else:
            self._place(row)
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
        row = self._make_row(role, text)
        self._place(row)
        self._rows.append(row)
        self._after_insert()
        return row

    def _make_row(self, role: str, text: str) -> _MessageRow:
        row = _MessageRow(role, text, self._context_actions_now)
        row.set_max_width(self._bubble_width())
        return row

    def _place(self, row: QWidget) -> None:
        self._list.insertWidget(self._list.count() - 1, row)

    def _insert_thinking(self) -> None:
        note = _ThinkingNote()
        note.set_max_width(self._bubble_width())
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
        if not self._rows and not self._notes:
            self._placeholder.show()

    def _after_insert(self) -> None:
        self._placeholder.hide()
        self._scroll_bottom()

    def _remove_stream_row(self) -> None:
        row = self._stream_row
        self._stream_row = None
        if row is None:
            return
        if row in self._rows:
            self._rows.remove(row)
        self._list.removeWidget(row)
        row.deleteLater()
        if not self._rows and not self._notes:
            self._placeholder.show()

    def _bubble_width(self) -> int:
        return max(140, int(self._scroll.viewport().width() * 0.8))

    def _sync_widths(self) -> None:
        width = self._bubble_width()
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
