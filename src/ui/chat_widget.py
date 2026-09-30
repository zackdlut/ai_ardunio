from __future__ import annotations

import html

from PySide6.QtCore import Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


def _esc(text: str) -> str:
    return html.escape(text).replace("\n", "<br/>")


class ChatWidget(QWidget):
    send_requested = Signal(str)
    voice_ptt_pressed = Signal()
    voice_ptt_released = Signal()

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QLabel("与助手对话")
        header.setObjectName("sectionEyebrow")
        layout.addWidget(header)

        self._log = QTextEdit()
        self._log.setObjectName("chatLog")
        self._log.setReadOnly(True)
        self._log.setPlaceholderText("提问会显示在这里…")
        layout.addWidget(self._log, stretch=1)

        row = QHBoxLayout()
        row.setSpacing(8)
        self._input = QLineEdit()
        self._input.setPlaceholderText("例如：当前到哪一步？")
        send_btn = QPushButton("发送")
        send_btn.setObjectName("primaryButton")
        send_btn.clicked.connect(self._on_send)
        self._input.returnPressed.connect(self._on_send)

        self._ptt = QPushButton("按住说话")
        self._ptt.setObjectName("voiceButton")
        self._ptt.setCheckable(True)
        self._ptt.pressed.connect(self.voice_ptt_pressed.emit)
        self._ptt.released.connect(self.voice_ptt_released.emit)

        row.addWidget(self._input, stretch=1)
        row.addWidget(send_btn)
        row.addWidget(self._ptt)
        layout.addLayout(row)

    def _append_html(self, block: str) -> None:
        self._log.moveCursor(QTextCursor.MoveOperation.End)
        self._log.insertHtml(block)
        self._log.insertPlainText("\n")
        self._log.moveCursor(QTextCursor.MoveOperation.End)

    def _on_send(self) -> None:
        text = self._input.text().strip()
        if text:
            self.append_user(text)
            self.send_requested.emit(text)
            self._input.clear()

    def append_user(self, text: str) -> None:
        self._append_html(
            f"<p style='margin:6px 0 10px 48px;'>"
            f"<span style='color:#8A9A90;font-size:10px;'>你</span><br/>"
            f"<span style='background:#1E3328;border:1px solid #2A4A3A;"
            f"border-radius:6px;padding:6px 10px;'>{_esc(text)}</span></p>"
        )

    def append_assistant(self, text: str) -> None:
        self._append_html(
            f"<p style='margin:6px 0 10px 0;'>"
            f"<span style='color:#C49A6C;font-size:10px;'>助手</span><br/>"
            f"<span style='background:#1A1F1C;border-left:3px solid #008184;"
            f"padding:6px 10px 6px 12px;'>{_esc(text)}</span></p>"
        )

    def append_system(self, text: str) -> None:
        self._append_html(
            f"<p style='margin:4px 0;color:#6B7A72;font-family:Consolas,monospace;"
            f"font-size:11px;'>↳ {_esc(text)}</p>"
        )
