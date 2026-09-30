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
    voice_session_toggled = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self._messages: list[tuple[str, str]] = []
        self._stream_open = False

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

        self._voice_status = QLabel("")
        self._voice_status.setObjectName("voiceStatus")
        self._voice_status.setMinimumWidth(108)

        self._ptt = QPushButton("语音辅导")
        self._ptt.setObjectName("voiceButton")
        self._ptt.setCheckable(True)
        self._ptt.setToolTip("点一下开始语音辅导，再说一次结束。开口即可打断老师。")
        self._ptt.toggled.connect(self._on_voice_toggled)

        row.addWidget(self._input, stretch=1)
        row.addWidget(send_btn)
        row.addWidget(self._voice_status)
        row.addWidget(self._ptt)
        layout.addLayout(row)

    def set_voice_checked(self, checked: bool) -> None:
        self._ptt.blockSignals(True)
        self._ptt.setChecked(checked)
        self._ptt.setText("结束辅导" if checked else "语音辅导")
        self._ptt.blockSignals(False)

    def set_voice_status(self, text: str) -> None:
        self._voice_status.setText(text)

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
        self._messages.append(("user", text))
        self._rebuild()

    def append_assistant(self, text: str) -> None:
        self.end_assistant_stream()
        self._messages.append(("assistant", text))
        self._rebuild()

    def append_system(self, text: str) -> None:
        if self._stream_open and self._messages:
            stream = self._messages.pop()
            self._messages.append(("system", text))
            self._messages.append(stream)
            self._rebuild()
            return
        self._messages.append(("system", text))
        self._rebuild()

    def begin_assistant_stream(self) -> None:
        self.end_assistant_stream()
        self._messages.append(("assistant", ""))
        self._stream_open = True
        self._rebuild()

    def append_assistant_chunk(self, text: str) -> None:
        if not text:
            return
        if not self._stream_open:
            self._messages.append(("assistant", text))
            self._stream_open = True
        else:
            role, prev = self._messages[-1]
            self._messages[-1] = (role, prev + text)
        self._rebuild()

    def discard_assistant_stream(self) -> None:
        if not self._stream_open:
            return
        if self._messages and self._messages[-1][0] == "assistant":
            self._messages.pop()
        self._stream_open = False
        self._rebuild()

    def end_assistant_stream(self) -> None:
        if not self._stream_open:
            return
        self._stream_open = False
        if self._messages and self._messages[-1] == ("assistant", ""):
            self._messages.pop()
            self._rebuild()

    def _rebuild(self) -> None:
        self._log.setHtml("".join(self._html_for(role, text) for role, text in self._messages))
        self._log.moveCursor(QTextCursor.MoveOperation.End)

    def _html_for(self, role: str, text: str) -> str:
        if role == "user":
            return (
                f"<p style='margin:6px 0 10px 48px;'>"
                f"<span style='color:#8A9A90;font-size:10px;'>你</span><br/>"
                f"<span style='background:#1E3328;border:1px solid #2A4A3A;"
                f"border-radius:6px;padding:6px 10px;'>{_esc(text)}</span></p>"
            )
        if role == "assistant":
            return (
                f"<p style='margin:6px 0 10px 0;'>"
                f"<span style='color:#C49A6C;font-size:10px;'>助手</span><br/>"
                f"<span style='background:#1A1F1C;border-left:3px solid #008184;"
                f"padding:6px 10px 6px 12px;'>{_esc(text)}</span></p>"
            )
        return (
            f"<p style='margin:4px 0;color:#6B7A72;font-family:Consolas,monospace;"
            f"font-size:11px;'>↳ {_esc(text)}</p>"
        )
