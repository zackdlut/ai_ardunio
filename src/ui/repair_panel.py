"""Sidebar for repair mode: case header, checklist, documents and the conclusion."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QModelIndex, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from ui.chrome import RailCard, ThemedMenu
from ui.icons import bind_editor_menu, icon_clear
from ui.theme import (
    COLOR_ACCENT,
    COLOR_ACCENT_INK,
    COLOR_ACCENT_SOFT,
    COLOR_CAUTION,
    COLOR_FAULT,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    COLOR_TRACK_EMPTY,
    FONT_MONO_FAMILIES,
)

STATUS_TEXT = {"pending": "待查", "pass": "正常", "fail": "异常", "skip": "跳过"}
_STATUS_COLOR = {
    "pending": COLOR_TEXT_MUTED,
    "pass": COLOR_ACCENT_INK,
    "fail": COLOR_FAULT,
    "skip": COLOR_CAUTION,
}
_STEP_COLOR = {
    "pending": COLOR_TRACK_EMPTY,
    "done": "#8ACFAF",
    "failed": COLOR_FAULT,
    "skipped": COLOR_CAUTION,
}
_STEP_WORD = {"pending": "待做", "done": "已完成", "failed": "未通过", "skipped": "已跳过"}
_TRACK_CHIP = {
    "tracking": ("跟踪中", COLOR_ACCENT_INK),
    "weak": ("跟踪弱", COLOR_CAUTION),
    "lost": ("跟丢了", COLOR_FAULT),
    "frozen": ("已定格", COLOR_CAUTION),
    "off": ("跟踪关", COLOR_TEXT_MUTED),
    "idle": ("未锚定", COLOR_TEXT_MUTED),
}
_ROLE_STATUS = Qt.ItemDataRole.UserRole + 1
_ROLE_SEQ = Qt.ItemDataRole.UserRole + 2
_ROLE_MEASURED = Qt.ItemDataRole.UserRole + 3
_ROW_HEIGHT = 26
_PANEL_MAX_HEIGHT = 470
_DOC_FILTER = "资料 (*.pdf *.png *.jpg *.jpeg *.bmp *.webp);;PDF (*.pdf);;图片 (*.png *.jpg *.jpeg *.bmp *.webp)"


class NewCaseDialog(QDialog):
    def __init__(self, device: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("settingsDialog")
        self.setWindowTitle("新建维修单")
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self._device = QLineEdit(device)
        self._device.setPlaceholderText("如：小米电饭煲 MFB1501、自制 STM32 板")
        self._symptom = QPlainTextEdit()
        self._symptom.setPlaceholderText("故障现象与经过：不开机、指示灯不亮、进过水、换过什么件…")
        self._symptom.setFixedHeight(96)
        bind_editor_menu(self._device)
        bind_editor_menu(self._symptom)
        form.addRow("设备", self._device)
        form.addRow("现象", self._symptom)
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("开始维修")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> tuple[str, str]:
        return self._device.text().strip(), self._symptom.toPlainText().strip()


def _mono(size: int, bold: bool = False) -> QFont:
    font = QFont()
    font.setFamilies(FONT_MONO_FAMILIES)
    font.setPixelSize(size)
    font.setBold(bold)
    return font


class _StepTrack(QWidget):
    """One segment per guide step, colored by its state; the current one stands taller."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._states: list[str] = []
        self._current = 0
        self.setFixedHeight(10)

    def set_steps(self, states: list[str], current: int) -> None:
        self._states = list(states)
        self._current = current
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        count = len(self._states)
        if not count:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        gap = 4.0
        width = (self.width() - gap * (count - 1)) / count
        for i, state in enumerate(self._states):
            current = i == self._current
            height = 6.0 if current else 4.0
            color = QColor(_STEP_COLOR.get(state, COLOR_TRACK_EMPTY))
            if current and state == "pending":
                color = QColor(COLOR_ACCENT)
            painter.setBrush(color)
            rect = QRectF(i * (width + gap), (self.height() - height) / 2, width, height)
            painter.drawRoundedRect(rect, height / 2, height / 2)
        painter.end()


class _CheckDelegate(QStyledItemDelegate):
    """Checklist row: status mark, mono sequence number, title, and the reading on the right."""

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        return QSize(40, _ROW_HEIGHT)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        status = index.data(_ROLE_STATUS)
        if status is None:
            super().paint(painter, option, index)
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(option.rect).adjusted(1, 1, -1, -1)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        if selected:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(COLOR_ACCENT_SOFT))
            painter.drawRoundedRect(rect, 6, 6)
        color = QColor(_STATUS_COLOR.get(status, COLOR_TEXT_MUTED))
        center = QPointF(rect.left() + 12, rect.center().y())
        self._mark(painter, center, status, color, selected)

        x = rect.left() + 24
        painter.setFont(_mono(11))
        painter.setPen(QColor(COLOR_TEXT_MUTED))
        seq = f"{int(index.data(_ROLE_SEQ) or 0):02d}"
        painter.drawText(QRectF(x, rect.top(), 22, rect.height()), int(Qt.AlignmentFlag.AlignVCenter), seq)
        x += 24

        measured = str(index.data(_ROLE_MEASURED) or "")
        right = rect.right() - 6
        if measured:
            font = _mono(12, bold=True)
            painter.setFont(font)
            width = min(painter.fontMetrics().horizontalAdvance(measured) + 4, int(rect.width() * 0.35))
            painter.setPen(color)
            box = QRectF(right - width, rect.top(), width, rect.height())
            text = painter.fontMetrics().elidedText(measured, Qt.TextElideMode.ElideRight, int(width))
            painter.drawText(box, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight), text)
            right -= width + 10

        font = QFont(option.font)
        font.setPixelSize(12)
        font.setBold(selected)
        font.setStrikeOut(status == "skip")
        painter.setFont(font)
        painter.setPen(QColor(COLOR_TEXT_MUTED if status == "skip" else COLOR_TEXT))
        title = painter.fontMetrics().elidedText(
            str(index.data(Qt.ItemDataRole.DisplayRole) or ""), Qt.TextElideMode.ElideRight, int(right - x)
        )
        painter.drawText(QRectF(x, rect.top(), right - x, rect.height()), int(Qt.AlignmentFlag.AlignVCenter), title)
        painter.restore()

    @staticmethod
    def _mark(painter: QPainter, c: QPointF, status: str, color: QColor, current: bool) -> None:
        r = 6.0
        if status == "pending":
            painter.setPen(QPen(QColor(COLOR_ACCENT if current else COLOR_TEXT_MUTED), 1.6 if current else 1.2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(c, r, r)
            if current:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(COLOR_ACCENT))
                painter.drawEllipse(c, 2.6, 2.6)
            return
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(c, r, r)
        pen = QPen(QColor("#FFFFFF"), 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        if status == "pass":
            painter.drawPolyline(
                QPolygonF([c + QPointF(-2.8, 0.2), c + QPointF(-0.8, 2.3), c + QPointF(3.0, -2.2)])
            )
        elif status == "fail":
            painter.drawLine(c + QPointF(-2.3, -2.3), c + QPointF(2.3, 2.3))
            painter.drawLine(c + QPointF(2.3, -2.3), c + QPointF(-2.3, 2.3))
        else:
            painter.drawLine(c + QPointF(-2.8, 0), c + QPointF(2.8, 0))


class RepairPanel(QWidget):
    new_case_requested = Signal(str, str)
    result_reported = Signal(int, str, str)
    check_focused = Signal(int)
    files_chosen = Signal(list)
    document_removed = Signal(int)
    ifixit_requested = Signal()
    freeze_toggled = Signal(bool)
    clear_annotations_requested = Signal()
    realign_requested = Signal()
    guide_prev = Signal()
    guide_next = Signal()
    guide_verify = Signal()
    guide_end = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._case: dict[str, Any] | None = None
        self._guide: dict[str, Any] | None = None
        self._guide_index = 0
        self._verifying = False
        self._syncing = False
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setObjectName("chatScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Leaves the chat below at least half of the sidebar.
        scroll.setMaximumHeight(_PANEL_MAX_HEIGHT)
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        self._scroll, self._body = scroll, body
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.setSpacing(6)

        head = QHBoxLayout()
        head.setSpacing(6)
        eyebrow = QLabel("维修单")
        eyebrow.setObjectName("sectionEyebrow")
        self._case_no = QLabel("")
        self._case_no.setObjectName("readoutMuted")
        head.addWidget(eyebrow)
        head.addWidget(self._case_no)
        head.addStretch(1)
        layout.addLayout(head)

        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        self._title = QLabel("还没有维修单")
        self._title.setObjectName("stepTitle")
        self._title.setWordWrap(True)
        new_btn = QPushButton("新建")
        new_btn.setObjectName("primaryButton")
        new_btn.setToolTip("填写设备和故障现象，开始新的维修单")
        new_btn.clicked.connect(self._ask_new_case)
        title_row.addWidget(self._title, stretch=1)
        title_row.addWidget(new_btn, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addLayout(title_row)

        self._symptom = QLabel("点「新建」写下设备和现象，或者直接在下面对话里描述")
        self._symptom.setObjectName("stepHint")
        self._symptom.setWordWrap(True)
        layout.addWidget(self._symptom)

        tools = QHBoxLayout()
        tools.setSpacing(6)
        self._freeze = QPushButton("定格画面")
        self._freeze.setObjectName("voiceButton")
        self._freeze.setCheckable(True)
        self._freeze.setToolTip("定住当前画面，拿起板子测量时标注也不会错位")
        self._freeze.toggled.connect(self._on_freeze)
        clear_btn = QPushButton("清除标注")
        clear_btn.setObjectName("voiceButton")
        clear_btn.clicked.connect(self.clear_annotations_requested.emit)
        realign = QPushButton("重新对准")
        realign.setObjectName("voiceButton")
        realign.setToolTip("跟丢后把板子摆回原位，或以当前画面为准重新锚定图形")
        realign.clicked.connect(self.realign_requested.emit)
        self._track_chip = QLabel("")
        self._track_chip.setObjectName("trackChip")
        self._track_chip.setTextFormat(Qt.TextFormat.RichText)
        self._track_chip.setToolTip("图形是否贴住板子：跟踪中 / 跟踪弱 / 跟丢了 / 已定格")
        tools.addWidget(self._freeze)
        tools.addWidget(clear_btn)
        tools.addWidget(realign)
        tools.addStretch(1)
        tools.addWidget(self._track_chip, alignment=Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(tools)
        self.set_tracking("idle")

        self._guide_card = RailCard(rail=COLOR_ACCENT)
        self._guide_card.setObjectName("guideCard")
        card = QVBoxLayout(self._guide_card)
        card.setContentsMargins(14, 9, 12, 10)
        card.setSpacing(6)
        guide_head = QHBoxLayout()
        guide_head.setSpacing(6)
        guide_eyebrow = QLabel("操作指引")
        guide_eyebrow.setObjectName("sectionEyebrow")
        self._guide_name = QLabel("")
        self._guide_name.setObjectName("stepHint")
        self._guide_progress = QLabel("")
        self._guide_progress.setObjectName("readout")
        guide_head.addWidget(guide_eyebrow)
        guide_head.addWidget(self._guide_name, stretch=1)
        guide_head.addWidget(self._guide_progress)
        card.addLayout(guide_head)
        self._guide_track = _StepTrack()
        card.addWidget(self._guide_track)
        self._guide_title = QLabel("")
        self._guide_title.setObjectName("guideStepTitle")
        self._guide_title.setWordWrap(True)
        card.addWidget(self._guide_title)
        self._guide_text = QLabel("")
        self._guide_text.setObjectName("guideInstruction")
        self._guide_text.setWordWrap(True)
        self._guide_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        card.addWidget(self._guide_text)
        self._guide_verify = QLabel("")
        self._guide_verify.setObjectName("guideVerify")
        self._guide_verify.setWordWrap(True)
        self._guide_verify.setTextFormat(Qt.TextFormat.RichText)
        self._guide_verify.setToolTip("点「做好了」后，助手看画面按这条判断")
        card.addWidget(self._guide_verify)
        self._guide_feedback = QLabel("")
        self._guide_feedback.setObjectName("guideFeedback")
        self._guide_feedback.setWordWrap(True)
        self._guide_feedback.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        card.addWidget(self._guide_feedback)
        guide_buttons = QHBoxLayout()
        guide_buttons.setSpacing(6)
        self._prev_btn = QPushButton("上一步")
        self._prev_btn.setObjectName("voiceButton")
        self._prev_btn.clicked.connect(self.guide_prev.emit)
        self._next_btn = QPushButton("跳过")
        self._next_btn.setObjectName("voiceButton")
        self._next_btn.setToolTip("不经确认直接进入下一步，这一步记为跳过")
        self._next_btn.clicked.connect(self.guide_next.emit)
        self._verify_btn = QPushButton("做好了，帮我确认")
        self._verify_btn.setObjectName("primaryButton")
        self._verify_btn.setToolTip("把当前画面发给助手，判断这一步是否完成（也可以竖起大拇指）")
        self._verify_btn.clicked.connect(self.guide_verify.emit)
        end_btn = QPushButton("结束")
        end_btn.setObjectName("voiceButton")
        end_btn.clicked.connect(self.guide_end.emit)
        guide_buttons.addWidget(self._prev_btn)
        guide_buttons.addWidget(self._next_btn)
        guide_buttons.addWidget(self._verify_btn, stretch=1)
        guide_buttons.addWidget(end_btn)
        card.addLayout(guide_buttons)
        self._guide_card.hide()
        layout.addWidget(self._guide_card)

        checks_head = QHBoxLayout()
        checks_eyebrow = QLabel("检查清单")
        checks_eyebrow.setObjectName("sectionEyebrow")
        self._progress = QLabel("")
        self._progress.setObjectName("modelLabel")
        checks_head.addWidget(checks_eyebrow)
        checks_head.addStretch(1)
        checks_head.addWidget(self._progress)
        layout.addLayout(checks_head)

        self._checks = QListWidget()
        self._checks.setObjectName("checkList")
        self._checks.setItemDelegate(_CheckDelegate(self._checks))
        self._checks.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._checks.currentItemChanged.connect(self._on_check_selected)
        layout.addWidget(self._checks)

        self._detail = QLabel("")
        self._detail.setObjectName("stepHint")
        self._detail.setWordWrap(True)
        self._detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._detail)

        report = QHBoxLayout()
        report.setSpacing(6)
        self._measured = QLineEdit()
        self._measured.setObjectName("composerInput")
        self._measured.setPlaceholderText("实测值，如 0.2V / 短路")
        self._measured.returnPressed.connect(lambda: self._report("fail"))
        bind_editor_menu(self._measured)
        report.addWidget(self._measured, stretch=1)
        self._report_buttons: list[QPushButton] = []
        for status, text in (("pass", "正常"), ("fail", "异常"), ("skip", "跳过")):
            btn = QPushButton(text)
            btn.setObjectName("voiceButton")
            btn.clicked.connect(lambda _=False, s=status: self._report(s))
            report.addWidget(btn)
            self._report_buttons.append(btn)
        layout.addLayout(report)

        docs_head = QHBoxLayout()
        docs_eyebrow = QLabel("资料")
        docs_eyebrow.setObjectName("sectionEyebrow")
        upload = QPushButton("上传")
        upload.setObjectName("voiceButton")
        upload.setToolTip("上传原理图、维修手册（PDF 或图片）")
        upload.clicked.connect(self._choose_files)
        ifixit = QPushButton("查 iFixit")
        ifixit.setObjectName("voiceButton")
        ifixit.setToolTip("让助手去 iFixit 查这台设备的拆机与维修指南")
        ifixit.clicked.connect(self.ifixit_requested.emit)
        self._ifixit_btn = ifixit
        docs_head.addWidget(docs_eyebrow)
        docs_head.addStretch(1)
        docs_head.addWidget(upload)
        docs_head.addWidget(ifixit)
        layout.addLayout(docs_head)

        self._docs = QListWidget()
        self._docs.setObjectName("checkList")
        self._docs.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._docs.customContextMenuRequested.connect(self._doc_menu)
        layout.addWidget(self._docs)

        self._conclusion = QLabel("")
        self._conclusion.setObjectName("conclusionCard")
        self._conclusion.setWordWrap(True)
        self._conclusion.setTextFormat(Qt.TextFormat.RichText)
        self._conclusion.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._conclusion.hide()
        layout.addWidget(self._conclusion)
        self._sync_report_enabled()

    # -- public ---------------------------------------------------------------

    @property
    def device(self) -> str:
        return (self._case or {}).get("device") or ""

    def set_ifixit_enabled(self, enabled: bool) -> None:
        self._ifixit_btn.setVisible(enabled)

    def set_frozen(self, frozen: bool) -> None:
        self._freeze.blockSignals(True)
        self._freeze.setChecked(frozen)
        self._freeze.setText("取消定格" if frozen else "定格画面")
        self._freeze.blockSignals(False)

    def set_tracking(self, status: str) -> None:
        text, color = _TRACK_CHIP.get(status, _TRACK_CHIP["idle"])
        self._track_chip.setText(f'<span style="color:{color}">●</span>&nbsp;{text}')
        self._track_chip.setStyleSheet(f"color: {color};")

    def set_verifying(self, busy: bool) -> None:
        self._verifying = busy
        self._verify_btn.setEnabled(not busy)
        self._verify_btn.setText("正在看画面…" if busy else "做好了，帮我确认")
        self._sync_guide_buttons()

    def show_guide(self, guide: dict[str, Any] | None) -> None:
        self._guide = guide
        steps = (guide or {}).get("steps") or []
        if not guide or guide.get("status") != "active" or not steps:
            self._guide_card.hide()
            self._fit_height()
            return
        index = min(max(int(guide.get("current_step") or 0), 0), len(steps) - 1)
        step = steps[index]
        state = step.get("status") or "pending"
        self._guide_index = index
        self._guide_name.setText(guide.get("title") or "")
        self._guide_progress.setText(f"{index + 1:02d} / {len(steps):02d}")
        self._guide_track.set_steps([s.get("status") or "pending" for s in steps], index)
        self._guide_track.setToolTip(
            "\n".join(
                f"{i + 1}. {s.get('title') or ''}  {_STEP_WORD.get(s.get('status') or 'pending', '')}"
                for i, s in enumerate(steps)
            )
        )
        self._guide_title.setText(step.get("title") or "")
        self._guide_text.setText(step.get("instruction") or "")
        self._guide_text.setVisible(bool(step.get("instruction")))
        verify = step.get("verify") or ""
        self._guide_verify.setText(
            f'<span style="color:{COLOR_TEXT_MUTED}; font-weight:600">确认标准</span>&nbsp;&nbsp;{_esc(verify)}'
        )
        self._guide_verify.setVisible(bool(verify))
        feedback = step.get("feedback") or ""
        failed = state == "failed"
        self._guide_feedback.setProperty("tone", "fail" if failed else "ok")
        self._guide_feedback.style().unpolish(self._guide_feedback)
        self._guide_feedback.style().polish(self._guide_feedback)
        self._guide_feedback.setText(("还没好：" if failed else "助手：") + feedback)
        self._guide_feedback.setVisible(bool(feedback))
        self._guide_card.set_tone(COLOR_FAULT if failed else COLOR_ACCENT)
        self._next_btn.setText("跳过并结束" if index == len(steps) - 1 else "跳过")
        self._sync_guide_buttons()
        self._guide_card.show()
        self._fit_height()

    def show_case(self, case: dict[str, Any] | None) -> None:
        self._case = case
        if not case:
            self._case_no.setText("")
            self._title.setText("还没有维修单")
            self._symptom.setText("点「新建」写下设备和现象，或者直接在下面对话里描述")
            self._fill_checks([], None)
            self._conclusion.hide()
        else:
            self._case_no.setText(f"#{case['id']:03d}")
            self._title.setText(case.get("device") or "设备未填写")
            self._symptom.setText(case.get("symptom") or "故障现象未填写")
            self._fill_checks(case.get("checks") or [], case.get("current_check_id"))
            self._show_conclusion(case)
        self._fit_height()

    def show_documents(self, docs: list[dict[str, Any]]) -> None:
        self._docs.clear()
        for doc in docs:
            tag = f"{doc['pages']} 页" if doc["kind"] == "pdf" else "图片"
            if doc["kind"] == "pdf" and not doc["has_text"]:
                tag += "，扫描件"
            item = QListWidgetItem(f"[{doc['id']}] {doc['name']}（{tag}）")
            item.setData(Qt.ItemDataRole.UserRole, doc["id"])
            item.setToolTip(doc.get("path") or "")
            self._docs.addItem(item)
        if not docs:
            empty = QListWidgetItem("还没有资料，可上传原理图或维修手册")
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            self._docs.addItem(empty)
        _fit_rows(self._docs, max_rows=3)
        self._fit_height()

    def selected_check(self) -> dict[str, Any] | None:
        item = self._checks.currentItem()
        if item is None:
            return None
        check_id = item.data(Qt.ItemDataRole.UserRole)
        for check in (self._case or {}).get("checks") or []:
            if check["id"] == check_id:
                return check
        return None

    # -- internals ------------------------------------------------------------

    def _fill_checks(self, checks: list[dict[str, Any]], current_id: int | None) -> None:
        self._syncing = True
        self._checks.clear()
        done = 0
        current_row = -1
        for row, check in enumerate(checks):
            status = check.get("status") or "pending"
            if status != "pending":
                done += 1
            item = QListWidgetItem(check["title"])
            item.setData(Qt.ItemDataRole.UserRole, check["id"])
            item.setData(_ROLE_STATUS, status)
            item.setData(_ROLE_SEQ, check["seq"])
            item.setData(_ROLE_MEASURED, check.get("measured") or "")
            tip = [f"{check['seq']}. {check['title']}  [{STATUS_TEXT.get(status, status)}]", check.get("instruction") or ""]
            if check.get("expected"):
                tip.append(f"预期：{check['expected']}")
            item.setToolTip("\n".join(t for t in tip if t))
            self._checks.addItem(item)
            if check["id"] == current_id:
                current_row = row
        if current_row >= 0:
            self._checks.setCurrentRow(current_row)
        _fit_rows(self._checks, max_rows=6)
        self._progress.setText(f"{done} / {len(checks)}" if checks else "")
        self._syncing = False
        self._show_detail(self.selected_check())
        self._sync_report_enabled()

    def _show_detail(self, check: dict[str, Any] | None) -> None:
        if not check:
            self._detail.setText("助手了解故障后会在这里列出要检查的项目" if not self._checks.count() else "")
            return
        parts = [check.get("instruction") or ""]
        if check.get("expected"):
            parts.append(f"预期：{check['expected']}")
        if check.get("note"):
            parts.append(f"备注：{check['note']}")
        self._detail.setText("\n".join(p for p in parts if p))

    def _show_conclusion(self, case: dict[str, Any]) -> None:
        diagnosis = (case.get("diagnosis") or "").strip()
        if not diagnosis:
            self._conclusion.hide()
            return
        html = [f"<b>结论</b><br>{_esc(diagnosis)}"]
        actions = case.get("actions") or []
        if actions:
            html.append("<b>处置</b><br>" + "<br>".join(f"{i}. {_esc(a)}" for i, a in enumerate(actions, 1)))
        if case.get("safety_notes"):
            html.append(f"<b>安全</b><br>{_esc(case['safety_notes'])}")
        self._conclusion.setText("<br><br>".join(html))
        self._conclusion.show()

    def _on_check_selected(self, current: QListWidgetItem | None, _previous) -> None:
        check = self.selected_check()
        self._show_detail(check)
        self._sync_report_enabled()
        if self._syncing or current is None or check is None:
            return
        self.check_focused.emit(int(check["id"]))

    def _fit_height(self) -> None:
        """Grow to the content up to the cap; QScrollArea's own hint stops short of it."""
        self._body.layout().activate()
        self._scroll.setMinimumHeight(min(self._body.sizeHint().height() + 2, _PANEL_MAX_HEIGHT))

    def _sync_guide_buttons(self) -> None:
        self._prev_btn.setEnabled(self._guide_index > 0 and not self._verifying)
        self._next_btn.setEnabled(not self._verifying)

    def _sync_report_enabled(self) -> None:
        enabled = self.selected_check() is not None
        self._measured.setEnabled(enabled)
        for btn in self._report_buttons:
            btn.setEnabled(enabled)

    def _report(self, status: str) -> None:
        check = self.selected_check()
        if check is None:
            return
        measured = self._measured.text().strip()
        self._measured.clear()
        self.result_reported.emit(int(check["id"]), status, measured)

    def _ask_new_case(self) -> None:
        dialog = NewCaseDialog(self.device, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        device, symptom = dialog.values()
        self.new_case_requested.emit(device, symptom)

    def _choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "上传原理图或维修手册", "", _DOC_FILTER)
        if paths:
            self.files_chosen.emit(list(paths))

    def _doc_menu(self, pos) -> None:
        item = self._docs.itemAt(pos)
        if item is None or item.data(Qt.ItemDataRole.UserRole) is None:
            return
        menu: QMenu = ThemedMenu(self)
        remove = menu.addAction(icon_clear(), "移除这份资料")
        doc_id = int(item.data(Qt.ItemDataRole.UserRole))
        remove.triggered.connect(lambda: self.document_removed.emit(doc_id))
        menu.exec(self._docs.mapToGlobal(pos))

    def _on_freeze(self, checked: bool) -> None:
        self._freeze.setText("取消定格" if checked else "定格画面")
        self.freeze_toggled.emit(checked)


def _fit_rows(widget: QListWidget, max_rows: int) -> None:
    rows = min(max(widget.count(), 1), max_rows)
    row_h = widget.sizeHintForRow(0) if widget.count() else 22
    widget.setFixedHeight(rows * max(row_h, 20) + 12)


def _esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )
