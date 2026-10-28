"""Sidebar for repair mode: case header, checklist, documents and the conclusion."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
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
    QVBoxLayout,
    QWidget,
)

from ui.chrome import ThemedMenu
from ui.icons import bind_editor_menu, icon_clear
from ui.theme import COLOR_ACCENT_INK, COLOR_TEXT, COLOR_TEXT_MUTED

STATUS_TEXT = {"pending": "待查", "pass": "正常", "fail": "异常", "skip": "跳过"}
_STATUS_COLOR = {
    "pending": COLOR_TEXT,
    "pass": COLOR_ACCENT_INK,
    "fail": "#B3261E",
    "skip": COLOR_TEXT_MUTED,
}
_PANEL_MAX_HEIGHT = 360
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


class RepairPanel(QWidget):
    new_case_requested = Signal(str, str)
    result_reported = Signal(int, str, str)
    check_focused = Signal(int)
    files_chosen = Signal(list)
    document_removed = Signal(int)
    ifixit_requested = Signal()
    freeze_toggled = Signal(bool)
    clear_annotations_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._case: dict[str, Any] | None = None
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
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.setSpacing(6)

        eyebrow = QLabel("维修单")
        eyebrow.setObjectName("sectionEyebrow")
        layout.addWidget(eyebrow)

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
        tools.addWidget(self._freeze)
        tools.addWidget(clear_btn)
        tools.addStretch(1)
        layout.addLayout(tools)

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

    def show_case(self, case: dict[str, Any] | None) -> None:
        self._case = case
        if not case:
            self._title.setText("还没有维修单")
            self._symptom.setText("点「新建」写下设备和现象，或者直接在下面对话里描述")
            self._fill_checks([], None)
            self._conclusion.hide()
            return
        self._title.setText(f"#{case['id']} · {case.get('device') or '设备未填写'}")
        self._symptom.setText(case.get("symptom") or "故障现象未填写")
        self._fill_checks(case.get("checks") or [], case.get("current_check_id"))
        self._show_conclusion(case)

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
            text = f"{check['seq']}. {check['title']}  [{STATUS_TEXT.get(status, status)}]"
            if check.get("measured"):
                text += f"  {check['measured']}"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, check["id"])
            item.setForeground(QColor(_STATUS_COLOR.get(status, COLOR_TEXT)))
            tip = [check.get("instruction") or ""]
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
