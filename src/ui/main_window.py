from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt, QObject, QSize, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from agent.frame_image import encode_frame_data_url, fit_frame
from agent.prompts import REPAIR_SYSTEM_PROMPT
from agent.repair_context import build_repair_extra_system
from agent.repair_messages import (
    check_result_message,
    ifixit_message,
    new_case_message,
    verify_step_message,
)
from agent.repair_nudge import NUDGE_TEXT, needs_drawing
from agent.repair_tools import RepairContext, build_repair_handlers, build_repair_tools
from agent.tools import AgentContext, build_openai_tools, build_tool_handlers
from vision.marks import components_in_mark, make_region_mark, nearest_component
from agent.tutor_context import build_extra_system
from arduino.board_manager import list_boards
from config.settings import get_settings
from project.repository import create_session
from project.state_machine import ProjectStateMachine, load_project
from repair.annotations import sent_image_size
from overlay.engine import OverlayEngine
from repair.case_manager import CaseManager
from repair.debug_log import RepairDebugLog
from repair.documents import DocumentLibrary
from repair.geometry import warp_marks
from repair.ifixit import IFixitClient
from ui.camera_devices import list_cameras
from ui.camera_worker import CameraWorker
from ui.camera_widget import CameraWidget
from ui.chat_widget import ChatWidget
from ui.chrome import RoundedCard, ThemedMenu
from ui.repair_panel import STATUS_TEXT, RepairPanel
from ui.settings_dialog import CHAT_KEYS, REPAIR_KEYS, SettingsDialog
from ui.icons import (
    bind_editor_menu,
    icon_camera,
    icon_clear,
    icon_hardware,
    icon_local,
    icon_log,
    icon_next,
    icon_project,
    icon_refresh,
    icon_remote,
    icon_repair,
    icon_settings,
    icon_speaker,
    icon_stop,
    icon_upload,
    icon_view,
    icon_voice,
    with_label_gap,
)
from voice.asr import AsrBackend, create_asr
from voice.sentences import SentenceBuffer
from voice.session import VoiceSession
from voice.tts import create_tts, prepare_speech_text, spoken_text_from_reply

if TYPE_CHECKING:
    from agent.llm_client import LLMClient

_BACKGROUND_THREADS: list[QThread] = []


def _keep_alive(thread: QThread) -> None:
    """Keep a QThread alive after the window closes so Qt does not destroy it mid-run."""
    _BACKGROUND_THREADS.append(thread)

    def _release() -> None:
        try:
            _BACKGROUND_THREADS.remove(thread)
        except ValueError:
            pass

    thread.finished.connect(_release)


class _TtsBridge(QObject):
    idle = Signal()
    failed = Signal(str)


class ASRWorker(QThread):
    succeeded = Signal(int, str)
    failed = Signal(int, str)

    def __init__(self, asr, audio: np.ndarray, sample_rate: int, turn: int) -> None:
        super().__init__()
        self._asr = asr
        self._audio = audio
        self._sample_rate = sample_rate
        self._turn = turn

    def run(self) -> None:
        if self.isInterruptionRequested():
            return
        try:
            text = self._asr.transcribe_pcm(self._audio, self._sample_rate)
            if not self.isInterruptionRequested():
                self.succeeded.emit(self._turn, text)
        except Exception as exc:  # noqa: BLE001
            if not self.isInterruptionRequested():
                self.failed.emit(self._turn, str(exc))


class LLMWorker(QThread):
    delta = Signal(int, str)
    reasoning = Signal(int, str)
    revoke = Signal(int)
    succeeded = Signal(int, str)
    failed = Signal(int, str)

    def __init__(
        self,
        client: LLMClient,
        message: str,
        tools,
        handlers,
        history: list[dict[str, str]],
        extra_system: str,
        voice_mode: bool,
        turn: int,
        stream: bool,
        frame: np.ndarray | None = None,
        frame_max_side: int = 768,
        system_prompt: str | None = None,
        max_rounds: int = 6,
        drain_images=None,
        keep_preface: bool = False,
        options: dict | None = None,
    ) -> None:
        super().__init__()
        self._keep_preface = keep_preface
        # Passed through to LLMClient: voice_max_tokens, extra_body, follow_up.
        self._options = dict(options or {})
        self._client = client
        self._message = message
        self._tools = tools
        self._handlers = handlers
        self._history = history
        self._extra_system = extra_system
        self._voice_mode = voice_mode
        self._turn = turn
        self._stream = stream
        self._frame = frame
        self._frame_max_side = frame_max_side
        self._system_prompt = system_prompt
        self._max_rounds = max_rounds
        self._drain_images = drain_images

    def run(self) -> None:
        if self.isInterruptionRequested():
            return
        try:
            image_url = encode_frame_data_url(self._frame, self._frame_max_side)
            if self._stream:
                reply = self._client.chat_with_tools_stream(
                    self._message,
                    self._tools,
                    self._handlers,
                    extra_system=self._extra_system,
                    history=self._history,
                    voice_mode=self._voice_mode,
                    on_delta=lambda text: self.delta.emit(self._turn, text),
                    on_reasoning=lambda text: self.reasoning.emit(self._turn, text),
                    on_revoke=lambda: self.revoke.emit(self._turn),
                    should_stop=self.isInterruptionRequested,
                    image_url=image_url,
                    system_prompt=self._system_prompt,
                    max_rounds=self._max_rounds,
                    drain_images=self._drain_images,
                    keep_preface=self._keep_preface,
                    **self._options,
                )
            else:
                reply = self._client.chat_with_tools(
                    self._message,
                    self._tools,
                    self._handlers,
                    extra_system=self._extra_system,
                    history=self._history,
                    voice_mode=self._voice_mode,
                    should_stop=self.isInterruptionRequested,
                    image_url=image_url,
                    system_prompt=self._system_prompt,
                    max_rounds=self._max_rounds,
                    drain_images=self._drain_images,
                    keep_preface=self._keep_preface,
                    **self._options,
                )
            if not self.isInterruptionRequested():
                self.succeeded.emit(self._turn, reply)
        except Exception as exc:  # noqa: BLE001
            if not self.isInterruptionRequested():
                self.failed.emit(self._turn, str(exc))


class _RepairBridge(QObject):
    changed = Signal(str)


class DocIngestWorker(QThread):
    ingested = Signal(object)
    failed = Signal(str)

    def __init__(self, ctx: RepairContext, paths: list[str]) -> None:
        super().__init__()
        self._ctx = ctx
        self._paths = list(paths)

    def run(self) -> None:
        for path in self._paths:
            if self.isInterruptionRequested():
                return
            try:
                self.ingested.emit(self._ctx.ingest_document(Path(path)))
            except Exception as exc:  # noqa: BLE001
                self.failed.emit(f"{Path(path).name}: {exc}")


class BoardScanWorker(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def run(self) -> None:
        try:
            boards = list_boards()
        except Exception as exc:  # noqa: BLE001
            if not self.isInterruptionRequested():
                self.failed.emit(str(exc).strip())
            return
        if not self.isInterruptionRequested():
            self.succeeded.emit(boards)


class LLMConnectWorker(QThread):
    ready = Signal(object, bool, str)

    def run(self) -> None:
        try:
            from agent.llm_client import LLMClient

            if self.isInterruptionRequested():
                return
            client = LLMClient()
            ok, msg = client.health_check()
        except Exception as exc:  # noqa: BLE001
            if not self.isInterruptionRequested():
                self.ready.emit(
                    None,
                    False,
                    f"LLM 连接失败: {exc}（请检查 .env 里的 OPENAI_* 与代理设置）",
                )
            return
        if not self.isInterruptionRequested():
            self.ready.emit(client, ok, msg)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Arduino 学习助手 · AI Copilot")
        self.resize(1280, 840)

        self._settings = get_settings()
        settings = self._settings
        blink_path = settings.projects_dir / "blink.json"
        project = load_project(blink_path)
        session_id = create_session(project.id)
        self._state_machine = ProjectStateMachine(project, session_id)

        self._scene: dict = {"nodes": {}, "summary": {}}
        self._verify_ok = True
        self._verify_msg = ""
        self._user_marks: list[dict] = []
        self._mark_seq = 0
        self._vision_paused = False
        self._agent_ctx = AgentContext(
            state_machine=self._state_machine,
            get_scene=lambda: self._scene,
            get_detections_json=self._detections_json,
        )
        self._tools = build_openai_tools()
        self._handlers = build_tool_handlers(self._agent_ctx)

        self._mode = "tutor"
        self._repair_frozen = False
        self._repair_bridge = _RepairBridge(self)
        self._repair = RepairContext(
            cases=CaseManager(),
            library=DocumentLibrary(settings.repair_docs_dir),
            ifixit=self._make_ifixit(settings),
            box_format=settings.repair_box_format,
            on_change=self._repair_bridge.changed.emit,
            debug=self._make_repair_debug(settings),
        )
        self._repair_clean_view = False
        self._marks_overlay = OverlayEngine()
        latest = self._repair.cases.latest_open_case()
        if latest:
            self._repair.case_id = latest["id"]
            self._repair.load_guide()
        self._guide_seen = self._snapshot_guide(self._repair.guide)
        self._track_status = "idle"
        self._lost_notified = False
        self._repair_tools = build_repair_tools(settings.ifixit_enabled)
        self._repair_handlers = build_repair_handlers(self._repair)
        self._ingest_worker: DocIngestWorker | None = None

        self._llm: LLMClient | None = None
        self._tts = create_tts(self._settings)
        self._asr = create_asr(self._settings)
        self._asr_worker: ASRWorker | None = None
        self._llm_worker: LLMWorker | None = None
        self._board_worker: BoardScanWorker | None = None
        self._llm_connect_worker: LLMConnectWorker | None = None
        self._closing = False
        self._dialog_history: list[dict[str, str]] = []
        self._pending_user_text = ""
        self._pending_voice_mode = False
        self._turn = 0
        self._voice_active = False
        self._voice_phase = "idle"
        self._syncing_voice_btn = False
        self._sentence_buf = SentenceBuffer()
        self._voice_streamed = False
        self._answer_shown = False

        self._build_ui()
        self._camera_widget.marked.connect(self._on_camera_mark)
        self._camera_widget.region_marked.connect(self._on_camera_region)
        self._repair_bridge.changed.connect(self._on_repair_changed)
        panel = self._repair_panel
        panel.new_case_requested.connect(self._on_repair_new_case)
        panel.result_reported.connect(self._on_repair_result)
        panel.check_focused.connect(self._on_repair_focus)
        panel.files_chosen.connect(self._on_repair_files)
        panel.document_removed.connect(self._on_repair_remove_doc)
        panel.ifixit_requested.connect(self._on_repair_ifixit)
        panel.freeze_toggled.connect(self._on_repair_freeze)
        panel.clear_annotations_requested.connect(lambda: self._repair.set_annotations([]))
        panel.realign_requested.connect(self._on_repair_realign)
        panel.guide_prev.connect(self._repair.back)
        panel.guide_next.connect(self._on_guide_next)
        panel.guide_verify.connect(self._verify_guide_step)
        panel.guide_end.connect(self._on_guide_end)
        panel.set_ifixit_enabled(settings.ifixit_enabled)
        self._seed_camera_menu()
        self._update_step_label()
        self._chat.append_system(self._asr_mode_label())

        self._camera = CameraWorker(camera_index=self._selected_camera_index())
        self._camera.frame_ready.connect(self._camera_widget.show_frame)
        self._camera.scene_updated.connect(self._on_scene)
        self._camera.gesture_detected.connect(self._on_gesture)
        self._camera.status_message.connect(self._on_camera_status)
        self._camera.tracking_changed.connect(self._on_tracking_changed)
        self._camera.configure_tracking(
            settings.repair_tracking,
            settings.repair_track_features,
            settings.repair_track_min_inliers,
        )
        self._camera.start()

        self._voice_session = VoiceSession(self._settings, self)
        self._voice_session.utterance_ready.connect(self._on_utterance)
        self._voice_session.speech_started.connect(self._on_speech_started)
        self._voice_session.barge_in.connect(self._on_barge_in)
        self._voice_session.capture_rejected.connect(self._on_capture_rejected)
        self._voice_session.error.connect(self._on_voice_error)
        self._tts_bridge = _TtsBridge(self)
        self._tts.set_idle_callback(self._tts_bridge.idle.emit)
        self._tts.set_error_callback(self._tts_bridge.failed.emit)
        self._tts_bridge.idle.connect(self._on_tts_idle)
        self._tts_bridge.failed.connect(self._on_tts_failed)

        self._apply_overlay_from_step()
        self._start_llm_connect()
        self._refresh_boards()

    def _build_ui(self) -> None:
        self._build_menus()

        root = QWidget()
        root.setObjectName("appCanvas")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(12, 8, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(self._build_top_bar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("mainSplitter")
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(12)

        workspace = QWidget()
        workspace.setObjectName("workspace")
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(8)

        camera_frame = RoundedCard(16)
        camera_frame.setObjectName("cameraFrame")
        camera_layout = QVBoxLayout(camera_frame)
        camera_layout.setContentsMargins(10, 10, 10, 10)
        self._camera_widget = CameraWidget()
        self._camera_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._camera_widget.customContextMenuRequested.connect(self._show_camera_menu)
        camera_layout.addWidget(self._camera_widget)
        workspace_layout.addWidget(camera_frame, stretch=1)

        self._log_panel = RoundedCard(16)
        self._log_panel.setObjectName("logPanel")
        log_layout = QVBoxLayout(self._log_panel)
        log_layout.setContentsMargins(14, 12, 14, 12)
        log_layout.setSpacing(6)
        log_header = QLabel("编译与烧录")
        log_header.setObjectName("sectionEyebrow")
        log_layout.addWidget(log_header)
        self._monitor = QTextEdit()
        self._monitor.setObjectName("buildLog")
        self._monitor.setReadOnly(True)
        self._monitor.setMinimumHeight(96)
        self._monitor.setMaximumHeight(160)
        bind_editor_menu(self._monitor)
        log_layout.addWidget(self._monitor)
        self._log_panel.hide()
        workspace_layout.addWidget(self._log_panel)

        sidebar = RoundedCard(16)
        sidebar.setObjectName("sidePanel")
        sidebar.setMinimumWidth(340)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(16, 14, 16, 12)
        side.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(8)
        presence = QLabel("●")
        presence.setObjectName("presenceDot")
        self._copilot_title = QLabel("助教")
        self._copilot_title.setObjectName("copilotTitle")
        header.addWidget(presence)
        header.addWidget(self._copilot_title)
        header.addStretch(1)
        side.addLayout(header)

        tutor_page = QWidget()
        tutor = QVBoxLayout(tutor_page)
        tutor.setContentsMargins(0, 0, 0, 0)
        tutor.setSpacing(8)
        step_eyebrow = QLabel("当前步骤")
        step_eyebrow.setObjectName("sectionEyebrow")
        tutor.addWidget(step_eyebrow)

        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        self._step_title = QLabel()
        self._step_title.setObjectName("stepTitle")
        self._step_title.setWordWrap(True)
        self._advance_btn = QPushButton("下一步")
        self._advance_btn.setObjectName("primaryButton")
        self._advance_btn.setToolTip("手动进入下一教学步骤")
        self._advance_btn.clicked.connect(self._advance_step)
        title_row.addWidget(self._step_title, stretch=1)
        title_row.addWidget(self._advance_btn, alignment=Qt.AlignmentFlag.AlignTop)
        tutor.addLayout(title_row)

        self._step_hint = QLabel()
        self._step_hint.setObjectName("stepHint")
        self._step_hint.setWordWrap(True)
        tutor.addWidget(self._step_hint)

        self._step_progress = QProgressBar()
        self._step_progress.setObjectName("stepProgress")
        self._step_progress.setTextVisible(False)
        self._step_progress.setFixedHeight(3)
        tutor.addWidget(self._step_progress)

        self._repair_panel = RepairPanel()
        self._mode_stack = QStackedWidget()
        self._mode_stack.addWidget(tutor_page)
        self._mode_stack.addWidget(self._repair_panel)
        side.addWidget(self._mode_stack)
        self._show_mode_page(0)

        divider = QFrame()
        divider.setObjectName("sideDivider")
        divider.setFrameShape(QFrame.Shape.NoFrame)
        divider.setFixedHeight(1)
        side.addWidget(divider)

        self._chat = ChatWidget(model_name=self._settings.openai_model)
        self._chat.send_requested.connect(lambda t: self._on_chat(t, voice_mode=False))
        self._chat.voice_session_toggled.connect(self._on_voice_toggled)
        self._chat.set_context_actions(
            [self._clear_action, self._tts_action, self._stop_tts_action]
        )
        side.addWidget(self._chat, stretch=1)

        splitter.addWidget(workspace)
        splitter.addWidget(sidebar)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([860, 400])
        layout.addWidget(splitter, stretch=1)

    def _build_top_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("topBar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(4, 2, 4, 2)
        row.setSpacing(8)

        mark = QLabel("助")
        mark.setObjectName("appMark")
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setFixedSize(28, 28)
        self._app_title = title = QLabel("学习助手")
        title.setObjectName("appTitle")
        row.addWidget(mark)
        row.addWidget(title)

        self._port_chip = QLabel("未连接开发板")
        self._port_chip.setObjectName("sessionChip")
        self._port_chip.setProperty("connected", False)
        self._port_chip.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._port_chip.customContextMenuRequested.connect(self._show_hardware_menu)
        row.addWidget(self._port_chip)
        row.addStretch(1)

        for text, menu, icon in (
            ("模式", self._mode_menu, icon_repair()),
            ("项目", self._project_menu, icon_project()),
            ("硬件", self._hardware_menu, icon_hardware()),
            ("摄像头", self._camera_menu, icon_camera()),
            ("语音", self._voice_menu, icon_voice()),
            ("视图", self._view_menu, icon_view()),
        ):
            button = QToolButton()
            button.setObjectName("toolbarButton")
            button.setText(f"{text}  ▾")
            button.setIcon(with_label_gap(icon, 8))
            button.setIconSize(QSize(24, 16))
            button.setMenu(menu)
            button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            row.addWidget(button)

        settings_btn = QToolButton()
        settings_btn.setObjectName("toolbarButton")
        settings_btn.setText("设置")
        settings_btn.setIcon(with_label_gap(icon_settings(), 8))
        settings_btn.setIconSize(QSize(24, 16))
        settings_btn.setToolTip("对话模型与语音识别")
        settings_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        settings_btn.clicked.connect(self._open_settings)
        row.addWidget(settings_btn)
        return bar

    def _show_camera_menu(self, pos) -> None:
        self._camera_menu.exec(self._camera_widget.mapToGlobal(pos))

    def _show_hardware_menu(self, pos) -> None:
        self._hardware_menu.exec(self._port_chip.mapToGlobal(pos))

    def _build_menus(self) -> None:
        self._camera_placeholders: list[QAction] = []
        self._port_placeholders: list[QAction] = []
        self._preferred_port = ""

        self.menuBar().hide()

        mode_menu = ThemedMenu(self)
        mode_menu.setObjectName("appMenu")
        self._mode_group = QActionGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_actions: dict[str, QAction] = {}
        for mode, icon, label, tip in (
            ("tutor", icon_project(), "教学模式", "跟着步骤搭建 Arduino 实验"),
            ("repair", icon_repair(), "维修模式", "诊断电路板或设备故障，在画面上标出检查点"),
        ):
            action = mode_menu.addAction(icon, label)
            action.setCheckable(True)
            action.setData(mode)
            action.setToolTip(tip)
            self._mode_group.addAction(action)
            self._mode_actions[mode] = action
        self._mode_actions["tutor"].setChecked(True)
        self._mode_group.triggered.connect(lambda action: self._set_mode(str(action.data())))
        self._mode_menu = mode_menu

        project = ThemedMenu(self)
        project.setObjectName("appMenu")
        next_action = project.addAction(icon_next(), "下一步")
        next_action.setToolTip("手动进入下一教学步骤")
        next_action.triggered.connect(self._advance_step)
        self._clear_action = project.addAction(icon_clear(), "清空对话")
        self._clear_action.setToolTip("清除与助手的多轮记忆")
        self._clear_action.triggered.connect(self._clear_dialog)
        self._project_menu = project

        view = ThemedMenu(self)
        view.setObjectName("appMenu")
        self._show_log_action = view.addAction(icon_log(), "编译与烧录")
        self._show_log_action.setCheckable(True)
        self._show_log_action.setChecked(False)
        self._show_log_action.setToolTip("显示或隐藏编译与烧录日志")
        self._show_log_action.toggled.connect(self._set_log_visible)
        self._pause_action = view.addAction(icon_stop(), "暂停识别")
        self._pause_action.setCheckable(True)
        self._pause_action.setToolTip("暂停器件识别，画面继续显示。竖起拇指可继续。")
        self._pause_action.toggled.connect(self._set_vision_paused)
        self._clear_marks_action = view.addAction(icon_clear(), "清除画面标记")
        self._clear_marks_action.setToolTip("去掉你在画面上点出、框出或圈出的标记")
        self._clear_marks_action.triggered.connect(self._clear_marks)
        self._view_menu = view

        self._hardware_menu = ThemedMenu(self)
        self._hardware_menu.setObjectName("appMenu")
        refresh_boards = self._hardware_menu.addAction(icon_refresh(), "刷新开发板")
        refresh_boards.setToolTip("重新扫描已连接的 Arduino 开发板")
        refresh_boards.triggered.connect(self._refresh_boards)
        self._hardware_menu.addSeparator()
        self._port_insert_before = self._hardware_menu.addSeparator()
        upload_action = self._hardware_menu.addAction(icon_upload(), "烧录 Blink")
        upload_action.setToolTip("编译并上传 Blink 示例到所选串口")
        upload_action.triggered.connect(self._upload_blink)
        self._port_group = QActionGroup(self)
        self._port_group.setExclusive(True)

        self._camera_menu = ThemedMenu(self)
        self._camera_menu.setObjectName("appMenu")
        refresh_cameras = self._camera_menu.addAction(icon_refresh(), "刷新")
        refresh_cameras.setToolTip("重新扫描本机摄像头")
        refresh_cameras.triggered.connect(self._refresh_cameras)
        self._camera_menu.addSeparator()
        self._camera_group = QActionGroup(self)
        self._camera_group.setExclusive(True)
        self._camera_group.triggered.connect(self._on_camera_changed)
        self._camera_insert_before = self._camera_menu.addSeparator()
        pause_cam = self._camera_menu.addAction(icon_stop(), "暂停 / 继续识别")
        pause_cam.setToolTip("暂停后画面还在，竖起拇指也会继续")
        pause_cam.triggered.connect(lambda: self._set_vision_paused(not self._vision_paused))
        clear_cam = self._camera_menu.addAction(icon_clear(), "清除画面标记")
        clear_cam.triggered.connect(self._clear_marks)

        voice = ThemedMenu(self)
        voice.setObjectName("appMenu")
        self._asr_group = QActionGroup(self)
        self._asr_group.setExclusive(True)
        self._asr_actions: dict[str, QAction] = {}
        for backend, icon, label in (
            ("local", icon_local(), "本地 Whisper"),
            ("ollama", icon_remote(), "远程 Ollama"),
            ("openai", icon_remote(), "远程 API"),
        ):
            action = voice.addAction(icon, label)
            action.setCheckable(True)
            action.setData(backend)
            self._asr_group.addAction(action)
            self._asr_actions[backend] = action
        self._asr_actions[self._settings.asr_backend].setChecked(True)
        self._refresh_asr_tooltips()
        self._asr_group.triggered.connect(self._on_asr_mode_changed)
        voice.addSeparator()
        self._tts_action = voice.addAction(icon_speaker(), "语音回复")
        self._tts_action.setCheckable(True)
        self._tts_action.setChecked(self._settings.tts_enabled)
        self._tts_action.setToolTip("文字提问后朗读回复。语音辅导会话中总会朗读。")
        self._stop_tts_action = voice.addAction(icon_stop(), "停止朗读")
        self._stop_tts_action.triggered.connect(self._tts.stop)
        self._voice_menu = voice

    def _set_log_visible(self, visible: bool) -> None:
        self._log_panel.setVisible(visible)
        if self._show_log_action.isChecked() != visible:
            self._show_log_action.blockSignals(True)
            self._show_log_action.setChecked(visible)
            self._show_log_action.blockSignals(False)

    def _clear_dialog(self) -> None:
        self._dialog_history.clear()
        self._chat.append_system("已清空对话记忆")

    def _tts_on(self) -> bool:
        return self._tts_action.isChecked()

    def _should_speak_reply(self, voice_mode: bool) -> bool:
        return self._tts_on() or voice_mode

    def _speak_if_enabled(self, text: str, force: bool = False) -> None:
        if not text.strip():
            return
        if force or self._tts_on():
            self._tts.speak(prepare_speech_text(text))

    def _extra_system_for_chat(self, voice_mode: bool, has_image: bool) -> str:
        if self._mode == "repair":
            return build_repair_extra_system(
                self._repair.case(),
                self._repair.documents(),
                self._repair.annotations,
                self._user_marks,
                voice_mode,
                has_image=has_image,
                box_format=self._repair.box_format,
                frozen=self._repair_panel_frozen(),
                ifixit_enabled=self._repair.ifixit is not None,
                guide=self._repair.guide,
                track_status=self._track_status,
                image_size=self._repair.image_size,
                clean_image=self._repair_clean_view,
            )
        return build_extra_system(
            self._state_machine.progress(),
            self._verify_ok,
            self._verify_msg,
            self._scene,
            voice_mode,
            has_image=has_image,
        )

    def _frame_for_chat(self) -> np.ndarray | None:
        if not self._settings.llm_send_frame or self._llm is None:
            return None
        if self._llm.supports_vision is False:
            return None
        if self._mode == "repair":
            return self._repair_anchor_frame()
        return self._camera_widget.latest_frame()

    def _repair_anchor_frame(self) -> np.ndarray | None:
        """The frame for the model, made the new tracking reference.

        Shapes are carried into its coordinates first, so whatever the model
        reads or draws lines up with the picture it gets. The model sees the
        clean camera frame with only the user's marks: our own boxes and labels
        on the picture made it box the overlay or the wrong part.
        """
        self._repair_clean_view = False
        snap = self._camera.anchor_snapshot()
        if snap is None:
            self._repair.anchor_frame = None
            return self._camera_widget.latest_frame()
        raw, _shown, H = snap
        self._repair.rebase(H)
        self._rebase_marks(H)
        self._repair.anchor_frame = raw
        self._refresh_repair_overlay()
        self._repair_clean_view = True
        return self._marks_overlay.render_marks(raw, self._user_marks)

    def _detections_json(self) -> list[dict]:
        return [
            {
                "type": node.get("type"),
                "bbox": node.get("bbox"),
                "confidence": node.get("confidence"),
            }
            for node in (self._scene.get("nodes") or {}).values()
        ]

    @Slot(dict)
    def _on_scene(self, scene: dict) -> None:
        scene = dict(scene)
        scene["user_marks"] = [dict(mark) for mark in self._user_marks]
        self._scene = scene
        if self._mode == "repair":
            return
        ok, msg = self._state_machine.verify_step(scene)
        self._verify_ok = ok
        self._verify_msg = msg
        self._step_hint.setToolTip(msg)
        hint = dict(self._agent_ctx.overlay_hint)
        if ok:
            hint.pop("status", None)
            hint.pop("status_text", None)
        else:
            hint["status"] = "missing"
            hint["status_text"] = msg
        self._agent_ctx.overlay_hint = hint
        self._camera.set_overlay_hint(hint)

    def _on_gesture(self, cmd: str) -> None:
        if cmd == "pause":
            self._set_vision_paused(True)
            return
        if cmd == "next":
            if self._vision_paused:
                self._set_vision_paused(False)
                return
            if self._mode == "repair" and self._repair.current_step() is not None:
                self._verify_guide_step()
                return
            self._advance_step()
            return
        if cmd == "rescan":
            if self._vision_paused:
                self._set_vision_paused(False)
            self._camera.request_rescan()
            self._chat.append_system("手势：重新识别")

    def _set_vision_paused(self, paused: bool) -> None:
        if paused == self._vision_paused:
            return
        self._vision_paused = paused
        self._camera.set_paused(paused)
        if self._pause_action.isChecked() != paused:
            self._pause_action.blockSignals(True)
            self._pause_action.setChecked(paused)
            self._pause_action.blockSignals(False)
        if paused:
            self._chat.append_system("识别已暂停。竖起拇指，或在「视图」里再点一次即可继续。")
        else:
            self._chat.append_system("已继续识别画面")

    def _on_camera_mark(self, nx: float, ny: float) -> None:
        probe = self._camera.to_reference({"nx": nx, "ny": ny})
        for mark in self._user_marks:
            if mark.get("kind", "point") != "point":
                continue
            if abs(float(mark["nx"]) - probe["nx"]) < 0.03 and abs(float(mark["ny"]) - probe["ny"]) < 0.03:
                return
        label = self._next_mark_label()
        self._add_mark({"label": label, "kind": "point", "nx": round(nx, 3), "ny": round(ny, 3)})
        near = nearest_component(self._scene, nx, ny)
        ask = self._mark_question(label)
        if near:
            self._chat.append_system(f"已添加{label}，靠近{near}。可以直接问「{ask}」。")
        else:
            self._chat.append_system(f"已添加{label}。可以直接问「{ask}」。")

    def _mark_question(self, label: str) -> str:
        if self._mode == "repair":
            return f"{label}这里是什么元件、该怎么测"
        return f"{label}这里该怎么接"

    def _on_camera_region(self, shape: dict) -> None:
        mark = make_region_mark(shape, "")
        if mark is None:
            return
        label = self._next_mark_label()
        mark["label"] = label
        self._add_mark(mark)
        verb = "框出" if mark["kind"] == "box" else "圈出"
        names = components_in_mark(self._scene, mark)
        near = None if names else nearest_component(self._scene, mark["nx"], mark["ny"])
        if names:
            where = f"，里面有{'、'.join(names)}"
        elif near:
            where = f"，靠近{near}"
        else:
            where = ""
        self._chat.append_system(f"已{verb}{label}{where}。可以直接问「{self._mark_question(label)}」。")

    def _next_mark_label(self) -> str:
        self._mark_seq += 1
        return f"标记 {self._mark_seq}"

    def _add_mark(self, mark: dict) -> None:
        """``mark`` is in displayed-video coordinates; it is stored in reference ones."""
        if len(self._user_marks) >= 6:
            self._user_marks.pop(0)
        self._user_marks.append(self._camera.to_reference(mark))
        self._push_marks()

    def _rebase_marks(self, H: np.ndarray | None) -> None:
        """Follow the camera's rebase, which moved its copy of the marks by ``H``."""
        if H is None or not self._user_marks:
            return
        self._user_marks = warp_marks(self._user_marks, H)
        scene = dict(self._scene)
        scene["user_marks"] = [dict(mark) for mark in self._user_marks]
        self._scene = scene

    def _push_marks(self) -> None:
        self._camera.set_user_marks(self._user_marks)
        scene = dict(self._scene)
        scene["user_marks"] = [dict(mark) for mark in self._user_marks]
        self._scene = scene

    def _clear_marks(self) -> None:
        if not self._user_marks:
            return
        self._user_marks.clear()
        self._push_marks()
        self._chat.append_system("已清除画面标记")

    def _apply_overlay_from_step(self) -> None:
        step = self._state_machine.current_step
        progress = self._state_machine.progress()
        hint = {
            "hint": step.hint,
            "requires": list(step.requires),
            "pin": step.pin,
            "step_label": f"第 {progress['step_index'] + 1}/{progress['total_steps']} 步",
        }
        self._agent_ctx.overlay_hint = hint
        self._camera.set_overlay_hint(hint)

    def _update_step_label(self) -> None:
        p = self._state_machine.progress()
        self._step_title.setText(
            f"{p['title']} · 第 {p['step_index'] + 1} / {p['total_steps']} 步"
        )
        self._step_hint.setText(p["current_hint"])
        total = max(p["total_steps"], 1)
        self._step_progress.setMaximum(total)
        self._step_progress.setValue(min(p["step_index"] + 1, total))
        at_end = p["step_index"] + 1 >= p["total_steps"]
        self._advance_btn.setEnabled(not at_end)
        self._advance_btn.setText("已完成" if at_end else "下一步")

    def _announce_current_step(self) -> None:
        p = self._state_machine.progress()
        hint = p["current_hint"]
        self._chat.append_system(f"现在这一步：{hint}")
        if self._voice_active:
            self._cancel_inflight()
            self._speak_in_session(hint)
        else:
            self._speak_if_enabled(hint)

    def _advance_step(self) -> None:
        if self._mode == "repair":
            if self._repair.current_step() is not None:
                self._on_guide_next()
            else:
                self._next_check()
            return
        if self._state_machine.advance():
            self._apply_overlay_from_step()
            self._update_step_label()
            self._announce_current_step()
            return
        self._chat.append_system("已经是最后一步")

    def _on_camera_status(self, msg: str) -> None:
        if self._closing:
            return
        self._chat.append_system(msg)

    def _seed_camera_menu(self) -> None:
        index = int(self._settings.camera_index)
        self._replace_camera_actions([(index, f"摄像头 {index}")], pick=index)

    def _start_llm_connect(self) -> None:
        previous = self._llm_connect_worker
        if previous is not None and previous.isRunning():
            previous.requestInterruption()
        self._llm_connect_worker = LLMConnectWorker()
        _keep_alive(self._llm_connect_worker)
        self._llm_connect_worker.ready.connect(self._on_llm_ready)
        self._llm_connect_worker.start()

    @Slot(object, bool, str)
    def _on_llm_ready(self, client: object, ok: bool, msg: str) -> None:
        if self._closing or self.sender() is not self._llm_connect_worker:
            return
        self._llm = client  # type: ignore[assignment]
        self._chat.append_system(msg if ok else f"警告: {msg}")

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self._settings, self._selected_asr_backend(), self)
        if dialog.exec() != SettingsDialog.DialogCode.Accepted or not dialog.changed:
            return
        self._settings = dialog.settings
        self._chat.append_system("设置已保存到 .env")
        self._refresh_asr_tooltips()
        if "ASR_BACKEND" in dialog.changed:
            self._asr_actions[self._settings.asr_backend].setChecked(True)
        shared = {"OPENAI_BASE_URL", "OPENAI_API_KEY"} & dialog.changed
        asr_changed = dialog.changed - CHAT_KEYS - REPAIR_KEYS
        if asr_changed or (shared and self._selected_asr_backend() != "local"):
            self._on_asr_mode_changed()
        if dialog.changed & CHAT_KEYS:
            self._chat.set_model_name(self._settings.openai_model)
            self._chat.append_system(f"正在连接对话模型 {self._settings.openai_model}…")
            self._start_llm_connect()
        if dialog.changed & REPAIR_KEYS:
            self._apply_repair_settings()

    def _refresh_asr_tooltips(self) -> None:
        cfg = self._settings
        tips = {
            "local": f"faster-whisper {cfg.whisper_model_size}（{cfg.whisper_device}）",
            "ollama": f"{cfg.ollama_asr_model} @ {cfg.resolve_ollama_base_url()}",
            "openai": f"{cfg.asr_model} @ {cfg.resolve_asr_base_url()}（{cfg.asr_api_mode}）",
        }
        for backend, action in self._asr_actions.items():
            action.setToolTip(tips[backend])

    def _selected_camera_index(self) -> int:
        action = self._camera_group.checkedAction()
        if action is not None and action.data() not in (None, ""):
            return int(action.data())
        return get_settings().camera_index

    def _replace_camera_actions(self, entries: list[tuple[object, str]], pick: object = None) -> None:
        self._camera_group.blockSignals(True)
        for action in list(self._camera_group.actions()):
            self._camera_group.removeAction(action)
            self._camera_menu.removeAction(action)
            action.deleteLater()
        for action in self._camera_placeholders:
            self._camera_menu.removeAction(action)
            action.deleteLater()
        self._camera_placeholders.clear()

        chosen: QAction | None = None
        for data, label in entries:
            if data in (None, ""):
                placeholder = QAction(icon_camera(), label, self)
                placeholder.setEnabled(False)
                self._camera_menu.insertAction(self._camera_insert_before, placeholder)
                self._camera_placeholders.append(placeholder)
                continue
            action = QAction(icon_camera(), label, self)
            action.setCheckable(True)
            action.setData(data)
            self._camera_group.addAction(action)
            self._camera_menu.insertAction(self._camera_insert_before, action)
            if pick is not None and data == pick:
                chosen = action
        if chosen is None and self._camera_group.actions():
            chosen = self._camera_group.actions()[0]
        if chosen is not None:
            chosen.setChecked(True)
        self._camera_group.blockSignals(False)

    def _refresh_cameras(self) -> None:
        preferred = self._selected_camera_index()
        try:
            devices = list_cameras()
            if not devices:
                self._replace_camera_actions([("", "未检测到摄像头")])
                self._chat.append_system("未检测到可用摄像头")
            else:
                pick = preferred if any(index == preferred for index, _ in devices) else devices[0][0]
                self._replace_camera_actions(list(devices), pick=pick)
        except Exception as exc:  # noqa: BLE001
            self._replace_camera_actions([("", f"扫描失败: {exc}")])
            self._chat.append_system(f"摄像头扫描: {exc}")
        if hasattr(self, "_camera"):
            self._on_camera_changed()

    def _on_camera_changed(self, _action: QAction | None = None) -> None:
        if not hasattr(self, "_camera"):
            return
        checked = self._camera_group.checkedAction()
        if checked is None or checked.data() in (None, ""):
            return
        self._camera.set_camera_index(int(checked.data()))

    def _replace_port_actions(self, entries: list[tuple[object, str]], pick: str = "") -> None:
        self._port_group.blockSignals(True)
        for action in list(self._port_group.actions()):
            self._port_group.removeAction(action)
            self._hardware_menu.removeAction(action)
            action.deleteLater()
        for action in self._port_placeholders:
            self._hardware_menu.removeAction(action)
            action.deleteLater()
        self._port_placeholders.clear()

        chosen: QAction | None = None
        for data, label in entries:
            if not data:
                placeholder = QAction(icon_hardware(), label, self)
                placeholder.setEnabled(False)
                self._hardware_menu.insertAction(self._port_insert_before, placeholder)
                self._port_placeholders.append(placeholder)
                continue
            action = QAction(icon_hardware(), label, self)
            action.setCheckable(True)
            action.setData(data)
            self._port_group.addAction(action)
            self._hardware_menu.insertAction(self._port_insert_before, action)
            if pick and data == pick:
                chosen = action
        if chosen is None and self._port_group.actions():
            chosen = self._port_group.actions()[0]
        if chosen is not None:
            chosen.setChecked(True)
        self._port_group.blockSignals(False)
        self._sync_port_chip()

    def _sync_port_chip(self) -> None:
        if not hasattr(self, "_port_chip"):
            return
        action = self._port_group.checkedAction()
        port = action.data() if action is not None else ""
        if port:
            self._port_chip.setText(f"●  {port}")
            self._port_chip.setToolTip(action.text())
            connected = True
        else:
            text = self._port_placeholders[-1].text() if self._port_placeholders else "未连接开发板"
            self._port_chip.setText(text)
            self._port_chip.setToolTip("")
            connected = False
        self._port_chip.setProperty("connected", connected)
        self._port_chip.style().unpolish(self._port_chip)
        self._port_chip.style().polish(self._port_chip)

    def _refresh_boards(self) -> None:
        if self._board_worker is not None and self._board_worker.isRunning():
            return
        self._preferred_port = self._selected_port()
        self._replace_port_actions([("", "正在扫描…")])
        self._board_worker = BoardScanWorker()
        _keep_alive(self._board_worker)
        self._board_worker.succeeded.connect(self._on_boards_ready)
        self._board_worker.failed.connect(self._on_boards_failed)
        self._board_worker.start()

    @Slot(object)
    def _on_boards_ready(self, boards: object) -> None:
        if self._closing:
            return
        preferred = self._preferred_port
        if not boards:
            self._replace_port_actions([("", "未检测到开发板")])
            return
        entries = [(board.port, f"{board.port} ({board.label})") for board in boards]
        pick = preferred if any(board.port == preferred for board in boards) else entries[0][0]
        self._replace_port_actions(entries, pick=pick)

    @Slot(str)
    def _on_boards_failed(self, msg: str) -> None:
        if self._closing:
            return
        self._replace_port_actions([("", "无法列出串口")])
        self._chat.append_system(f"硬件连接: {msg}")

    def _selected_port(self) -> str:
        action = self._port_group.checkedAction()
        if action is None:
            return ""
        return action.data() or ""

    def _upload_blink(self) -> None:
        self._set_log_visible(True)
        self._agent_ctx.selected_port = self._selected_port()
        log = self._handlers["compile_and_upload"](sketch_name="blink", port=self._agent_ctx.selected_port)
        self._monitor.append(log)
        self._chat.append_system(log[:500])

    def _selected_asr_backend(self) -> AsrBackend:
        action = self._asr_group.checkedAction()
        return action.data() if action is not None else "local"

    def _asr_mode_label(self) -> str:
        action = self._asr_group.checkedAction()
        if action is None:
            return "ASR：本地 Whisper"
        return f"ASR：{action.text()}（{action.toolTip()}）"

    def _on_asr_mode_changed(self, _action: QAction | None = None) -> None:
        backend = self._selected_asr_backend()
        try:
            self._asr = create_asr(self._settings, backend=backend)
        except Exception as exc:  # noqa: BLE001
            self._chat.append_system(f"ASR 切换失败: {exc}")
            return
        self._chat.append_system(self._asr_mode_label())

    def _on_voice_toggled(self, active: bool) -> None:
        if self._syncing_voice_btn:
            return
        if active:
            self._start_voice_session()
        else:
            self._stop_voice_session()

    def _set_voice_button(self, active: bool) -> None:
        self._syncing_voice_btn = True
        self._chat.set_voice_checked(active)
        self._syncing_voice_btn = False

    def _start_voice_session(self) -> None:
        self._cancel_inflight()
        try:
            self._voice_session.start()
        except Exception as exc:  # noqa: BLE001
            self._voice_active = False
            self._voice_phase = "idle"
            self._set_voice_button(False)
            self._chat.set_voice_status("")
            self._chat.append_system(f"麦克风不可用: {exc}")
            return
        self._voice_active = True
        self._enter_listening()

    def _stop_voice_session(self) -> None:
        self._voice_active = False
        self._voice_phase = "idle"
        self._cancel_inflight()
        self._voice_session.stop()
        self._chat.end_assistant_stream()
        self._chat.set_voice_status("")
        self._set_voice_button(False)

    def _enter_listening(self) -> None:
        if not self._voice_active:
            return
        self._voice_phase = "listening"
        self._chat.set_voice_status("在听，请说话")
        self._voice_session.set_mode("listen")

    def _cancel_inflight(self) -> None:
        self._turn += 1
        self._sentence_buf.clear()
        self._tts.stop()
        for worker in (self._llm_worker, self._asr_worker):
            if worker is not None and worker.isRunning():
                worker.requestInterruption()

    def _speak_in_session(self, text: str) -> None:
        spoken = prepare_speech_text(text)
        if not spoken or not self._voice_active:
            return
        self._voice_phase = "speaking"
        self._chat.set_voice_status("讲解中")
        self._voice_session.set_mode("monitor")
        self._voice_session.set_barge_enabled(False)
        self._tts.enqueue(spoken)

    def _on_voice_error(self, message: str) -> None:
        if self._closing:
            return
        self._chat.append_system(f"语音辅导: {message}")
        if self._voice_active:
            self._stop_voice_session()

    def _on_speech_started(self) -> None:
        if self._voice_active and self._voice_phase == "listening":
            self._chat.set_voice_status("正在说")

    def _on_tts_failed(self, message: str) -> None:
        if self._closing:
            return
        self._chat.append_system(f"TTS 失败: {message}")

    def _on_tts_idle(self) -> None:
        # The model may still be flushing the next sentence onto the queue.
        QTimer.singleShot(200, self._maybe_resume_listen)

    def _maybe_resume_listen(self) -> None:
        if not self._voice_active or self._voice_phase != "speaking":
            return
        if self._llm_worker is not None and self._llm_worker.isRunning():
            return
        if self._tts.is_busy():
            return
        self._enter_listening()

    def _on_barge_in(self) -> None:
        if not self._voice_active:
            return
        self._voice_phase = "listening"
        self._chat.set_voice_status("正在说")
        self._chat.end_assistant_stream()
        self._cancel_inflight()

    def _on_capture_rejected(self) -> None:
        if not self._voice_active or self._voice_phase == "recognizing":
            return
        self._enter_listening()

    def _on_utterance(self, audio: object) -> None:
        if not self._voice_active:
            return
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return
        self._turn += 1
        turn = self._turn
        self._sentence_buf.clear()
        self._tts.stop()
        self._voice_phase = "recognizing"
        self._chat.set_voice_status("识别中")
        self._voice_session.set_mode("paused")
        self._asr_worker = ASRWorker(self._asr, samples, self._voice_session.sample_rate, turn)
        _keep_alive(self._asr_worker)
        self._asr_worker.succeeded.connect(self._on_asr_finished)
        self._asr_worker.failed.connect(self._on_asr_failed)
        self._asr_worker.start()

    @Slot(int, str)
    def _on_asr_finished(self, turn: int, text: str) -> None:
        if turn != self._turn or not self._voice_active:
            return
        text = text.strip()
        if not text:
            self._enter_listening()
            self._chat.set_voice_status("没听清，请再说")
            return
        self._chat.append_user(text)
        if self._llm is None:
            self._chat.append_system("助手仍在连接，请稍候")
            self._enter_listening()
            return
        self._start_llm(text, voice_mode=True, stream=True)

    @Slot(int, str)
    def _on_asr_failed(self, turn: int, message: str) -> None:
        if turn != self._turn:
            return
        self._chat.append_system(f"ASR 失败: {message}")
        if self._voice_active:
            self._enter_listening()

    def _on_chat(self, text: str, voice_mode: bool = False) -> None:
        if self._voice_active:
            self._cancel_inflight()
            self._chat.end_assistant_stream()
            voice_mode = True
        if self._llm is None:
            self._chat.append_system("助手仍在连接，请稍候")
            if self._voice_active:
                self._enter_listening()
            return
        if not self._voice_active and self._llm_worker is not None and self._llm_worker.isRunning():
            self._chat.append_system("助手仍在回复上一问，请稍候")
            return
        self._start_llm(text, voice_mode=voice_mode, stream=True)

    def _start_llm(self, text: str, voice_mode: bool, stream: bool) -> None:
        self._agent_ctx.selected_port = self._selected_port()
        self._pending_user_text = text
        self._pending_voice_mode = voice_mode
        self._turn += 1
        turn = self._turn
        self._sentence_buf.clear()
        self._voice_streamed = False
        self._answer_shown = False
        if self._voice_active:
            self._voice_phase = "thinking"
            self._chat.set_voice_status("老师在想")
            self._voice_session.set_mode("monitor")
            self._voice_session.set_barge_enabled(True)
        frame = self._frame_for_chat()
        repair = self._mode == "repair"
        max_side = (
            self._settings.repair_frame_max_side if repair else self._settings.llm_frame_max_side
        )
        if repair:
            self._repair.drain_images()
            self._repair.begin_turn(
                self._repair.anchor_frame if frame is not None else None,
                sent_image_size(frame.shape[1], frame.shape[0], max_side)
                if frame is not None
                else None,
            )
        extra = self._extra_system_for_chat(voice_mode, has_image=frame is not None)
        if repair and self._repair.debug is not None:
            self._repair.debug.begin(
                turn,
                text,
                extra,
                None if frame is None else fit_frame(frame, max_side),
                self._repair.anchor_frame,
                {
                    "model": self._settings.openai_model,
                    "box_format": self._repair.box_format,
                    "track_status": self._track_status,
                    "frozen": self._repair_frozen,
                    "clean_image": self._repair_clean_view,
                },
            )
        self._llm_worker = LLMWorker(
            self._llm,  # type: ignore[arg-type]
            text,
            self._repair_tools if repair else self._tools,
            self._repair_handlers if repair else self._handlers,
            list(self._dialog_history),
            extra,
            voice_mode,
            turn,
            stream,
            frame=frame,
            frame_max_side=max_side,
            system_prompt=REPAIR_SYSTEM_PROMPT if repair else None,
            max_rounds=10 if repair else 6,
            drain_images=self._repair.drain_images if repair else None,
            keep_preface=repair and not voice_mode,
            options=self._repair_llm_options(voice_mode, frame is not None) if repair else None,
        )
        _keep_alive(self._llm_worker)
        self._llm_worker.delta.connect(self._on_llm_delta)
        self._llm_worker.reasoning.connect(self._on_llm_reasoning)
        self._llm_worker.revoke.connect(self._on_llm_revoke)
        self._llm_worker.succeeded.connect(self._on_llm_reply)
        self._llm_worker.failed.connect(self._on_llm_failed)
        self._chat.begin_thinking()
        self._llm_worker.start()

    def _repair_llm_options(self, voice_mode: bool, has_image: bool) -> dict:
        ctx = self._repair

        def follow_up(reply: str) -> str | None:
            if not needs_drawing(reply, has_image, ctx.drew_this_turn):
                return None
            if ctx.debug is not None:
                ctx.debug.note(nudged=True)
            return NUDGE_TEXT

        # Voice repair turns still have to call drawing tools, so no small token cap.
        options: dict = {"voice_max_tokens": None, "follow_up": follow_up}
        if voice_mode and not self._settings.repair_voice_think:
            options["extra_body"] = {"reasoning_effort": "none"}
        return options

    @Slot(int, str)
    def _on_llm_reasoning(self, turn: int, text: str) -> None:
        if turn != self._turn or not text:
            return
        self._chat.append_thinking(text)

    @Slot(int, str)
    def _on_llm_delta(self, turn: int, text: str) -> None:
        if turn != self._turn or not text:
            return
        self._chat.finish_thinking()
        if self._pending_voice_mode:
            for sentence in self._sentence_buf.push(text):
                self._emit_spoken_sentence(sentence)
            return
        self._answer_shown = True
        self._chat.append_assistant_chunk(text)

    def _emit_spoken_sentence(self, sentence: str) -> None:
        self._voice_streamed = True
        self._chat.append_assistant_chunk(sentence)
        if not self._pending_voice_mode:
            return
        spoken = prepare_speech_text(sentence)
        if not spoken:
            return
        if self._voice_active and self._voice_phase != "speaking":
            self._voice_phase = "speaking"
            self._chat.set_voice_status("讲解中")
            self._voice_session.set_barge_enabled(False)
            self._voice_session.refresh_barge_grace()
        self._tts.enqueue(spoken)

    @Slot(int)
    def _on_llm_revoke(self, turn: int) -> None:
        if turn != self._turn:
            return
        self._tts.stop()
        self._sentence_buf.clear()
        self._voice_streamed = False
        self._answer_shown = False
        self._chat.discard_assistant_stream()
        self._chat.reopen_thinking()
        if self._voice_active:
            self._voice_phase = "thinking"
            self._chat.set_voice_status("老师在想")
            self._voice_session.set_mode("monitor")
            self._voice_session.set_barge_enabled(True)

    def _remember_turn(self, reply: str) -> None:
        self._dialog_history.append({"role": "user", "content": self._pending_user_text})
        self._dialog_history.append({"role": "assistant", "content": reply})
        if len(self._dialog_history) > 20:
            self._dialog_history = self._dialog_history[-20:]

    def _apply_overlay_update(self) -> None:
        if self._mode == "repair":
            self._refresh_repair_overlay()
            return
        if self._agent_ctx.overlay_hint:
            self._camera.set_overlay_hint(self._agent_ctx.overlay_hint)

    @Slot(int, str)
    def _on_llm_reply(self, turn: int, reply: str) -> None:
        if turn != self._turn:
            return
        self._repair_panel.set_verifying(False)
        self._finish_repair_debug(turn, reply)
        unanswered = not reply.strip() and not self._voice_streamed and not self._answer_shown
        if unanswered:
            self._chat.finish_thinking()
            self._chat.end_assistant_stream()
            self._chat.append_system("助手没说完，再问一次")
            self._apply_overlay_update()
            if self._voice_active:
                self._enter_listening()
            return
        if self._pending_voice_mode and self._voice_active:
            rest = self._sentence_buf.flush()
            if rest:
                self._emit_spoken_sentence(rest)
            if reply.strip() and not self._voice_streamed:
                if reply.strip() == self._chat.thinking_text().strip():
                    self._chat.discard_thinking()
                self._emit_spoken_sentence(reply.strip())
            self._chat.end_assistant_stream()
            if reply.strip():
                self._remember_turn(reply.strip())
            self._apply_overlay_update()
            if self._tts.is_busy():
                self._voice_phase = "speaking"
                self._chat.set_voice_status("讲解中")
                self._voice_session.set_barge_enabled(False)
            else:
                self._enter_listening()
            return

        if self._answer_shown:
            self._chat.finish_thinking()
            if reply.strip():
                self._chat.set_stream_text(reply.strip())
            self._chat.end_assistant_stream()
        else:
            thinking = self._chat.thinking_text().strip()
            if reply.strip() and reply.strip() == thinking:
                self._chat.discard_thinking()
            else:
                self._chat.finish_thinking()
            if reply.strip():
                self._chat.append_assistant(reply)
        self._remember_turn(reply)
        if self._should_speak_reply(self._pending_voice_mode):
            spoken = spoken_text_from_reply(reply, self._pending_voice_mode)
            if spoken:
                self._tts.speak(spoken)
            elif self._pending_voice_mode:
                self._chat.append_system("TTS：回复为空或无法朗读，请查看文字回复")
        self._apply_overlay_update()

    @Slot(int, str)
    def _on_llm_failed(self, turn: int, message: str) -> None:
        if turn != self._turn:
            return
        self._repair_panel.set_verifying(False)
        self._finish_repair_debug(turn, f"错误: {message}")
        self._chat.end_assistant_stream()
        self._chat.append_assistant(f"错误: {message}")
        if self._voice_active:
            self._enter_listening()

    @staticmethod
    def _make_ifixit(settings) -> IFixitClient | None:
        if not settings.ifixit_enabled:
            return None
        return IFixitClient(settings.ifixit_base_url, settings.ifixit_timeout_s)

    def _finish_repair_debug(self, turn: int, reply: str) -> None:
        debug = self._repair.debug
        if debug is None or self._mode != "repair":
            return
        folder = debug.finish(turn, reply, self._repair.current_shapes(), self._repair.anchor_frame)
        if folder is not None:
            self._chat.append_system(f"调试记录：{folder}")

    @staticmethod
    def _make_repair_debug(settings) -> RepairDebugLog | None:
        return RepairDebugLog(settings.repair_debug_dir) if settings.repair_debug else None

    def _apply_repair_settings(self) -> None:
        cfg = self._settings
        self._repair.debug = self._make_repair_debug(cfg)
        self._repair.box_format = cfg.repair_box_format
        self._repair.ifixit = self._make_ifixit(cfg)
        self._repair.library = DocumentLibrary(cfg.repair_docs_dir)
        self._repair_tools = build_repair_tools(cfg.ifixit_enabled)
        self._repair_panel.set_ifixit_enabled(cfg.ifixit_enabled)
        self._camera.configure_tracking(
            cfg.repair_tracking, cfg.repair_track_features, cfg.repair_track_min_inliers
        )

    def _show_mode_page(self, index: int) -> None:
        self._mode_stack.setCurrentIndex(index)
        for i in range(self._mode_stack.count()):
            vertical = QSizePolicy.Policy.Preferred if i == index else QSizePolicy.Policy.Ignored
            self._mode_stack.widget(i).setSizePolicy(QSizePolicy.Policy.Preferred, vertical)
        self._mode_stack.adjustSize()

    def _set_mode(self, mode: str) -> None:
        if mode == self._mode or mode not in ("tutor", "repair"):
            return
        if self._llm_worker is not None and self._llm_worker.isRunning():
            self._cancel_inflight()
            self._chat.end_assistant_stream()
        self._mode = mode
        if not self._mode_actions[mode].isChecked():
            self._mode_actions[mode].setChecked(True)
        repair = mode == "repair"
        self._show_mode_page(1 if repair else 0)
        self._camera.set_mode(mode)
        self._dialog_history.clear()
        self._user_marks.clear()
        self._push_marks()
        self._app_title.setText("维修助手" if repair else "学习助手")
        self._copilot_title.setText("维修顾问" if repair else "助教")
        self._chat.set_assistant_name("维修顾问" if repair else "助教")
        self._chat.set_input_placeholder(
            "描述故障现象，或问「标记 1 是什么」…" if repair else "问问当前这一步…"
        )
        if repair:
            self._refresh_repair_panel()
            self._refresh_repair_overlay()
            case = self._repair.case()
            if case:
                self._chat.append_system(
                    f"已进入维修模式，继续维修单 #{case['id']}。可以在侧栏点「新建」开始新的维修。"
                )
            else:
                self._chat.append_system(
                    "已进入维修模式。先说说设备型号和故障现象，我会列出检查清单并在画面上标出要查的位置。"
                    "需要时可上传原理图或维修手册。"
                )
        else:
            self._repair_frozen = False
            self._camera.set_frozen(False)
            self._repair_panel.set_frozen(False)
            self._repair_panel.set_verifying(False)
            self._camera.set_shapes([])
            self._apply_overlay_from_step()
            self._chat.append_system("已回到教学模式")
        if self._voice_active:
            self._enter_listening()

    def _repair_panel_frozen(self) -> bool:
        return self._repair_frozen

    def _refresh_repair_panel(self) -> None:
        self._repair_panel.show_case(self._repair.case())
        self._repair_panel.show_guide(self._repair.guide)
        self._repair_panel.show_documents(self._repair.documents())

    def _repair_hint(self, case: dict | None) -> dict:
        current_step = self._repair.current_step()
        if current_step is not None:
            index, step = current_step
            guide = self._repair.guide or {}
            text = step.get("title") or ""
            if step.get("status") == "failed" and step.get("feedback"):
                text += f" · {step['feedback']}"
            elif step.get("instruction"):
                text += f" · {step['instruction']}"
            return {
                "hint": text[:80],
                "step_label": f"步骤 {index + 1}/{len(guide.get('steps') or [])}",
                "check_id": guide.get("check_id"),
            }
        if not case:
            return {"step_label": "维修"}
        checks = case.get("checks") or []
        current = next((c for c in checks if c["id"] == case.get("current_check_id")), None)
        if current is None:
            if case.get("diagnosis"):
                return {"hint": "已给出结论，详见侧栏", "step_label": "维修"}
            if checks:
                return {"hint": "清单已查完，还有什么现象可以继续说", "step_label": "维修"}
            return {"step_label": "维修"}
        text = current["title"]
        if current.get("expected"):
            text += f" · 预期 {current['expected']}"
        return {
            "hint": text,
            "step_label": f"检查 {current['seq']}/{len(checks)}",
            "check_id": current["id"],
        }

    def _refresh_repair_overlay(self) -> None:
        if self._mode != "repair":
            return
        self._camera.set_overlay_hint(self._repair_hint(self._repair.case()))
        self._camera.set_shapes(self._repair.current_shapes(), self._repair.board_box)

    @Slot(str)
    def _on_repair_changed(self, what: str) -> None:
        if self._closing:
            return
        if what == "documents":
            self._repair_panel.show_documents(self._repair.documents())
            return
        if what == "case":
            self._refresh_repair_panel()
        elif what == "conclusion":
            self._repair_panel.show_case(self._repair.case())
        elif what == "guide":
            self._repair_panel.show_guide(self._repair.guide)
            self._announce_guide()
        self._refresh_repair_overlay()

    @Slot(str)
    def _on_tracking_changed(self, status: str) -> None:
        self._track_status = status
        self._repair_panel.set_tracking(status)
        if self._mode != "repair":
            return
        if status == "tracking":
            self._lost_notified = False
        elif status == "lost" and not self._lost_notified and self._repair.current_shapes():
            self._lost_notified = True
            self._chat.append_system(
                "画面跟丢了，图形先隐藏。把板子移回原来的位置会自动恢复；"
                "换了角度的话点「重新对准」，或让我重新标注。"
            )

    def _on_repair_realign(self) -> None:
        if self._repair_frozen:
            self._repair_panel.set_frozen(False)
            self._on_repair_freeze(False)
        snap = self._camera.anchor_snapshot()
        if snap is None:
            self._chat.append_system("摄像头还没有画面")
            return
        self._repair.rebase(snap[2])
        self._rebase_marks(snap[2])
        self._refresh_repair_overlay()
        self._lost_notified = False
        self._chat.append_system("已按当前画面重新锚定图形。位置对不上的话，让我重新标一下。")

    def _on_guide_next(self) -> None:
        current = self._repair.current_step()
        if current is None:
            return
        index, step = current
        guide = self._repair.advance(mark="skipped")
        nxt = self._repair.current_step()
        ended = guide is None or guide.get("status") != "active" or nxt is None
        self._chat.append_event(
            f"第 {index + 1} 步 · 已跳过",
            step.get("title") or "",
            "指引已结束" if ended else f"下一步：{nxt[1]['title']}",
            tone="skip",
        )

    def _on_guide_end(self) -> None:
        guide = self._repair.guide
        if not guide:
            return
        steps = guide.get("steps") or []
        done = sum(1 for s in steps if s.get("status") == "done")
        self._repair.end_guide("用户手动结束", "cancelled" if done < len(steps) else "done")
        self._chat.append_event(
            "指引已结束", guide.get("title") or "", f"确认完成 {done} / {len(steps)} 步", tone="info"
        )

    def _snapshot_guide(self, guide: dict | None) -> dict | None:
        if not guide or guide.get("status") != "active":
            return None
        return {
            "id": guide["id"],
            "title": guide.get("title") or "",
            "steps": [(s.get("status"), s.get("feedback") or "") for s in guide.get("steps") or []],
        }

    def _announce_guide(self) -> None:
        """Turn guide changes made by the assistant into cards: started, step confirmed, not yet, finished."""
        guide = self._repair.guide
        before, after = self._guide_seen, self._snapshot_guide(guide)
        self._guide_seen = after
        if after is not None and (before is None or before["id"] != after["id"]):
            steps = guide.get("steps") or []
            self._chat.append_event(
                f"操作指引 · 共 {len(steps)} 步",
                after["title"],
                f"第 1 步：{steps[0].get('title') or ''}" if steps else "",
                tone="info",
            )
            return
        if before is None:
            return
        record = guide if after is not None else self._repair.cases.get_guide(before["id"])
        if not record:
            return
        steps = record.get("steps") or []
        for i, (old, new) in enumerate(zip(before["steps"], [(s.get("status"), s.get("feedback") or "") for s in steps])):
            if old == new:
                continue
            title = steps[i].get("title") or ""
            if new[0] == "done":
                self._chat.append_event(
                    f"第 {i + 1} 步 · 已确认", title, new[1] or "助手看了画面，这一步完成了", tone="pass", stamp="通过"
                )
            elif new[0] == "failed":
                self._chat.append_event(
                    f"第 {i + 1} 步 · 还没完成", title, new[1] or "按画面上的提示再试一次", tone="fail", stamp="未通过"
                )
        if after is None and record.get("status") == "done" and all(s.get("status") == "done" for s in steps):
            self._chat.append_event(
                "指引完成", record.get("title") or "", f"{len(steps)} 步全部确认通过", tone="pass", stamp="完成"
            )

    def _verify_guide_step(self) -> None:
        current = self._repair.current_step()
        if current is None:
            return
        if self._llm is None or (self._llm_busy() and not self._voice_active):
            self._chat.append_system("助手正忙，等它说完再确认")
            return
        if not self._settings.llm_send_frame or self._llm.supports_vision is False:
            self._chat.append_system("确认需要把画面发给助手：请在设置里开启发送画面，或点「下一步」手动继续")
            return
        if self._repair_frozen:
            self._repair_panel.set_frozen(False)
            self._on_repair_freeze(False, quiet=True)
        index, step = current
        text = verify_step_message(index, step.get("title") or "")
        self._repair_panel.set_verifying(True)
        verify = step.get("verify")
        self._send_repair_message(
            text,
            event=(
                f"请助手确认 · 第 {index + 1} 步",
                step.get("title") or "",
                f"确认标准：{verify}" if verify else "看画面判断这一步是否完成",
                "ask",
            ),
        )

    def _llm_busy(self) -> bool:
        return self._llm_worker is not None and self._llm_worker.isRunning()

    def _send_repair_message(self, text: str, event: tuple[str, str, str, str] | None = None) -> None:
        """Send ``text`` to the assistant; ``event`` shows a card in its place instead of a user bubble."""
        if self._llm is None or (self._llm_busy() and not self._voice_active):
            self._repair_panel.set_verifying(False)
            self._chat.append_system("已记录。助手空闲后可以接着问。")
            return
        if event is None:
            self._chat.append_user(text)
        else:
            eyebrow, title, detail, tone = event
            self._chat.append_event(eyebrow, title, detail, tone=tone)
        self._on_chat(text)

    def _on_repair_new_case(self, device: str, symptom: str) -> None:
        case = self._repair.new_case(device, symptom)
        self._dialog_history.clear()
        event = (f"新建维修单 #{case['id']:03d}", device or "设备未填写", symptom, "info")
        if device or symptom:
            self._send_repair_message(new_case_message(device, symptom), event=event)
        else:
            self._chat.append_event(*event[:3], tone="info")

    def _on_repair_result(self, check_id: int, status: str, measured: str) -> None:
        try:
            result = self._repair.record_result(check_id, status, measured)
        except ValueError as exc:
            self._chat.append_system(str(exc))
            return
        check = result["recorded"]
        nxt = result["next_check"]
        word = STATUS_TEXT.get(status, status)
        text = check_result_message(check["seq"], check["title"], status, measured, last=nxt is None)
        tone = {"pass": "pass", "fail": "fail", "skip": "skip"}.get(status, "info")
        eyebrow = f"检查 {check['seq']:02d} · {word}"
        reading = f"实测 {measured}" if measured else ""
        if status == "fail":
            detail = " · ".join(p for p in (reading, "已请助手分析下一步") if p)
            self._send_repair_message(text, event=(eyebrow, check["title"], detail, tone))
        elif nxt is None:
            detail = " · ".join(p for p in (reading, "清单查完了，已请助手给结论") if p)
            self._send_repair_message(text, event=(eyebrow, check["title"], detail, tone))
        else:
            detail = " · ".join(p for p in (reading, f"下一项：{nxt['seq']}. {nxt['title']}") if p)
            self._chat.append_event(eyebrow, check["title"], detail, tone=tone)

    def _on_repair_focus(self, check_id: int) -> None:
        try:
            self._repair.focus(check_id)
        except ValueError as exc:
            self._chat.append_system(str(exc))

    def _next_check(self) -> None:
        case = self._repair.case()
        if not case or not case.get("checks"):
            self._chat.append_system("还没有检查清单，先描述一下故障")
            return
        nxt = self._repair.cases.next_pending(case["id"], case.get("current_check_id"))
        if nxt is None:
            self._chat.append_system("清单里没有待查项了")
            return
        self._repair.focus(nxt["id"])
        self._chat.append_system(f"下一项：{nxt['seq']}. {nxt['title']}")

    def _on_repair_files(self, paths: list) -> None:
        if self._ingest_worker is not None and self._ingest_worker.isRunning():
            self._chat.append_system("上一批资料还在导入，请稍候")
            return
        self._chat.append_system(f"正在导入 {len(paths)} 份资料…")
        self._ingest_worker = DocIngestWorker(self._repair, [str(p) for p in paths])
        _keep_alive(self._ingest_worker)
        self._ingest_worker.ingested.connect(self._on_doc_ingested)
        self._ingest_worker.failed.connect(lambda msg: self._chat.append_system(f"导入失败 {msg}"))
        self._ingest_worker.start()

    @Slot(object)
    def _on_doc_ingested(self, doc: object) -> None:
        if self._closing or not isinstance(doc, dict):
            return
        if doc["kind"] == "pdf":
            state = "可检索文字" if doc["has_text"] else "扫描件，只能看图"
            detail = f"{doc['pages']} 页，{state}"
        else:
            detail = "图片"
        self._chat.append_system(f"已导入资料 [{doc['id']}] {doc['name']}（{detail}）")

    def _on_repair_remove_doc(self, doc_id: int) -> None:
        self._repair.remove_document(doc_id)
        self._chat.append_system(f"已移除资料 [{doc_id}]")

    def _on_repair_ifixit(self) -> None:
        device = self._repair_panel.device or "这台设备"
        self._send_repair_message(ifixit_message(device))

    def _on_repair_freeze(self, frozen: bool, quiet: bool = False) -> None:
        self._repair_frozen = frozen
        H = self._camera.set_frozen(frozen)
        if frozen:
            self._repair.rebase(H)
            self._rebase_marks(H)
            self._refresh_repair_overlay()
        if quiet:
            return
        self._chat.append_system(
            "画面已定格，图形固定在这张图上。测量完点「取消定格」回到实时画面。"
            if frozen
            else "已回到实时画面，图形会继续贴着板子移动。"
        )

    def closeEvent(self, event) -> None:  # noqa: N802
        self._closing = True
        self._voice_active = False
        self._voice_phase = "idle"
        self._turn += 1
        if hasattr(self, "_voice_session"):
            self._voice_session.stop()
        self._tts.close()
        self._camera.stop()
        self._camera.wait(2000)
        for worker in (
            self._board_worker,
            self._llm_connect_worker,
            self._llm_worker,
            self._asr_worker,
            self._ingest_worker,
        ):
            if worker is not None and worker.isRunning():
                worker.requestInterruption()
        super().closeEvent(event)
