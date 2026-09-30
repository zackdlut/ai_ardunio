from __future__ import annotations

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
    QSplitter,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from agent.tools import AgentContext, build_openai_tools, build_tool_handlers
from vision.marks import nearest_component
from agent.tutor_context import build_extra_system
from arduino.board_manager import list_boards
from config.settings import get_settings
from project.repository import create_session
from project.state_machine import ProjectStateMachine, load_project
from ui.camera_devices import list_cameras
from ui.camera_worker import CameraWorker
from ui.camera_widget import CameraWidget
from ui.chat_widget import ChatWidget
from ui.chrome import RoundedCard, ThemedMenu
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
    ) -> None:
        super().__init__()
        self._client = client
        self._message = message
        self._tools = tools
        self._handlers = handlers
        self._history = history
        self._extra_system = extra_system
        self._voice_mode = voice_mode
        self._turn = turn
        self._stream = stream

    def run(self) -> None:
        if self.isInterruptionRequested():
            return
        try:
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
                )
            if not self.isInterruptionRequested():
                self.succeeded.emit(self._turn, reply)
        except Exception as exc:  # noqa: BLE001
            if not self.isInterruptionRequested():
                self.failed.emit(self._turn, str(exc))


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
                    f"LLM 连接失败: {exc}（请检查 OPENAI_BASE_URL 与 NO_PROXY）",
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
        self._seed_camera_menu()
        self._update_step_label()
        self._chat.append_system(self._asr_mode_label())

        self._camera = CameraWorker(camera_index=self._selected_camera_index())
        self._camera.frame_ready.connect(self._camera_widget.show_frame)
        self._camera.scene_updated.connect(self._on_scene)
        self._camera.gesture_detected.connect(self._on_gesture)
        self._camera.status_message.connect(self._on_camera_status)
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
        copilot_title = QLabel("助教")
        copilot_title.setObjectName("copilotTitle")
        header.addWidget(presence)
        header.addWidget(copilot_title)
        header.addStretch(1)
        side.addLayout(header)

        step_eyebrow = QLabel("当前步骤")
        step_eyebrow.setObjectName("sectionEyebrow")
        side.addWidget(step_eyebrow)

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
        side.addLayout(title_row)

        self._step_hint = QLabel()
        self._step_hint.setObjectName("stepHint")
        self._step_hint.setWordWrap(True)
        side.addWidget(self._step_hint)

        self._step_progress = QProgressBar()
        self._step_progress.setObjectName("stepProgress")
        self._step_progress.setTextVisible(False)
        self._step_progress.setFixedHeight(3)
        side.addWidget(self._step_progress)

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
        title = QLabel("学习助手")
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
        self._clear_marks_action.setToolTip("去掉你在画面上点出的标记")
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
        self._asr_local = voice.addAction(icon_local(), "本地 Whisper")
        self._asr_local.setCheckable(True)
        self._asr_local.setData("local")
        self._asr_local.setToolTip("faster-whisper（CPU）")
        self._asr_remote = voice.addAction(icon_remote(), "远程 Ollama")
        self._asr_remote.setCheckable(True)
        self._asr_remote.setData("remote")
        self._asr_remote.setToolTip("Ollama qwen3-asr（见 .env）")
        self._asr_group.addAction(self._asr_local)
        self._asr_group.addAction(self._asr_remote)
        if self._settings.asr_backend == "remote":
            self._asr_remote.setChecked(True)
        else:
            self._asr_local.setChecked(True)
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

    def _extra_system_for_chat(self, voice_mode: bool) -> str:
        return build_extra_system(
            self._state_machine.progress(),
            self._verify_ok,
            self._verify_msg,
            self._scene,
            voice_mode,
        )

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
        for mark in self._user_marks:
            if abs(float(mark["nx"]) - nx) < 0.03 and abs(float(mark["ny"]) - ny) < 0.03:
                return
        if len(self._user_marks) >= 6:
            self._user_marks.pop(0)
        self._mark_seq += 1
        label = f"标记 {self._mark_seq}"
        self._user_marks.append({"nx": round(nx, 3), "ny": round(ny, 3), "label": label})
        self._push_marks()
        near = nearest_component(self._scene, nx, ny)
        if near:
            self._chat.append_system(f"已添加{label}，靠近{near}。可以直接问「{label}这里该怎么接」。")
        else:
            self._chat.append_system(f"已添加{label}。可以直接问「{label}这里该怎么接」。")

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
        self._llm_connect_worker = LLMConnectWorker()
        _keep_alive(self._llm_connect_worker)
        self._llm_connect_worker.ready.connect(self._on_llm_ready)
        self._llm_connect_worker.start()

    @Slot(object, bool, str)
    def _on_llm_ready(self, client: object, ok: bool, msg: str) -> None:
        if self._closing:
            return
        self._llm = client  # type: ignore[assignment]
        self._chat.append_system(msg if ok else f"警告: {msg}")

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
        if self._asr_remote.isChecked():
            return "remote"
        return "local"

    def _asr_mode_label(self) -> str:
        if self._selected_asr_backend() == "remote":
            return (
                f"ASR：远程 Ollama（{self._settings.ollama_asr_model} @ "
                f"{self._settings.resolve_ollama_base_url()}）"
            )
        return f"ASR：本地 Whisper（{self._settings.whisper_model_size}）"

    def _on_asr_mode_changed(self, _action: QAction | None = None) -> None:
        backend = self._selected_asr_backend()
        self._asr = create_asr(self._settings, backend=backend)
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
        extra = self._extra_system_for_chat(voice_mode)
        self._llm_worker = LLMWorker(
            self._llm,  # type: ignore[arg-type]
            text,
            self._tools,
            self._handlers,
            list(self._dialog_history),
            extra,
            voice_mode,
            turn,
            stream,
        )
        _keep_alive(self._llm_worker)
        self._llm_worker.delta.connect(self._on_llm_delta)
        self._llm_worker.reasoning.connect(self._on_llm_reasoning)
        self._llm_worker.revoke.connect(self._on_llm_revoke)
        self._llm_worker.succeeded.connect(self._on_llm_reply)
        self._llm_worker.failed.connect(self._on_llm_failed)
        self._chat.begin_thinking()
        self._llm_worker.start()

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
        if self._agent_ctx.overlay_hint:
            self._camera.set_overlay_hint(self._agent_ctx.overlay_hint)

    @Slot(int, str)
    def _on_llm_reply(self, turn: int, reply: str) -> None:
        if turn != self._turn:
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
        self._chat.end_assistant_stream()
        self._chat.append_assistant(f"错误: {message}")
        if self._voice_active:
            self._enter_listening()

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
        ):
            if worker is not None and worker.isRunning():
                worker.requestInterruption()
        super().closeEvent(event)
