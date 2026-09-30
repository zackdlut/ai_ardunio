from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from agent.tools import AgentContext, build_openai_tools, build_tool_handlers
from agent.tutor_context import build_extra_system
from arduino.board_manager import list_boards
from config.settings import get_settings
from project.repository import create_session
from project.state_machine import ProjectStateMachine, load_project
from ui.camera_devices import list_cameras
from ui.camera_worker import CameraWorker
from ui.camera_widget import CameraWidget
from ui.chat_widget import ChatWidget
from voice.asr import AsrBackend, create_asr
from voice.tts import create_tts, prepare_speech_text, spoken_text_from_reply

if TYPE_CHECKING:
    from agent.llm_client import LLMClient

PROACTIVE_COOLDOWN_S = 30.0
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


class ASRWorker(QThread):
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, asr, audio: np.ndarray, sample_rate: int) -> None:
        super().__init__()
        self._asr = asr
        self._audio = audio
        self._sample_rate = sample_rate

    def run(self) -> None:
        try:
            text = self._asr.transcribe_pcm(self._audio, self._sample_rate)
            self.finished.emit(text)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class LLMWorker(QThread):
    finished = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        client: LLMClient,
        message: str,
        tools,
        handlers,
        history: list[dict[str, str]],
        extra_system: str,
        voice_mode: bool,
    ) -> None:
        super().__init__()
        self._client = client
        self._message = message
        self._tools = tools
        self._handlers = handlers
        self._history = history
        self._extra_system = extra_system
        self._voice_mode = voice_mode

    def run(self) -> None:
        try:
            reply = self._client.chat_with_tools(
                self._message,
                self._tools,
                self._handlers,
                extra_system=self._extra_system,
                history=self._history,
                voice_mode=self._voice_mode,
            )
            self.finished.emit(reply)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


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
        self._agent_ctx = AgentContext(
            state_machine=self._state_machine,
            get_scene=lambda: self._scene,
            get_detections_json=lambda: [],
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
        self._recording: list[np.ndarray] = []
        self._sample_rate = 16000
        self._dialog_history: list[dict[str, str]] = []
        self._pending_user_text = ""
        self._pending_voice_mode = False
        self._last_proactive_ts = 0.0

        self._build_ui()
        self._seed_camera_combo()
        self._update_step_label()
        self._chat.append_system(self._asr_mode_label())

        self._camera = CameraWorker(camera_index=self._selected_camera_index())
        self._camera.frame_ready.connect(self._camera_widget.show_frame)
        self._camera.scene_updated.connect(self._on_scene)
        self._camera.gesture_detected.connect(self._on_gesture)
        self._camera.status_message.connect(self._on_camera_status)
        self._camera.start()
        self._camera_combo.currentIndexChanged.connect(self._on_camera_changed)

        self._apply_overlay_from_step()
        self._start_llm_connect()
        self._refresh_boards()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("Arduino 学习助手")
        title.setObjectName("appTitle")
        layout.addWidget(title)

        top = QHBoxLayout()
        top.setSpacing(12)

        camera_frame = QFrame()
        camera_frame.setObjectName("cameraFrame")
        camera_layout = QVBoxLayout(camera_frame)
        camera_layout.setContentsMargins(10, 10, 10, 10)
        cam_header = QHBoxLayout()
        cam_label = QLabel("工作台视野")
        cam_label.setObjectName("sectionEyebrow")
        cam_header.addWidget(cam_label)
        cam_header.addStretch()
        self._camera_combo = QComboBox()
        self._camera_combo.setMinimumWidth(160)
        refresh_cam_btn = QPushButton("刷新")
        refresh_cam_btn.setToolTip("重新扫描本机摄像头")
        refresh_cam_btn.clicked.connect(self._refresh_cameras)
        cam_header.addWidget(QLabel("摄像头"))
        cam_header.addWidget(self._camera_combo)
        cam_header.addWidget(refresh_cam_btn)
        camera_layout.addLayout(cam_header)
        self._camera_widget = CameraWidget()
        camera_layout.addWidget(self._camera_widget, stretch=1)
        top.addWidget(camera_frame, stretch=3)

        side_panel = QFrame()
        side_panel.setObjectName("sidePanel")
        side = QVBoxLayout(side_panel)
        side.setContentsMargins(14, 14, 14, 14)
        side.setSpacing(10)

        step_eyebrow = QLabel("当前步骤")
        step_eyebrow.setObjectName("sectionEyebrow")
        side.addWidget(step_eyebrow)

        self._step_title = QLabel()
        self._step_title.setObjectName("stepTitle")
        side.addWidget(self._step_title)

        self._step_progress = QProgressBar()
        self._step_progress.setObjectName("stepProgress")
        self._step_progress.setTextVisible(False)
        self._step_progress.setFixedHeight(6)
        side.addWidget(self._step_progress)

        self._step_hint = QLabel()
        self._step_hint.setObjectName("stepHint")
        self._step_hint.setWordWrap(True)
        side.addWidget(self._step_hint)

        side.addSpacing(4)
        hw_eyebrow = QLabel("硬件连接")
        hw_eyebrow.setObjectName("sectionEyebrow")
        side.addWidget(hw_eyebrow)

        board_row = QHBoxLayout()
        board_row.setSpacing(8)
        self._port_combo = QComboBox()
        refresh_btn = QPushButton("刷新")
        refresh_btn.setToolTip("重新扫描已连接的 Arduino 开发板")
        refresh_btn.clicked.connect(self._refresh_boards)
        board_row.addWidget(self._port_combo, stretch=1)
        board_row.addWidget(refresh_btn)
        side.addLayout(board_row)

        asr_row = QHBoxLayout()
        asr_row.setSpacing(8)
        self._asr_combo = QComboBox()
        self._asr_combo.addItem("本地 Whisper", "local")
        self._asr_combo.addItem("远程 Ollama ASR", "remote")
        asr_idx = self._asr_combo.findData(self._settings.asr_backend)
        if asr_idx >= 0:
            self._asr_combo.setCurrentIndex(asr_idx)
        self._asr_combo.setToolTip(
            "本地：faster-whisper（CPU）；远程：Ollama qwen3-asr（见 .env）"
        )
        self._asr_combo.currentIndexChanged.connect(self._on_asr_mode_changed)
        asr_row.addWidget(QLabel("语音识别"))
        asr_row.addWidget(self._asr_combo, stretch=1)
        side.addLayout(asr_row)

        tts_row = QHBoxLayout()
        tts_row.setSpacing(8)
        self._tts_enabled = QCheckBox("语音回复")
        self._tts_enabled.setChecked(self._settings.tts_enabled)
        self._tts_enabled.setToolTip("助手回复后用本机中文朗读（可停止）")
        stop_tts_btn = QPushButton("停止朗读")
        stop_tts_btn.clicked.connect(self._tts.stop)
        tts_row.addWidget(self._tts_enabled)
        tts_row.addWidget(stop_tts_btn)
        side.addLayout(tts_row)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        adv_btn = QPushButton("下一步")
        adv_btn.setToolTip("手动进入下一教学步骤")
        adv_btn.clicked.connect(self._advance_step)
        clear_chat_btn = QPushButton("清空对话")
        clear_chat_btn.setToolTip("清除与助手的多轮记忆")
        clear_chat_btn.clicked.connect(self._clear_dialog)
        upload_btn = QPushButton("烧录 Blink")
        upload_btn.setObjectName("primaryButton")
        upload_btn.setToolTip("编译并上传 Blink 示例到所选串口")
        upload_btn.clicked.connect(self._upload_blink)
        btn_row.addWidget(adv_btn)
        btn_row.addWidget(clear_chat_btn)
        btn_row.addWidget(upload_btn)
        side.addLayout(btn_row)
        side.addStretch(1)
        top.addWidget(side_panel, stretch=1)
        layout.addLayout(top, stretch=5)

        self._chat = ChatWidget()
        self._chat.send_requested.connect(lambda t: self._on_chat(t, voice_mode=False))
        self._chat.voice_ptt_pressed.connect(self._start_voice)
        self._chat.voice_ptt_released.connect(self._stop_voice)
        layout.addWidget(self._chat, stretch=3)

        log_panel = QFrame()
        log_panel.setObjectName("logPanel")
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(12, 10, 12, 10)
        log_layout.setSpacing(6)
        log_header = QLabel("编译与烧录")
        log_header.setObjectName("sectionEyebrow")
        log_layout.addWidget(log_header)
        self._monitor = QTextEdit()
        self._monitor.setObjectName("buildLog")
        self._monitor.setReadOnly(True)
        self._monitor.setMaximumHeight(120)
        log_layout.addWidget(self._monitor)
        layout.addWidget(log_panel)

    def _clear_dialog(self) -> None:
        self._dialog_history.clear()
        self._chat.append_system("已清空对话记忆")

    def _tts_on(self) -> bool:
        return self._tts_enabled.isChecked()

    def _should_speak_reply(self, voice_mode: bool) -> bool:
        return self._tts_on() or voice_mode

    def _speak_if_enabled(self, text: str, force: bool = False) -> None:
        if not text.strip():
            return
        if force or self._tts_on():
            self._tts.speak(
                prepare_speech_text(text),
                on_error=lambda e: self._chat.append_system(f"TTS 失败: {e}"),
            )

    def _agent_busy(self) -> bool:
        if self._llm_worker is not None and self._llm_worker.isRunning():
            return True
        if self._asr_worker is not None and self._asr_worker.isRunning():
            return True
        return False

    def _extra_system_for_chat(self, voice_mode: bool) -> str:
        return build_extra_system(
            self._state_machine.progress(),
            self._verify_ok,
            self._verify_msg,
            self._scene,
            voice_mode,
        )

    @Slot(dict)
    def _on_scene(self, scene: dict) -> None:
        self._scene = scene
        ok, msg = self._state_machine.verify_step(scene)
        self._verify_ok = ok
        self._verify_msg = msg
        self._step_hint.setToolTip(msg)

        if ok or self._agent_busy():
            return
        now = time.monotonic()
        if now - self._last_proactive_ts < PROACTIVE_COOLDOWN_S:
            return
        self._last_proactive_ts = now
        step = self._state_machine.current_step
        tip = (
            f"导师提醒：{msg} 请按当前步骤操作——{step.hint}。"
            "看不清或识别不准可以调整摄像头或语音问我。"
        )
        self._chat.append_system(tip)
        self._speak_if_enabled(step.hint)

    def _on_gesture(self, cmd: str) -> None:
        if cmd == "next":
            self._advance_step()
        elif cmd == "pause":
            self._camera.set_paused(True)
            self._chat.append_system("手势：已暂停视觉处理")
        elif cmd == "rescan":
            self._camera.set_paused(False)
            self._camera.request_rescan()
            self._chat.append_system("手势：重新识别")

    def _apply_overlay_from_step(self) -> None:
        step = self._state_machine.current_step
        hint = {"hint": step.hint, "requires": step.requires, "pin": step.pin}
        self._agent_ctx.overlay_hint = hint
        self._camera.set_overlay_hint(hint)

    def _update_step_label(self) -> None:
        p = self._state_machine.progress()
        self._step_title.setText(
            f"{p['title']} · 第 {p['step_index'] + 1} / {p['total_steps']} 步"
        )
        self._step_hint.setText(f"{p['current_step']}\n{p['current_hint']}")
        total = max(p["total_steps"], 1)
        self._step_progress.setMaximum(total)
        self._step_progress.setValue(min(p["step_index"] + 1, total))

    def _announce_current_step(self) -> None:
        p = self._state_machine.progress()
        hint = p["current_hint"]
        self._chat.append_system(f"进入步骤：{p['current_step']} — {hint}")
        self._speak_if_enabled(hint)

    def _advance_step(self) -> None:
        if self._state_machine.advance():
            self._apply_overlay_from_step()
            self._update_step_label()
            self._announce_current_step()

    def _on_camera_status(self, msg: str) -> None:
        if self._closing:
            return
        self._chat.append_system(msg)

    def _seed_camera_combo(self) -> None:
        index = int(self._settings.camera_index)
        self._camera_combo.blockSignals(True)
        self._camera_combo.clear()
        self._camera_combo.addItem(f"摄像头 {index}", index)
        self._camera_combo.blockSignals(False)

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
        data = self._camera_combo.currentData()
        if data is not None and data != "":
            return int(data)
        return get_settings().camera_index

    def _refresh_cameras(self) -> None:
        preferred = self._selected_camera_index()
        self._camera_combo.blockSignals(True)
        self._camera_combo.clear()
        try:
            devices = list_cameras()
            for index, label in devices:
                self._camera_combo.addItem(label, index)
            if not devices:
                self._camera_combo.addItem("未检测到摄像头", "")
                self._chat.append_system("未检测到可用摄像头")
            else:
                pick = preferred
                if not any(i == pick for i, _ in devices):
                    pick = devices[0][0]
                idx = self._camera_combo.findData(pick)
                if idx >= 0:
                    self._camera_combo.setCurrentIndex(idx)
        except Exception as exc:  # noqa: BLE001
            self._camera_combo.addItem(f"扫描失败: {exc}", "")
            self._chat.append_system(f"摄像头扫描: {exc}")
        finally:
            self._camera_combo.blockSignals(False)
        if hasattr(self, "_camera"):
            self._on_camera_changed()

    def _on_camera_changed(self) -> None:
        if not hasattr(self, "_camera"):
            return
        index = self._camera_combo.currentData()
        if index is None or index == "":
            return
        self._camera.set_camera_index(int(index))

    def _refresh_boards(self) -> None:
        if self._board_worker is not None and self._board_worker.isRunning():
            return
        self._port_combo.clear()
        self._port_combo.addItem("正在扫描…", "")
        self._board_worker = BoardScanWorker()
        _keep_alive(self._board_worker)
        self._board_worker.succeeded.connect(self._on_boards_ready)
        self._board_worker.failed.connect(self._on_boards_failed)
        self._board_worker.start()

    @Slot(object)
    def _on_boards_ready(self, boards: object) -> None:
        if self._closing:
            return
        preferred = self._selected_port()
        self._port_combo.clear()
        for board in boards:
            self._port_combo.addItem(f"{board.port} ({board.label})", board.port)
        if not boards:
            self._port_combo.addItem("未检测到开发板", "")
            return
        if preferred:
            idx = self._port_combo.findData(preferred)
            if idx >= 0:
                self._port_combo.setCurrentIndex(idx)

    @Slot(str)
    def _on_boards_failed(self, msg: str) -> None:
        if self._closing:
            return
        self._port_combo.clear()
        self._port_combo.addItem("无法列出串口", "")
        self._chat.append_system(f"硬件连接: {msg}")

    def _selected_port(self) -> str:
        return self._port_combo.currentData() or ""

    def _upload_blink(self) -> None:
        self._agent_ctx.selected_port = self._selected_port()
        log = self._handlers["compile_and_upload"](sketch_name="blink", port=self._agent_ctx.selected_port)
        self._monitor.append(log)
        self._chat.append_system(log[:500])

    def _asr_mode_label(self) -> str:
        backend = self._asr_combo.currentData() if hasattr(self, "_asr_combo") else self._settings.asr_backend
        if backend == "remote":
            return (
                f"ASR：远程 Ollama（{self._settings.ollama_asr_model} @ "
                f"{self._settings.resolve_ollama_base_url()}）"
            )
        return f"ASR：本地 Whisper（{self._settings.whisper_model_size}）"

    def _on_asr_mode_changed(self) -> None:
        backend: AsrBackend = self._asr_combo.currentData() or "local"
        self._asr = create_asr(self._settings, backend=backend)
        self._chat.append_system(self._asr_mode_label())

    def _start_voice(self) -> None:
        self._tts.stop()
        if self._asr_worker is not None and self._asr_worker.isRunning():
            self._chat.append_system("上一段语音仍在识别中，请稍候")
            return
        try:
            import sounddevice as sd

            self._recording = []
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._chat.append_system("录音中…")
        except Exception as exc:  # noqa: BLE001
            self._chat.append_system(f"麦克风不可用: {exc}")

    def _audio_callback(self, indata, frames, time, status) -> None:  # noqa: ARG001
        self._recording.append(indata.copy())

    def _stop_voice(self) -> None:
        if hasattr(self, "_stream"):
            self._stream.stop()
            self._stream.close()
        if not self._recording:
            return
        audio = np.concatenate(self._recording, axis=0).flatten().astype(np.float32)
        self._chat.append_system("识别中…")
        self._asr_worker = ASRWorker(self._asr, audio, self._sample_rate)
        self._asr_worker.finished.connect(self._on_asr_finished)
        self._asr_worker.failed.connect(
            lambda exc: self._chat.append_system(f"ASR 失败: {exc}")
        )
        self._asr_worker.start()

    @Slot(str)
    def _on_asr_finished(self, text: str) -> None:
        if text:
            self._chat.append_user(text)
            self._on_chat(text, voice_mode=True)
        else:
            self._chat.append_system("未识别到语音")

    def _on_chat(self, text: str, voice_mode: bool = False) -> None:
        if self._llm is None:
            self._chat.append_system("助手仍在连接，请稍候")
            return
        if self._llm_worker is not None and self._llm_worker.isRunning():
            self._chat.append_system("助手仍在回复上一问，请稍候")
            return
        self._agent_ctx.selected_port = self._selected_port()
        self._pending_user_text = text
        self._pending_voice_mode = voice_mode
        extra = self._extra_system_for_chat(voice_mode)
        self._llm_worker = LLMWorker(
            self._llm,
            text,
            self._tools,
            self._handlers,
            self._dialog_history,
            extra,
            voice_mode,
        )
        self._llm_worker.finished.connect(self._on_llm_reply)
        self._llm_worker.failed.connect(lambda e: self._chat.append_assistant(f"错误: {e}"))
        self._chat.append_system("思考中…")
        self._llm_worker.start()

    @Slot(str)
    def _on_llm_reply(self, reply: str) -> None:
        self._chat.append_assistant(reply)
        self._dialog_history.append({"role": "user", "content": self._pending_user_text})
        self._dialog_history.append({"role": "assistant", "content": reply})
        if len(self._dialog_history) > 20:
            self._dialog_history = self._dialog_history[-20:]
        if self._should_speak_reply(self._pending_voice_mode):
            spoken = spoken_text_from_reply(reply, self._pending_voice_mode)
            if spoken:
                self._tts.speak(
                    spoken,
                    on_error=lambda e: self._chat.append_system(f"TTS 失败: {e}"),
                )
            elif self._pending_voice_mode:
                self._chat.append_system("TTS：回复为空或无法朗读，请查看文字回复")
        if self._agent_ctx.overlay_hint:
            self._camera.set_overlay_hint(self._agent_ctx.overlay_hint)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._closing = True
        self._tts.stop()
        self._camera.stop()
        self._camera.wait(2000)
        for worker in (self._board_worker, self._llm_connect_worker):
            if worker is not None and worker.isRunning():
                worker.requestInterruption()
        super().closeEvent(event)
