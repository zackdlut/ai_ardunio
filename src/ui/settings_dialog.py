"""Chat model, ASR and repair-mode settings, saved to .env and applied without a restart."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from pydantic import ValidationError

from config.settings import Settings, save_settings

CHAT_KEYS = frozenset(
    {"OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL", "LLM_VISION", "LLM_EXTRA_BODY", "LLM_TIMEOUT_S"}
)

REPAIR_KEYS = frozenset(
    {
        "IFIXIT_ENABLED",
        "IFIXIT_BASE_URL",
        "IFIXIT_TIMEOUT_S",
        "REPAIR_DOCS_DIR",
        "REPAIR_FRAME_MAX_SIDE",
        "REPAIR_BOX_FORMAT",
        "REPAIR_TRACKING",
        "REPAIR_TRACK_FEATURES",
        "REPAIR_TRACK_MIN_INLIERS",
        "REPAIR_DEBUG",
        "REPAIR_DEBUG_DIR",
    }
)

_ON_OFF = (("true", "开启"), ("false", "关闭"))
_BOX_FORMATS = (
    ("norm1000", "0–1000 相对坐标（Qwen-VL 等）"),
    ("norm1", "0–1 相对坐标"),
    ("pixel", "所发图片的像素坐标"),
)
_ASR_BACKENDS = (("local", "本地 Whisper"), ("ollama", "远程 Ollama"), ("openai", "远程 API"))
_VISION_CHOICES = (("auto", "自动判断"), ("true", "发送"), ("false", "不发送"))
_API_MODES = (
    ("transcriptions", "transcriptions（/audio/transcriptions）"),
    ("chat", "chat（/chat/completions + input_audio）"),
)
_WHISPER_MODELS = ("tiny", "base", "small", "medium", "large-v3", "large-v3-turbo")


def _line(value: str, placeholder: str = "", secret: bool = False) -> QLineEdit:
    edit = QLineEdit(value)
    edit.setPlaceholderText(placeholder)
    if secret:
        edit.setEchoMode(QLineEdit.EchoMode.PasswordEchoOnEdit)
    return edit


def _choice(options: tuple[tuple[str, str], ...], current: str) -> QComboBox:
    box = QComboBox()
    for value, label in options:
        box.addItem(label, value)
    index = box.findData(current)
    box.setCurrentIndex(max(index, 0))
    return box


def _editable(options: tuple[str, ...], current: str) -> QComboBox:
    box = QComboBox()
    box.setEditable(True)
    box.addItems(options)
    box.setCurrentText(current)
    return box


def _seconds(value: float) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(5, 3600)
    spin.setSuffix(" 秒")
    spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
    spin.setValue(int(value))
    return spin


def _form() -> tuple[QWidget, QFormLayout]:
    page = QWidget()
    form = QFormLayout(page)
    form.setContentsMargins(16, 14, 16, 14)
    form.setHorizontalSpacing(14)
    form.setVerticalSpacing(10)
    return page, form


def _row(form: QFormLayout, text: str, widget: QWidget) -> None:
    label = QLabel(text)
    label.setMinimumWidth(104)
    form.addRow(label, widget)


def _note(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("settingsNote")
    label.setWordWrap(True)
    return label


class SettingsDialog(QDialog):
    """``changed`` holds the env names that were saved; ``settings`` the reloaded config."""

    def __init__(self, settings: Settings, asr_backend: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("settingsDialog")
        self.setWindowTitle("设置")
        self.setMinimumWidth(560)
        self.settings = settings
        self.changed: set[str] = set()

        tabs = QTabWidget()
        tabs.addTab(self._build_chat_tab(settings), "对话模型")
        tabs.addTab(self._build_asr_tab(settings, asr_backend), "语音识别")
        tabs.addTab(self._build_repair_tab(settings), "维修模式")

        self._error = QLabel()
        self._error.setObjectName("settingsError")
        self._error.setWordWrap(True)
        self._error.hide()

        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        save = QPushButton("保存")
        save.setObjectName("primaryButton")
        save.setDefault(True)
        save.clicked.connect(self._save)
        footer = _note("保存到 .env，立即生效")
        footer.setWordWrap(False)
        buttons = QHBoxLayout()
        buttons.addWidget(footer)
        buttons.addStretch(1)
        buttons.addWidget(cancel)
        buttons.addWidget(save)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 14)
        layout.setSpacing(10)
        layout.addWidget(tabs)
        layout.addWidget(self._error)
        layout.addLayout(buttons)

        self._initial = self._values()

    def _build_chat_tab(self, cfg: Settings) -> QWidget:
        page, form = _form()
        self._chat_url = _line(cfg.openai_base_url, "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self._chat_key = _line(cfg.openai_api_key, "Ollama 可随便填", secret=True)
        self._chat_model = _line(cfg.openai_model, "qwen3.8:27b")
        self._chat_vision = _choice(_VISION_CHOICES, cfg.llm_vision)
        self._chat_extra = _line(cfg.llm_extra_body, '{"enable_thinking": false}')
        self._chat_timeout = _seconds(cfg.llm_timeout_s)
        _row(form, "服务地址", self._chat_url)
        _row(form, "API Key", self._chat_key)
        _row(form, "模型", self._chat_model)
        _row(form, "发送摄像头画面", self._chat_vision)
        _row(form, "附加参数 JSON", self._chat_extra)
        _row(form, "超时", self._chat_timeout)
        form.addRow(_note(
            "任意 OpenAI 兼容服务，需支持工具调用与流式输出。"
            "内网地址需在 NO_PROXY 中，云端服务经 HTTPS_PROXY 出网。"
        ))
        return page

    def _build_asr_tab(self, cfg: Settings, backend: str) -> QWidget:
        page, form = _form()
        self._asr_backend = _choice(_ASR_BACKENDS, backend)
        self._asr_language = _line(cfg.asr_language, "留空自动检测")
        self._asr_timeout = _seconds(cfg.asr_timeout_s)
        _row(form, "识别后端", self._asr_backend)
        _row(form, "语言", self._asr_language)
        _row(form, "超时", self._asr_timeout)

        self._whisper_model = _editable(_WHISPER_MODELS, cfg.whisper_model_size)
        self._whisper_device = _editable(("cpu", "cuda", "auto"), cfg.whisper_device)
        self._whisper_compute = _editable(("int8", "int8_float16", "float16", "float32"), cfg.whisper_compute_type)
        self._ollama_url = _line(cfg.ollama_base_url, "留空则用对话服务地址去掉 /v1")
        self._ollama_model = _line(cfg.ollama_asr_model, "frozenlab/qwen3-asr:0.6b")
        self._asr_url = _line(cfg.asr_base_url, "留空沿用对话服务地址")
        self._asr_key = _line(cfg.asr_api_key, "留空沿用对话服务 Key", secret=True)
        self._asr_model = _line(cfg.asr_model, "whisper-1 / FunAudioLLM/SenseVoiceSmall / qwen3-asr-flash")
        self._asr_mode = _choice(_API_MODES, cfg.asr_api_mode)
        self._asr_extra = _line(cfg.asr_extra_body, '{"asr_options": {"language": "zh"}}')
        backend_rows = {
            "local": (
                ("模型", self._whisper_model),
                ("设备", self._whisper_device),
                ("精度", self._whisper_compute),
                (None, _note("模型可填名称或本地 CTranslate2 目录，首次使用会下载。")),
            ),
            "ollama": (
                ("Ollama 地址", self._ollama_url),
                ("模型", self._ollama_model),
            ),
            "openai": (
                ("服务地址", self._asr_url),
                ("API Key", self._asr_key),
                ("模型", self._asr_model),
                ("协议", self._asr_mode),
                ("附加参数 JSON", self._asr_extra),
                (None, _note("chat 协议适用于 qwen3-asr-flash 等，语言写在附加参数里。")),
            ),
        }
        self._asr_form = form
        self._asr_rows: dict[str, list[QWidget]] = {}
        for backend, rows in backend_rows.items():
            for label, widget in rows:
                if label is None:
                    form.addRow(widget)
                else:
                    _row(form, label, widget)
                self._asr_rows.setdefault(backend, []).append(widget)
        self._asr_backend.currentIndexChanged.connect(self._show_backend_rows)
        self._show_backend_rows()
        return page

    def _build_repair_tab(self, cfg: Settings) -> QWidget:
        page, form = _form()
        self._ifixit_enabled = _choice(_ON_OFF, "true" if cfg.ifixit_enabled else "false")
        self._ifixit_url = _line(cfg.ifixit_base_url, "https://www.ifixit.com/api/2.0")
        self._ifixit_timeout = _seconds(cfg.ifixit_timeout_s)
        self._repair_docs = _line(str(cfg.repair_docs_dir), "data/repair_docs")
        self._repair_side = QSpinBox()
        self._repair_side.setRange(480, 2560)
        self._repair_side.setSingleStep(64)
        self._repair_side.setSuffix(" px")
        self._repair_side.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self._repair_side.setValue(int(cfg.repair_frame_max_side))
        self._repair_box = _choice(_BOX_FORMATS, cfg.repair_box_format)
        self._repair_tracking = _choice(_ON_OFF, "true" if cfg.repair_tracking else "false")
        self._track_features = QSpinBox()
        self._track_features.setRange(200, 5000)
        self._track_features.setSingleStep(100)
        self._track_features.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self._track_features.setValue(int(cfg.repair_track_features))
        self._track_inliers = QSpinBox()
        self._track_inliers.setRange(6, 200)
        self._track_inliers.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self._track_inliers.setValue(int(cfg.repair_track_min_inliers))
        _row(form, "iFixit 查询", self._ifixit_enabled)
        _row(form, "iFixit 地址", self._ifixit_url)
        _row(form, "iFixit 超时", self._ifixit_timeout)
        _row(form, "资料存放目录", self._repair_docs)
        _row(form, "附图长边", self._repair_side)
        _row(form, "标注坐标格式", self._repair_box)
        _row(form, "图形跟踪板子", self._repair_tracking)
        _row(form, "跟踪特征点数", self._track_features)
        _row(form, "最少匹配点", self._track_inliers)
        self._repair_debug = _choice(_ON_OFF, "true" if cfg.repair_debug else "false")
        self._repair_debug_dir = _line(str(cfg.repair_debug_dir), "data/repair_debug")
        _row(form, "调试记录", self._repair_debug)
        _row(form, "调试记录目录", self._repair_debug_dir)
        form.addRow(_note(
            "iFixit 是外网服务，经 HTTPS_PROXY 出网。附图长边越大，丝印看得越清，请求也越大。"
            "标注坐标格式要与视觉模型的输出习惯一致，框总是偏移时可以换一种试试。"
            "图形跟踪用板子上的纹理把图形贴住：特征点越多越稳但越耗 CPU；"
            "最少匹配点调高能减少误跟，调低在纹理少的板子上更不容易跟丢。"
            "调试记录会把每轮发给模型的图片、模型的原始坐标和画出的结果存到目录里，标注不准时打开对照。"
        ))
        return page

    def _show_backend_rows(self) -> None:
        current = self._asr_backend.currentData()
        for backend, widgets in self._asr_rows.items():
            for widget in widgets:
                self._asr_form.setRowVisible(widget, backend == current)

    def _values(self) -> dict[str, str]:
        return {
            "OPENAI_BASE_URL": self._chat_url.text().strip(),
            "OPENAI_API_KEY": self._chat_key.text().strip(),
            "OPENAI_MODEL": self._chat_model.text().strip(),
            "LLM_VISION": self._chat_vision.currentData(),
            "LLM_EXTRA_BODY": self._chat_extra.text().strip(),
            "LLM_TIMEOUT_S": str(self._chat_timeout.value()),
            "ASR_BACKEND": self._asr_backend.currentData(),
            "ASR_LANGUAGE": self._asr_language.text().strip(),
            "ASR_TIMEOUT_S": str(self._asr_timeout.value()),
            "WHISPER_MODEL_SIZE": self._whisper_model.currentText().strip(),
            "WHISPER_DEVICE": self._whisper_device.currentText().strip(),
            "WHISPER_COMPUTE_TYPE": self._whisper_compute.currentText().strip(),
            "OLLAMA_BASE_URL": self._ollama_url.text().strip(),
            "OLLAMA_ASR_MODEL": self._ollama_model.text().strip(),
            "ASR_BASE_URL": self._asr_url.text().strip(),
            "ASR_API_KEY": self._asr_key.text().strip(),
            "ASR_MODEL": self._asr_model.text().strip(),
            "ASR_API_MODE": self._asr_mode.currentData(),
            "ASR_EXTRA_BODY": self._asr_extra.text().strip(),
            "IFIXIT_ENABLED": self._ifixit_enabled.currentData(),
            "IFIXIT_BASE_URL": self._ifixit_url.text().strip(),
            "IFIXIT_TIMEOUT_S": str(self._ifixit_timeout.value()),
            "REPAIR_DOCS_DIR": self._repair_docs.text().strip(),
            "REPAIR_FRAME_MAX_SIDE": str(self._repair_side.value()),
            "REPAIR_BOX_FORMAT": self._repair_box.currentData(),
            "REPAIR_TRACKING": self._repair_tracking.currentData(),
            "REPAIR_TRACK_FEATURES": str(self._track_features.value()),
            "REPAIR_TRACK_MIN_INLIERS": str(self._track_inliers.value()),
            "REPAIR_DEBUG": self._repair_debug.currentData(),
            "REPAIR_DEBUG_DIR": self._repair_debug_dir.text().strip(),
        }

    def _save(self) -> None:
        values = self._values()
        changes = {k: v for k, v in values.items() if v != self._initial[k]}
        try:
            self.settings = save_settings(changes)
        except ValidationError as exc:
            self._show_error(exc)
            return
        except OSError as exc:
            self._error.setText(f"写入 .env 失败: {exc}")
            self._error.show()
            return
        self.changed = set(changes)
        self.accept()

    def _show_error(self, exc: ValidationError) -> None:
        lines = []
        for err in exc.errors():
            field = ".".join(str(part) for part in err.get("loc", ()))
            lines.append(f"{field}: {err.get('msg', '')}")
        self._error.setText("无法保存：\n" + "\n".join(lines))
        self._error.show()
