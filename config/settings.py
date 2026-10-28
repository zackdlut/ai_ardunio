import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

AsrBackend = Literal["local", "ollama", "openai"]
AsrApiMode = Literal["transcriptions", "chat"]
LlmVision = Literal["auto", "true", "false"]
TtsBackend = Literal["local"]
YoloDevice = Literal["auto", "cpu", "gpu", "npu"]
RepairBoxFormat = Literal["norm1000", "norm1", "pixel"]

_ASR_BACKEND_ALIASES = {"remote": "ollama", "whisper": "local"}


def _json_object(text: str) -> dict[str, Any]:
    if not text.strip():
        return {}
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("must be a JSON object")
    return value

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(default="ollama", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(
        default="http://127.0.0.1:11434/v1", alias="OPENAI_BASE_URL"
    )
    openai_model: str = Field(default="qwen3.8:27b", alias="OPENAI_MODEL")
    llm_send_frame: bool = Field(default=True, alias="LLM_SEND_FRAME")
    llm_frame_max_side: int = Field(default=768, alias="LLM_FRAME_MAX_SIDE")
    llm_vision: LlmVision = Field(default="auto", alias="LLM_VISION")
    llm_extra_body: str = Field(default="", alias="LLM_EXTRA_BODY")
    llm_timeout_s: float = Field(default=600.0, alias="LLM_TIMEOUT_S")

    asr_backend: AsrBackend = Field(default="local", alias="ASR_BACKEND")
    asr_language: str = Field(default="zh", alias="ASR_LANGUAGE")
    asr_timeout_s: float = Field(default=300.0, alias="ASR_TIMEOUT_S")
    whisper_model_size: str = Field(default="small", alias="WHISPER_MODEL_SIZE")
    whisper_device: str = Field(default="cpu", alias="WHISPER_DEVICE")
    whisper_compute_type: str = Field(default="int8", alias="WHISPER_COMPUTE_TYPE")
    ollama_asr_model: str = Field(
        default="frozenlab/qwen3-asr:0.6b", alias="OLLAMA_ASR_MODEL"
    )
    ollama_base_url: str = Field(default="", alias="OLLAMA_BASE_URL")
    asr_base_url: str = Field(default="", alias="ASR_BASE_URL")
    asr_api_key: str = Field(default="", alias="ASR_API_KEY")
    asr_model: str = Field(default="whisper-1", alias="ASR_MODEL")
    asr_api_mode: AsrApiMode = Field(default="transcriptions", alias="ASR_API_MODE")
    asr_extra_body: str = Field(default="", alias="ASR_EXTRA_BODY")

    tts_backend: TtsBackend = Field(default="local", alias="TTS_BACKEND")
    tts_enabled: bool = Field(default=True, alias="TTS_ENABLED")
    tts_voice: str = Field(default="Microsoft Huihui Desktop", alias="TTS_VOICE")
    tts_rate: int = Field(default=180, alias="TTS_RATE")

    vad_noise_multiplier: float = Field(default=2.2, alias="VAD_NOISE_MULTIPLIER")
    vad_barge_noise_multiplier: float = Field(default=6.0, alias="VAD_BARGE_NOISE_MULTIPLIER")
    vad_min_rms: float = Field(default=0.004, alias="VAD_MIN_RMS")
    vad_barge_min_rms: float = Field(default=0.05, alias="VAD_BARGE_MIN_RMS")
    vad_onset_ms: int = Field(default=250, alias="VAD_ONSET_MS")
    vad_barge_ms: int = Field(default=200, alias="VAD_BARGE_MS")
    vad_silence_ms: int = Field(default=800, alias="VAD_SILENCE_MS")
    vad_min_speech_ms: int = Field(default=300, alias="VAD_MIN_SPEECH_MS")
    vad_max_utterance_s: float = Field(default=20.0, alias="VAD_MAX_UTTERANCE_S")
    vad_preroll_ms: int = Field(default=300, alias="VAD_PREROLL_MS")
    vad_barge_grace_ms: int = Field(default=800, alias="VAD_BARGE_GRACE_MS")

    arduino_fqbn: str = Field(default="arduino:avr:uno", alias="ARDUINO_FQBN")
    arduino_cli: str = Field(default="arduino-cli", alias="ARDUINO_CLI")
    camera_index: int = Field(default=0, alias="CAMERA_INDEX")
    yolo_model_path: Path = Field(
        default=ROOT_DIR / "models" / "best.pt", alias="YOLO_MODEL_PATH"
    )
    yolo_device: YoloDevice = Field(default="gpu", alias="YOLO_DEVICE")
    database_path: Path = Field(
        default=ROOT_DIR / "data" / "copilot.db", alias="DATABASE_PATH"
    )
    projects_dir: Path = Field(default=ROOT_DIR / "config" / "projects")

    ifixit_enabled: bool = Field(default=True, alias="IFIXIT_ENABLED")
    ifixit_base_url: str = Field(
        default="https://www.ifixit.com/api/2.0", alias="IFIXIT_BASE_URL"
    )
    ifixit_timeout_s: float = Field(default=15.0, alias="IFIXIT_TIMEOUT_S")
    repair_docs_dir: Path = Field(
        default=ROOT_DIR / "data" / "repair_docs", alias="REPAIR_DOCS_DIR"
    )
    repair_frame_max_side: int = Field(default=1280, alias="REPAIR_FRAME_MAX_SIDE")
    repair_box_format: RepairBoxFormat = Field(default="norm1000", alias="REPAIR_BOX_FORMAT")

    @field_validator(
        "asr_backend", "asr_api_mode", "llm_vision", "yolo_device", "repair_box_format", mode="before"
    )
    @classmethod
    def _normalize_choice(cls, value: object) -> str:
        if isinstance(value, str):
            value = value.strip().lower()
            return _ASR_BACKEND_ALIASES.get(value, value)
        return value

    @field_validator("llm_extra_body", "asr_extra_body")
    @classmethod
    def _check_json_object(cls, value: str) -> str:
        _json_object(value)
        return value

    def llm_extra_body_dict(self) -> dict[str, Any]:
        return _json_object(self.llm_extra_body)

    def asr_extra_body_dict(self) -> dict[str, Any]:
        return _json_object(self.asr_extra_body)

    def resolve_ollama_base_url(self) -> str:
        if self.ollama_base_url.strip():
            return self.ollama_base_url.rstrip("/")
        url = self.openai_base_url.rstrip("/")
        if url.endswith("/v1"):
            return url[:-3]
        return url

    def resolve_asr_base_url(self) -> str:
        return (self.asr_base_url.strip() or self.openai_base_url).rstrip("/")

    def resolve_asr_api_key(self) -> str:
        return self.asr_api_key.strip() or self.openai_api_key

    @property
    def firmware_dir(self) -> Path:
        return ROOT_DIR / "firmware"


def get_settings() -> Settings:
    return Settings()


def save_settings(values: dict[str, str], env_path: Path | None = None) -> Settings:
    """Validate ``values`` (env names), then persist them to .env and this process.

    Raises ``pydantic.ValidationError`` before anything is written.
    """
    from config.env_file import update_env_file

    Settings(**values)
    if values:
        update_env_file(env_path or ROOT_DIR / ".env", values)
        os.environ.update(values)
    return get_settings()
