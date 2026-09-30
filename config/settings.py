from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

AsrBackend = Literal["local", "remote"]
TtsBackend = Literal["local"]
YoloDevice = Literal["auto", "cpu", "npu"]

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

    asr_backend: AsrBackend = Field(default="local", alias="ASR_BACKEND")
    whisper_model_size: str = Field(default="small", alias="WHISPER_MODEL_SIZE")
    ollama_asr_model: str = Field(
        default="frozenlab/qwen3-asr:0.6b", alias="OLLAMA_ASR_MODEL"
    )
    ollama_base_url: str = Field(default="", alias="OLLAMA_BASE_URL")

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
    yolo_device: YoloDevice = Field(default="auto", alias="YOLO_DEVICE")
    database_path: Path = Field(
        default=ROOT_DIR / "data" / "copilot.db", alias="DATABASE_PATH"
    )
    projects_dir: Path = Field(default=ROOT_DIR / "config" / "projects")

    @field_validator("asr_backend", "yolo_device", mode="before")
    @classmethod
    def _normalize_choice(cls, value: object) -> str:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    def resolve_ollama_base_url(self) -> str:
        if self.ollama_base_url.strip():
            return self.ollama_base_url.rstrip("/")
        url = self.openai_base_url.rstrip("/")
        if url.endswith("/v1"):
            return url[:-3]
        return url

    @property
    def firmware_dir(self) -> Path:
        return ROOT_DIR / "firmware"


def get_settings() -> Settings:
    return Settings()
