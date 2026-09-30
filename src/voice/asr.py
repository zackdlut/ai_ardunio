from __future__ import annotations

import base64
import io
import wave
from pathlib import Path
from typing import Literal, Protocol

import httpx
import numpy as np

from config.settings import Settings, get_settings

AsrBackend = Literal["local", "remote"]


class SpeechRecognizer(Protocol):
    def transcribe_pcm(self, samples: np.ndarray, sample_rate: int = 16000) -> str: ...


def pcm_to_wav_bytes(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    if samples.dtype == np.float32 or samples.dtype == np.float64:
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    else:
        pcm = samples.astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def parse_ollama_asr_text(content: str) -> str:
    text = content.strip()
    if "<asr_text>" in text:
        return text.split("<asr_text>", 1)[1].strip()
    return text


class WhisperASR:
    def __init__(self, model_size: str = "small") -> None:
        self._model_size = model_size
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        self._model = WhisperModel(self._model_size, device="cpu", compute_type="int8")

    def transcribe_file(self, audio_path: Path) -> str:
        self._ensure_model()
        segments, _ = self._model.transcribe(str(audio_path), language="zh")
        return "".join(s.text for s in segments).strip()

    def transcribe_pcm(self, samples: np.ndarray, sample_rate: int = 16000) -> str:
        self._ensure_model()
        segments, _ = self._model.transcribe(
            samples, language="zh", sample_rate=sample_rate
        )
        return "".join(s.text for s in segments).strip()


class OllamaASR:
    """Remote ASR via Ollama /api/chat (e.g. frozenlab/qwen3-asr)."""

    def __init__(self, base_url: str, model: str, timeout_s: float = 300.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_s = timeout_s

    def transcribe_pcm(self, samples: np.ndarray, sample_rate: int = 16000) -> str:
        wav = pcm_to_wav_bytes(samples, sample_rate)
        payload = {
            "model": self._model,
            "stream": False,
            "messages": [
                {"role": "user", "content": "", "images": [base64.b64encode(wav).decode("ascii")]}
            ],
        }
        url = f"{self._base_url}/api/chat"
        with httpx.Client(timeout=httpx.Timeout(self._timeout_s)) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            content = response.json()["message"]["content"]
        return parse_ollama_asr_text(content)


def create_asr(settings: Settings | None = None, backend: AsrBackend | None = None) -> SpeechRecognizer:
    cfg = settings or get_settings()
    mode: AsrBackend = backend or cfg.asr_backend
    if mode == "remote":
        return OllamaASR(cfg.resolve_ollama_base_url(), cfg.ollama_asr_model)
    return WhisperASR(model_size=cfg.whisper_model_size)
