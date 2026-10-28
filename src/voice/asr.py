from __future__ import annotations

import base64
import io
import wave
from pathlib import Path
from typing import Any, Protocol

import httpx
import numpy as np

from config.settings import AsrApiMode, AsrBackend, Settings, get_settings

WHISPER_SAMPLE_RATE = 16000


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


def parse_asr_text(content: str) -> str:
    """Qwen3-ASR prefixes the transcript with ``language X<asr_text>``."""
    text = content.strip()
    if "<asr_text>" in text:
        return text.split("<asr_text>", 1)[1].strip()
    return text


def _to_float32(samples: np.ndarray) -> np.ndarray:
    if samples.dtype == np.int16:
        return samples.astype(np.float32) / 32768.0
    return samples.astype(np.float32, copy=False)


class WhisperASR:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "zh",
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language or None
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self._model_size, device=self._device, compute_type=self._compute_type
        )

    def transcribe_file(self, audio_path: Path) -> str:
        self._ensure_model()
        segments, _ = self._model.transcribe(str(audio_path), language=self._language)
        return "".join(s.text for s in segments).strip()

    def transcribe_pcm(self, samples: np.ndarray, sample_rate: int = 16000) -> str:
        if sample_rate != WHISPER_SAMPLE_RATE:
            raise ValueError(f"faster-whisper 需要 16 kHz 音频，收到 {sample_rate} Hz")
        self._ensure_model()
        segments, _ = self._model.transcribe(_to_float32(samples), language=self._language)
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
            if response.is_error:
                raise RuntimeError(self._describe_error(response))
            content = response.json()["message"]["content"]
        return parse_asr_text(content)

    def _describe_error(self, response: httpx.Response) -> str:
        try:
            detail = str(response.json().get("error") or response.text)
        except ValueError:
            detail = response.text
        detail = detail.strip()[:300] or response.reason_phrase
        if response.status_code == 404 and "not found" in detail:
            return (
                f"Ollama 上没有 ASR 模型 {self._model}（{self._base_url}）。"
                f"请在服务器上执行 ollama pull {self._model}，或在设置里换模型/后端。"
            )
        return f"Ollama ASR 请求失败 HTTP {response.status_code}: {detail}"


class OpenAIASR:
    """Remote ASR over an OpenAI-compatible API.

    ``transcriptions`` posts the WAV to ``/audio/transcriptions`` (Whisper API,
    SenseVoice, vLLM, speaches...). ``chat`` sends it as an ``input_audio`` data
    URL to ``/chat/completions`` (e.g. qwen3-asr-flash on DashScope).
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        mode: AsrApiMode = "transcriptions",
        language: str = "zh",
        extra_body: dict[str, Any] | None = None,
        timeout_s: float = 300.0,
        client: Any = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key or "none",
                base_url=base_url,
                timeout=timeout_s,
                max_retries=0,
            )
        self._client = client
        self._model = model
        self._mode = mode
        self._language = language
        self._extra_body = extra_body or None

    def transcribe_pcm(self, samples: np.ndarray, sample_rate: int = 16000) -> str:
        wav = pcm_to_wav_bytes(samples, sample_rate)
        if self._mode == "chat":
            return self._via_chat(wav)
        return self._via_transcriptions(wav)

    def _via_transcriptions(self, wav: bytes) -> str:
        kwargs: dict[str, Any] = {}
        if self._language:
            kwargs["language"] = self._language
        result = self._client.audio.transcriptions.create(
            model=self._model,
            file=("speech.wav", wav, "audio/wav"),
            extra_body=self._extra_body,
            **kwargs,
        )
        text = result if isinstance(result, str) else getattr(result, "text", "")
        return parse_asr_text(text or "")

    def _via_chat(self, wav: bytes) -> str:
        data_url = "data:audio/wav;base64," + base64.b64encode(wav).decode("ascii")
        response = self._client.chat.completions.create(
            model=self._model,
            stream=False,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "input_audio", "input_audio": {"data": data_url}}],
                }
            ],
            extra_body=self._extra_body,
        )
        return parse_asr_text(response.choices[0].message.content or "")


def create_asr(settings: Settings | None = None, backend: AsrBackend | None = None) -> SpeechRecognizer:
    cfg = settings or get_settings()
    mode: AsrBackend = backend or cfg.asr_backend
    if mode == "ollama":
        return OllamaASR(cfg.resolve_ollama_base_url(), cfg.ollama_asr_model, cfg.asr_timeout_s)
    if mode == "openai":
        return OpenAIASR(
            cfg.resolve_asr_base_url(),
            cfg.resolve_asr_api_key(),
            cfg.asr_model,
            mode=cfg.asr_api_mode,
            language=cfg.asr_language,
            extra_body=cfg.asr_extra_body_dict(),
            timeout_s=cfg.asr_timeout_s,
        )
    return WhisperASR(
        cfg.whisper_model_size,
        device=cfg.whisper_device,
        compute_type=cfg.whisper_compute_type,
        language=cfg.asr_language,
    )
