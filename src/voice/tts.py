from __future__ import annotations

import re
import threading
from collections.abc import Callable
from typing import Literal, Protocol

from config.settings import Settings, get_settings

TtsBackend = Literal["local"]
SPOKEN_MARKER = "【朗读】"
_MAX_SPEECH_CHARS = 400


class SpeechSynthesizer(Protocol):
    def speak(self, text: str, on_error: Callable[[str], None] | None = None) -> None: ...

    def stop(self) -> None: ...


def prepare_speech_text(text: str, max_chars: int = _MAX_SPEECH_CHARS) -> str:
    if not text.strip():
        return ""
    cleaned = text
    cleaned = re.sub(r"```[\s\S]*?```", " ", cleaned)
    cleaned = re.sub(r"`[^`]+`", " ", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"[#*_>]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    cut = cleaned[:max_chars]
    for sep in ("。", "！", "？", ".", "!", "?"):
        idx = cut.rfind(sep)
        if idx > max_chars // 2:
            cut = cut[: idx + 1]
            break
    return cut + " 其余内容请看聊天窗口。"


def spoken_text_from_reply(reply: str, voice_mode: bool) -> str:
    """Pick TTS text: prefer 【朗读】 segment in voice mode, else cleaned full reply."""
    text = reply.strip()
    if voice_mode and SPOKEN_MARKER in text:
        spoken = text.split(SPOKEN_MARKER, 1)[1].strip()
        if spoken:
            return prepare_speech_text(spoken)
    return prepare_speech_text(text)


class LocalTextToSpeech:
    def __init__(self, voice_name: str = "", rate: int = 180) -> None:
        self._voice_name = voice_name.strip()
        self._rate = rate
        self._stop_flag = threading.Event()
        self._active_engine = None
        self._engine_lock = threading.Lock()

    def _configure_engine(self, engine) -> None:
        if self._voice_name:
            for voice in engine.getProperty("voices"):
                if self._voice_name.lower() in voice.name.lower():
                    engine.setProperty("voice", voice.id)
                    break
        if self._rate > 0:
            engine.setProperty("rate", self._rate)

    def stop(self) -> None:
        self._stop_flag.set()
        with self._engine_lock:
            engine = self._active_engine
        if engine is not None:
            try:
                engine.stop()
            except Exception:  # noqa: BLE001
                pass

    def speak(
        self, text: str, on_error: Callable[[str], None] | None = None
    ) -> None:
        if not text.strip():
            return
        self.stop()
        self._stop_flag.clear()
        utterance = text.strip()

        def _run() -> None:
            com_initialized = False
            try:
                try:
                    import pythoncom

                    pythoncom.CoInitialize()
                    com_initialized = True
                except ImportError:
                    pass

                if self._stop_flag.is_set():
                    return

                import pyttsx3

                engine = pyttsx3.init()
                self._configure_engine(engine)
                with self._engine_lock:
                    self._active_engine = engine

                if self._stop_flag.is_set():
                    return

                engine.say(utterance)
                engine.runAndWait()
            except Exception as exc:  # noqa: BLE001
                if on_error:
                    on_error(str(exc))
            finally:
                with self._engine_lock:
                    self._active_engine = None
                if com_initialized:
                    try:
                        import pythoncom

                        pythoncom.CoUninitialize()
                    except Exception:  # noqa: BLE001
                        pass

        threading.Thread(target=_run, daemon=True).start()


TextToSpeech = LocalTextToSpeech


def create_tts(settings: Settings | None = None) -> SpeechSynthesizer:
    cfg = settings or get_settings()
    return LocalTextToSpeech(voice_name=cfg.tts_voice, rate=cfg.tts_rate)
