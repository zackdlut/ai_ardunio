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

    def enqueue(self, text: str) -> None: ...

    def stop(self) -> None: ...

    def is_busy(self) -> bool: ...

    def set_idle_callback(self, callback: Callable[[], None] | None) -> None: ...

    def set_error_callback(self, callback: Callable[[str], None] | None) -> None: ...


def prepare_speech_text(text: str, max_chars: int = _MAX_SPEECH_CHARS) -> str:
    if not text.strip():
        return ""
    cleaned = text.replace(SPOKEN_MARKER, " ")
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
    """One worker thread. Each sentence gets a new SAPI engine so playback can continue."""

    def __init__(self, voice_name: str = "", rate: int = 180) -> None:
        self._voice_name = voice_name.strip()
        self._rate = rate
        self._lock = threading.Lock()
        self._pending: list[tuple[int, str]] = []
        self._generation = 0
        self._speaking = False
        self._closed = False
        self._active_engine = None
        self._on_idle: Callable[[], None] | None = None
        self._on_error: Callable[[str], None] | None = None
        self._wakeup = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="tts", daemon=True)
        self._thread.start()

    def set_idle_callback(self, callback: Callable[[], None] | None) -> None:
        self._on_idle = callback

    def set_error_callback(self, callback: Callable[[str], None] | None) -> None:
        self._on_error = callback

    def is_busy(self) -> bool:
        with self._lock:
            return bool(self._pending) or self._speaking

    def enqueue(self, text: str) -> None:
        utterance = text.strip()
        if not utterance or self._closed:
            return
        with self._lock:
            self._pending.append((self._generation, utterance))
        self._wakeup.set()

    def stop(self) -> None:
        with self._lock:
            self._generation += 1
            self._pending.clear()
            engine = self._active_engine
        if engine is not None:
            try:
                engine.stop()
            except Exception:  # noqa: BLE001
                pass
        self._wakeup.set()

    def speak(
        self, text: str, on_error: Callable[[str], None] | None = None
    ) -> None:
        del on_error
        if not text.strip():
            return
        self.stop()
        self.enqueue(text)

    def _loop(self) -> None:
        com_initialized = False
        try:
            try:
                import pythoncom

                pythoncom.CoInitialize()
                com_initialized = True
            except ImportError:
                pass

            while not self._closed:
                self._wakeup.wait()
                self._wakeup.clear()
                while not self._closed:
                    with self._lock:
                        if not self._pending:
                            break
                        gen, text = self._pending.pop(0)
                        current = self._generation
                    if gen != current:
                        continue
                    try:
                        self._speak_one(text, gen)
                    except Exception as exc:  # noqa: BLE001
                        callback = self._on_error
                        if callback is not None and gen == self._generation_now():
                            callback(str(exc))
                    with self._lock:
                        more = any(item[0] == self._generation for item in self._pending)
                        same = gen == self._generation
                    if same and not more:
                        idle = self._on_idle
                        if idle is not None:
                            idle()
        finally:
            if com_initialized:
                try:
                    import pythoncom

                    pythoncom.CoUninitialize()
                except Exception:  # noqa: BLE001
                    pass

    def _speak_one(self, text: str, gen: int) -> None:
        if gen != self._generation_now():
            return
        import pyttsx3

        engine = pyttsx3.init()
        self._configure_engine(engine)
        with self._lock:
            if gen != self._generation:
                self._release_engine(engine)
                return
            self._speaking = True
            self._active_engine = engine
        try:
            if gen != self._generation_now():
                return
            engine.say(text)
            engine.runAndWait()
        finally:
            with self._lock:
                self._speaking = False
                if self._active_engine is engine:
                    self._active_engine = None
            self._release_engine(engine)

    def _release_engine(self, engine) -> None:
        try:
            engine.stop()
        except Exception:  # noqa: BLE001
            pass

    def _generation_now(self) -> int:
        with self._lock:
            return self._generation

    def _configure_engine(self, engine) -> None:
        if self._voice_name:
            for voice in engine.getProperty("voices"):
                if self._voice_name.lower() in voice.name.lower():
                    engine.setProperty("voice", voice.id)
                    break
        if self._rate > 0:
            engine.setProperty("rate", self._rate)


TextToSpeech = LocalTextToSpeech


def create_tts(settings: Settings | None = None) -> SpeechSynthesizer:
    cfg = settings or get_settings()
    return LocalTextToSpeech(voice_name=cfg.tts_voice, rate=cfg.tts_rate)
