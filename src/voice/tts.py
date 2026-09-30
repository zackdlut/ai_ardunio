from __future__ import annotations

import re
import threading
import time
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

    def close(self) -> None: ...

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
    """One STA worker thread owns the SAPI engine.

    Other threads may only set flags. Calling pyttsx3 from the UI thread finalizes
    the COM connection point there, and ``Unadvise`` then waits forever for this
    thread while it is blocked in ``Event.wait()``.
    """

    def __init__(self, voice_name: str = "", rate: int = 180) -> None:
        self._voice_name = voice_name.strip()
        self._rate = rate
        self._lock = threading.Lock()
        self._pending: list[tuple[int, str]] = []
        self._generation = 0
        self._speaking = False
        self._closed = False
        self._on_idle: Callable[[], None] | None = None
        self._on_error: Callable[[str], None] | None = None
        self._wakeup = threading.Event()
        self._cancel = threading.Event()
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
        """Cancel queued and in-progress speech without touching COM."""
        with self._lock:
            self._generation += 1
            self._pending.clear()
        self._cancel.set()
        self._wakeup.set()

    def close(self) -> None:
        """Stop speech and release the engine on the TTS thread."""
        self._closed = True
        self.stop()
        self._thread.join(timeout=2.0)

    def speak(
        self, text: str, on_error: Callable[[str], None] | None = None
    ) -> None:
        del on_error
        if not text.strip():
            return
        self.stop()
        self.enqueue(text)

    def _loop(self) -> None:
        pythoncom = None
        com_initialized = False
        engine = None
        try:
            try:
                import pythoncom as pythoncom_mod

                pythoncom_mod.CoInitialize()
                pythoncom = pythoncom_mod
                com_initialized = True
            except Exception:  # noqa: BLE001
                pythoncom = None

            while not self._closed:
                # Wake often enough to dispatch COM calls posted onto this STA.
                if not self._wakeup.wait(0.2):
                    if pythoncom is not None:
                        pythoncom.PumpWaitingMessages()
                    continue
                self._wakeup.clear()
                while not self._closed:
                    with self._lock:
                        if not self._pending:
                            break
                        gen, text = self._pending.pop(0)
                    if gen != self._generation_now():
                        continue
                    try:
                        if engine is None:
                            engine = self._new_engine()
                        if not self._speak_one(engine, text, gen):
                            engine = self._discard_engine(engine)
                            _collect_garbage()
                    except Exception as exc:  # noqa: BLE001
                        engine = self._discard_engine(engine)
                        _collect_garbage()
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
            engine = self._discard_engine(engine)
            _collect_garbage()
            if com_initialized and pythoncom is not None:
                try:
                    pythoncom.CoUninitialize()
                except Exception:  # noqa: BLE001
                    pass

    def _new_engine(self):
        import pyttsx3

        engine = pyttsx3.init()
        self._configure_engine(engine)
        return engine

    def _speak_one(self, engine, text: str, gen: int) -> bool:
        """Speak on this thread. Return False when the engine must be recreated."""
        with self._lock:
            if gen != self._generation:
                return True
            self._speaking = True
            self._cancel.clear()
            if gen != self._generation:
                self._speaking = False
                return True
        started = False
        broken = False
        try:
            engine.say(text)
            engine.startLoop(False)
            started = True
            while not self._closed and gen == self._generation_now():
                engine.iterate()
                if not engine.isBusy():
                    break
                if self._cancel.wait(0.05):
                    break
            if self._closed or gen != self._generation_now():
                self._halt_engine(engine)
        finally:
            if started:
                try:
                    engine.endLoop()
                except Exception:  # noqa: BLE001
                    broken = True
            with self._lock:
                self._speaking = False
        return not broken

    def _halt_engine(self, engine) -> None:
        try:
            engine.stop()
        except Exception:  # noqa: BLE001
            return
        for _ in range(20):
            try:
                engine.iterate()
            except Exception:  # noqa: BLE001
                return
            if not engine.isBusy():
                return
            time.sleep(0.02)

    def _discard_engine(self, engine):
        """Stop ``engine`` and drop the last reference on this thread."""
        if engine is not None:
            try:
                if getattr(engine, "_inLoop", False):
                    engine.endLoop()
            except Exception:  # noqa: BLE001
                pass
            try:
                engine.stop()
            except Exception:  # noqa: BLE001
                pass
        # Caller must rebind this return value before collecting, so the
        # COM finalizer runs on this STA thread rather than the UI thread.
        return None

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


def _collect_garbage() -> None:
    import gc

    gc.collect()


TextToSpeech = LocalTextToSpeech


def create_tts(settings: Settings | None = None) -> SpeechSynthesizer:
    cfg = settings or get_settings()
    return LocalTextToSpeech(voice_name=cfg.tts_voice, rate=cfg.tts_rate)
