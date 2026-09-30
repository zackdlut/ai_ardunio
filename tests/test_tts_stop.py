"""stop() must not touch SAPI from the caller thread."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import pyttsx3  # noqa: E402

from voice.tts import LocalTextToSpeech  # noqa: E402


class FakeEngine:
    def __init__(self) -> None:
        self._inLoop = False
        self._busy = False
        self.stops: list[str] = []
        self.said: list[str] = []
        self.entered = threading.Event()

    def say(self, text: str) -> None:
        self.said.append(text)
        self._busy = True

    def startLoop(self, useDriverLoop: bool = True) -> None:
        del useDriverLoop
        self._inLoop = True

    def endLoop(self) -> None:
        if not self._inLoop:
            raise RuntimeError("run loop not started")
        self._inLoop = False

    def iterate(self) -> None:
        self.entered.set()
        time.sleep(0.2)

    def isBusy(self) -> bool:
        return self._busy

    def stop(self) -> None:
        self.stops.append(threading.current_thread().name)
        self._busy = False

    def getProperty(self, name: str):
        if name == "voices":
            return []
        return 180

    def setProperty(self, name: str, value) -> None:
        del name, value


def test_stop_stays_off_caller_thread() -> None:
    engine = FakeEngine()
    pyttsx3.init = lambda: engine  # type: ignore[assignment]
    caller = threading.current_thread().name
    tts = LocalTextToSpeech()
    try:
        tts.enqueue("hello")
        assert engine.entered.wait(2), "tts thread did not start speaking"
        started = time.perf_counter()
        tts.stop()
        elapsed = time.perf_counter() - started
        assert elapsed < 0.1, elapsed
        deadline = time.time() + 2
        while not engine.stops and time.time() < deadline:
            time.sleep(0.01)
        assert engine.stops, "engine.stop was never called"
        assert set(engine.stops) == {"tts"}
        assert caller not in engine.stops
    finally:
        tts.close()
        assert not tts._thread.is_alive()


if __name__ == "__main__":
    test_stop_stays_off_caller_thread()
    print("ok")
