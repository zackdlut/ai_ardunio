from __future__ import annotations

_DELIMS = "。！？!?\n"


class SentenceBuffer:
    """Accumulate streamed text and release it one spoken sentence at a time."""

    def __init__(self) -> None:
        self._buf = ""

    def clear(self) -> None:
        self._buf = ""

    def push(self, text: str) -> list[str]:
        if not text:
            return []
        self._buf += text
        ready: list[str] = []
        while self._buf:
            indexes = [self._buf.find(mark) for mark in _DELIMS]
            cuts = [index for index in indexes if index >= 0]
            if not cuts:
                break
            cut = min(cuts) + 1
            sentence = self._buf[:cut].strip()
            self._buf = self._buf[cut:]
            if sentence:
                ready.append(sentence)
        return ready

    def flush(self) -> str:
        rest = self._buf.strip()
        self._buf = ""
        return rest
