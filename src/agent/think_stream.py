"""Split a token stream into scratchpad text and the spoken answer."""

from __future__ import annotations

_OPEN = "<think>"
_CLOSE = "</think>"


def _partial_suffix(text: str, token: str) -> str:
    """Keep a trailing slice that might still grow into ``token``."""
    limit = min(len(text), len(token) - 1)
    for size in range(limit, 0, -1):
        if token.startswith(text[-size:]):
            return text[-size:]
    return ""


class ThinkSplitter:
    """Pull ``<think>`` blocks out of streamed content. Tags may span chunks."""

    def __init__(self) -> None:
        self._mode = "scan"
        self._hold = ""

    def push(self, piece: str) -> list[tuple[str, str]]:
        if not piece:
            return []
        self._hold += piece
        out: list[tuple[str, str]] = []
        while self._hold:
            if self._mode == "answer":
                out.append(("content", self._hold))
                self._hold = ""
                break
            if self._mode == "think":
                close_at = self._hold.find(_CLOSE)
                if close_at < 0:
                    keep = _partial_suffix(self._hold, _CLOSE)
                    emit = self._hold[: len(self._hold) - len(keep)]
                    self._hold = keep
                    if emit:
                        out.append(("reasoning", emit))
                    break
                emit = self._hold[:close_at]
                if emit:
                    out.append(("reasoning", emit))
                self._hold = self._hold[close_at + len(_CLOSE) :].lstrip("\r\n")
                self._mode = "answer"
                continue
            open_at = self._hold.find(_OPEN)
            if open_at < 0:
                keep = _partial_suffix(self._hold, _OPEN)
                emit = self._hold[: len(self._hold) - len(keep)]
                self._hold = keep
                if emit:
                    out.append(("content", emit))
                break
            prefix = self._hold[:open_at]
            if prefix.strip():
                out.append(("content", prefix))
            self._hold = self._hold[open_at + len(_OPEN) :]
            self._mode = "think"
        return out

    def flush(self) -> list[tuple[str, str]]:
        if not self._hold:
            return []
        kind = "reasoning" if self._mode == "think" else "content"
        text = self._hold
        self._hold = ""
        return [(kind, text)]
