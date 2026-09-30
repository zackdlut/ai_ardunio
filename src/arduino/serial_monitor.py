from __future__ import annotations

import subprocess
from typing import Callable

from config.settings import get_settings


class SerialMonitor:
    """Wraps arduino-cli monitor; logs lines via callback."""

    def __init__(self, on_line: Callable[[str], None] | None = None) -> None:
        self._on_line = on_line
        self._proc: subprocess.Popen[str] | None = None

    def start(self, port: str) -> None:
        settings = get_settings()
        self._proc = subprocess.Popen(
            [settings.arduino_cli, "monitor", "-p", port],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    def poll_lines(self) -> list[str]:
        if not self._proc or not self._proc.stdout:
            return []
        lines: list[str] = []
        while True:
            line = self._proc.stdout.readline()
            if not line:
                break
            line = line.rstrip("\n")
            lines.append(line)
            if self._on_line:
                self._on_line(line)
        return lines

    def stop(self) -> None:
        if self._proc:
            self._proc.terminate()
            self._proc = None
