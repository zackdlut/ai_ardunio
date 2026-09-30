from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from arduino.board_manager import _run_cli
from config.settings import get_settings


@dataclass
class CompileResult:
    success: bool
    stdout: str
    stderr: str


def compile_project(sketch_path: Path, fqbn: str | None = None) -> CompileResult:
    settings = get_settings()
    board = fqbn or settings.arduino_fqbn
    code, stdout, stderr = _run_cli(
        ["compile", "-b", board, str(sketch_path)],
        timeout=300,
    )
    return CompileResult(success=code == 0, stdout=stdout, stderr=stderr)
