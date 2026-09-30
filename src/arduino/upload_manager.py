from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from arduino.board_manager import _run_cli
from config.settings import get_settings


@dataclass
class UploadResult:
    success: bool
    stdout: str
    stderr: str


def upload_firmware(
    sketch_path: Path, port: str, fqbn: str | None = None
) -> UploadResult:
    settings = get_settings()
    board = fqbn or settings.arduino_fqbn
    code, stdout, stderr = _run_cli(
        ["upload", "-p", port, "-b", board, str(sketch_path)],
        timeout=300,
    )
    return UploadResult(success=code == 0, stdout=stdout, stderr=stderr)
