from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

from config.settings import get_settings

_CLI_NOT_FOUND_HINT = (
    "未找到 arduino-cli。请安装 Arduino CLI 并在 .env 中设置 ARDUINO_CLI=完整路径，"
    "例如 ARDUINO_CLI=C:\\\\tools\\\\arduino-cli.exe"
)


@dataclass
class BoardInfo:
    port: str
    label: str
    fqbn: str | None = None


def resolve_arduino_cli() -> str:
    settings = get_settings()
    configured = settings.arduino_cli.strip()
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return str(path.resolve())
        found = shutil.which(configured)
        if found:
            return found

    found = shutil.which("arduino-cli")
    if found:
        return found
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", "")
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        candidates = [
            Path(local) / "Programs" / "arduino-cli" / "arduino-cli.exe",
            Path(local) / "Arduino15" / "arduino-cli" / "arduino-cli.exe",
            Path(program_files) / "Arduino CLI" / "arduino-cli.exe",
            Path(program_files) / "Arduino IDE" / "arduino-cli.exe",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate.resolve())

    raise FileNotFoundError(_CLI_NOT_FOUND_HINT)


def _run_cli(args: list[str], timeout: int = 60) -> tuple[int, str, str]:
    cli = resolve_arduino_cli()
    try:
        proc = subprocess.run(
            [cli, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        raise RuntimeError(_CLI_NOT_FOUND_HINT) from None
    except OSError as exc:
        if getattr(exc, "winerror", None) == 2 or "cannot find the file" in str(exc).lower():
            raise RuntimeError(_CLI_NOT_FOUND_HINT) from exc
        raise
    return proc.returncode, proc.stdout, proc.stderr


def _list_serial_ports() -> List[BoardInfo]:
    try:
        from serial.tools import list_ports
    except ImportError:
        return []

    boards: list[BoardInfo] = []
    for port in list_ports.comports():
        desc = (port.description or "").strip()
        label = f"{desc} · 仅串口" if desc else "串口设备 · 仅串口"
        boards.append(BoardInfo(port=port.device, label=label, fqbn=None))
    return boards


def _parse_board_list_json(stdout: str) -> List[dict]:
    if not stdout.strip():
        return []
    parsed = json.loads(stdout)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        ports = parsed.get("detected_ports")
        if isinstance(ports, list):
            return ports
    raise ValueError("unexpected arduino-cli board list JSON shape")


def _list_boards_via_cli() -> List[BoardInfo]:
    code, stdout, stderr = _run_cli(["board", "list", "--format", "json"])
    if code != 0:
        raise RuntimeError(stderr.strip() or stdout.strip() or "arduino-cli board list failed")

    boards: list[BoardInfo] = []
    for entry in _parse_board_list_json(stdout):
        port = entry.get("port", {}) or {}
        address = port.get("address") or entry.get("address")
        if not address:
            continue
        label = port.get("label") or address
        boards_info = entry.get("matching_boards") or []
        fqbn = boards_info[0].get("fqbn") if boards_info else None
        boards.append(BoardInfo(port=address, label=label, fqbn=fqbn))
    return boards


def list_boards() -> List[BoardInfo]:
    try:
        return _list_boards_via_cli()
    except (RuntimeError, FileNotFoundError, ValueError, json.JSONDecodeError, KeyError, TypeError):
        fallback = _list_serial_ports()
        if fallback:
            return fallback
        raise RuntimeError(_CLI_NOT_FOUND_HINT) from None
