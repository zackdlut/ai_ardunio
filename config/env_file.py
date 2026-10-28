"""Rewrite selected keys of a .env file in place, keeping comments and order."""

from __future__ import annotations

import re
from pathlib import Path

_PLAIN = re.compile(r"[\w.:/@+\-,]*")


def format_env_value(value: str) -> str:
    if _PLAIN.fullmatch(value):
        return value
    # python-dotenv only unescapes \\ and \' inside single quotes.
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def update_env_file(path: Path, values: dict[str, str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    pending = dict(values)
    out: list[str] = []
    for line in lines:
        match = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
        key = match.group(1) if match else None
        if key in values:
            if key in pending:
                out.append(f"{key}={format_env_value(pending.pop(key))}")
            continue
        out.append(line)
    if pending:
        if out and out[-1].strip():
            out.append("")
        out.extend(f"{key}={format_env_value(value)}" for key, value in pending.items())
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
