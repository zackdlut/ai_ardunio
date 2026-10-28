"""Per-turn record of what the repair model saw and drew, for checking bad marks.

Each turn gets ``<root>/<time>-t<turn>/``:

- ``sent.jpg``: the picture exactly as sent to the model
- ``raw.jpg``: the full-resolution reference frame (zoom crops come from here)
- ``meta.json``: question, extra system text, sizes, coordinate format
- ``tools.jsonl``: every tool call with its raw arguments and result
- ``img_N.jpg``: images returned by tools (zoom views, document pages)
- ``result.jpg``: the shapes after the turn drawn on ``raw.jpg``
- ``reply.txt``: the final answer or the error
"""

from __future__ import annotations

import base64
import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from overlay.engine import OverlayEngine

_KEEP_TURNS = 60


class RepairDebugLog:
    def __init__(self, root: Path | str, keep: int = _KEEP_TURNS) -> None:
        self.root = Path(root)
        self.keep = keep
        self._lock = threading.Lock()
        self._dir: Path | None = None
        self._turn: int | None = None
        self._images = 0
        self._engine = OverlayEngine()

    @property
    def current_dir(self) -> Path | None:
        return self._dir

    def begin(
        self,
        turn: int,
        question: str,
        extra_system: str,
        sent: np.ndarray | None,
        raw: np.ndarray | None,
        meta: dict[str, Any] | None = None,
    ) -> Path | None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        folder = self.root / f"{stamp}-t{turn}"
        try:
            folder.mkdir(parents=True, exist_ok=True)
            if sent is not None:
                _write_jpg(folder / "sent.jpg", sent)
            if raw is not None:
                _write_jpg(folder / "raw.jpg", raw)
            info = {
                "turn": turn,
                "question": question,
                "sent_size": None if sent is None else [int(sent.shape[1]), int(sent.shape[0])],
                "raw_size": None if raw is None else [int(raw.shape[1]), int(raw.shape[0])],
                **(meta or {}),
                "extra_system": extra_system,
            }
            (folder / "meta.json").write_text(
                json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            return None
        with self._lock:
            self._dir, self._turn, self._images = folder, turn, 0
        self._prune()
        return folder

    def note(self, **fields: Any) -> None:
        """Add fields to the current turn's ``meta.json``."""
        with self._lock:
            folder = self._dir
        if folder is None:
            return
        path = folder / "meta.json"
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
            info.update(fields)
            path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        except (OSError, ValueError):
            pass

    def tool(self, name: str, args: dict[str, Any], result: Any) -> None:
        with self._lock:
            folder = self._dir
        if folder is None:
            return
        line = json.dumps(
            {"t": round(time.time(), 2), "tool": name, "args": args, "result": str(result)[:4000]},
            ensure_ascii=False,
            default=str,
        )
        try:
            with (folder / "tools.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass

    def image(self, caption: str, url: str) -> None:
        with self._lock:
            folder = self._dir
            if folder is None:
                return
            self._images += 1
            index = self._images
        try:
            data = base64.b64decode(url.split(",", 1)[1])
            (folder / f"img_{index}.jpg").write_bytes(data)
            with (folder / "images.txt").open("a", encoding="utf-8") as fh:
                fh.write(f"img_{index}.jpg\t{caption}\n")
        except (OSError, IndexError, ValueError):
            pass

    def finish(
        self,
        turn: int,
        reply: str,
        shapes: list[dict[str, Any]],
        raw: np.ndarray | None,
    ) -> Path | None:
        with self._lock:
            folder = self._dir if self._turn == turn else None
            if folder is not None:
                self._dir = self._turn = None
        if folder is None:
            return None
        try:
            (folder / "reply.txt").write_text(reply or "", encoding="utf-8")
            (folder / "shapes.json").write_text(
                json.dumps(shapes, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            if raw is not None:
                drawn = self._engine.render(raw, [], mode="repair", annotations=shapes, pulse=0.25)
                _write_jpg(folder / "result.jpg", drawn)
        except OSError:
            pass
        return folder

    def _prune(self) -> None:
        try:
            dirs = sorted(p for p in self.root.iterdir() if p.is_dir())
        except OSError:
            return
        for old in dirs[: max(0, len(dirs) - self.keep)]:
            shutil.rmtree(old, ignore_errors=True)


def _write_jpg(path: Path, frame: np.ndarray) -> None:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        path.write_bytes(buf.tobytes())
