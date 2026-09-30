from __future__ import annotations

import os
import sys
from pathlib import Path


def _bootstrap_paths() -> Path:
    root = Path(__file__).resolve().parent.parent
    src = root / "src"
    for p in (str(root), str(src)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


def _load_env_early(root: Path) -> None:
    env_path = root / ".env"
    if env_path.is_file():
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)

    # Ensure httpx honors no_proxy for exact Ollama host IP
    for key in ("NO_PROXY", "no_proxy"):
        val = os.environ.get(key)
        if val:
            os.environ[key] = val


def run() -> int:
    root = _bootstrap_paths()
    _load_env_early(root)

    from PySide6.QtWidgets import QApplication

    from project.models import init_db
    from ui.main_window import MainWindow

    init_db()
    app = QApplication(sys.argv)

    from ui.theme import apply_theme

    apply_theme(app)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
