"""Global Qt palette and stylesheet — workshop bench + PCB solder mask."""

from __future__ import annotations

# Named tokens (FR4 substrate, silkscreen, copper trace, Arduino teal)
COLOR_BG = "#121514"
COLOR_SURFACE = "#1A1F1C"
COLOR_SURFACE_RAISED = "#222924"
COLOR_BORDER = "#2E3832"
COLOR_TEXT = "#E4EBE6"
COLOR_TEXT_MUTED = "#8A9A90"
COLOR_ACCENT = "#008184"
COLOR_ACCENT_HOVER = "#00A3A6"
COLOR_COPPER = "#C49A6C"
COLOR_DANGER = "#D45D48"


def app_stylesheet() -> str:
    return f"""
    QWidget {{
        background-color: {COLOR_BG};
        color: {COLOR_TEXT};
        font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
        font-size: 13px;
    }}

    QMainWindow {{
        background-color: {COLOR_BG};
    }}

    QLabel#appTitle {{
        font-size: 18px;
        font-weight: 600;
        color: {COLOR_TEXT};
        letter-spacing: 0.5px;
    }}

    QLabel#sectionEyebrow {{
        font-size: 10px;
        font-weight: 600;
        color: {COLOR_COPPER};
        letter-spacing: 1.2px;
        text-transform: none;
        padding-top: 4px;
    }}

    QLabel#stepTitle {{
        font-size: 15px;
        font-weight: 600;
        color: {COLOR_TEXT};
        padding: 2px 0 6px 0;
    }}

    QLabel#stepHint {{
        font-size: 12px;
        color: {COLOR_TEXT_MUTED};
        line-height: 1.45;
    }}

    QFrame#sidePanel, QFrame#logPanel {{
        background-color: {COLOR_SURFACE};
        border: 1px solid {COLOR_BORDER};
        border-radius: 8px;
    }}

    QFrame#cameraFrame {{
        background-color: {COLOR_SURFACE_RAISED};
        border: 1px solid {COLOR_BORDER};
        border-radius: 8px;
    }}

    QLabel#cameraViewport {{
        background-color: #0A0C0B;
        color: {COLOR_TEXT_MUTED};
        border: 1px solid #1E2622;
        border-radius: 6px;
        font-size: 12px;
    }}

    QProgressBar#stepProgress {{
        background-color: {COLOR_SURFACE_RAISED};
        border: none;
        border-radius: 3px;
        height: 6px;
        text-align: center;
        color: transparent;
    }}

    QProgressBar#stepProgress::chunk {{
        background-color: qlineargradient(
            x1:0, y1:0, x2:1, y2:0,
            stop:0 {COLOR_ACCENT}, stop:1 {COLOR_COPPER}
        );
        border-radius: 3px;
    }}

    QLineEdit {{
        background-color: {COLOR_SURFACE_RAISED};
        border: 1px solid {COLOR_BORDER};
        border-radius: 6px;
        padding: 8px 12px;
        color: {COLOR_TEXT};
        selection-background-color: {COLOR_ACCENT};
    }}

    QLineEdit:focus {{
        border: 1px solid {COLOR_ACCENT};
    }}

    QComboBox {{
        background-color: {COLOR_SURFACE_RAISED};
        border: 1px solid {COLOR_BORDER};
        border-radius: 6px;
        padding: 6px 10px;
        min-height: 28px;
    }}

    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}

    QComboBox QAbstractItemView {{
        background-color: {COLOR_SURFACE};
        border: 1px solid {COLOR_BORDER};
        selection-background-color: {COLOR_ACCENT};
        color: {COLOR_TEXT};
    }}

    QTextEdit#chatLog, QTextEdit#buildLog {{
        background-color: {COLOR_SURFACE_RAISED};
        border: 1px solid {COLOR_BORDER};
        border-radius: 6px;
        padding: 8px;
        color: {COLOR_TEXT};
        font-size: 13px;
    }}

    QTextEdit#buildLog {{
        font-family: "Cascadia Mono", "Consolas", monospace;
        font-size: 11px;
        color: {COLOR_TEXT_MUTED};
    }}

    QPushButton {{
        background-color: {COLOR_SURFACE_RAISED};
        border: 1px solid {COLOR_BORDER};
        border-radius: 6px;
        padding: 8px 14px;
        color: {COLOR_TEXT};
        font-weight: 500;
    }}

    QPushButton:hover {{
        border-color: {COLOR_ACCENT};
        color: {COLOR_TEXT};
    }}

    QPushButton:pressed {{
        background-color: {COLOR_BORDER};
    }}

    QPushButton#primaryButton {{
        background-color: {COLOR_ACCENT};
        border: 1px solid {COLOR_ACCENT_HOVER};
        color: #FFFFFF;
        font-weight: 600;
    }}

    QPushButton#primaryButton:hover {{
        background-color: {COLOR_ACCENT_HOVER};
        border-color: {COLOR_ACCENT_HOVER};
    }}

    QPushButton#voiceButton {{
        border-color: {COLOR_COPPER};
        color: {COLOR_COPPER};
    }}

    QPushButton#voiceButton:checked, QPushButton#voiceButton:pressed {{
        background-color: #3D3024;
        border-color: {COLOR_COPPER};
        color: {COLOR_TEXT};
    }}

    QLabel#voiceStatus {{
        color: {COLOR_COPPER};
        font-size: 12px;
    }}

    QScrollBar:vertical {{
        background: {COLOR_SURFACE};
        width: 10px;
        margin: 0;
    }}

    QScrollBar::handle:vertical {{
        background: {COLOR_BORDER};
        border-radius: 5px;
        min-height: 24px;
    }}

    QScrollBar::handle:vertical:hover {{
        background: {COLOR_TEXT_MUTED};
    }}
    """


def apply_theme(app) -> None:
    app.setStyleSheet(app_stylesheet())
