"""Pale mint canvas and green copilot chrome, matched to the reference terminal app."""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette

COLOR_BG = "#F3F6F4"
COLOR_SURFACE = "#FFFFFF"
COLOR_BORDER = "#E3EAE6"
COLOR_TEXT = "#1B2A22"
COLOR_TEXT_MUTED = "#8B9892"
COLOR_ACCENT = "#1B9A6A"
COLOR_ACCENT_HOVER = "#15885C"
COLOR_ACCENT_SOFT = "#E7F6EF"
COLOR_ACCENT_INK = "#137A52"
COLOR_USER_BUBBLE = "#E7F6EF"
COLOR_THINK = "#5E6D66"
COLOR_VIEWPORT = "#1A1F1C"


def app_stylesheet() -> str:
    return f"""
    QWidget {{
        color: {COLOR_TEXT};
        font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
        font-size: 13px;
    }}

    QLabel {{
        background: transparent;
    }}

    QMainWindow, QWidget#appCanvas, QWidget#workspace, QWidget#topBar {{
        background-color: {COLOR_BG};
    }}

    QLabel#appMark {{
        background-color: {COLOR_ACCENT};
        color: #FFFFFF;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 700;
    }}

    QLabel#appTitle {{
        font-size: 14px;
        font-weight: 600;
        color: {COLOR_TEXT};
    }}

    QLabel#sessionChip {{
        background-color: {COLOR_ACCENT_SOFT};
        color: {COLOR_ACCENT_INK};
        border-radius: 11px;
        padding: 3px 10px;
        font-weight: 600;
    }}

    QLabel#sessionChip[connected="false"] {{
        background-color: #EEF2F0;
        color: {COLOR_TEXT_MUTED};
        font-weight: 500;
    }}

    QToolButton#toolbarButton {{
        background: transparent;
        border: 1px solid transparent;
        border-radius: 8px;
        padding: 4px 10px 4px 8px;
        color: {COLOR_TEXT};
        font-weight: 500;
    }}

    QToolButton#toolbarButton:hover,
    QToolButton#toolbarButton:pressed,
    QToolButton#toolbarButton:open {{
        background: {COLOR_SURFACE};
        border-color: {COLOR_BORDER};
    }}

    QToolButton#toolbarButton::menu-indicator {{
        image: none;
        width: 0;
        height: 0;
    }}

    QMenu {{
        background-color: {COLOR_SURFACE};
        color: {COLOR_TEXT};
        border: 1px solid {COLOR_BORDER};
        border-radius: 12px;
        padding: 6px;
    }}

    QMenu::item {{
        padding: 7px 14px 7px 8px;
        margin: 2px 4px;
        border-radius: 8px;
        background: transparent;
    }}

    QMenu::item:selected,
    QMenu::item:checked {{
        background-color: {COLOR_ACCENT_SOFT};
        color: {COLOR_TEXT};
    }}

    QMenu::item:disabled {{
        color: {COLOR_TEXT_MUTED};
        background: transparent;
    }}

    QMenu::icon {{
        padding-left: 8px;
    }}

    QMenu::indicator {{
        width: 0px;
        height: 0px;
    }}

    QMenu::separator {{
        height: 1px;
        background: {COLOR_BORDER};
        margin: 5px 12px;
    }}

    QWidget#chatSidebar {{
        background: transparent;
    }}

    QFrame#sideDivider {{
        background-color: {COLOR_BORDER};
        border: none;
        max-height: 1px;
    }}

    QLabel#presenceDot {{
        color: {COLOR_ACCENT};
        font-size: 9px;
        padding-top: 1px;
    }}

    QLabel#copilotTitle {{
        font-size: 15px;
        font-weight: 600;
        color: {COLOR_TEXT};
    }}

    QLabel#sectionEyebrow {{
        font-size: 11px;
        font-weight: 600;
        color: {COLOR_TEXT_MUTED};
        padding: 0;
    }}

    QLabel#stepTitle {{
        font-size: 14px;
        font-weight: 600;
        color: {COLOR_TEXT};
        padding: 0;
    }}

    QLabel#stepHint {{
        font-size: 12px;
        color: {COLOR_TEXT_MUTED};
    }}

    QLabel#cameraViewport {{
        background-color: {COLOR_VIEWPORT};
        color: #A7B0AB;
        border: none;
        border-radius: 12px;
        font-size: 13px;
    }}

    QFrame#markModeBar {{
        background-color: rgba(22, 32, 28, 210);
        border: 1px solid rgba(255, 255, 255, 36);
        border-radius: 10px;
    }}

    QToolButton#markModeButton {{
        background: transparent;
        border: none;
        border-radius: 7px;
        padding: 3px 10px;
        color: #D7E2DC;
        font-size: 12px;
        font-weight: 600;
    }}

    QToolButton#markModeButton:hover {{
        background-color: rgba(255, 255, 255, 30);
    }}

    QToolButton#markModeButton:checked {{
        background-color: {COLOR_ACCENT};
        color: #FFFFFF;
    }}

    QProgressBar#stepProgress {{
        background-color: #E6EEE9;
        border: none;
        border-radius: 2px;
        height: 3px;
        color: transparent;
    }}

    QProgressBar#stepProgress::chunk {{
        background-color: {COLOR_ACCENT};
        border-radius: 2px;
    }}

    QSplitter#mainSplitter::handle {{
        background-color: {COLOR_BG};
    }}

    QLineEdit#composerInput {{
        background: transparent;
        border: none;
        padding: 2px 2px 4px 2px;
        color: {COLOR_TEXT};
        selection-background-color: {COLOR_ACCENT};
        selection-color: #FFFFFF;
    }}

    QLineEdit#composerInput:focus {{
        border: none;
    }}

    QScrollArea#chatScroll {{
        background: transparent;
        border: none;
    }}

    QLabel#bubbleRole {{
        color: {COLOR_TEXT_MUTED};
        font-size: 11px;
        background: transparent;
    }}

    QLabel#bubbleText {{
        color: {COLOR_TEXT};
        font-size: 13px;
        background: transparent;
    }}

    QPushButton#thinkToggle {{
        background: transparent;
        border: 1px solid transparent;
        border-radius: 6px;
        color: {COLOR_TEXT_MUTED};
        font-size: 12px;
        font-weight: 600;
        text-align: left;
        padding: 2px 6px;
    }}

    QPushButton#thinkToggle:hover {{
        background: {COLOR_ACCENT_SOFT};
        border-color: transparent;
        color: {COLOR_ACCENT_INK};
    }}

    QPushButton#thinkToggle:focus {{
        border: 1px solid {COLOR_ACCENT};
        background: transparent;
    }}

    QLabel#thinkBody {{
        color: {COLOR_THINK};
        font-size: 12px;
        background: transparent;
        padding: 1px 2px 2px 10px;
        margin-left: 8px;
        border-left: 2px solid {COLOR_ACCENT};
    }}

    QLabel#systemDot {{
        color: {COLOR_ACCENT};
        font-size: 9px;
        background: transparent;
    }}

    QLabel#systemText {{
        color: {COLOR_TEXT_MUTED};
        font-size: 12px;
        background: transparent;
    }}

    QLabel#chatPlaceholder {{
        color: {COLOR_TEXT_MUTED};
        font-size: 13px;
        background: transparent;
    }}

    QTextEdit#buildLog {{
        background-color: #F7FAF8;
        border: none;
        border-radius: 8px;
        padding: 8px;
        color: #314038;
        font-family: "Cascadia Mono", "Consolas", monospace;
        font-size: 11px;
    }}

    QLabel#modelLabel {{
        color: {COLOR_TEXT_MUTED};
        font-size: 12px;
    }}

    QPushButton {{
        background-color: {COLOR_SURFACE};
        border: 1px solid {COLOR_BORDER};
        border-radius: 8px;
        padding: 6px 12px;
        color: {COLOR_TEXT};
        font-weight: 500;
    }}

    QPushButton:hover {{
        border-color: #C9D5CE;
        background-color: #FAFCFB;
    }}

    QPushButton:pressed {{
        background-color: #E7EEEA;
    }}

    QPushButton:focus {{
        border: 1px solid {COLOR_ACCENT};
    }}

    QPushButton#primaryButton {{
        background-color: {COLOR_ACCENT};
        border: 1px solid {COLOR_ACCENT};
        color: #FFFFFF;
        font-weight: 600;
        border-radius: 14px;
        padding: 5px 12px;
    }}

    QPushButton#primaryButton:hover {{
        background-color: {COLOR_ACCENT_HOVER};
        border-color: {COLOR_ACCENT_HOVER};
    }}

    QPushButton#primaryButton:pressed {{
        background-color: #0F6E48;
        border-color: #0F6E48;
    }}

    QPushButton#sendButton {{
        background-color: {COLOR_ACCENT};
        border: none;
        color: #FFFFFF;
        border-radius: 16px;
        min-width: 32px;
        max-width: 32px;
        min-height: 32px;
        max-height: 32px;
        padding: 0;
        font-size: 16px;
        font-weight: 700;
    }}

    QPushButton#sendButton:hover {{
        background-color: {COLOR_ACCENT_HOVER};
    }}

    QPushButton#sendButton:pressed {{
        background-color: #0F6E48;
    }}

    QPushButton#voiceButton {{
        background: transparent;
        border: 1px solid {COLOR_BORDER};
        border-radius: 14px;
        padding: 4px 10px;
        color: {COLOR_TEXT};
        font-weight: 500;
    }}

    QPushButton#voiceButton:hover {{
        background: #F7FAF8;
    }}

    QPushButton#voiceButton:checked {{
        background-color: {COLOR_ACCENT_SOFT};
        border-color: {COLOR_ACCENT};
        color: {COLOR_ACCENT_INK};
    }}

    QLabel#voiceStatus {{
        color: {COLOR_ACCENT_INK};
        font-size: 12px;
        background: transparent;
    }}

    QScrollBar:vertical {{
        background: transparent;
        width: 10px;
        margin: 2px;
    }}

    QScrollBar::handle:vertical {{
        background: #D3DDD7;
        border-radius: 4px;
        min-height: 24px;
    }}

    QScrollBar::handle:vertical:hover {{
        background: #B7C4BC;
    }}

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}

    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        background: transparent;
    }}
    """


def apply_theme(app) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(COLOR_BG))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(COLOR_TEXT))
    palette.setColor(QPalette.ColorRole.Base, QColor(COLOR_SURFACE))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(COLOR_BG))
    palette.setColor(QPalette.ColorRole.Text, QColor(COLOR_TEXT))
    palette.setColor(QPalette.ColorRole.Button, QColor(COLOR_SURFACE))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(COLOR_TEXT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(COLOR_ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(COLOR_SURFACE))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(COLOR_TEXT))
    app.setPalette(palette)
    app.setStyleSheet(app_stylesheet())
