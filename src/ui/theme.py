"""Pale mint canvas and green copilot chrome, matched to the reference terminal app."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QColor, QPalette

_CHEVRON_DOWN = (Path(__file__).parent / "assets" / "chevron_down.png").as_posix()

COLOR_BG = "#F3F6F4"
COLOR_SURFACE = "#FFFFFF"
COLOR_BORDER = "#E3EAE6"
COLOR_TEXT = "#1B2A22"
COLOR_TEXT_MUTED = "#8B9892"
COLOR_TEXT_FAINT = "#A3AFA9"
COLOR_ACCENT = "#1B9A6A"
COLOR_ACCENT_HOVER = "#15885C"
COLOR_ACCENT_SOFT = "#E7F6EF"
COLOR_ACCENT_INK = "#137A52"
COLOR_USER_BUBBLE = "#E7F6EF"
COLOR_THINK = "#5E6D66"
COLOR_VIEWPORT = "#1A1F1C"
COLOR_FAULT = "#B3261E"
COLOR_FAULT_SOFT = "#FCF1F0"
COLOR_CAUTION = "#B07A10"
COLOR_CAUTION_SOFT = "#FDF6E8"
COLOR_TRACK_EMPTY = "#DCE5E0"
# Readings and step counters: what an instrument shows is set in mono.
FONT_MONO = '"Cascadia Mono", "Consolas", monospace'
FONT_MONO_FAMILIES = ["Cascadia Mono", "Consolas"]


def app_stylesheet() -> str:
    # Must stay a single selector: it is prefixed onto descendant rules below.
    sidebar = "QWidget#sidePanel"
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

    QLabel#menuSection {{
        color: {COLOR_TEXT_MUTED};
        font-size: 11px;
        font-weight: 600;
        padding: 6px 14px 2px 14px;
        background: transparent;
    }}

    QFrame#toolbarDivider {{
        background-color: {COLOR_BORDER};
        border: none;
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

    QScrollArea#chatScroll,
    QScrollArea#chatScroll QWidget#qt_scrollarea_viewport,
    QWidget#chatTranscript {{
        background: transparent;
        border: none;
    }}

    QLabel#speakerName {{
        color: {COLOR_ACCENT_INK};
        font-size: 12px;
        font-weight: 700;
    }}

    QLabel#speakerTime {{
        color: {COLOR_TEXT_FAINT};
        font-family: {FONT_MONO};
        font-size: 11px;
    }}

    QLabel#bubbleText {{
        color: {COLOR_TEXT};
        font-size: 13px;
        background: transparent;
    }}

    QPushButton#thinkToggle,
    QPushButton#logToggle {{
        background: transparent;
        border: 1px solid transparent;
        border-radius: 5px;
        color: {COLOR_TEXT_MUTED};
        font-size: 12px;
        font-weight: 500;
        text-align: left;
        padding: 1px 6px 1px 2px;
    }}

    QPushButton#logToggle {{
        color: {COLOR_TEXT_FAINT};
        font-size: 11px;
    }}

    QPushButton#thinkToggle:hover,
    QPushButton#logToggle:hover {{
        color: {COLOR_ACCENT_INK};
    }}

    QPushButton#thinkToggle:focus,
    QPushButton#logToggle:focus {{
        border-color: {COLOR_ACCENT};
    }}

    QLabel#thinkBody {{
        color: {COLOR_THINK};
        font-size: 12px;
        background: transparent;
        padding: 0 0 0 10px;
        margin-left: 3px;
        border-left: 2px solid {COLOR_TRACK_EMPTY};
    }}

    QLabel#logTime {{
        color: {COLOR_TEXT_FAINT};
        font-family: {FONT_MONO};
        font-size: 11px;
        padding-top: 1px;
    }}

    QLabel#logText {{
        color: {COLOR_TEXT_MUTED};
        font-size: 12px;
    }}

    QLabel#logText[alert="true"] {{
        color: {COLOR_FAULT};
    }}

    QLabel#chatPlaceholder {{
        color: {COLOR_TEXT_MUTED};
        font-size: 12px;
        padding: 28px 24px;
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

    QWidget#chatSidebar QLabel#modelLabel {{
        font-size: 11px;
        font-weight: 500;
        padding: 0 2px;
    }}

    QWidget#chatSidebar QLabel#voiceStatus {{
        color: {COLOR_ACCENT_INK};
        font-size: 11px;
        font-weight: 600;
        padding: 0 4px;
    }}

    {sidebar} QPushButton#chatControl,
    {sidebar} QToolButton#chatControl {{
        background-color: #F4F8F6;
        border: 1px solid {COLOR_BORDER};
        border-radius: 14px;
        padding: 0 10px;
        min-height: 28px;
        max-height: 28px;
        color: {COLOR_TEXT};
        font-size: 12px;
        font-weight: 600;
    }}

    {sidebar} QToolButton#chatControl {{
        padding: 0 8px 0 6px;
    }}

    {sidebar} QPushButton#chatControl:hover,
    {sidebar} QToolButton#chatControl:hover {{
        background-color: #FAFCFB;
        border-color: #C9D5CE;
    }}

    {sidebar} QPushButton#chatControl:pressed,
    {sidebar} QToolButton#chatControl:pressed {{
        background-color: #E7EEEA;
    }}

    {sidebar} QPushButton#chatControl:focus,
    {sidebar} QToolButton#chatControl:focus {{
        border-color: {COLOR_ACCENT};
    }}

    {sidebar} QToolButton#chatControl::menu-indicator {{
        image: none;
        width: 0;
    }}

    {sidebar} QPushButton#chatControl:checked {{
        background-color: {COLOR_ACCENT_SOFT};
        border-color: {COLOR_ACCENT};
        color: {COLOR_ACCENT_INK};
    }}

    {sidebar} QPushButton#chatControl:disabled,
    {sidebar} QToolButton#chatControl:disabled {{
        color: #B5C0BA;
        border-color: #E8EEEA;
        background-color: #F7FAF8;
    }}

    {sidebar} QPushButton#sidebarPrimary {{
        background-color: {COLOR_ACCENT};
        border: 1px solid {COLOR_ACCENT};
        color: #FFFFFF;
        font-weight: 600;
        font-size: 12px;
        border-radius: 14px;
        padding: 0 12px;
        min-height: 28px;
        max-height: 28px;
    }}

    {sidebar} QPushButton#sidebarPrimary:hover {{
        background-color: {COLOR_ACCENT_HOVER};
        border-color: {COLOR_ACCENT_HOVER};
    }}

    {sidebar} QPushButton#sidebarPrimary:pressed {{
        background-color: #0F6E48;
        border-color: #0F6E48;
    }}

    {sidebar} QPushButton#sidebarPrimary:focus {{
        border: 2px solid {COLOR_ACCENT_SOFT};
    }}

    {sidebar} QPushButton#sidebarPrimary:disabled {{
        background-color: #8ACFAF;
        border-color: #8ACFAF;
        color: #FFFFFF;
    }}

    QWidget#chatSidebar QPushButton#chatSend {{
        background-color: {COLOR_ACCENT};
        border: none;
        border-radius: 14px;
        min-width: 28px;
        max-width: 28px;
        min-height: 28px;
        max-height: 28px;
        padding: 0;
        color: #FFFFFF;
        font-size: 15px;
        font-weight: 700;
    }}

    QWidget#chatSidebar QPushButton#chatSend:hover {{
        background-color: {COLOR_ACCENT_HOVER};
    }}

    QWidget#chatSidebar QPushButton#chatSend:pressed {{
        background-color: #0F6E48;
    }}

    QWidget#chatSidebar QPushButton#chatSend:focus {{
        border: 2px solid {COLOR_ACCENT_SOFT};
    }}

    QWidget#chatSidebar QPushButton#chatSend:disabled {{
        background-color: {COLOR_TRACK_EMPTY};
        color: #FFFFFF;
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

    QPushButton#primaryButton:disabled {{
        background-color: #8ACFAF;
        border-color: #8ACFAF;
        color: #FFFFFF;
    }}

    QPushButton:disabled {{
        color: #B5C0BA;
        border-color: #E8EEEA;
        background-color: #F7FAF8;
    }}

    QListWidget#checkList {{
        background-color: #F7FAF8;
        border: 1px solid {COLOR_BORDER};
        border-radius: 8px;
        padding: 4px;
        font-size: 12px;
        outline: none;
    }}

    QListWidget#checkList::item {{
        padding: 3px 6px;
        border-radius: 6px;
    }}

    QListWidget#checkList::item:selected {{
        background-color: {COLOR_ACCENT_SOFT};
        color: {COLOR_TEXT};
    }}

    QLabel#conclusionCard {{
        background-color: {COLOR_ACCENT_SOFT};
        border: 1px solid #CDEBDD;
        border-radius: 10px;
        padding: 8px 10px;
        color: {COLOR_TEXT};
        font-size: 12px;
    }}

    QFrame#guideCard QLabel {{
        background: transparent;
        border: none;
    }}

    QLabel#readout {{
        font-family: {FONT_MONO};
        font-size: 13px;
        font-weight: 700;
        color: {COLOR_ACCENT_INK};
    }}

    QLabel#readoutMuted {{
        font-family: {FONT_MONO};
        font-size: 12px;
        color: {COLOR_TEXT_MUTED};
    }}

    QLabel#guideStepTitle {{
        font-size: 15px;
        font-weight: 700;
        color: {COLOR_TEXT};
    }}

    QLabel#guideInstruction {{
        font-size: 13px;
        color: {COLOR_TEXT};
    }}

    QLabel#guideVerify {{
        font-size: 12px;
        color: {COLOR_THINK};
        padding: 5px 8px;
        background-color: #EEF4F1;
        border-radius: 6px;
    }}

    QLabel#guideFeedback {{
        font-size: 12px;
        font-weight: 600;
        padding: 6px 8px;
        border-radius: 6px;
        color: {COLOR_ACCENT_INK};
        background-color: {COLOR_ACCENT_SOFT};
    }}

    QLabel#guideFeedback[tone="fail"] {{
        color: {COLOR_FAULT};
        background-color: {COLOR_FAULT_SOFT};
    }}

    QLabel#trackChip {{
        font-size: 11px;
        font-weight: 600;
        padding: 0 2px;
        color: {COLOR_TEXT_MUTED};
    }}

    QLabel#eventEyebrow {{
        font-size: 11px;
        font-weight: 700;
    }}

    QLabel#eventTitle {{
        font-size: 13px;
        font-weight: 600;
        color: {COLOR_TEXT};
    }}

    QLabel#eventDetail {{
        font-size: 12px;
        color: {COLOR_THINK};
    }}

    QDialog#settingsDialog {{
        background-color: {COLOR_BG};
    }}

    QDialog#settingsDialog QTabWidget::pane {{
        background: {COLOR_SURFACE};
        border: 1px solid {COLOR_BORDER};
        border-radius: 12px;
        top: -1px;
    }}

    QDialog#settingsDialog QTabBar::tab {{
        background: transparent;
        color: {COLOR_TEXT_MUTED};
        padding: 6px 14px;
        margin: 0 4px 6px 0;
        border-radius: 8px;
        font-weight: 500;
    }}

    QDialog#settingsDialog QTabBar::tab:selected {{
        background: {COLOR_ACCENT_SOFT};
        color: {COLOR_ACCENT_INK};
        font-weight: 600;
    }}

    QDialog#settingsDialog QTabBar::tab:hover:!selected {{
        color: {COLOR_TEXT};
    }}

    QDialog#settingsDialog QLineEdit,
    QDialog#settingsDialog QComboBox,
    QDialog#settingsDialog QSpinBox {{
        background: {COLOR_SURFACE};
        border: 1px solid {COLOR_BORDER};
        border-radius: 8px;
        padding: 5px 8px;
        min-height: 18px;
        selection-background-color: {COLOR_ACCENT};
        selection-color: #FFFFFF;
    }}

    QDialog#settingsDialog QLineEdit:focus,
    QDialog#settingsDialog QComboBox:focus,
    QDialog#settingsDialog QSpinBox:focus {{
        border-color: {COLOR_ACCENT};
    }}

    QDialog#settingsDialog QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}

    QDialog#settingsDialog QComboBox::down-arrow {{
        image: url("{_CHEVRON_DOWN}");
        width: 12px;
        height: 12px;
    }}

    QDialog#settingsDialog QComboBox QAbstractItemView {{
        background: {COLOR_SURFACE};
        border: 1px solid {COLOR_BORDER};
        border-radius: 8px;
        padding: 4px;
        selection-background-color: {COLOR_ACCENT_SOFT};
        selection-color: {COLOR_TEXT};
        outline: none;
    }}

    QLabel#settingsNote {{
        color: {COLOR_TEXT_MUTED};
        font-size: 12px;
    }}

    QLabel#settingsError {{
        color: #B3261E;
        background: #FCEEEE;
        border-radius: 8px;
        padding: 6px 10px;
        font-size: 12px;
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
