"""Shared pill controls for the right sidebar (repair panel + chat composer)."""

from __future__ import annotations

from PySide6.QtWidgets import QPushButton, QToolButton

SIDEBAR_CONTROL_HEIGHT = 28


def style_sidebar_chip(widget: QPushButton | QToolButton, *, compact: bool = False) -> None:
    widget.setObjectName("chatControl")
    if compact:
        widget.setFixedSize(SIDEBAR_CONTROL_HEIGHT, SIDEBAR_CONTROL_HEIGHT)
    else:
        widget.setFixedHeight(SIDEBAR_CONTROL_HEIGHT)


def style_sidebar_primary(button: QPushButton) -> None:
    button.setObjectName("sidebarPrimary")
    button.setFixedHeight(SIDEBAR_CONTROL_HEIGHT)
