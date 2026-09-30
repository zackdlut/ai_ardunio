from __future__ import annotations

from gesture.mediapipe_ctrl import GestureCommand


class GestureDebouncer:
    """Fire a gesture once it holds steady, then wait until the hand relaxes."""

    def __init__(self, stable_frames: int = 3, release_frames: int = 2) -> None:
        self._stable_frames = max(1, stable_frames)
        self._release_frames = max(1, release_frames)
        self._pending = GestureCommand.NONE
        self._pending_count = 0
        self._latched = False
        self._release_count = 0

    def push(self, command: GestureCommand) -> GestureCommand | None:
        if self._latched:
            if command == GestureCommand.NONE:
                self._release_count += 1
                if self._release_count >= self._release_frames:
                    self._latched = False
                    self._pending = GestureCommand.NONE
                    self._pending_count = 0
                    self._release_count = 0
            else:
                self._release_count = 0
            return None

        if command == GestureCommand.NONE:
            self._pending = GestureCommand.NONE
            self._pending_count = 0
            return None

        if command == self._pending:
            self._pending_count += 1
        else:
            self._pending = command
            self._pending_count = 1

        if self._pending_count < self._stable_frames:
            return None

        self._latched = True
        self._release_count = 0
        return command
