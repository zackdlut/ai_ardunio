from __future__ import annotations

from pathlib import Path

import numpy as np

_CACHE_LIMIT = 96
_FONT_FILES = (
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\msyhbd.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
)


class TextSpriteCache:
    """Cache antialiased labels, including Chinese, as BGRA sprites.

    Drawn with Pillow so the camera thread never touches Qt fonts.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple, np.ndarray] = {}
        self._fonts: dict[int, object] = {}
        self._failed = False

    def get(
        self,
        text: str,
        bgr: tuple[int, int, int],
        px: int = 22,
        max_width: int = 0,
    ) -> np.ndarray | None:
        cleaned = " ".join(text.split())
        if not cleaned or self._failed:
            return None
        key = (cleaned, bgr, px, max_width)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        sprite = self._render(cleaned, bgr, px, max_width)
        if sprite is None:
            return None
        if len(self._cache) >= _CACHE_LIMIT:
            self._cache.clear()
        self._cache[key] = sprite
        return sprite

    def _font(self, px: int):
        cached = self._fonts.get(px)
        if cached is not None:
            return cached
        from PIL import ImageFont

        for path in _FONT_FILES:
            if not path.is_file():
                continue
            font = ImageFont.truetype(str(path), px)
            self._fonts[px] = font
            return font
        self._failed = True
        return None

    def _render(
        self,
        text: str,
        bgr: tuple[int, int, int],
        px: int,
        max_width: int,
    ) -> np.ndarray | None:
        try:
            from PIL import Image, ImageDraw
        except Exception:  # noqa: BLE001
            self._failed = True
            return None
        font = self._font(px)
        if font is None:
            return None
        probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        shown = _fit_text(probe, text, font, max_width)
        left, top, right, bottom = probe.textbbox((0, 0), shown, font=font)
        width = max(8, right - left + 8)
        height = max(8, bottom - top + 6)
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.text(
            (4 - left, 3 - top),
            shown,
            font=font,
            fill=(int(bgr[2]), int(bgr[1]), int(bgr[0]), 255),
        )
        rgba = np.asarray(image)
        bgra = rgba[:, :, [2, 1, 0, 3]]
        return np.ascontiguousarray(bgra)


def _fit_text(draw, text: str, font, max_width: int) -> str:
    if max_width <= 8 or draw.textlength(text, font=font) <= max_width:
        return text
    ellipsis = "…"
    best = ellipsis
    low = 0
    high = len(text)
    while low <= high:
        mid = (low + high) // 2
        candidate = text[:mid].rstrip() + ellipsis
        if draw.textlength(candidate, font=font) <= max_width:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best


def blit_bgra(dest_bgr: np.ndarray, sprite_bgra: np.ndarray, x: int, y: int) -> None:
    """Alpha-blend a BGRA sprite onto a BGR frame. Coordinates may fall outside."""
    if sprite_bgra.size == 0:
        return
    frame_h, frame_w = dest_bgr.shape[:2]
    sprite_h, sprite_w = sprite_bgra.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(frame_w, x + sprite_w)
    y1 = min(frame_h, y + sprite_h)
    if x0 >= x1 or y0 >= y1:
        return
    sx0 = x0 - x
    sy0 = y0 - y
    src = sprite_bgra[sy0 : sy0 + (y1 - y0), sx0 : sx0 + (x1 - x0)]
    alpha = src[:, :, 3:4].astype(np.float32) * (1.0 / 255.0)
    if float(alpha.max()) <= 0:
        return
    roi = dest_bgr[y0:y1, x0:x1]
    blended = src[:, :, :3].astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)
    roi[:] = blended.astype(np.uint8)
