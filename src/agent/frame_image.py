"""Turn a camera frame into an OpenAI-style image part for the chat model."""

from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np


def encode_frame_data_url(
    frame_bgr: np.ndarray | None,
    max_side: int = 768,
    quality: int = 80,
) -> str | None:
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.size == 0:
        return None
    height, width = frame_bgr.shape[:2]
    longest = max(height, width)
    if max_side > 0 and longest > max_side:
        scale = max_side / longest
        size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        frame_bgr = cv2.resize(frame_bgr, size, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def user_message(text: str, image_url: str | None = None) -> dict[str, Any]:
    if not image_url:
        return {"role": "user", "content": text}
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    }


def drop_images(messages: list[dict[str, Any]]) -> bool:
    """Replace image parts with their text in place. Returns True if any were removed."""
    changed = False
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        texts = [part.get("text", "") for part in content if part.get("type") == "text"]
        msg["content"] = "\n".join(t for t in texts if t)
        changed = True
    return changed
