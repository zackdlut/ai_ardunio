"""Camera frames go to the chat model as a separate image part."""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import httpx
import numpy as np
from openai import BadRequestError

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from agent.frame_image import drop_images, encode_frame_data_url, user_message  # noqa: E402
from agent.llm_client import LLMClient  # noqa: E402
from agent.tutor_context import build_extra_system  # noqa: E402


def _decode(url: str) -> np.ndarray:
    raw = base64.b64decode(url.split(",", 1)[1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def test_frame_is_shrunk_to_max_side() -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    url = encode_frame_data_url(frame, max_side=768)
    assert url is not None and url.startswith("data:image/jpeg;base64,")
    assert _decode(url).shape[:2] == (432, 768)


def test_small_frame_keeps_its_size() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    assert _decode(encode_frame_data_url(frame, max_side=768)).shape[:2] == (240, 320)


def test_missing_frame_gives_no_image() -> None:
    assert encode_frame_data_url(None) is None
    assert encode_frame_data_url(np.zeros((0, 0, 3), dtype=np.uint8)) is None


def test_user_message_without_image_stays_plain_text() -> None:
    assert user_message("下一步？") == {"role": "user", "content": "下一步？"}


def test_drop_images_keeps_the_question() -> None:
    messages = [
        {"role": "system", "content": "s"},
        user_message("这样接对吗？", "data:image/jpeg;base64,AAAA"),
    ]
    assert drop_images(messages) is True
    assert messages[1] == {"role": "user", "content": "这样接对吗？"}
    assert drop_images(messages) is False


def test_tutor_context_explains_the_overlay_only_when_image_attached() -> None:
    progress = {"title": "Blink", "project_id": "blink", "step_index": 0, "total_steps": 5}
    with_image = build_extra_system(progress, True, "", {}, False, has_image=True)
    without = build_extra_system(progress, True, "", {}, False)
    assert "摄像头画面" in with_image
    assert "摄像头画面" not in without


def _bad_request() -> BadRequestError:
    request = httpx.Request("POST", "http://test/v1/chat/completions")
    response = httpx.Response(400, request=request)
    return BadRequestError("image not supported", response=response, body=None)


class _FakeCompletions:
    def __init__(self) -> None:
        self.sent: list[list] = []

    def create(self, *, messages, **_kwargs):
        self.sent.append([dict(m) for m in messages])
        if isinstance(messages[-1]["content"], list):
            raise _bad_request()
        msg = SimpleNamespace(content="看到了", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _client_with(completions: _FakeCompletions) -> LLMClient:
    client = LLMClient.__new__(LLMClient)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client._model = "m"
    client._base_url = "http://test/v1"
    client._extra_body = {}
    client.supports_vision = None
    return client


def test_rejected_image_is_dropped_and_retried_once() -> None:
    completions = _FakeCompletions()
    client = _client_with(completions)
    reply = client.chat_with_tools(
        "这样接对吗？", [], {}, image_url="data:image/jpeg;base64,AAAA"
    )
    assert reply == "看到了"
    assert len(completions.sent) == 2
    assert completions.sent[1][-1] == {"role": "user", "content": "这样接对吗？"}
    assert client.supports_vision is False


def test_known_text_only_model_never_gets_the_image() -> None:
    completions = _FakeCompletions()
    client = _client_with(completions)
    client.supports_vision = False
    client.chat_with_tools("下一步？", [], {}, image_url="data:image/jpeg;base64,AAAA")
    assert len(completions.sent) == 1
    assert completions.sent[0][-1]["content"] == "下一步？"
