"""Images produced by tools reach the model as a follow-up user message."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
from openai import BadRequestError

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from agent.llm_client import LLMClient  # noqa: E402

_URL = "data:image/jpeg;base64,AAAA"


def _tool_call(name: str, arguments: str = "{}") -> SimpleNamespace:
    return SimpleNamespace(id="call_1", function=SimpleNamespace(name=name, arguments=arguments))


class _Completions:
    def __init__(self, reject_images: bool = False) -> None:
        self.sent: list[list[dict]] = []
        self.reject_images = reject_images

    def create(self, *, messages, **_kwargs):
        self.sent.append([dict(m) for m in messages])
        if self.reject_images and isinstance(messages[-1]["content"], list):
            request = httpx.Request("POST", "http://test/v1/chat/completions")
            raise BadRequestError("no images", response=httpx.Response(400, request=request), body=None)
        if len(self.sent) == 1:
            msg = SimpleNamespace(content="", tool_calls=[_tool_call("view_document_page", '{"doc_id": 1, "page": 2}')])
        else:
            msg = SimpleNamespace(content="R12 开路", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _client(completions: _Completions, vision: bool | None = None) -> LLMClient:
    client = LLMClient.__new__(LLMClient)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client._model = "m"
    client._base_url = "http://test/v1"
    client._extra_body = {}
    client.supports_vision = vision
    return client


def _run(client: LLMClient):
    pending: list[tuple[str, str]] = []

    def view_document_page(doc_id: int, page: int) -> str:
        pending.append((f"doc {doc_id} 第 {page} 页", _URL))
        return "已附上图片"

    def drain():
        images = list(pending)
        pending.clear()
        return images

    return client.chat_with_tools(
        "看看原理图",
        [],
        {"view_document_page": view_document_page},
        system_prompt="REPAIR",
        drain_images=drain,
    )


def test_tool_image_is_appended_after_tool_results() -> None:
    completions = _Completions()
    assert _run(_client(completions)) == "R12 开路"
    second = completions.sent[1]
    assert second[0]["content"].startswith("REPAIR")
    assert [m["role"] for m in second[-3:]] == ["assistant", "tool", "user"]
    parts = second[-1]["content"]
    assert parts[0]["type"] == "text" and "第 2 页" in parts[0]["text"]
    assert parts[1] == {"type": "image_url", "image_url": {"url": _URL}}


def test_text_only_model_gets_a_note_instead() -> None:
    completions = _Completions()
    _run(_client(completions, vision=False))
    note = completions.sent[1][-1]
    assert note["role"] == "user" and isinstance(note["content"], str)
    assert "不支持看图" in note["content"]


def test_rejected_tool_image_is_dropped_and_retried() -> None:
    completions = _Completions(reject_images=True)
    client = _client(completions)
    assert _run(client) == "R12 开路"
    assert len(completions.sent) == 3
    assert isinstance(completions.sent[2][-1]["content"], str)
    assert client.supports_vision is False


class _StreamCompletions:
    def __init__(self) -> None:
        self.sent: list[list[dict]] = []

    def create(self, *, messages, stream=False, **_kwargs):
        assert stream
        self.sent.append([dict(m) for m in messages])
        if len(self.sent) == 1:
            call = SimpleNamespace(
                index=0,
                id="call_1",
                function=SimpleNamespace(name="view_document_page", arguments='{"doc_id": 1, "page": 2}'),
            )
            delta = SimpleNamespace(content=None, tool_calls=[call])
        else:
            delta = SimpleNamespace(content="看到了 R12", tool_calls=None)
        return iter([SimpleNamespace(choices=[SimpleNamespace(delta=delta)])])


def test_streaming_path_appends_tool_images() -> None:
    completions = _StreamCompletions()
    client = _client(completions)  # type: ignore[arg-type]
    pending = [("doc 1 第 2 页", _URL)]
    reply = client.chat_with_tools_stream(
        "看原理图",
        [],
        {"view_document_page": lambda **_: "已附上图片"},
        system_prompt="REPAIR",
        drain_images=lambda: [pending.pop()] if pending else [],
    )
    assert reply == "看到了 R12"
    last = completions.sent[1][-1]
    assert last["role"] == "user" and last["content"][1]["image_url"]["url"] == _URL


def test_tool_errors_become_tool_results() -> None:
    completions = _Completions()

    def broken(**_kwargs) -> str:
        raise RuntimeError("disk full")

    client = _client(completions)
    client.chat_with_tools("x", [], {"view_document_page": broken})
    tool_msg = completions.sent[1][-1]
    assert tool_msg["role"] == "tool" and "disk full" in tool_msg["content"]
