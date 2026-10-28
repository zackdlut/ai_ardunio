"""Answers that point at something in view get it drawn, and cut-off thinking is never the answer."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from agent.llm_client import LLMClient  # noqa: E402
from agent.repair_context import build_repair_extra_system  # noqa: E402
from agent.repair_nudge import NUDGE_TEXT, needs_drawing  # noqa: E402
from agent.prompts import REPAIR_SYSTEM_PROMPT  # noqa: E402

_REPLY = "把扁的那头插到手机底部那个椭圆口里，插紧会咔一下。"
_THINKING = "The user wants me to use AR annotations. Looking at the image"


def _client(completions) -> LLMClient:
    client = LLMClient.__new__(LLMClient)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client._model = "m"
    client._base_url = "http://test/v1"
    client._extra_body = {"keep": 1}
    client.supports_vision = None
    return client


def _call(name: str = "annotate") -> SimpleNamespace:
    return SimpleNamespace(index=0, id="call_1", function=SimpleNamespace(name=name, arguments="{}"))


class _Plain:
    """Answers in words first; only draws once told to."""

    def __init__(self) -> None:
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        last = kwargs["messages"][-1]
        if last["role"] == "user" and last["content"] == NUDGE_TEXT:
            msg = SimpleNamespace(content="好的我画上了", tool_calls=[_call()])
        elif last["role"] == "tool":
            msg = SimpleNamespace(content="画好了，你看绿框。", tool_calls=None)
        else:
            msg = SimpleNamespace(content=_REPLY, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])


class _Stream:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def create(self, *, stream=False, **kwargs):
        assert stream
        self.requests.append(kwargs)
        last = kwargs["messages"][-1]
        if last["role"] == "user" and last["content"] == NUDGE_TEXT:
            deltas = [
                SimpleNamespace(content="好的我画上了", tool_calls=None),
                SimpleNamespace(content=None, tool_calls=[_call()]),
            ]
        elif last["role"] == "tool":
            deltas = [SimpleNamespace(content="画好了，你看绿框。", tool_calls=None)]
        else:
            deltas = [SimpleNamespace(content=_REPLY, tool_calls=None)]
        return iter(
            SimpleNamespace(choices=[SimpleNamespace(delta=d, finish_reason=None)]) for d in deltas
        )


def _drawing_handlers(drawn: list) -> dict:
    return {"annotate": lambda **_: drawn.append(1) or "{}"}


def _follow_up(asked: list):
    def follow_up(reply: str):
        asked.append(reply)
        return NUDGE_TEXT if needs_drawing(reply, True, False) else None

    return follow_up


def test_needs_drawing_examples() -> None:
    assert needs_drawing(_REPLY, has_image=True, drew=False)
    assert needs_drawing("红笔点在 C12 左边那个焊盘上，黑笔接地。", True, False)
    assert not needs_drawing(_REPLY, has_image=False, drew=False)
    assert not needs_drawing(_REPLY, has_image=True, drew=True)
    assert not needs_drawing("这手机之前摔过或者进过水吗？", True, False)
    assert not needs_drawing("换原装电池大概一百多。", True, False)
    assert not needs_drawing("", True, False)


def test_follow_up_draws_once_and_keeps_the_answer() -> None:
    fake, drawn, asked = _Plain(), [], []
    reply = _client(fake).chat_with_tools(
        "怎么充电", [], _drawing_handlers(drawn), follow_up=_follow_up(asked)
    )
    assert reply == _REPLY
    assert drawn == [1]
    assert asked == [_REPLY]
    assert len(fake.requests) == 3


def test_follow_up_round_streams_nothing() -> None:
    fake, drawn, asked, shown, revoked = _Stream(), [], [], [], []
    reply = _client(fake).chat_with_tools_stream(
        "怎么充电",
        [],
        _drawing_handlers(drawn),
        on_delta=shown.append,
        on_revoke=lambda: revoked.append(1),
        follow_up=_follow_up(asked),
    )
    assert reply == _REPLY
    assert "".join(shown) == _REPLY and not revoked
    assert drawn == [1] and asked == [_REPLY]
    assert len(fake.requests) == 3


def test_no_follow_up_when_not_needed() -> None:
    fake, drawn = _Plain(), []
    reply = _client(fake).chat_with_tools("x", [], _drawing_handlers(drawn), follow_up=lambda _r: None)
    assert reply == _REPLY and not drawn and len(fake.requests) == 1


def test_voice_options_reach_the_request() -> None:
    fake = _Plain()
    client = _client(fake)
    client.chat_with_tools("x", [], {}, voice_mode=True)
    assert fake.requests[-1]["max_tokens"] == 512
    client.chat_with_tools(
        "x", [], {}, voice_mode=True, voice_max_tokens=None, extra_body={"reasoning_effort": "none"}
    )
    request = fake.requests[-1]
    assert "max_tokens" not in request
    assert request["extra_body"] == {"keep": 1, "reasoning_effort": "none"}


class _Truncated:
    def __init__(self, stream: bool) -> None:
        self.stream = stream

    def create(self, **kwargs):
        if self.stream:
            assert kwargs["stream"]
            chunks = [
                SimpleNamespace(choices=[SimpleNamespace(
                    delta=SimpleNamespace(content=None, tool_calls=None, reasoning=_THINKING),
                    finish_reason=None,
                )]),
                SimpleNamespace(choices=[SimpleNamespace(
                    delta=SimpleNamespace(content=None, tool_calls=None), finish_reason="length",
                )]),
            ]
            return iter(chunks)
        msg = SimpleNamespace(content="", tool_calls=None, reasoning=_THINKING)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="length")])


def test_cut_off_thinking_is_not_the_answer() -> None:
    assert _client(_Truncated(False)).chat_with_tools("x", [], {}, voice_mode=True) == ""
    thinking: list[str] = []
    reply = _client(_Truncated(True)).chat_with_tools_stream(
        "x", [], {}, voice_mode=True, on_reasoning=thinking.append
    )
    assert reply == "" and "".join(thinking) == _THINKING


def test_prompts_require_drawing_in_voice_too() -> None:
    assert "必须在同一轮先把它画出来" in REPAIR_SYSTEM_PROMPT
    assert "board_box" in REPAIR_SYSTEM_PROMPT
    voice = build_repair_extra_system(None, [], [], [], voice_mode=True, has_image=True)
    assert "annotate" in voice and "再开口" in voice
    blind = build_repair_extra_system(None, [], [], [], voice_mode=True, has_image=False)
    assert "再开口" not in blind
