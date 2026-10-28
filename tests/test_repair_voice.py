"""Repair chat reads like a person: analysis said before a tool call is kept, and button messages sound natural."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from agent.llm_client import LLMClient  # noqa: E402
from agent.prompts import REPAIR_SYSTEM_PROMPT  # noqa: E402
from agent.repair_messages import check_result_message, new_case_message, verify_step_message  # noqa: E402

_ANALYSIS = "308V 在、5V 没有，毛病在开关电源这一级。"
_TAIL = "先给大电容放电。"


def _client(completions) -> LLMClient:
    client = LLMClient.__new__(LLMClient)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client._model = "m"
    client._base_url = "http://test/v1"
    client._extra_body = {}
    client.supports_vision = None
    return client


def _call() -> SimpleNamespace:
    return SimpleNamespace(
        index=0, id="call_1", function=SimpleNamespace(name="update_repair_case", arguments="{}")
    )


class _Plain:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            msg = SimpleNamespace(content=_ANALYSIS, tool_calls=[_call()])
        else:
            msg = SimpleNamespace(content=_TAIL, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class _Stream:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, *, stream=False, **_kwargs):
        assert stream
        self.calls += 1
        if self.calls == 1:
            chunks = [
                SimpleNamespace(content=_ANALYSIS, tool_calls=None),
                SimpleNamespace(content=None, tool_calls=[_call()]),
            ]
        else:
            chunks = [SimpleNamespace(content=_TAIL, tool_calls=None)]
        return iter(SimpleNamespace(choices=[SimpleNamespace(delta=d)]) for d in chunks)


def _handlers() -> dict:
    return {"update_repair_case": lambda **_: "{}"}


def test_preface_dropped_by_default() -> None:
    assert _client(_Plain()).chat_with_tools("x", [], _handlers()) == _TAIL


def test_preface_kept_in_plain_path() -> None:
    reply = _client(_Plain()).chat_with_tools("x", [], _handlers(), keep_preface=True)
    assert reply == f"{_ANALYSIS}\n\n{_TAIL}"


def test_stream_revokes_preface_by_default() -> None:
    shown, revoked = [], []
    reply = _client(_Stream()).chat_with_tools_stream(
        "x", [], _handlers(), on_delta=shown.append, on_revoke=lambda: revoked.append(1)
    )
    assert reply == _TAIL and revoked == [1]


def test_stream_keeps_preface_and_continues_after_it() -> None:
    shown, revoked = [], []
    reply = _client(_Stream()).chat_with_tools_stream(
        "x",
        [],
        _handlers(),
        on_delta=shown.append,
        on_revoke=lambda: revoked.append(1),
        keep_preface=True,
    )
    assert reply == f"{_ANALYSIS}\n\n{_TAIL}"
    assert "".join(shown) == reply and not revoked


def test_button_messages_sound_like_a_person() -> None:
    texts = [
        new_case_message("Redmi K40 Pro", ""),
        new_case_message("电饭煲", "插电没反应。"),
        check_result_message(3, "5V 输出", "fail", "0V"),
        check_result_message(5, "K1 线圈", "pass", "", last=True),
        verify_step_message(1, "撬开卡扣"),
    ]
    assert texts[1] == "我要修电饭煲，插电没反应。"
    assert texts[2] == "第 3 项「5V 输出」测了，0V，不对。接下来查哪？"
    assert texts[3].endswith("清单上的都查完了，你怎么看？")
    assert texts[4] == "第 2 步「撬开卡扣」我弄好了，你看看画面行不行。"
    for text in texts:
        assert "请" not in text and "调用" not in text and "confirm_step" not in text


def test_prompt_puts_voice_before_procedure() -> None:
    assert REPAIR_SYSTEM_PROMPT.index("怎么说话") < REPAIR_SYSTEM_PROMPT.index("你的工作方法")
    assert "不要念给用户听" in REPAIR_SYSTEM_PROMPT
