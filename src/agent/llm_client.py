from __future__ import annotations

import json
from typing import Any, Callable

from openai import OpenAI

from agent.prompts import SYSTEM_PROMPT
from agent.think_stream import ThinkSplitter
from config.settings import get_settings

_REASONING_KEYS = ("reasoning", "reasoning_content", "thinking")


def _field_text(obj: Any, names: tuple[str, ...]) -> str:
    if obj is None:
        return ""
    for name in names:
        value = getattr(obj, name, None)
        if isinstance(value, str) and value:
            return value
    extra = getattr(obj, "model_extra", None)
    if isinstance(extra, dict):
        for name in names:
            value = extra.get(name)
            if isinstance(value, str) and value:
                return value
    return ""


class _StreamInterrupted(Exception):
    """Raised when a voice turn is cancelled mid-stream."""


class LLMClient:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self._model = settings.openai_model

    def health_check(self) -> tuple[bool, str]:
        try:
            page = self._client.with_options(max_retries=0).models.list(timeout=3.0)
            names = {model.id for model in page}
            if self._model not in names:
                return False, f"LLM 已连接，但未找到模型 {self._model}"
            return True, "LLM 连接正常"
        except Exception as exc:  # noqa: BLE001
            return False, f"LLM 连接失败: {exc}（请检查 OPENAI_BASE_URL 与 NO_PROXY）"

    def chat_with_tools(
        self,
        user_message: str,
        tools: list[dict[str, Any]],
        tool_handlers: dict[str, Callable[..., str]],
        extra_system: str = "",
        history: list[dict[str, str]] | None = None,
        voice_mode: bool = False,
        max_rounds: int = 6,
        max_history_messages: int = 20,
        should_stop: Callable[[], bool] | None = None,
    ) -> str:
        system_content = SYSTEM_PROMPT + ("\n" + extra_system if extra_system else "")
        prior = list(history or [])[-max_history_messages:]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
            *prior,
            {"role": "user", "content": user_message},
        ]
        completion_kwargs: dict[str, Any] = {}
        if voice_mode:
            completion_kwargs["max_tokens"] = 512

        for _ in range(max_rounds):
            if should_stop and should_stop():
                return ""
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tools if tools else None,
                **completion_kwargs,
            )
            msg = response.choices[0].message
            tool_calls = msg.tool_calls or []

            if not tool_calls:
                content = (msg.content or "").strip()
                if content:
                    return content
                reasoning = getattr(msg, "reasoning", None) or ""
                if isinstance(reasoning, str) and reasoning.strip():
                    return reasoning.strip()
                return ""

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            for tc in tool_calls:
                if should_stop and should_stop():
                    return ""
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                handler = tool_handlers.get(name)
                if handler:
                    result = handler(**args)
                else:
                    result = f"Unknown tool: {name}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )

        return "（达到工具调用轮次上限，请重试或简化问题）"

    def chat_with_tools_stream(
        self,
        user_message: str,
        tools: list[dict[str, Any]],
        tool_handlers: dict[str, Callable[..., str]],
        extra_system: str = "",
        history: list[dict[str, str]] | None = None,
        voice_mode: bool = False,
        max_rounds: int = 6,
        max_history_messages: int = 20,
        on_delta: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
        on_revoke: Callable[[], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> str:
        """Stream the spoken answer. Thinking tokens go to ``on_reasoning``."""
        system_content = SYSTEM_PROMPT + ("\n" + extra_system if extra_system else "")
        prior = list(history or [])[-max_history_messages:]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
            *prior,
            {"role": "user", "content": user_message},
        ]
        completion_kwargs: dict[str, Any] = {}
        if voice_mode:
            completion_kwargs["max_tokens"] = 512

        try:
            return self._run_stream_rounds(
                messages,
                tools,
                tool_handlers,
                completion_kwargs,
                max_rounds,
                on_delta,
                on_reasoning,
                on_revoke,
                should_stop,
            )
        except _StreamInterrupted:
            return ""

    def _run_stream_rounds(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_handlers: dict[str, Callable[..., str]],
        completion_kwargs: dict[str, Any],
        max_rounds: int,
        on_delta: Callable[[str], None] | None,
        on_reasoning: Callable[[str], None] | None,
        on_revoke: Callable[[], None] | None,
        should_stop: Callable[[], bool] | None,
    ) -> str:
        for _ in range(max_rounds):
            if should_stop and should_stop():
                raise _StreamInterrupted
            content, tool_calls = self._stream_completion(
                messages,
                tools,
                completion_kwargs,
                on_delta=on_delta,
                on_reasoning=on_reasoning,
                on_revoke=on_revoke,
                should_stop=should_stop,
            )
            if not tool_calls:
                return content

            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": slot["id"],
                            "type": "function",
                            "function": {
                                "name": slot["name"],
                                "arguments": slot["arguments"] or "{}",
                            },
                        }
                        for slot in tool_calls
                    ],
                }
            )
            for slot in tool_calls:
                if should_stop and should_stop():
                    raise _StreamInterrupted
                name = slot["name"]
                try:
                    args = json.loads(slot["arguments"] or "{}")
                except json.JSONDecodeError:
                    result = f"工具参数无法解析: {name}"
                else:
                    handler = tool_handlers.get(name)
                    if handler:
                        result = handler(**args)
                    else:
                        result = f"Unknown tool: {name}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": slot["id"],
                        "content": result,
                    }
                )

        return "（达到工具调用轮次上限，请重试或简化问题）"

    def _stream_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        completion_kwargs: dict[str, Any],
        on_delta: Callable[[str], None] | None,
        on_reasoning: Callable[[str], None] | None,
        on_revoke: Callable[[], None] | None,
        should_stop: Callable[[], bool] | None,
    ) -> tuple[str, list[dict[str, str]] | None]:
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools if tools else None,
            stream=True,
            **completion_kwargs,
        )
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_acc: dict[int, dict[str, str]] = {}
        emitted = False
        saw_tool = False
        splitter = ThinkSplitter()

        def emit_reasoning(text: str) -> None:
            if not text:
                return
            reasoning_parts.append(text)
            if on_reasoning is not None:
                on_reasoning(text)

        def emit_content(text: str) -> None:
            nonlocal emitted
            if not text or saw_tool:
                return
            if not content_parts:
                text = text.lstrip()
                if not text:
                    return
            content_parts.append(text)
            if on_delta is not None:
                on_delta(text)
                emitted = True

        try:
            for chunk in stream:
                if should_stop and should_stop():
                    raise _StreamInterrupted
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta is None:
                    continue
                emit_reasoning(_field_text(delta, _REASONING_KEYS))
                piece = getattr(delta, "content", None) or ""
                if piece:
                    for kind, text in splitter.push(piece):
                        if kind == "reasoning":
                            emit_reasoning(text)
                        else:
                            emit_content(text)
                tool_deltas = getattr(delta, "tool_calls", None) or []
                if tool_deltas:
                    if emitted and not saw_tool and on_revoke is not None:
                        on_revoke()
                    saw_tool = True
                    emitted = False
                    _merge_tool_call_deltas(tool_acc, tool_deltas)
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass

        for kind, text in splitter.flush():
            if kind == "reasoning":
                emit_reasoning(text)
            else:
                emit_content(text)

        content = "".join(content_parts).strip()
        if not saw_tool and not content:
            content = "".join(reasoning_parts).strip()
        if not saw_tool:
            return content, None
        ordered = [tool_acc[index] for index in sorted(tool_acc)]
        return content, ordered


def _merge_tool_call_deltas(acc: dict[int, dict[str, str]], deltas: Any) -> None:
    for fallback, delta in enumerate(deltas):
        index = getattr(delta, "index", None)
        slot_index = fallback if index is None else int(index)
        slot = acc.setdefault(
            slot_index, {"id": "", "name": "", "arguments": ""}
        )
        call_id = getattr(delta, "id", None)
        if call_id:
            slot["id"] = call_id
        function = getattr(delta, "function", None)
        if function is None:
            continue
        name = getattr(function, "name", None)
        if name:
            slot["name"] += name
        arguments = getattr(function, "arguments", None)
        if arguments:
            slot["arguments"] += arguments
        if not slot["id"]:
            slot["id"] = f"call_{slot_index}"
