from __future__ import annotations

import json
from typing import Any, Callable

import httpx
from openai import AuthenticationError, BadRequestError, NotFoundError, OpenAI

from agent.frame_image import drop_images, user_message as build_user_message
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


_CONNECT_HINT = "请检查 OPENAI_BASE_URL、OPENAI_API_KEY 与代理（NO_PROXY / HTTPS_PROXY）"


class LLMClient:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=settings.llm_timeout_s,
        )
        self._model = settings.openai_model
        self._base_url = settings.openai_base_url
        self._vision_mode = settings.llm_vision
        self._extra_body = settings.llm_extra_body_dict()
        # None means the server did not say; the image is sent and dropped if rejected.
        self.supports_vision: bool | None = {"true": True, "false": False}.get(
            self._vision_mode
        )

    def health_check(self) -> tuple[bool, str]:
        listed = True
        try:
            page = self._client.with_options(max_retries=0).models.list(timeout=10.0)
            names = {model.id for model in page}
            if self._model not in names:
                return False, f"LLM 已连接，但未找到模型 {self._model}"
        except NotFoundError:
            # Some OpenAI-compatible gateways do not expose /models.
            listed = False
        except AuthenticationError as exc:
            return False, f"LLM 鉴权失败: {exc}（请检查 OPENAI_API_KEY）"
        except Exception as exc:  # noqa: BLE001
            return False, f"LLM 连接失败: {exc}（{_CONNECT_HINT}）"
        if self._vision_mode == "auto":
            self.supports_vision = self._probe_vision()
        head = f"LLM 连接正常（{self._model}）"
        if not listed:
            head += "，服务未提供模型列表，未校验模型名"
        if self.supports_vision:
            return True, head + "，可以看摄像头画面"
        if self.supports_vision is False:
            return True, head + "，该模型不支持图片，只发送检测结果"
        return True, head

    def _probe_vision(self) -> bool | None:
        """Ask Ollama's /api/show for capabilities. Other servers return None."""
        root = self._base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        try:
            response = httpx.post(f"{root}/api/show", json={"model": self._model}, timeout=3.0)
            response.raise_for_status()
            caps = response.json().get("capabilities")
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(caps, list):
            return None
        return "vision" in caps

    def _build_messages(
        self,
        user_message: str,
        extra_system: str,
        history: list[dict[str, str]] | None,
        max_history_messages: int,
        image_url: str | None,
    ) -> list[dict[str, Any]]:
        system_content = SYSTEM_PROMPT + ("\n" + extra_system if extra_system else "")
        prior = list(history or [])[-max_history_messages:]
        if self.supports_vision is False:
            image_url = None
        return [
            {"role": "system", "content": system_content},
            *prior,
            build_user_message(user_message, image_url),
        ]

    def _create(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        completion_kwargs: dict[str, Any],
        first_round: bool,
        **extra: Any,
    ):
        def call():
            return self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tools if tools else None,
                extra_body=self._extra_body or None,
                **completion_kwargs,
                **extra,
            )

        try:
            return call()
        except BadRequestError:
            # Only before any tool ran: a retry must not repeat compile_and_upload.
            if not first_round or not drop_images(messages):
                raise
            self.supports_vision = False
            return call()

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
        image_url: str | None = None,
    ) -> str:
        messages = self._build_messages(
            user_message, extra_system, history, max_history_messages, image_url
        )
        completion_kwargs: dict[str, Any] = {}
        if voice_mode:
            completion_kwargs["max_tokens"] = 512

        for round_index in range(max_rounds):
            if should_stop and should_stop():
                return ""
            response = self._create(
                messages, tools, completion_kwargs, first_round=round_index == 0
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
        image_url: str | None = None,
    ) -> str:
        """Stream the spoken answer. Thinking tokens go to ``on_reasoning``."""
        messages = self._build_messages(
            user_message, extra_system, history, max_history_messages, image_url
        )
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
        for round_index in range(max_rounds):
            if should_stop and should_stop():
                raise _StreamInterrupted
            content, tool_calls = self._stream_completion(
                messages,
                tools,
                completion_kwargs,
                first_round=round_index == 0,
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
        first_round: bool,
        on_delta: Callable[[str], None] | None,
        on_reasoning: Callable[[str], None] | None,
        on_revoke: Callable[[], None] | None,
        should_stop: Callable[[], bool] | None,
    ) -> tuple[str, list[dict[str, str]] | None]:
        stream = self._create(
            messages, tools, completion_kwargs, first_round=first_round, stream=True
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
