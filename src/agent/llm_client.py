from __future__ import annotations

import json
from typing import Any, Callable

from openai import OpenAI

from agent.prompts import SYSTEM_PROMPT
from config.settings import get_settings


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
