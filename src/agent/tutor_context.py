from __future__ import annotations

import json
from typing import Any

from voice.tts import SPOKEN_MARKER


def build_extra_system(
    progress: dict[str, Any],
    verify_ok: bool,
    verify_msg: str,
    scene_summary: dict[str, Any],
    voice_mode: bool,
) -> str:
    summary = scene_summary.get("summary", {})
    parts = [
        "【当前教学上下文】",
        f"项目: {progress.get('title')} ({progress.get('project_id')})",
        f"步骤 {progress.get('step_index', 0) + 1}/{progress.get('total_steps')}: "
        f"{progress.get('current_step')} — {progress.get('current_hint')}",
        f"已完成步骤: {', '.join(progress.get('completed_steps', [])) or '无'}",
        f"视觉摘要(counts): {json.dumps(summary, ensure_ascii=False)}",
        f"步骤校验: {'通过' if verify_ok else '未通过'} — {verify_msg}",
    ]
    if voice_mode:
        parts.append(
            f"用户通过语音提问。正文可稍详细；另起一行必须以 {SPOKEN_MARKER} 开头，"
            "后接 2–4 句口语化中文，适合朗读，不要列表和代码。"
        )
    return "\n".join(parts)
