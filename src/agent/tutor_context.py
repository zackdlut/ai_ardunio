from __future__ import annotations

import json
from typing import Any

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
            "学生正在用语音和你面对面交流。请只输出要说出口的话："
            "2到4句口语化中文，先接住学生的问题，再给一个明确的下一步。"
            "不要使用 Markdown、列表、代码，也不要输出【朗读】标记。"
            "当前步骤和视觉摘要已经在上面，非必要不要调用工具，以便尽快开口。"
        )
    return "\n".join(parts)
