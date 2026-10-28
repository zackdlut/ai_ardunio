from __future__ import annotations

import json
from typing import Any

def build_extra_system(
    progress: dict[str, Any],
    verify_ok: bool,
    verify_msg: str,
    scene_summary: dict[str, Any],
    voice_mode: bool,
    has_image: bool = False,
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
    marks = scene_summary.get("user_marks") or []
    if marks:
        compact = [{k: v for k, v in mark.items() if k != "points"} for mark in marks]
        parts.append(
            "学生在画面上的标记（坐标均为相对宽高 0 到 1；kind=point 是单击点，"
            "box 是拖出的矩形框，lasso 是自由圈选；nx/ny 为中心，box 为 [x1,y1,x2,y2] 范围）: "
            + json.dumps(compact, ensure_ascii=False)
        )
        parts.append("学生若问起「这里」或某个标记，就按这些坐标和最近的器件回答。")
    if has_image:
        parts.append(
            "学生这条消息附了一张当前摄像头画面。画面上的角框、器件名、引脚高亮、"
            "箭头、顶部提示条、「标记 N」以及橙色方框和圈线都是程序叠加的，不是实物。"
            "判断接线、极性、插孔位置时以图片为准；检测框可能漏检或误检，"
            "与视觉摘要冲突时说明你看到的情况。看不清就直说并请学生调整镜头，不要猜。"
        )
    if voice_mode:
        parts.append(
            "学生正在用语音和你面对面交流。请只输出要说出口的话："
            "2到4句口语化中文，先接住学生的问题，再给一个明确的下一步。"
            "不要使用 Markdown、列表、代码，也不要输出【朗读】标记。"
            "当前步骤和视觉摘要已经在上面，非必要不要调用工具，以便尽快开口。"
        )
    return "\n".join(parts)
