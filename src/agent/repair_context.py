from __future__ import annotations

import json
from typing import Any

_STATUS_TEXT = {"pending": "待查", "pass": "正常", "fail": "异常", "skip": "跳过"}
_BOX_FORMAT_TEXT = {
    "norm1000": "0 到 1000 的相对坐标（左上角为 0,0，右下角为 1000,1000）",
    "norm1": "0 到 1 的相对坐标（左上角为 0,0，右下角为 1,1）",
    "pixel": "所附图片的像素坐标",
}


def build_repair_extra_system(
    case: dict[str, Any] | None,
    documents: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    user_marks: list[dict[str, Any]],
    voice_mode: bool,
    has_image: bool = False,
    box_format: str = "norm1000",
    frozen: bool = False,
    ifixit_enabled: bool = True,
) -> str:
    parts = ["【当前维修上下文】"]
    if case:
        parts.append(f"维修单 #{case.get('id')}，状态: {case.get('status') or 'open'}")
        parts.append(f"设备: {case.get('device') or '未填写'}")
        parts.append(f"故障现象: {case.get('symptom') or '未填写'}")
        checks = case.get("checks") or []
        if checks:
            parts.append("检查清单（id. 标题 [状态] 实测/备注）:")
            current = case.get("current_check_id")
            for check in checks:
                status = _STATUS_TEXT.get(check.get("status"), check.get("status"))
                line = f"  {check['id']}. {check.get('title')} [{status}]"
                if check.get("measured"):
                    line += f" 实测: {check['measured']}"
                if check.get("note"):
                    line += f" 备注: {check['note']}"
                if check.get("id") == current:
                    line += "  ← 当前项"
                parts.append(line)
        else:
            parts.append("检查清单: 还没有，了解故障后用 propose_checklist 生成。")
        if case.get("diagnosis"):
            parts.append(f"已给出的结论: {case['diagnosis']}")
    else:
        parts.append("还没有维修单。先问清设备和故障现象，再用 update_repair_case 记录。")

    if documents:
        brief = [
            {
                "doc_id": doc.get("id"),
                "name": doc.get("name"),
                "pages": doc.get("pages"),
                "has_text": doc.get("has_text"),
            }
            for doc in documents
        ]
        parts.append(
            "用户上传的资料（has_text=false 表示扫描件或图片，只能用 view_document_page 看图）: "
            + json.dumps(brief, ensure_ascii=False)
        )
    else:
        parts.append("用户还没有上传原理图或手册。")
    if not ifixit_enabled:
        parts.append("iFixit 查询已在设置中关闭。")

    if annotations:
        compact = [
            {k: v for k, v in item.items() if k in ("label", "kind", "note", "check_id")}
            for item in annotations
        ]
        parts.append("画面上当前的标注: " + json.dumps(compact, ensure_ascii=False))

    if user_marks:
        compact = [{k: v for k, v in mark.items() if k != "points"} for mark in user_marks]
        parts.append(
            "用户在画面上的标记（相对宽高 0 到 1；point 是单击点，box 是矩形 [x1,y1,x2,y2]，"
            "lasso 是圈选的外接框）: " + json.dumps(compact, ensure_ascii=False)
        )
        parts.append("用户问「这里」或「标记 N」时，按这些坐标回答。")

    fmt = _BOX_FORMAT_TEXT.get(box_format, _BOX_FORMAT_TEXT["norm1000"])
    parts.append(f"annotate 和 propose_checklist 的 box 一律写成 [x1,y1,x2,y2]，使用{fmt}。")

    if has_image:
        state = "已定格，标注会固定在这张图上" if frozen else "是实时画面，用户移动设备后标注会错位"
        parts.append(
            f"这条消息附了一张当前摄像头画面（{state}）。画面上的彩色框、编号、顶部提示条、"
            "「标记 N」以及橙色方框和圈线都是程序叠加的，不是实物。"
            "坐标按整张图计算。丝印、焊点看不清时请用户拉近镜头或补光，不要猜。"
        )
    else:
        parts.append("这条消息没有附画面，无法给出画面坐标，只能文字指导。")

    if voice_mode:
        parts.append(
            "用户正在用语音交流，手上可能正拿着表笔。请只输出要说出口的话："
            "2到4句口语化中文，先说结论或下一步要测的点，再说预期值。"
            "不要使用 Markdown、列表、代码，也不要输出【朗读】标记。"
        )
    return "\n".join(parts)
