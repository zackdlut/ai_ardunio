from __future__ import annotations

import json
from typing import Any

_STATUS_TEXT = {"pending": "待查", "pass": "正常", "fail": "异常", "skip": "跳过"}
_BOX_FORMAT_TEXT = {
    "norm1000": "0 到 1000 的相对坐标（左上角为 0,0，右下角为 1000,1000）",
    "norm1": "0 到 1 的相对坐标（左上角为 0,0，右下角为 1,1）",
    "pixel": "所附图片的像素坐标",
}
_STEP_TEXT = {"pending": "未做", "done": "已完成", "failed": "未通过", "skipped": "跳过"}
_TRACK_TEXT = {
    "tracking": "实时画面，图形已贴在板子上跟随移动",
    "weak": "实时画面，跟踪信号弱，图形可能轻微漂移",
    "lost": "实时画面，跟踪已丢失，旧图形可能错位，必要时重新标注",
    "off": "实时画面，跟踪已关闭，用户移动设备后图形会错位",
}


def _in_format(
    point: list[float], box_format: str, image_size: tuple[int, int] | None
) -> list[float]:
    x, y = float(point[0]), float(point[1])
    if box_format == "norm1":
        return [round(x, 3), round(y, 3)]
    if box_format == "pixel" and image_size:
        return [round(x * image_size[0]), round(y * image_size[1])]
    return [round(x * 1000), round(y * 1000)]


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
    guide: dict[str, Any] | None = None,
    track_status: str = "",
    image_size: tuple[int, int] | None = None,
    clean_image: bool = False,
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

    def located(shape: dict[str, Any]) -> dict[str, Any]:
        out = {
            k: v
            for k, v in shape.items()
            if k in ("id", "type", "label", "kind", "note", "check_id", "text", "meter", "expected") and v
        }
        pts = shape.get("pts") or []
        if pts and has_image:
            out["pts"] = [_in_format(p, box_format, image_size) for p in pts]
        return out

    if guide and guide.get("status") == "active" and guide.get("steps"):
        steps = guide["steps"]
        current = min(max(int(guide.get("current_step") or 0), 0), len(steps) - 1)
        parts.append(f"进行中的指引 #{guide['id']}「{guide.get('title')}」，共 {len(steps)} 步:")
        for i, step in enumerate(steps):
            status = _STEP_TEXT.get(step.get("status"), step.get("status"))
            line = f"  {i + 1}. {step.get('title')} [{status}]"
            if step.get("feedback"):
                line += f" 反馈: {step['feedback']}"
            if i == current:
                line += "  ← 当前步"
            parts.append(line)
        step = steps[current]
        parts.append(f"当前步说明: {step.get('instruction') or '无'}")
        parts.append(f"当前步判断标准(verify): {step.get('verify') or '未写，按说明判断'}")
        if step.get("shapes"):
            parts.append(
                "当前步画面图形（pts 为图形各点，rect/circle 是四角 左上,右上,右下,左下）: "
                + json.dumps([located(s) for s in step["shapes"]], ensure_ascii=False)
            )

    if annotations:
        parts.append(
            "画面上的其他标注: "
            + json.dumps([located(s) for s in annotations], ensure_ascii=False)
        )

    if user_marks:
        compact = [{k: v for k, v in mark.items() if k != "points"} for mark in user_marks]
        parts.append(
            "用户在画面上的标记（相对宽高 0 到 1；point 是单击点，box 是矩形 [x1,y1,x2,y2]，"
            "lasso 是圈选的外接框）: " + json.dumps(compact, ensure_ascii=False)
        )
        parts.append("用户问「这里」或「标记 N」时，按这些坐标回答。")

    fmt = _BOX_FORMAT_TEXT.get(box_format, _BOX_FORMAT_TEXT["norm1000"])
    parts.append(
        f"所有坐标使用{fmt}：box 写成 [x1,y1,x2,y2]，at/from/to/red/black 等点写成 [x,y]。"
    )

    if has_image:
        if frozen:
            state = "已定格，图形固定在这张图上"
        else:
            state = _TRACK_TEXT.get(track_status, "实时画面")
        if clean_image:
            overlay = (
                "附图是干净的摄像头原始画面，没有画已有图形，只叠加了用户的「标记 N」（橙色圆圈、方框和圈线）。"
                "已有图形的位置见上面列出的坐标。"
            )
        else:
            overlay = (
                "画面上的彩色框、箭头、虚线、表笔圆点、编号、顶部提示条、「标记 N」"
                "以及橙色方框和圈线都是程序叠加的，不是实物。"
            )
        parts.append(
            f"这条消息附了一张当前摄像头画面（{state}）。{overlay}"
            "坐标按整张图计算，上面列出的图形坐标也已换算到这张图上。"
            "丝印、焊点看不清时先用 zoom_view 放大；还看不清就请用户拉近镜头或补光，不要猜。"
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
