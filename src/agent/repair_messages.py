"""What the app says to the repair assistant on the user's behalf (buttons, not typing).

Phrased the way a person at the bench would say it. Telling the model what to
do step by step here ("analyze the causes, then list checks...") makes it
answer like a form; how to respond belongs in the system prompt.
"""

from __future__ import annotations

_RESULT_WORD = {"pass": "正常", "fail": "不对", "skip": "先跳过了"}


def new_case_message(device: str, symptom: str) -> str:
    device = device.strip() or "一台设备"
    symptom = symptom.strip().rstrip("。")
    if symptom:
        return f"我要修{device}，{symptom}。"
    return f"我要修{device}，毛病还没说。"


def check_result_message(seq: int, title: str, status: str, measured: str = "", last: bool = False) -> str:
    word = _RESULT_WORD.get(status, status)
    if status == "skip":
        text = f"第 {seq} 项「{title}」先跳过了。"
    elif measured:
        text = f"第 {seq} 项「{title}」测了，{measured}，{word}。"
    else:
        text = f"第 {seq} 项「{title}」查了，{word}。"
    if last:
        text += "清单上的都查完了，你怎么看？"
    elif status == "fail":
        text += "接下来查哪？"
    return text


def verify_step_message(index: int, title: str) -> str:
    return f"第 {index + 1} 步「{title}」我弄好了，你看看画面行不行。"


def ifixit_message(device: str) -> str:
    return f"帮我在 iFixit 上找找{device}的拆机和维修指南，还有常见毛病。"
