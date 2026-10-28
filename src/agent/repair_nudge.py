"""Catch repair answers that tell the user to act on something in view without drawing it."""

from __future__ import annotations

import re

_ACTIONS = re.compile(
    r"插|拔|拧|撬|按住|按下|按一下|测|量一下|量它|焊|拆|对准|接到|接在|放到|放在|夹|剪|撕|掀|揭|"
    r"推|拉开|拨|扣|卡进|装回|换掉|擦|刮|吹|涂"
)
_PLACES = re.compile(
    r"这里|那里|这儿|那儿|这边|那边|[这那](个|颗|根|条|块|排|片|处|几)|"
    r"左|右|上面|下面|上边|下边|上方|下方|底部|顶部|底下|旁边|中间|角|侧|边上|"
    r"口|接口|座|脚|焊盘|触点|位置|附近"
)

NUDGE_TEXT = (
    "（程序提示）你刚才让用户去操作画面里的东西，但没有画出来。"
    "现在只调用 annotate（一个位置或动作）或 start_guide（几步操作），"
    "把刚才说的位置和动作画在本轮附图上，并带上 board_box。不要再输出任何文字。"
    "如果附图里确实看不到你说的东西，就不调用工具，直接结束。"
)


def needs_drawing(reply: str, has_image: bool, drew: bool) -> bool:
    """True when ``reply`` points the user at an action in view and nothing was drawn."""
    if not has_image or drew or not reply:
        return False
    return bool(_ACTIONS.search(reply)) and bool(_PLACES.search(reply))
