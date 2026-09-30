"""Thinking tags can arrive split across stream chunks."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from agent.think_stream import ThinkSplitter  # noqa: E402


def _feed(parts: list[str]) -> dict[str, str]:
    splitter = ThinkSplitter()
    buckets = {"reasoning": [], "content": []}
    for part in parts:
        for kind, text in splitter.push(part):
            buckets[kind].append(text)
    for kind, text in splitter.flush():
        buckets[kind].append(text)
    return {kind: "".join(pieces) for kind, pieces in buckets.items()}


def test_plain_answer_is_content() -> None:
    assert _feed(["先把长脚接到 D13。"]) == {
        "reasoning": "",
        "content": "先把长脚接到 D13。",
    }


def test_think_block_in_one_chunk() -> None:
    assert _feed(["<think>核对极性</think>\n\n长脚接 D13。"]) == {
        "reasoning": "核对极性",
        "content": "长脚接 D13。",
    }


def test_tags_split_across_chunks() -> None:
    parts = ["<thi", "nk>先看", "电阻</thi", "nk>\n短脚", "回 GND。"]
    assert _feed(parts) == {
        "reasoning": "先看电阻",
        "content": "短脚回 GND。",
    }


def test_whitespace_before_tag_is_dropped() -> None:
    assert _feed(["\n<think>想</think>回答"]) == {
        "reasoning": "想",
        "content": "回答",
    }


def test_unfinished_tag_is_flushed() -> None:
    assert _feed(["答案以 <"]) == {"reasoning": "", "content": "答案以 <"}
