"""Tutor replies render as styled rich text instead of raw Markdown."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from ui.markdown_render import code_blocks, render_markdown  # noqa: E402


def test_inline_marks_become_tags() -> None:
    out = render_markdown("LED 的**长脚**接 `D13`，*注意*方向，~~旧方法~~")
    assert "<b>长脚</b>" in out
    assert "<i>注意</i>" in out
    assert "<s>旧方法</s>" in out
    assert "D13" in out and "`" not in out
    assert "**" not in out


def test_identifiers_with_underscores_stay_plain() -> None:
    out = render_markdown("调用 digital_write_fast 和 LED_BUILTIN")
    assert "<i>" not in out
    assert "digital_write_fast" in out


def test_code_is_escaped_and_not_formatted() -> None:
    out = render_markdown("```cpp\nif (a < b && **p) {}\n```")
    assert "&lt; b &amp;&amp; **p" in out
    assert "<b>" not in out
    assert ">cpp<" in out


def test_unclosed_fence_keeps_streaming_text_in_code() -> None:
    out = render_markdown("看这里：\n```arduino\nvoid setup() {")
    assert "void setup() {" in out
    assert "```" not in out


def test_lists_headings_and_tables() -> None:
    text = "## 步骤\n1. 接线\n2. 上传\n   - 选端口\n\n| 引脚 | 作用 |\n| --- | --- |\n| D13 | LED |"
    out = render_markdown(text)
    assert "##" not in out and "步骤" in out
    assert "1." in out and "2." in out and "◦" in out
    assert "<th" in out and "D13" in out
    assert "---" not in out


def test_html_in_text_is_escaped() -> None:
    out = render_markdown("<script>alert(1)</script>")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_code_blocks_extracts_fenced_bodies() -> None:
    text = "说明\n```cpp\nint a = 1;\n```\n再看\n~~~\nint b = 2;\n~~~"
    assert code_blocks(text) == ["int a = 1;", "int b = 2;"]
