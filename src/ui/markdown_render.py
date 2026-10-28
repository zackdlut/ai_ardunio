"""Markdown to the rich-text subset QLabel understands, styled for the tutor bubble."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

from ui.theme import (
    COLOR_ACCENT,
    COLOR_ACCENT_INK,
    COLOR_ACCENT_SOFT,
    COLOR_BORDER,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    COLOR_VIEWPORT,
)

MONO = "'Cascadia Mono', 'Consolas', monospace"
CODE_TEXT = "#DCE6E0"
CODE_LABEL = "#6FCB9F"
INLINE_CODE_BG = "#EEF4F0"
BLOCK_GAP = 8
LINE_HEIGHT = "150%"

_FENCE = re.compile(r"^\s*(`{3,}|~{3,})\s*([^\s`]*)")
_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_RULE = re.compile(r"^\s{0,3}([-*_])(\s*\1){2,}\s*$")
_QUOTE = re.compile(r"^\s{0,3}>\s?(.*)$")
_ITEM = re.compile(r"^(\s*)([-*+]|\d{1,3}[.)])\s+(.*)$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_CODE_SPAN = re.compile(r"(`+)(.+?)(?<!`)\1(?!`)")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_BOLD = re.compile(r"\*\*(?!\s)(.+?)(?<!\s)\*\*|__(?!\s)(.+?)(?<!\s)__")
_ITALIC = re.compile(r"(?<![*\w])\*(?![\s*])(.+?)(?<![\s*])\*(?!\*)|(?<![_\w])_(?![\s_])(.+?)(?<![\s_])_(?![_\w])")
_STRIKE = re.compile(r"~~(?!\s)(.+?)(?<!\s)~~")

_HEADING_SIZE = {1: 16, 2: 15, 3: 14}


def render_markdown(text: str) -> str:
    """Return QLabel rich text; an unclosed fence keeps streaming text inside its code block."""
    blocks = _blocks(text.replace("\r\n", "\n").replace("\r", "\n").split("\n"))
    out: list[str] = []
    for index, block in enumerate(blocks):
        out.append(_wrap_gap(block, first=index == 0))
    return "".join(out)


def code_blocks(text: str) -> list[str]:
    lines = text.replace("\r\n", "\n").split("\n")
    found: list[str] = []
    i = 0
    while i < len(lines):
        match = _FENCE.match(lines[i])
        if not match:
            i += 1
            continue
        body, i = _take_fence(lines, i, match.group(1))
        found.append("\n".join(body))
    return found


def _wrap_gap(block: str, first: bool) -> str:
    top = 0 if first else BLOCK_GAP
    return f'<div style="margin-top:{top}px;">{block}</div>'


def _blocks(lines: list[str]) -> list[str]:
    blocks: list[str] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            body = "<br>".join(_inline(line.strip()) for line in paragraph)
            blocks.append(f'<p style="margin:0; line-height:{LINE_HEIGHT};">{body}</p>')
            paragraph.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            flush()
            i += 1
            continue

        fence = _FENCE.match(line)
        if fence:
            flush()
            body, i = _take_fence(lines, i, fence.group(1))
            blocks.append(_code_block("\n".join(body), fence.group(2)))
            continue

        heading = _HEADING.match(line)
        if heading:
            flush()
            blocks.append(_heading(len(heading.group(1)), heading.group(2)))
            i += 1
            continue

        if _RULE.match(line):
            flush()
            blocks.append(_rule())
            i += 1
            continue

        if "|" in line and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1]):
            flush()
            rows = [line]
            i += 2
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append(lines[i])
                i += 1
            blocks.append(_table(rows))
            continue

        if _QUOTE.match(line):
            flush()
            inner: list[str] = []
            while i < len(lines) and (quote := _QUOTE.match(lines[i])):
                inner.append(quote.group(1))
                i += 1
            blocks.append(_quote(inner))
            continue

        if _ITEM.match(line):
            flush()
            items, i = _take_list(lines, i)
            blocks.append(_list(items))
            continue

        paragraph.append(line)
        i += 1

    flush()
    return blocks


def _take_fence(lines: list[str], start: int, marker: str) -> tuple[list[str], int]:
    body: list[str] = []
    i = start + 1
    while i < len(lines):
        if lines[i].strip().startswith(marker[0] * len(marker)) and not lines[i].strip().strip(marker[0]):
            return body, i + 1
        body.append(lines[i])
        i += 1
    return body, i


@dataclass
class _Item:
    indent: int
    marker: str
    lines: list[str]
    children: list["_Item"] = field(default_factory=list)


def _take_list(lines: list[str], start: int) -> tuple[list[_Item], int]:
    flat: list[_Item] = []
    i = start
    while i < len(lines):
        line = lines[i]
        match = _ITEM.match(line)
        if match:
            indent = len(match.group(1).expandtabs(4))
            flat.append(_Item(indent, match.group(2), [match.group(3)]))
            i += 1
            continue
        if not line.strip():
            nxt = i + 1
            if nxt < len(lines) and _ITEM.match(lines[nxt]):
                i = nxt
                continue
            break
        if line.startswith((" ", "\t")) and flat:
            flat[-1].lines.append(line.strip())
            i += 1
            continue
        break

    roots: list[_Item] = []
    stack: list[_Item] = []
    for item in flat:
        while stack and item.indent <= stack[-1].indent:
            stack.pop()
        (stack[-1].children if stack else roots).append(item)
        stack.append(item)
    return roots, i


def _list(items: list[_Item], depth: int = 0) -> str:
    ordered = items[0].marker[0].isdigit()
    counter = int(re.sub(r"\D", "", items[0].marker) or 1) if ordered else 0
    rows: list[str] = []
    for item in items:
        if ordered:
            mark = f'<span style="color:{COLOR_ACCENT_INK}; font-weight:600;">{counter}.</span>'
            counter += 1
        else:
            glyph = "•" if depth == 0 else "◦"
            mark = f'<span style="color:{COLOR_ACCENT}; font-weight:700;">{glyph}</span>'
        body = "<br>".join(_inline(part) for part in item.lines)
        nested = _list(item.children, depth + 1) if item.children else ""
        rows.append(
            "<tr>"
            f'<td width="{22 if ordered else 16}" valign="top" style="line-height:{LINE_HEIGHT};">{mark}</td>'
            f'<td valign="top"><p style="margin:0; line-height:{LINE_HEIGHT};">{body}</p>{nested}</td>'
            "</tr>"
        )
    pad = "margin-top:2px;" if depth else ""
    return f'<table cellspacing="0" cellpadding="1" style="{pad}">{"".join(rows)}</table>'


def _heading(level: int, text: str) -> str:
    size = _HEADING_SIZE.get(level, 13)
    color = COLOR_ACCENT_INK if level >= 4 else COLOR_TEXT
    return (
        f'<p style="margin:2px 0 0 0; font-size:{size}px; font-weight:700; color:{color};">'
        f"{_inline(text)}</p>"
    )


def _rule() -> str:
    return f'<hr style="background-color:{COLOR_BORDER}; height:1px; border:none;">'


def _code_block(code: str, lang: str) -> str:
    label = ""
    if lang:
        label = (
            f'<p style="margin:0 0 6px 0; font-family:{MONO}; font-size:11px; '
            f'color:{CODE_LABEL};">{html.escape(lang.lower())}</p>'
        )
    body = html.escape(code.rstrip("\n")) or "&nbsp;"
    return (
        f'<table width="100%" cellspacing="0" cellpadding="10" bgcolor="{COLOR_VIEWPORT}">'
        f"<tr><td>{label}"
        f'<p style="margin:0; white-space:pre-wrap; font-family:{MONO}; font-size:12px; '
        f'color:{CODE_TEXT}; line-height:140%;">{body}</p>'
        "</td></tr></table>"
    )


def _quote(lines: list[str]) -> str:
    inner = "".join(_wrap_gap(block, first=index == 0) for index, block in enumerate(_blocks(lines)))
    return (
        '<table cellspacing="0" cellpadding="0"><tr>'
        f'<td width="3" bgcolor="{COLOR_ACCENT}"></td>'
        f'<td style="padding-left:10px; color:{COLOR_TEXT_MUTED};">{inner}</td>'
        "</tr></table>"
    )


def _cells(row: str) -> list[str]:
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    return [cell.strip() for cell in row.split("|")]


def _table(rows: list[str]) -> str:
    header, *body = rows
    head_cells = "".join(
        f'<th align="left" style="color:{COLOR_ACCENT_INK};">{_inline(cell)}</th>' for cell in _cells(header)
    )
    body_rows = "".join(
        "<tr>" + "".join(f"<td>{_inline(cell)}</td>" for cell in _cells(row)) + "</tr>" for row in body
    )
    return (
        f'<table cellspacing="0" cellpadding="6" border="1" '
        f'style="border-collapse:collapse; border-style:solid; border-color:{COLOR_BORDER};">'
        f'<tr bgcolor="{COLOR_ACCENT_SOFT}">{head_cells}</tr>{body_rows}</table>'
    )


def _inline(text: str) -> str:
    spans: list[str] = []

    def stash(match: re.Match[str]) -> str:
        spans.append(match.group(2).strip())
        return f"\x00{len(spans) - 1}\x00"

    text = _CODE_SPAN.sub(stash, text)
    text = html.escape(text, quote=False)
    text = _LINK.sub(
        lambda m: f'<a href="{html.escape(m.group(2))}" style="color:{COLOR_ACCENT_INK};">{m.group(1)}</a>',
        text,
    )
    text = _BOLD.sub(lambda m: f"<b>{m.group(1) or m.group(2)}</b>", text)
    text = _ITALIC.sub(lambda m: f"<i>{m.group(1) or m.group(2)}</i>", text)
    text = _STRIKE.sub(r"<s>\1</s>", text)

    def restore(match: re.Match[str]) -> str:
        code = html.escape(spans[int(match.group(1))], quote=False)
        return (
            f'<span style="font-family:{MONO}; font-size:12px; background-color:{INLINE_CODE_BG}; '
            f'color:{COLOR_ACCENT_INK};">&nbsp;{code}&nbsp;</span>'
        )

    return re.sub(r"\x00(\d+)\x00", restore, text)
