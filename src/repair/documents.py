"""Uploaded schematics and service manuals: page text search and page rendering."""

from __future__ import annotations

import base64
import json
import math
import re
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PDF_SUFFIXES = {".pdf"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
SUPPORTED_SUFFIXES = PDF_SUFFIXES | IMAGE_SUFFIXES
_PAGES_FILE = "pages.json"
_ASCII_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+\-./]*")
_CJK_RE = re.compile(r"[\u3400-\u9fff]+")
_SPLIT_RE = re.compile(r"[_\-./+]+")


def _open_pdf(path: Path):
    try:
        import pymupdf
    except ImportError:  # older releases only ship the fitz name
        import fitz as pymupdf  # type: ignore[no-redef]
    return pymupdf, pymupdf.open(str(path))


def tokenize(text: str) -> list[str]:
    """Part numbers and net names stay whole (r12, vcc_3v3) and also split (vcc, 3v3).

    Chinese has no spaces, so it is cut into overlapping two-character pieces.
    """
    tokens: list[str] = []
    for word in _ASCII_RE.findall(text):
        word = word.lower().strip("-./+")
        if not word:
            continue
        tokens.append(word)
        parts = [p for p in _SPLIT_RE.split(word) if p]
        if len(parts) > 1:
            tokens.extend(parts)
    for run in _CJK_RE.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


class DocumentLibrary:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._text_cache: dict[str, list[str]] = {}

    def ingest(self, source: Path) -> dict[str, Any]:
        """Copy ``source`` into the library and extract per-page text."""
        source = Path(source)
        suffix = source.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(f"不支持的文件类型: {suffix}（支持 PDF、PNG、JPG、BMP、WEBP）")
        folder = self.root / uuid.uuid4().hex[:12]
        folder.mkdir(parents=True, exist_ok=True)
        stored = folder / source.name
        shutil.copy2(source, stored)
        try:
            if suffix in PDF_SUFFIXES:
                kind = "pdf"
                _module, doc = _open_pdf(stored)
                with doc:
                    texts = [page.get_text() or "" for page in doc]
            else:
                kind = "image"
                if _read_image(stored) is None:
                    raise ValueError(f"无法读取图片: {source.name}")
                texts = [""]
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        (folder / _PAGES_FILE).write_text(
            json.dumps(texts, ensure_ascii=False), encoding="utf-8"
        )
        self._text_cache[str(stored)] = texts
        return {
            "name": source.name,
            "path": str(stored),
            "kind": kind,
            "pages": max(1, len(texts)),
            "has_text": any(t.strip() for t in texts),
        }

    def delete(self, doc: dict[str, Any]) -> None:
        path = Path(doc.get("path") or "")
        self._text_cache.pop(str(path), None)
        folder = path.parent
        if folder.is_dir() and folder.parent.resolve() == self.root.resolve():
            shutil.rmtree(folder, ignore_errors=True)

    def page_texts(self, doc: dict[str, Any]) -> list[str]:
        key = str(doc.get("path") or "")
        if key in self._text_cache:
            return self._text_cache[key]
        pages_file = Path(key).parent / _PAGES_FILE
        try:
            texts = json.loads(pages_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            texts = []
        if not isinstance(texts, list):
            texts = []
        texts = [str(t) for t in texts]
        self._text_cache[key] = texts
        return texts

    def search(
        self,
        docs: list[dict[str, Any]],
        query: str,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms:
            return []
        pages: list[tuple[dict[str, Any], int, str, Counter]] = []
        for doc in docs:
            for index, text in enumerate(self.page_texts(doc)):
                if text.strip():
                    pages.append((doc, index + 1, text, Counter(tokenize(text))))
        if not pages:
            return []
        total = len(pages)
        df = {t: sum(1 for *_rest, counts in pages if counts.get(t)) for t in terms}

        hits: list[dict[str, Any]] = []
        for doc, page_no, text, counts in pages:
            score = 0.0
            matched: list[str] = []
            for term in terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                weight = 1.0 if _CJK_RE.fullmatch(term) else 3.0
                idf = math.log(1.0 + total / df[term])
                score += weight * idf * (1.0 + math.log(tf))
                matched.append(term)
            if score <= 0:
                continue
            hits.append(
                {
                    "doc_id": doc.get("id"),
                    "name": doc.get("name"),
                    "page": page_no,
                    "score": round(score, 3),
                    "matched": matched,
                    "snippet": _snippet(text, matched),
                }
            )
        hits.sort(key=lambda hit: hit["score"], reverse=True)
        return hits[:limit]

    def render(
        self,
        doc: dict[str, Any],
        page: int = 1,
        region: list[float] | None = None,
        max_side: int = 1600,
        quality: int = 85,
    ) -> tuple[str, str]:
        """Return ``(caption, JPEG data URL)`` for one page, optionally cropped.

        ``region`` is ``[x1, y1, x2, y2]`` within 0..1 of the page.
        """
        path = Path(doc.get("path") or "")
        pages = int(doc.get("pages") or 1)
        if page < 1 or page > pages:
            raise ValueError(f"页码超出范围：共 {pages} 页")
        clip = _clean_region(region)
        if doc.get("kind") == "pdf":
            data = self._render_pdf(path, page, clip, max_side, quality)
        else:
            data = self._render_image(path, clip, max_side, quality)
        caption = f"{doc.get('name')} 第 {page} 页"
        if clip is not None:
            caption += f"（局部 {', '.join(f'{v:.2f}' for v in clip)}）"
        return caption, "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")

    def _render_pdf(
        self,
        path: Path,
        page: int,
        clip: list[float] | None,
        max_side: int,
        quality: int,
    ) -> bytes:
        pymupdf, doc = _open_pdf(path)
        with doc:
            pdf_page = doc[page - 1]
            rect = pdf_page.rect
            if clip is not None:
                rect = pymupdf.Rect(
                    rect.x0 + clip[0] * rect.width,
                    rect.y0 + clip[1] * rect.height,
                    rect.x0 + clip[2] * rect.width,
                    rect.y0 + clip[3] * rect.height,
                )
            longest = max(rect.width, rect.height, 1.0)
            zoom = min(max(max_side / longest, 0.5), 8.0)
            pix = pdf_page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=rect, alpha=False)
            return pix.tobytes("jpg", jpg_quality=int(quality))

    def _render_image(
        self,
        path: Path,
        clip: list[float] | None,
        max_side: int,
        quality: int,
    ) -> bytes:
        image = _read_image(path)
        if image is None:
            raise ValueError(f"无法读取图片: {path.name}")
        if clip is not None:
            h, w = image.shape[:2]
            x1, y1 = int(clip[0] * w), int(clip[1] * h)
            x2, y2 = max(x1 + 1, int(clip[2] * w)), max(y1 + 1, int(clip[3] * h))
            image = image[y1:y2, x1:x2]
        h, w = image.shape[:2]
        longest = max(h, w)
        if longest > max_side:
            scale = max_side / longest
            image = cv2.resize(
                image,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        elif longest < max_side // 2:
            scale = min(4.0, (max_side // 2) / longest)
            image = cv2.resize(
                image,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_CUBIC,
            )
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:
            raise ValueError("图片编码失败")
        return buf.tobytes()


def _read_image(path: Path) -> np.ndarray | None:
    # cv2.imread cannot open non-ASCII paths on Windows.
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if raw.size == 0:
        return None
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def _clean_region(region: Any) -> list[float] | None:
    if not region:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in region)
    except (TypeError, ValueError):
        return None
    if max(x1, y1, x2, y2) > 1.0:
        x1, y1, x2, y2 = (v / 1000.0 for v in (x1, y1, x2, y2))
    x1, x2 = sorted((min(max(x1, 0.0), 1.0), min(max(x2, 0.0), 1.0)))
    y1, y2 = sorted((min(max(y1, 0.0), 1.0), min(max(y2, 0.0), 1.0)))
    if x2 - x1 < 0.02 or y2 - y1 < 0.02:
        return None
    return [x1, y1, x2, y2]


def _snippet(text: str, matched: list[str], radius: int = 80) -> str:
    flat = " ".join(text.split())
    lower = flat.lower()
    pos = -1
    for term in sorted(matched, key=len, reverse=True):
        pos = lower.find(term)
        if pos >= 0:
            break
    if pos < 0:
        return flat[: radius * 2]
    start = max(0, pos - radius)
    end = min(len(flat), pos + radius)
    return ("…" if start else "") + flat[start:end] + ("…" if end < len(flat) else "")
