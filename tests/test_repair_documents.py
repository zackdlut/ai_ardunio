"""Uploaded schematics: page text search and page rendering."""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from repair.documents import DocumentLibrary, tokenize  # noqa: E402


def _decode(url: str) -> np.ndarray:
    raw = base64.b64decode(url.split(",", 1)[1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def _make_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Power input J1 F1 FUSE 2A D1 SS34")
    doc.new_page().insert_text((72, 72), "U3 AMS1117 output VCC_3V3 C12 22uF")
    doc.new_page()
    doc.save(str(path))
    return path


def test_tokens_keep_part_numbers_and_split_nets() -> None:
    tokens = tokenize("VCC_3V3 接 U3，保险丝")
    assert "vcc_3v3" in tokens and "3v3" in tokens and "u3" in tokens
    assert "保险" in tokens and "险丝" in tokens


def test_pdf_ingest_search_and_render(tmp_path: Path) -> None:
    library = DocumentLibrary(tmp_path / "lib")
    info = library.ingest(_make_pdf(tmp_path / "board.pdf"))
    assert info["kind"] == "pdf" and info["pages"] == 3 and info["has_text"] is True
    doc = {"id": 5, **info}

    hits = library.search([doc], "3V3 U3")
    assert hits[0]["page"] == 2 and hits[0]["doc_id"] == 5
    assert "AMS1117" in hits[0]["snippet"]
    assert library.search([doc], "fuse")[0]["page"] == 1
    assert library.search([doc], "nothing-here") == []

    fresh = DocumentLibrary(tmp_path / "lib")
    assert fresh.page_texts(doc)[1].startswith("U3")

    caption, url = library.render(doc, 2, max_side=800)
    assert "第 2 页" in caption
    whole = _decode(url)
    assert max(whole.shape[:2]) == 800
    _caption, crop_url = library.render(doc, 2, region=[0, 0, 0.5, 0.25], max_side=800)
    crop = _decode(crop_url)
    assert crop.shape[1] > crop.shape[0]
    with pytest.raises(ValueError):
        library.render(doc, 4)

    library.delete(doc)
    assert not Path(info["path"]).exists()


def test_image_ingest_and_crop(tmp_path: Path) -> None:
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    image[:, 200:] = 255
    src = tmp_path / "原理图.png"
    ok, buf = cv2.imencode(".png", image)
    assert ok
    buf.tofile(str(src))

    library = DocumentLibrary(tmp_path / "lib")
    info = library.ingest(src)
    assert info["kind"] == "image" and info["pages"] == 1 and info["has_text"] is False
    _caption, url = library.render({"id": 1, **info}, 1, region=[0.6, 0, 1, 1], max_side=400)
    assert int(_decode(url).mean()) > 200


def test_unsupported_file_is_rejected(tmp_path: Path) -> None:
    src = tmp_path / "notes.docx"
    src.write_bytes(b"x")
    with pytest.raises(ValueError):
        DocumentLibrary(tmp_path / "lib").ingest(src)
