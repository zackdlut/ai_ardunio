"""YOLO device selection helpers."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from vision.detector import (  # noqa: E402
    is_openvino_model,
    openvino_export_dir,
    openvino_export_is_fresh,
    select_yolo_runtime,
)


def test_select_yolo_runtime() -> None:
    assert select_yolo_runtime("auto", True) == "npu"
    assert select_yolo_runtime("auto", False) == "cpu"
    assert select_yolo_runtime("npu", True) == "npu"
    assert select_yolo_runtime("npu", False) == "cpu"
    assert select_yolo_runtime("cpu", True) == "cpu"
    assert select_yolo_runtime(" CPU ", False) == "cpu"


def test_openvino_export_cache(tmp_path: Path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"pt")
    export_dir = openvino_export_dir(weights)
    assert export_dir.name == "best_openvino_model"
    assert not openvino_export_is_fresh(weights, export_dir)

    export_dir.mkdir()
    xml = export_dir / "best.xml"
    xml.write_text("<net/>", encoding="utf-8")
    assert is_openvino_model(export_dir)
    assert openvino_export_is_fresh(weights, export_dir)

    time.sleep(0.05)
    weights.write_bytes(b"pt2")
    assert not openvino_export_is_fresh(weights, export_dir)
