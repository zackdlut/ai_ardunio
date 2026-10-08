"""把外部 YOLO 标签改写成 Phase 2 的六个类别。"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import yaml

CANON = ["ArduinoUNO", "Breadboard", "LED", "Resistor", "JumperWire", "Button"]
INDEX = {name: i for i, name in enumerate(CANON)}

ALIASES = {
    "arduino": "ArduinoUNO",
    "arduino_uno": "ArduinoUNO",
    "arduino-uno": "ArduinoUNO",
    "uno": "ArduinoUNO",
    "breadboard": "Breadboard",
    "bboard": "Breadboard",
    "led": "LED",
    "resistor": "Resistor",
    "wire": "JumperWire",
    "jumper": "JumperWire",
    "jumper_wire": "JumperWire",
    "jumper_wires": "JumperWire",
    "jumper-wires": "JumperWire",
    "jumper_cables": "JumperWire",
    "button": "Button",
    "push_button": "Button",
}


def canon_name(raw: str) -> str | None:
    key = raw.strip().lower().replace(" ", "_")
    if key in ALIASES:
        return ALIASES[key]
    if raw in INDEX:
        return raw
    return None


def load_names(data_yaml: Path) -> dict[int, str]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data["names"]
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(v) for i, v in enumerate(names)}


def remap_split(src: Path, dst: Path, names: dict[int, str], prefix: str) -> tuple[int, int]:
    kept = dropped = 0
    images = src / "images"
    labels = src / "labels"
    if not images.is_dir():
        return 0, 0
    for image in images.iterdir():
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        label = labels / f"{image.stem}.txt"
        lines_out: list[str] = []
        if label.is_file():
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                raw = names.get(int(float(parts[0])), "")
                mapped = canon_name(raw)
                if mapped is None:
                    dropped += 1
                    continue
                parts[0] = str(INDEX[mapped])
                lines_out.append(" ".join(parts[:5]))
                kept += 1
        if not lines_out:
            continue
        (dst / "images").mkdir(parents=True, exist_ok=True)
        (dst / "labels").mkdir(parents=True, exist_ok=True)
        out_name = f"{prefix}{image.name}"
        shutil.copy2(image, dst / "images" / out_name)
        (dst / "labels" / f"{Path(out_name).stem}.txt").write_text(
            "\n".join(lines_out) + "\n", encoding="utf-8"
        )
    return kept, dropped


def main() -> None:
    src_root = Path(sys.argv[1])
    dst_root = Path(sys.argv[2])
    prefix = sys.argv[3] if len(sys.argv) > 3 else ""
    yaml_path = next(src_root.rglob("data.yaml"))
    names = load_names(yaml_path)
    print(f"classes: {names}")
    total_kept = total_dropped = 0
    for split in ("train", "valid", "val", "test"):
        kept, dropped = remap_split(src_root / split, dst_root / "all", names, prefix)
        total_kept += kept
        total_dropped += dropped
        if kept or dropped:
            print(f"  {split}: kept_boxes={kept} dropped_boxes={dropped}")
    images = list((dst_root / "all" / "images").glob("*.*")) if (dst_root / "all" / "images").is_dir() else []
    print(f"kept_boxes={total_kept} dropped_boxes={total_dropped} images={len(images)} -> {dst_root / 'all'}")


if __name__ == "__main__":
    main()
