"""把映射后的公开图片按 85/15 划成 train/val。"""

from __future__ import annotations

import random
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ROOT / "dataset" / "external" / "circuits-resistor-canon" / "all",
    ROOT / "dataset" / "external" / "cv-lab-canon" / "all",
    ROOT / "dataset" / "external" / "iit-yolov8-tl-canon" / "all",
]
OUT = ROOT / "dataset"
NAMES = ["ArduinoUNO", "Breadboard", "LED", "Resistor", "JumperWire", "Button"]


def main() -> None:
    pairs: list[tuple[Path, Path]] = []
    for source in SOURCES:
        images = source / "images"
        labels = source / "labels"
        if not images.is_dir():
            print(f"skip missing {images}")
            continue
        for image in images.iterdir():
            label = labels / f"{image.stem}.txt"
            if label.is_file() and label.read_text(encoding="utf-8").strip():
                pairs.append((image, label))
    if not pairs:
        raise SystemExit("no labeled images")
    random.seed(42)
    random.shuffle(pairs)
    val_n = max(1, int(round(len(pairs) * 0.15)))
    splits = {"val": pairs[:val_n], "train": pairs[val_n:]}
    for split, items in splits.items():
        img_dir = OUT / "images" / split
        lbl_dir = OUT / "labels" / split
        if img_dir.exists():
            shutil.rmtree(img_dir)
        if lbl_dir.exists():
            shutil.rmtree(lbl_dir)
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        counts: Counter[int] = Counter()
        for image, label in items:
            shutil.copy2(image, img_dir / image.name)
            text = label.read_text(encoding="utf-8")
            (lbl_dir / label.name).write_text(text, encoding="utf-8")
            for line in text.splitlines():
                if line.strip():
                    counts[int(float(line.split()[0]))] += 1
        summary = ", ".join(f"{NAMES[k]}={counts[k]}" for k in range(6))
        print(f"{split}: images={len(items)} {summary}")


if __name__ == "__main__":
    main()
