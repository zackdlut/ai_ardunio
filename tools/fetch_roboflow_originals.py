"""Download original public Roboflow images and write Phase 2 YOLO labels.

Reads ROBOFLOW_API_KEY from the environment. Does not store the key.
"""

from __future__ import annotations

import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import yaml
from roboflow import Roboflow

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dataset" / "external" / "originals"

DATASETS = [
    {
        "workspace": "circuits-project",
        "project": "resistor-detection-5azes",
        "prefix": "circuits_",
        "license": "CC BY 4.0",
        "url": "https://universe.roboflow.com/circuits-project/resistor-detection-5azes",
    },
    {
        "workspace": "cv-laboratory",
        "project": "circuit-components-detection-7gniu",
        "prefix": "cvlab_",
        "license": "CC BY 4.0",
        "url": "https://universe.roboflow.com/cv-laboratory/circuit-components-detection-7gniu",
    },
    {
        "workspace": "iit-hyderabad-zvrsg",
        "project": "yolov8_tl",
        "prefix": "iit_",
        "license": "CC BY 4.0",
        "url": "https://universe.roboflow.com/iit-hyderabad-zvrsg/yolov8_tl",
    },
]

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


def our_split(name: str | None) -> str:
    if name in {"valid", "val", "test"}:
        return "val"
    return "train"


def yolo_line(box: dict, width: float, height: float) -> str | None:
    mapped = canon_name(str(box.get("label", "")))
    if mapped is None or width <= 0 or height <= 0:
        return None
    bw = float(box["width"])
    bh = float(box["height"])
    if bw < 2 or bh < 2:
        return None
    cx = min(1.0, max(0.0, float(box["x"]) / width))
    cy = min(1.0, max(0.0, float(box["y"]) / height))
    nw = min(1.0, max(0.0, bw / width))
    nh = min(1.0, max(0.0, bh / height))
    if nw <= 0 or nh <= 0:
        return None
    return f"{INDEX[mapped]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def search_all(project) -> list[dict]:
    records: list[dict] = []
    offset = 0
    limit = 100
    while True:
        batch = project.search(
            offset=offset,
            limit=limit,
            fields=["id", "name", "split", "width", "height"],
        )
        if not batch:
            break
        records.extend(batch)
        if len(batch) < limit:
            break
        offset += limit
    return records


def download_image(url: str, dest: Path) -> None:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    dest.write_bytes(response.content)


def fetch_record(spec: dict, record: dict, key: str) -> tuple[str, list[str]] | None:
    image_id = record["id"]
    split = our_split(record.get("split"))
    stem = f"{spec['prefix']}{image_id}"
    image_path = OUT / "images" / split / f"{stem}.jpg"
    label_path = OUT / "labels" / split / f"{stem}.txt"
    if image_path.is_file() and label_path.is_file() and image_path.stat().st_size > 0:
        lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return (split, lines) if lines else None

    url = (
        "https://api.roboflow.com/"
        f"{spec['workspace']}/{spec['project']}/images/{image_id}?api_key={key}"
    )
    payload = requests.get(url, timeout=60).json()
    node = payload.get("image", payload)
    annotation = node.get("annotation") or {}
    width = float(annotation.get("width") or record.get("width") or 0)
    height = float(annotation.get("height") or record.get("height") or 0)
    lines = [
        line
        for box in annotation.get("boxes") or []
        if (line := yolo_line(box, width, height))
    ]
    image_url = (node.get("urls") or {}).get("original")
    if not lines or not image_url:
        return None
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    download_image(image_url, image_path)
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return split, lines


def main() -> None:
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        raise SystemExit("ROBOFLOW_API_KEY is not set")
    rf = Roboflow(api_key=key)
    counts: Counter[str] = Counter()
    box_counts: Counter[int] = Counter()
    sources: list[str] = [
        "# Public dataset sources",
        "",
        "Downloaded as original images on 2026-09-30. Generated Roboflow versions were not used, because those exports add augmented copies.",
        "",
    ]

    for spec in DATASETS:
        project = rf.workspace(spec["workspace"]).project(spec["project"])
        records = search_all(project)
        print(f"{spec['project']}: {len(records)} images", flush=True)
        saved = 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(fetch_record, spec, record, key) for record in records]
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                if result is None:
                    continue
                split, lines = result
                saved += 1
                counts[split] += 1
                for line in lines:
                    box_counts[int(line.split()[0])] += 1
                if index % 50 == 0:
                    print(f"  {spec['prefix']}{index}/{len(records)} saved={saved}", flush=True)
        print(f"  saved {saved}", flush=True)
        sources.append(
            f"- {spec['project']} ({spec['license']}): {spec['url']} — {len(records)} listed, {saved} kept"
        )

    summary = ", ".join(f"{CANON[i]}={box_counts[i]}" for i in range(6))
    print(f"images train={counts['train']} val={counts['val']} boxes {summary}", flush=True)
    sources.extend(["", "Box counts after class mapping:", "", summary, ""])
    source_path = ROOT / "dataset" / "external" / "SOURCES.md"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("\n".join(sources), encoding="utf-8")

    data_yaml = {
        "path": str(OUT).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": 6,
        "names": CANON,
    }
    yaml_path = ROOT / "dataset" / "components.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    print(f"wrote {yaml_path}", flush=True)


if __name__ == "__main__":
    main()
