"""Download the public Roboflow datasets used as Phase 2 seeds.

The API key is read from ROBOFLOW_API_KEY and is never written to disk.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from roboflow import Roboflow

ROOT = Path(__file__).resolve().parents[1]
DATASETS = [
    ("circuits-project", "resistor-detection-5azes", "circuits-resistor"),
    ("cv-laboratory", "circuit-components-detection-7gniu", "cv-lab"),
    ("iit-hyderabad-zvrsg", "yolov8_tl", "iit-yolov8-tl"),
]


def version_number(version) -> int:
    for attr in ("version", "id", "name"):
        value = getattr(version, attr, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    text = str(version)
    digits = "".join(ch if ch.isdigit() else " " for ch in text).split()
    return int(digits[-1]) if digits else 0


def main() -> None:
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        raise SystemExit("ROBOFLOW_API_KEY is not set")
    list_only = "--list" in sys.argv
    rf = Roboflow(api_key=key)
    for workspace, project_slug, folder in DATASETS:
        project = rf.workspace(workspace).project(project_slug)
        versions = list(project.versions())
        print(f"\n{workspace}/{project_slug}: {len(versions)} versions")
        for version in versions:
            number = version_number(version)
            print(f"  version={number} type={type(version).__name__} repr={version!r}")
        if list_only or not versions:
            continue
        chosen = max(versions, key=version_number)
        number = version_number(chosen)
        dest = ROOT / "dataset" / "external" / folder
        dest.mkdir(parents=True, exist_ok=True)
        print(f"downloading version {number} -> {dest}")
        project.version(number).download("yolov8", location=str(dest), overwrite=True)


if __name__ == "__main__":
    main()
