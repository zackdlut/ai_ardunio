from __future__ import annotations

import math
from typing import Any

from vision.detector import Detection


def _center(bbox: list[int]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class SceneGraphBuilder:
    """MVP: geometric heuristics only."""

    WIRE_NEAR_PX = 120.0

    def build(self, detections: list[Detection]) -> dict[str, Any]:
        by_type: dict[str, list[Detection]] = {}
        for d in detections:
            by_type.setdefault(d.type, []).append(d)

        nodes: dict[str, Any] = {}
        for d in detections:
            key = f"{d.type.lower()}_{d.track_id or id(d)}"
            nodes[key] = {
                "type": d.type,
                "bbox": d.bbox,
                "confidence": d.confidence,
                "connected_to": None,
            }

        wires = by_type.get("JumperWire", [])
        leds = by_type.get("LED", [])
        resistors = by_type.get("Resistor", [])
        uno = by_type.get("ArduinoUNO", [])

        for wire in wires:
            wc = _center(wire.bbox)
            candidates: list[tuple[str, float]] = []
            for led in leds:
                candidates.append((f"led_{led.track_id}", _dist(wc, _center(led.bbox))))
            for res in resistors:
                candidates.append(
                    (f"resistor_{res.track_id}", _dist(wc, _center(res.bbox)))
                )
            if uno:
                candidates.append(("uno_d13_zone", _dist(wc, _center(uno[0].bbox))))
            if candidates:
                candidates.sort(key=lambda x: x[1])
                name, dist = candidates[0]
                if dist < self.WIRE_NEAR_PX:
                    wkey = f"jumperwire_{wire.track_id or id(wire)}"
                    if wkey in nodes:
                        nodes[wkey]["connected_to"] = name

        return {
            "nodes": nodes,
            "summary": {t: len(v) for t, v in by_type.items()},
        }
