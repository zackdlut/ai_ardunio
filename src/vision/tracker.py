from __future__ import annotations

from vision.detector import Detection


def _iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


class SimpleTracker:
    """Greedy IoU matching to stabilize track_id across frames."""

    def __init__(self, iou_threshold: float = 0.3) -> None:
        self._iou_threshold = iou_threshold
        self._next_id = 1
        self._tracks: dict[int, list[int]] = {}

    def update(self, detections: list[Detection]) -> list[Detection]:
        updated: list[Detection] = []
        used_tracks: set[int] = set()

        for det in detections:
            best_id = None
            best_iou = 0.0
            for tid, bbox in self._tracks.items():
                if tid in used_tracks:
                    continue
                score = _iou(det.bbox, bbox)
                if score > best_iou:
                    best_iou = score
                    best_id = tid

            if best_id is not None and best_iou >= self._iou_threshold:
                track_id = best_id
            else:
                track_id = self._next_id
                self._next_id += 1

            used_tracks.add(track_id)
            self._tracks[track_id] = det.bbox
            updated.append(
                Detection(
                    type=det.type,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    track_id=track_id,
                )
            )

        # Drop stale tracks not matched this frame
        self._tracks = {d.track_id: d.bbox for d in updated if d.track_id is not None}
        return updated
