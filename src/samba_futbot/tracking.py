from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .types import Box, Detection


def box_area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = box_area((ix1, iy1, ix2, iy2))
    if intersection <= 0:
        return 0.0
    union = box_area(a) + box_area(b) - intersection
    return intersection / union if union > 0 else 0.0


@dataclass(slots=True)
class _Track:
    track_id: int
    class_name: str
    box: Box
    last_frame: int
    misses: int = 0
    team: str | None = None


class IouTracker:
    """Small dependency-free tracker for repairing or replacing missing IDs."""

    def __init__(self, iou_threshold: float = 0.25, max_age: int = 12) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    def update(self, detections: list[Detection], frame_index: int) -> list[Detection]:
        active_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if frame_index - track.last_frame <= self.max_age
        ]
        candidate_pairs: list[tuple[float, int, int]] = []
        for det_idx, det in enumerate(detections):
            for track_id in active_ids:
                track = self._tracks[track_id]
                if track.class_name != det.class_name:
                    continue
                candidate_pairs.append((iou(det.box, track.box), det_idx, track_id))

        candidate_pairs.sort(reverse=True, key=lambda item: item[0])
        matched_dets: set[int] = set()
        matched_tracks: set[int] = set()

        for overlap, det_idx, track_id in candidate_pairs:
            if overlap < self.iou_threshold:
                break
            if det_idx in matched_dets or track_id in matched_tracks:
                continue
            self._assign(detections[det_idx], track_id, frame_index)
            matched_dets.add(det_idx)
            matched_tracks.add(track_id)

        for det_idx, det in enumerate(detections):
            if det_idx not in matched_dets:
                self._create(det, frame_index)

        for track_id, track in list(self._tracks.items()):
            if frame_index - track.last_frame > self.max_age:
                del self._tracks[track_id]
        return detections

    def _assign(self, detection: Detection, track_id: int, frame_index: int) -> None:
        track = self._tracks[track_id]
        track.box = detection.box
        track.last_frame = frame_index
        track.misses = 0
        track.team = detection.team or track.team
        detection.track_id = track_id
        if not detection.team and track.team:
            detection.team = track.team

    def _create(self, detection: Detection, frame_index: int) -> None:
        track_id = self._next_id
        self._next_id += 1
        self._tracks[track_id] = _Track(
            track_id=track_id,
            class_name=detection.class_name,
            box=detection.box,
            last_frame=frame_index,
            team=detection.team,
        )
        detection.track_id = track_id


def track_detections(
    detections: Iterable[Detection], iou_threshold: float = 0.25, max_age: int = 12
) -> list[Detection]:
    tracker = IouTracker(iou_threshold=iou_threshold, max_age=max_age)
    by_frame: dict[int, list[Detection]] = defaultdict(list)
    for detection in detections:
        by_frame[detection.frame_index].append(detection)

    tracked: list[Detection] = []
    for frame_index in sorted(by_frame):
        tracked.extend(tracker.update(by_frame[frame_index], frame_index))
    return tracked
