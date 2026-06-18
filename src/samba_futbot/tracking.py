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
    detections: Iterable[Detection],
    iou_threshold: float = 0.25,
    max_age: int = 12,
    *,
    backend: str = "iou",
    frame_rate: int = 30,
    track_activation_threshold: float = 0.05,
    minimum_matching_threshold: float = 0.8,
) -> list[Detection]:
    detections = list(detections)
    if backend == "bytetrack":
        return _track_with_bytetrack(
            detections,
            max_age=max_age,
            frame_rate=frame_rate,
            track_activation_threshold=track_activation_threshold,
            minimum_matching_threshold=minimum_matching_threshold,
        )
    if backend != "iou":
        raise ValueError(f"Unknown tracker backend: {backend}")
    tracker = IouTracker(iou_threshold=iou_threshold, max_age=max_age)
    by_frame: dict[int, list[Detection]] = defaultdict(list)
    for detection in detections:
        by_frame[detection.frame_index].append(detection)

    tracked: list[Detection] = []
    for frame_index in sorted(by_frame):
        tracked.extend(tracker.update(by_frame[frame_index], frame_index))
    return tracked


def _track_with_bytetrack(
    detections: list[Detection],
    *,
    max_age: int,
    frame_rate: int,
    track_activation_threshold: float,
    minimum_matching_threshold: float,
) -> list[Detection]:
    try:
        import numpy as np
        import supervision as sv
    except ImportError as exc:
        raise RuntimeError(
            "ByteTrack requires the optional 'supervision' dependency."
        ) from exc
    if not detections:
        return []

    by_frame: dict[int, list[Detection]] = defaultdict(list)
    classes: set[str] = set()
    for detection in detections:
        by_frame[detection.frame_index].append(detection)
        classes.add(detection.class_name)
    trackers = {
        class_name: sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=max_age,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
            minimum_consecutive_frames=1,
        )
        for class_name in classes
    }
    global_ids: dict[tuple[str, int], int] = {}
    next_global_id = 1
    tracked: list[Detection] = []

    for frame_index in range(min(by_frame), max(by_frame) + 1):
        frame_detections = by_frame.get(frame_index, [])
        by_class: dict[str, list[tuple[int, Detection]]] = defaultdict(list)
        for source_index, detection in enumerate(frame_detections):
            by_class[detection.class_name].append((source_index, detection))

        for class_name, tracker in trackers.items():
            class_items = by_class.get(class_name, [])
            if not class_items:
                tracker.update_with_detections(sv.Detections.empty())
                continue
            source_indices = np.asarray([item[0] for item in class_items], dtype=int)
            sv_detections = sv.Detections(
                xyxy=np.asarray([item[1].box for item in class_items], dtype=float),
                confidence=np.asarray([item[1].score for item in class_items], dtype=float),
                class_id=np.zeros(len(class_items), dtype=int),
                data={"source_index": source_indices},
            )
            updated = tracker.update_with_detections(sv_detections)
            assigned_sources: set[int] = set()
            for source_index, local_track_id in zip(
                updated.data.get("source_index", []),
                updated.tracker_id,
                strict=True,
            ):
                source_index = int(source_index)
                key = (class_name, int(local_track_id))
                if key not in global_ids:
                    global_ids[key] = next_global_id
                    next_global_id += 1
                frame_detections[source_index].track_id = global_ids[key]
                assigned_sources.add(source_index)
            for source_index, _ in class_items:
                if source_index not in assigned_sources:
                    frame_detections[source_index].track_id = next_global_id
                    next_global_id += 1
        tracked.extend(frame_detections)
    return tracked
