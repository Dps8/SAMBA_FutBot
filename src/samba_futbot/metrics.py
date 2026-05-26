from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev
from typing import Iterable

from .types import Detection


def summarize_tracks(detections: Iterable[Detection]) -> dict:
    detections_list = list(detections)
    frames = sorted({det.frame_index for det in detections_list})
    by_class: dict[str, list[Detection]] = defaultdict(list)
    by_track: dict[int, list[Detection]] = defaultdict(list)

    for det in detections_list:
        by_class[det.class_name].append(det)
        if det.track_id is not None:
            by_track[det.track_id].append(det)

    class_summary = {}
    for class_name, class_dets in by_class.items():
        tracks = {det.track_id for det in class_dets if det.track_id is not None}
        class_summary[class_name] = {
            "detections": len(class_dets),
            "unique_tracks": len(tracks),
            "mean_score": mean([det.score for det in class_dets]) if class_dets else 0.0,
        }

    lengths = [len(items) for items in by_track.values()]
    gaps = [_count_gaps(items) for items in by_track.values()]
    areas = [det.area for det in detections_list if det.area is not None and det.area > 0]

    return {
        "frames_observed": len(frames),
        "first_frame": frames[0] if frames else None,
        "last_frame": frames[-1] if frames else None,
        "detections": len(detections_list),
        "tracks": len(by_track),
        "mean_track_length": mean(lengths) if lengths else 0.0,
        "track_fragmentation_gaps": sum(gaps),
        "classes": class_summary,
        "mask_area_mean": mean(areas) if areas else None,
        "mask_area_std": pstdev(areas) if len(areas) > 1 else 0.0,
    }


def _count_gaps(track_dets: list[Detection]) -> int:
    frames = sorted(det.frame_index for det in track_dets)
    return sum(max(0, b - a - 1) for a, b in zip(frames, frames[1:]))
