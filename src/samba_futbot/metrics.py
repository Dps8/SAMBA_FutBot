from __future__ import annotations

from collections import defaultdict
import math
from statistics import mean, pstdev
from typing import Iterable

from .play_state import BALL_CLASSES, in_play_balls
from .types import Detection


def summarize_tracks(
    detections: Iterable[Detection],
    *,
    fps: float | None = None,
    possession_radius_px: float = 90.0,
    field_margin_px: float = 8.0,
) -> dict:
    detections_list = list(detections)
    in_play_ball_dets = in_play_balls(
        detections_list,
        possession_radius_px=possession_radius_px,
        field_margin_px=field_margin_px,
    )
    frames = sorted({det.frame_index for det in detections_list})
    by_class: dict[str, list[Detection]] = defaultdict(list)
    by_track: dict[int, list[Detection]] = defaultdict(list)

    for det in detections_list:
        by_class[det.class_name].append(det)
        if det.track_id is not None:
            by_track[det.track_id].append(det)

    class_summary = {}
    observed_span = (frames[-1] - frames[0] + 1) if frames else 0
    for class_name, class_dets in by_class.items():
        tracks = {det.track_id for det in class_dets if det.track_id is not None}
        class_frames = {det.frame_index for det in class_dets}
        class_gaps = [_count_gaps(items) for items in _tracks_for_class(class_dets).values()]
        in_play_frames = {
            det.frame_index
            for det in in_play_ball_dets
            if class_name in BALL_CLASSES and det.class_name == class_name
        }
        class_summary[class_name] = {
            "detections": len(class_dets),
            "unique_tracks": len(tracks),
            "mean_score": mean([det.score for det in class_dets]) if class_dets else 0.0,
            "frames_with_detection": len(class_frames),
            "frame_coverage_ratio": (
                len(class_frames) / observed_span if observed_span else 0.0
            ),
            "track_fragmentation_gaps": sum(class_gaps),
        }
        if class_name in BALL_CLASSES:
            class_summary[class_name]["in_play_detections"] = len(in_play_ball_dets)
            class_summary[class_name]["in_play_frames"] = len(in_play_frames)
            class_summary[class_name]["in_play_coverage_ratio"] = (
                len(in_play_frames) / observed_span if observed_span else 0.0
            )
            class_summary[class_name]["in_play_duration_seconds"] = (
                len(in_play_frames) / fps if fps and fps > 0 else None
            )

    lengths = [len(items) for items in by_track.values()]
    gaps = [_count_gaps(items) for items in by_track.values()]
    areas = [det.area for det in detections_list if det.area is not None and det.area > 0]
    motion = _motion_summary(detections_list, in_play_ball_dets=in_play_ball_dets, fps=fps)

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
        "motion": motion,
    }


def _count_gaps(track_dets: list[Detection]) -> int:
    frames = sorted(det.frame_index for det in track_dets)
    return sum(max(0, b - a - 1) for a, b in zip(frames, frames[1:]))


def _tracks_for_class(detections: Iterable[Detection]) -> dict[int, list[Detection]]:
    tracks: dict[int, list[Detection]] = defaultdict(list)
    for det in detections:
        if det.track_id is not None:
            tracks[det.track_id].append(det)
    return tracks


def _motion_summary(
    detections: list[Detection],
    *,
    in_play_ball_dets: list[Detection],
    fps: float | None = None,
) -> dict:
    candidate_ball_speeds = _class_speeds(
        [det for det in detections if det.class_name in BALL_CLASSES]
    )
    ball_summary = _speed_summary(_class_speeds(in_play_ball_dets))
    ball_summary["trajectory_scope"] = "in_play"
    ball_summary["raw_candidates"] = _speed_summary(candidate_ball_speeds)
    if fps and fps > 0:
        ball_summary["mean_speed_px_second"] = ball_summary["mean_speed_px_frame"] * fps
        ball_summary["max_speed_px_second"] = ball_summary["max_speed_px_frame"] * fps
        ball_summary["raw_candidates"]["mean_speed_px_second"] = (
            ball_summary["raw_candidates"]["mean_speed_px_frame"] * fps
        )
        ball_summary["raw_candidates"]["max_speed_px_second"] = (
            ball_summary["raw_candidates"]["max_speed_px_frame"] * fps
        )
    else:
        ball_summary["mean_speed_px_second"] = None
        ball_summary["max_speed_px_second"] = None
        ball_summary["raw_candidates"]["mean_speed_px_second"] = None
        ball_summary["raw_candidates"]["max_speed_px_second"] = None
    return {"ball": ball_summary}


def _class_speeds(detections: list[Detection]) -> list[float]:
    speeds: list[float] = []
    for track_dets in _tracks_for_class(detections).values():
        ordered = sorted(track_dets, key=lambda det: det.frame_index)
        for prev, current in zip(ordered, ordered[1:]):
            frame_delta = current.frame_index - prev.frame_index
            if frame_delta <= 0:
                continue
            dx = current.centroid[0] - prev.centroid[0]
            dy = current.centroid[1] - prev.centroid[1]
            speeds.append(math.hypot(dx, dy) / frame_delta)
    return speeds


def _speed_summary(speeds: list[float]) -> dict:
    return {
        "samples": len(speeds),
        "mean_speed_px_frame": mean(speeds) if speeds else 0.0,
        "max_speed_px_frame": max(speeds) if speeds else 0.0,
    }
