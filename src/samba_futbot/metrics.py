from __future__ import annotations

from collections import defaultdict
import math
from statistics import mean, pstdev
from typing import Iterable

from .events import estimate_possession
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
    possession = _possession_summary(
        detections_list,
        fps=fps,
        possession_radius_px=possession_radius_px,
    )

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
        "possession": possession,
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


def _possession_summary(
    detections: list[Detection],
    *,
    fps: float | None,
    possession_radius_px: float,
) -> dict:
    possession = estimate_possession(detections, possession_radius_px=possession_radius_px)
    by_team: dict[str, int] = defaultdict(int)
    by_track: dict[str, int] = defaultdict(int)
    possessed_frames = 0
    for owner in possession.values():
        if owner is None:
            continue
        possessed_frames += 1
        team = owner.team or "unknown"
        by_team[team] += 1
        by_track[str(owner.track_id or "unknown")] += 1

    total_frames = len(possession)
    streaks = _possession_streaks(possession, fps=fps)
    longest_streak = max(streaks, key=lambda item: item["frames"], default=None)
    dominance = _possession_dominance(by_team, possessed_frames)
    return {
        "frames_with_possession": possessed_frames,
        "frames_without_possession": max(0, total_frames - possessed_frames),
        "coverage_ratio": possessed_frames / total_frames if total_frames else 0.0,
        "seconds": possessed_frames / fps if fps and fps > 0 else None,
        "by_team": {
            team: {
                "frames": frames,
                "seconds": frames / fps if fps and fps > 0 else None,
                "ratio": frames / possessed_frames if possessed_frames else 0.0,
            }
            for team, frames in sorted(by_team.items())
        },
        "by_track": {
            track_id: {
                "frames": frames,
                "seconds": frames / fps if fps and fps > 0 else None,
                "ratio": frames / possessed_frames if possessed_frames else 0.0,
            }
            for track_id, frames in sorted(by_track.items())
        },
        "dominance": dominance,
        "longest_streak": longest_streak,
        "streaks": streaks[:20],
    }


def _possession_streaks(
    possession: dict[int, Detection | None],
    *,
    fps: float | None,
) -> list[dict]:
    streaks: list[dict] = []
    current_owner: Detection | None = None
    start_frame: int | None = None
    previous_frame: int | None = None
    for frame_index in sorted(possession):
        owner = possession[frame_index]
        same_owner = (
            owner is not None
            and current_owner is not None
            and owner.track_id == current_owner.track_id
            and frame_index == (previous_frame or frame_index - 1) + 1
        )
        if owner is None:
            _append_possession_streak(
                streaks,
                current_owner,
                start_frame,
                previous_frame,
                fps=fps,
            )
            current_owner = None
            start_frame = None
        elif not same_owner:
            _append_possession_streak(
                streaks,
                current_owner,
                start_frame,
                previous_frame,
                fps=fps,
            )
            current_owner = owner
            start_frame = frame_index
        previous_frame = frame_index
    _append_possession_streak(
        streaks,
        current_owner,
        start_frame,
        previous_frame,
        fps=fps,
    )
    return sorted(streaks, key=lambda item: item["frames"], reverse=True)


def _possession_dominance(by_team: dict[str, int], possessed_frames: int) -> dict:
    if possessed_frames <= 0 or not by_team:
        return {
            "team": "none",
            "frames": 0,
            "margin_frames": 0,
            "ratio": 0.0,
            "margin_ratio": 0.0,
        }
    ranked = sorted(by_team.items(), key=lambda item: (-item[1], item[0]))
    team, frames = ranked[0]
    runner_up_frames = ranked[1][1] if len(ranked) > 1 else 0
    margin = frames - runner_up_frames
    return {
        "team": team,
        "frames": frames,
        "margin_frames": margin,
        "ratio": frames / possessed_frames,
        "margin_ratio": margin / possessed_frames,
    }


def _append_possession_streak(
    streaks: list[dict],
    owner: Detection | None,
    start_frame: int | None,
    end_frame: int | None,
    *,
    fps: float | None,
) -> None:
    if owner is None or start_frame is None or end_frame is None or end_frame < start_frame:
        return
    frames = end_frame - start_frame + 1
    streaks.append(
        {
            "track_id": owner.track_id,
            "team": owner.team or "unknown",
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frames": frames,
            "seconds": frames / fps if fps and fps > 0 else None,
        }
    )
