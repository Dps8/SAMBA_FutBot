from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from .play_state import ROBOT_CLASSES
from .types import Detection
from .video import require_cv2


DEFAULT_TEAM_PALETTE = {
    "blue": (55, 115, 220),
    "yellow": (230, 210, 60),
}


def dominant_rgb(
    frame_rgb: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    min_saturation: int = 0,
    min_value: int = 0,
    min_pixels: int = 1,
) -> tuple[int, int, int]:
    pixels = frame_rgb[mask > 0] if mask is not None else frame_rgb.reshape(-1, 3)
    pixels = _filter_color_pixels(pixels, min_saturation=min_saturation, min_value=min_value)
    if len(pixels) < min_pixels:
        return (0, 0, 0)
    median = np.median(pixels, axis=0)
    return tuple(int(v) for v in median[:3])


def nearest_palette_team(
    rgb: tuple[int, int, int], palette: dict[str, tuple[int, int, int]]
) -> tuple[str, float]:
    color = np.asarray(rgb, dtype=np.float64)
    best_team = "unknown"
    best_distance = float("inf")
    for team, value in palette.items():
        distance = float(np.linalg.norm(color - np.asarray(value, dtype=np.float64)))
        if distance < best_distance:
            best_team = team
            best_distance = distance
    return best_team, best_distance


def assign_robot_teams_from_video(
    video_path: str | Path,
    detections: Iterable[Detection],
    *,
    palette: dict[str, tuple[int, int, int]] | None = None,
    max_color_distance: float = 170.0,
    min_saturation: int = 45,
    min_value: int = 40,
    min_pixels: int = 8,
) -> list[Detection]:
    detections_list = list(detections)
    robot_dets = [
        det
        for det in detections_list
        if det.class_name in ROBOT_CLASSES and det.track_id is not None
    ]
    if not robot_dets:
        return detections_list

    resolved_palette = palette or DEFAULT_TEAM_PALETTE
    observations = _team_observations(
        video_path,
        robot_dets,
        resolved_palette,
        max_color_distance=max_color_distance,
        min_saturation=min_saturation,
        min_value=min_value,
        min_pixels=min_pixels,
    )
    team_by_track = _team_by_track(observations, max_color_distance=max_color_distance)
    for det in detections_list:
        if det.class_name in ROBOT_CLASSES and det.track_id in team_by_track:
            det.team = team_by_track[det.track_id]
    return detections_list


def _team_observations(
    video_path: str | Path,
    robot_dets: list[Detection],
    palette: dict[str, tuple[int, int, int]],
    *,
    max_color_distance: float,
    min_saturation: int,
    min_value: int,
    min_pixels: int,
) -> dict[int, list[tuple[str, float]]]:
    cv2 = require_cv2()
    by_frame: dict[int, list[Detection]] = {}
    for det in robot_dets:
        by_frame.setdefault(det.frame_index, []).append(det)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    observations: dict[int, list[tuple[str, float]]] = {}
    frame_index = 0
    wanted = set(by_frame)
    while wanted:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_index not in wanted:
            frame_index += 1
            continue
        frame_rgb = frame_bgr[:, :, ::-1]
        for det in by_frame.get(frame_index, []):
            crop = _crop_detection(frame_rgb, det)
            if crop.size == 0 or det.track_id is None:
                continue
            team, distance = palette_team_vote(
                crop,
                palette,
                max_color_distance=max_color_distance,
                min_saturation=min_saturation,
                min_value=min_value,
                min_pixels=min_pixels,
            )
            if team != "unknown":
                observations.setdefault(det.track_id, []).append((team, distance))
        wanted.remove(frame_index)
        frame_index += 1

    cap.release()
    return observations


def palette_team_vote(
    crop_rgb: np.ndarray,
    palette: dict[str, tuple[int, int, int]],
    *,
    max_color_distance: float,
    min_saturation: int = 45,
    min_value: int = 40,
    min_pixels: int = 8,
) -> tuple[str, float]:
    pixels = _filter_color_pixels(
        crop_rgb.reshape(-1, 3),
        min_saturation=min_saturation,
        min_value=min_value,
    )
    if len(pixels) < min_pixels:
        return "unknown", float("inf")

    palette_items = list(palette.items())
    distances = np.stack(
        [
            np.linalg.norm(pixels.astype(np.float64) - np.asarray(value, dtype=np.float64), axis=1)
            for _, value in palette_items
        ],
        axis=1,
    )
    best_indices = np.argmin(distances, axis=1)
    best_distances = distances[np.arange(len(pixels)), best_indices]
    valid = best_distances <= max_color_distance
    if int(np.count_nonzero(valid)) < min_pixels:
        return "unknown", float("inf")

    votes = Counter(palette_items[index][0] for index in best_indices[valid])
    team, _ = votes.most_common(1)[0]
    team_index = next(index for index, item in enumerate(palette_items) if item[0] == team)
    team_distances = distances[valid, team_index]
    return team, float(np.mean(team_distances))


def _team_by_track(
    observations: dict[int, list[tuple[str, float]]],
    *,
    max_color_distance: float,
) -> dict[int, str]:
    result: dict[int, str] = {}
    for track_id, items in observations.items():
        filtered = [team for team, distance in items if distance <= max_color_distance]
        if not filtered:
            continue
        result[track_id] = Counter(filtered).most_common(1)[0][0]
    return result


def _crop_detection(frame_rgb: np.ndarray, det: Detection) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in det.box]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=frame_rgb.dtype)
    crop = frame_rgb[y1:y2, x1:x2]
    # Use the central area to reduce field/background influence around the box.
    h, w = crop.shape[:2]
    mx = max(0, int(w * 0.18))
    my = max(0, int(h * 0.18))
    return crop[my : h - my or h, mx : w - mx or w]


def _filter_color_pixels(
    pixels_rgb: np.ndarray,
    *,
    min_saturation: int,
    min_value: int,
) -> np.ndarray:
    if pixels_rgb.size == 0:
        return pixels_rgb.reshape(0, 3)
    if min_saturation <= 0 and min_value <= 0:
        return pixels_rgb.reshape(-1, 3)
    cv2 = require_cv2()
    pixels = pixels_rgb.reshape(-1, 1, 3).astype(np.uint8)
    hsv = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV).reshape(-1, 3)
    keep = (hsv[:, 1] >= min_saturation) & (hsv[:, 2] >= min_value)
    return pixels_rgb.reshape(-1, 3)[keep]
