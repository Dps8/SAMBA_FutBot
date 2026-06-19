from __future__ import annotations

from collections.abc import Iterable

from .play_state import ROBOT_CLASSES
from .types import Detection


def filter_tracking_artifacts(
    detections: Iterable[Detection],
    *,
    robot_fallback_min_area: float = 0.0,
    robot_fallback_max_area: float = float("inf"),
    robot_fallback_max_extent: float = 1.0,
    robot_fallback_max_aspect_ratio: float = float("inf"),
) -> tuple[list[Detection], dict]:
    """Remove geometric artifacts produced only by the color robot fallback."""
    if robot_fallback_min_area < 0:
        raise ValueError("robot_fallback_min_area must be non-negative")
    if robot_fallback_max_area <= 0:
        raise ValueError("robot_fallback_max_area must be positive")
    if not 0 < robot_fallback_max_extent <= 1:
        raise ValueError("robot_fallback_max_extent must be in (0, 1]")
    if robot_fallback_max_aspect_ratio < 1:
        raise ValueError("robot_fallback_max_aspect_ratio must be at least 1")

    kept: list[Detection] = []
    removed_by_reason = {
        "fallback_area": 0,
        "fallback_max_area": 0,
        "fallback_extent": 0,
        "fallback_aspect": 0,
    }
    input_count = 0
    for detection in detections:
        input_count += 1
        if detection.class_name not in ROBOT_CLASSES or not _is_color_fallback(detection):
            kept.append(detection)
            continue
        area = float(detection.area or 0.0)
        extent = float(detection.extra.get("extent", 0.0) or 0.0)
        if area < robot_fallback_min_area:
            removed_by_reason["fallback_area"] += 1
            continue
        if area > robot_fallback_max_area:
            removed_by_reason["fallback_max_area"] += 1
            continue
        if extent > robot_fallback_max_extent:
            removed_by_reason["fallback_extent"] += 1
            continue
        if _box_aspect_ratio(detection) > robot_fallback_max_aspect_ratio:
            removed_by_reason["fallback_aspect"] += 1
            continue
        kept.append(detection)

    removed = input_count - len(kept)
    return kept, {
        "input_detections": input_count,
        "detections": len(kept),
        "removed": removed,
        "removed_by_reason": removed_by_reason,
        "robot_fallback_min_area": robot_fallback_min_area,
        "robot_fallback_max_area": robot_fallback_max_area,
        "robot_fallback_max_extent": robot_fallback_max_extent,
        "robot_fallback_max_aspect_ratio": robot_fallback_max_aspect_ratio,
    }


def is_tracking_artifact(
    detection: Detection,
    *,
    robot_fallback_min_area: float,
    robot_fallback_max_area: float = float("inf"),
    robot_fallback_max_extent: float,
    robot_fallback_max_aspect_ratio: float = float("inf"),
) -> bool:
    if detection.class_name not in ROBOT_CLASSES or not _is_color_fallback(detection):
        return False
    area = float(detection.area or 0.0)
    extent = float(detection.extra.get("extent", 0.0) or 0.0)
    return (
        area < robot_fallback_min_area
        or area > robot_fallback_max_area
        or extent > robot_fallback_max_extent
        or _box_aspect_ratio(detection) > robot_fallback_max_aspect_ratio
    )


def _is_color_fallback(detection: Detection) -> bool:
    return detection.extra.get("source") == "color_robots" or detection.prompt == (
        "hsv_dark_robot_fallback"
    )


def _box_aspect_ratio(detection: Detection) -> float:
    x1, y1, x2, y2 = detection.box
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    if width <= 0 or height <= 0:
        return float("inf")
    return max(width / height, height / width)
