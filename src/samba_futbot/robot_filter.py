from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from math import hypot
from typing import Iterable

from .play_state import BALL_CLASSES, ROBOT_CLASSES
from .tracking import iou
from .types import Detection


def filter_robot_detections(
    detections: Iterable[Detection],
    *,
    frame_width: int | None = None,
    frame_height: int | None = None,
    max_per_frame: int | None = None,
    min_area: float = 0.0,
    max_area_ratio: float | None = None,
    containment_threshold: float = 0.82,
    iou_threshold: float = 0.55,
    min_center_distance_px: float = 0.0,
    max_center_distance_ratio: float = 0.0,
    min_center_y_ratio: float | None = None,
    max_center_y_ratio: float | None = None,
    protect_near_ball_px: float = 0.0,
) -> list[Detection]:
    grouped: dict[int, list[tuple[int, Detection]]] = defaultdict(list)
    passthrough: list[tuple[int, Detection]] = []
    balls_by_frame: dict[int, list[Detection]] = defaultdict(list)
    for index, det in enumerate(detections):
        if det.class_name in ROBOT_CLASSES:
            grouped[det.frame_index].append((index, det))
        else:
            passthrough.append((index, det))
            if det.class_name in BALL_CLASSES:
                balls_by_frame[det.frame_index].append(det)

    max_area = _max_area_from_ratio(
        frame_width=frame_width,
        frame_height=frame_height,
        max_area_ratio=max_area_ratio,
    )
    kept = list(passthrough)
    for frame_index in sorted(grouped):
        selected: list[tuple[int, Detection]] = []
        ball = _best_ball(balls_by_frame.get(frame_index, []))
        candidates = sorted(
            grouped[frame_index],
            key=lambda item: (
                not _near_ball(item[1], ball, protect_near_ball_px),
                _is_color_fragment(item[1]),
                -float(item[1].score),
                -_box_area(item[1].box),
                item[0],
            ),
        )
        for index, det in candidates:
            area = _detection_area(det)
            if area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue
            if not _inside_vertical_gate(
                det,
                frame_height=frame_height,
                min_center_y_ratio=min_center_y_ratio,
                max_center_y_ratio=max_center_y_ratio,
            ):
                continue
            if _conflicts_with_selected(
                det,
                [existing for _, existing in selected],
                containment_threshold=containment_threshold,
                iou_threshold=iou_threshold,
                min_center_distance_px=min_center_distance_px,
                max_center_distance_ratio=max_center_distance_ratio,
            ):
                continue
            selected.append((index, _mark_robot_filter(det)))
            if max_per_frame is not None and max_per_frame > 0 and len(selected) >= max_per_frame:
                break
        kept.extend(selected)
    return [det for _, det in sorted(kept, key=lambda item: item[0])]


def _best_ball(detections: list[Detection]) -> Detection | None:
    return max(detections, key=lambda det: det.score, default=None)


def _near_ball(det: Detection, ball: Detection | None, threshold_px: float) -> bool:
    if ball is None or threshold_px <= 0:
        return False
    return (
        hypot(det.centroid[0] - ball.centroid[0], det.centroid[1] - ball.centroid[1])
        <= threshold_px
    )


def _max_area_from_ratio(
    *,
    frame_width: int | None,
    frame_height: int | None,
    max_area_ratio: float | None,
) -> float | None:
    if not max_area_ratio or max_area_ratio <= 0 or not frame_width or not frame_height:
        return None
    return float(frame_width) * float(frame_height) * float(max_area_ratio)


def _detection_area(det: Detection) -> float:
    if isinstance(det.area, int | float) and float(det.area) > 0:
        return float(det.area)
    return _box_area(det.box)


def _box_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def _conflicts_with_selected(
    candidate: Detection,
    selected: list[Detection],
    *,
    containment_threshold: float,
    iou_threshold: float,
    min_center_distance_px: float,
    max_center_distance_ratio: float,
) -> bool:
    for existing in selected:
        if iou(candidate.box, existing.box) >= iou_threshold:
            return True
        if _containment_ratio(candidate.box, existing.box) >= containment_threshold:
            return True
        if (
            min_center_distance_px > 0
            and hypot(
                candidate.centroid[0] - existing.centroid[0],
                candidate.centroid[1] - existing.centroid[1],
            )
            < min_center_distance_px
        ):
            return True
        if max_center_distance_ratio > 0:
            center_distance = hypot(
                candidate.centroid[0] - existing.centroid[0],
                candidate.centroid[1] - existing.centroid[1],
            )
            adaptive_distance = max_center_distance_ratio * max(
                _box_diagonal(candidate.box),
                _box_diagonal(existing.box),
            )
            if center_distance < adaptive_distance:
                return True
    return False


def _inside_vertical_gate(
    detection: Detection,
    *,
    frame_height: int | None,
    min_center_y_ratio: float | None,
    max_center_y_ratio: float | None,
) -> bool:
    if not frame_height or frame_height <= 0:
        return True
    ratio = detection.centroid[1] / frame_height
    if min_center_y_ratio is not None and ratio < min_center_y_ratio:
        return False
    return not (max_center_y_ratio is not None and ratio > max_center_y_ratio)


def _box_diagonal(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return hypot(max(0.0, x2 - x1), max(0.0, y2 - y1))


def _containment_ratio(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = _box_area((ix1, iy1, ix2, iy2))
    smaller = min(_box_area(a), _box_area(b))
    if smaller <= 0:
        return 0.0
    return intersection / smaller


def _mark_robot_filter(det: Detection) -> Detection:
    extra = dict(det.extra)
    extra["robot_filter"] = "kept"
    return replace(det, extra=extra)


def _is_color_fragment(detection: Detection) -> bool:
    return detection.extra.get("source") == "color_robots" or detection.prompt == (
        "hsv_dark_robot_fallback"
    )
