from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

from .types import Detection


FIELD_CLASSES = {"field", "soccer_field", "green_field"}
ROBOT_CLASSES = {"robot", "robots", "robot_allied", "robot_rival"}
BALL_CLASSES = {"ball", "balon", "soccer_ball"}


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def group_by_frame(detections: Iterable[Detection]) -> dict[int, list[Detection]]:
    frames: dict[int, list[Detection]] = defaultdict(list)
    for detection in detections:
        frames[detection.frame_index].append(detection)
    return frames


def point_in_box(
    point: tuple[float, float],
    box: tuple[float, float, float, float],
    *,
    margin_px: float = 0,
) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return (x1 - margin_px) <= x <= (x2 + margin_px) and (y1 - margin_px) <= y <= (
        y2 + margin_px
    )


def ball_in_play(
    ball: Detection,
    frame_detections: Iterable[Detection],
    *,
    possession_radius_px: float = 90.0,
    field_margin_px: float = 8.0,
) -> bool:
    frame_dets = list(frame_detections)
    fields = [det for det in frame_dets if det.class_name in FIELD_CLASSES]
    if any(point_in_box(ball.centroid, field.box, margin_px=field_margin_px) for field in fields):
        return True

    robots = [det for det in frame_dets if det.class_name in ROBOT_CLASSES]
    return any(distance(ball.centroid, robot.centroid) <= possession_radius_px for robot in robots)


def in_play_balls(
    detections: Iterable[Detection],
    *,
    possession_radius_px: float = 90.0,
    field_margin_px: float = 8.0,
) -> list[Detection]:
    selected: list[Detection] = []
    for _, frame_dets in group_by_frame(detections).items():
        balls = [det for det in frame_dets if det.class_name in BALL_CLASSES]
        for ball in balls:
            if ball_in_play(
                ball,
                frame_dets,
                possession_radius_px=possession_radius_px,
                field_margin_px=field_margin_px,
            ):
                selected.append(ball)
    return sorted(selected, key=lambda det: (det.frame_index, det.track_id or -1, det.score))
