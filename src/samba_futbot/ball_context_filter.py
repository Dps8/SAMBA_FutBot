from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Iterable

from .play_state import BALL_CLASSES, FIELD_CLASSES, ROBOT_CLASSES
from .types import Detection


HUMAN_CONTEXT_CLASSES = {
    "person",
    "human",
    "referee",
    "arbitro",
    "hand",
    "mano",
    "arm",
    "brazo",
}


def filter_contextual_ball_candidates(
    detections: Iterable[Detection],
    *,
    reject_robot_overlap_ratio: float = 0.20,
    require_field_or_human_context: bool = True,
    max_ball_area: float | None = None,
) -> tuple[list[Detection], dict]:
    """Reject orange/object fragments that cannot be the unique game ball."""
    detections_list = list(detections)
    by_frame: dict[int, list[Detection]] = defaultdict(list)
    for detection in detections_list:
        by_frame[detection.frame_index].append(detection)

    kept_balls: list[Detection] = []
    removed = defaultdict(int)
    input_balls = 0
    for frame_index in sorted(by_frame):
        frame = by_frame[frame_index]
        robots = [det for det in frame if det.class_name in ROBOT_CLASSES]
        fields = [det for det in frame if det.class_name in FIELD_CLASSES]
        humans = [det for det in frame if det.class_name in HUMAN_CONTEXT_CLASSES]
        for ball in [det for det in frame if det.class_name in BALL_CLASSES]:
            input_balls += 1
            if max_ball_area is not None and _detection_area(ball) > max_ball_area:
                removed["oversized"] += 1
                continue
            if any(
                _ball_robot_conflict(ball, robot, reject_robot_overlap_ratio)
                for robot in robots
            ):
                removed["robot_overlap"] += 1
                continue
            if require_field_or_human_context and not (
                any(_centroid_inside(ball, field) for field in fields)
                or any(_centroid_inside(ball, human, margin=20.0) for human in humans)
            ):
                removed["outside_context"] += 1
                continue
            extra = dict(ball.extra)
            extra["ball_context_filter"] = "field_or_human"
            kept_balls.append(replace(ball, extra=extra))

    non_balls = [det for det in detections_list if det.class_name not in BALL_CLASSES]
    output = sorted(
        non_balls + kept_balls,
        key=lambda det: (det.frame_index, det.class_name, det.track_id or -1, -det.score),
    )
    return output, {
        "input_balls": input_balls,
        "kept_balls": len(kept_balls),
        "removed_balls": input_balls - len(kept_balls),
        "removed_by_reason": dict(sorted(removed.items())),
        "reject_robot_overlap_ratio": reject_robot_overlap_ratio,
        "require_field_or_human_context": require_field_or_human_context,
        "max_ball_area": max_ball_area,
    }


def _ball_robot_conflict(
    ball: Detection,
    robot: Detection,
    overlap_threshold: float,
) -> bool:
    if _centroid_inside(ball, robot):
        return True
    return _intersection_area(ball.box, robot.box) / max(_box_area(ball.box), 1.0) >= overlap_threshold


def _centroid_inside(
    candidate: Detection,
    context: Detection,
    *,
    margin: float = 0.0,
) -> bool:
    x, y = candidate.centroid
    x1, y1, x2, y2 = context.box
    return x1 - margin <= x <= x2 + margin and y1 - margin <= y <= y2 + margin


def _intersection_area(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    return _box_area((x1, y1, x2, y2))


def _detection_area(detection: Detection) -> float:
    if isinstance(detection.area, int | float) and detection.area > 0:
        return float(detection.area)
    return _box_area(detection.box)


def _box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
