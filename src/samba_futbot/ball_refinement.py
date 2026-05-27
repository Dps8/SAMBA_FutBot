from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import replace
from typing import Iterable

from .play_state import BALL_CLASSES
from .types import Detection


def refine_ball_trajectory(
    detections: Iterable[Detection],
    *,
    max_jump_px: float = 45.0,
    preferred_area: float = 650.0,
    score_weight: float = 2.0,
    area_weight: float = 1.0,
    max_candidates_per_frame: int = 6,
) -> list[Detection]:
    detections_list = list(detections)
    non_ball = [det for det in detections_list if det.class_name not in BALL_CLASSES]
    balls_by_frame: dict[int, list[Detection]] = defaultdict(list)
    for det in detections_list:
        if det.class_name in BALL_CLASSES:
            balls_by_frame[det.frame_index].append(det)

    if not balls_by_frame:
        return detections_list

    frames = sorted(balls_by_frame)
    candidates_by_frame = [
        _rank_candidates(
            balls_by_frame[frame],
            preferred_area=preferred_area,
            score_weight=score_weight,
            area_weight=area_weight,
            max_candidates=max_candidates_per_frame,
        )
        for frame in frames
    ]

    costs: list[list[float]] = []
    parents: list[list[int | None]] = []
    for frame_idx, candidates in enumerate(candidates_by_frame):
        frame_costs: list[float] = []
        frame_parents: list[int | None] = []
        if frame_idx == 0:
            for candidate in candidates:
                frame_costs.append(
                    _emission_cost(candidate, preferred_area, score_weight, area_weight)
                )
                frame_parents.append(None)
        else:
            prev_candidates = candidates_by_frame[frame_idx - 1]
            prev_costs = costs[frame_idx - 1]
            frame_delta = max(1, frames[frame_idx] - frames[frame_idx - 1])
            for candidate in candidates:
                emission = _emission_cost(candidate, preferred_area, score_weight, area_weight)
                best_cost = math.inf
                best_parent: int | None = None
                for prev_idx, previous in enumerate(prev_candidates):
                    transition = _transition_cost(
                        previous,
                        candidate,
                        max_jump_px=max_jump_px,
                        frame_delta=frame_delta,
                    )
                    cost = prev_costs[prev_idx] + transition + emission
                    if cost < best_cost:
                        best_cost = cost
                        best_parent = prev_idx
                frame_costs.append(best_cost)
                frame_parents.append(best_parent)
        costs.append(frame_costs)
        parents.append(frame_parents)

    selected_indices = _backtrack(costs, parents)
    refined_balls = []
    for path_index, candidate_index in enumerate(selected_indices):
        ball = candidates_by_frame[path_index][candidate_index]
        extra = dict(ball.extra)
        extra["ball_refinement"] = {
            "method": "temporal_dp",
            "path_cost": costs[path_index][candidate_index],
        }
        refined_balls.append(replace(ball, extra=extra))

    refined = non_ball + refined_balls
    return sorted(
        refined,
        key=lambda det: (det.frame_index, det.class_name, det.track_id or -1, det.score),
    )


def _rank_candidates(
    candidates: list[Detection],
    *,
    preferred_area: float,
    score_weight: float,
    area_weight: float,
    max_candidates: int,
) -> list[Detection]:
    ranked = sorted(
        candidates,
        key=lambda det: _emission_cost(det, preferred_area, score_weight, area_weight),
    )
    return ranked[:max_candidates]


def _emission_cost(
    det: Detection,
    preferred_area: float,
    score_weight: float,
    area_weight: float,
) -> float:
    area = det.area if det.area and det.area > 0 else _box_area(det)
    if preferred_area > 0 and area > 0:
        area_cost = abs(math.log(area / preferred_area))
    else:
        area_cost = 0.0
    return area_weight * area_cost - score_weight * det.score


def _transition_cost(
    previous: Detection,
    current: Detection,
    *,
    max_jump_px: float,
    frame_delta: int,
) -> float:
    dx = current.centroid[0] - previous.centroid[0]
    dy = current.centroid[1] - previous.centroid[1]
    distance = math.hypot(dx, dy)
    allowed = max_jump_px * max(1, frame_delta)
    if distance <= allowed:
        return distance / max(allowed, 1.0)
    return 50.0 + distance / max(allowed, 1.0)


def _backtrack(costs: list[list[float]], parents: list[list[int | None]]) -> list[int]:
    last_index = min(range(len(costs[-1])), key=lambda idx: costs[-1][idx])
    selected = [last_index]
    for frame_idx in range(len(costs) - 1, 0, -1):
        parent = parents[frame_idx][selected[-1]]
        selected.append(0 if parent is None else parent)
    return list(reversed(selected))


def _box_area(det: Detection) -> float:
    x1, y1, x2, y2 = det.box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)
