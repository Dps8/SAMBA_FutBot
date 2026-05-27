from __future__ import annotations

from itertools import combinations
from typing import Iterable

from .play_state import BALL_CLASSES, ROBOT_CLASSES, ball_in_play, distance, group_by_frame
from .types import Detection, Event


def estimate_possession(
    detections: Iterable[Detection], possession_radius_px: float = 90.0
) -> dict[int, Detection | None]:
    possession: dict[int, Detection | None] = {}
    for frame_index, frame_dets in group_by_frame(detections).items():
        balls = [det for det in frame_dets if det.class_name in BALL_CLASSES]
        robots = [det for det in frame_dets if det.class_name in ROBOT_CLASSES]
        if not balls or not robots:
            possession[frame_index] = None
            continue
        ball = max(balls, key=lambda det: det.score)
        nearest = min(robots, key=lambda det: distance(det.centroid, ball.centroid))
        if distance(nearest.centroid, ball.centroid) <= possession_radius_px:
            possession[frame_index] = nearest
        else:
            possession[frame_index] = None
    return possession


def detect_events(
    detections: Iterable[Detection],
    *,
    possession_radius_px: float = 90.0,
    collision_radius_px: float = 55.0,
    frame_width: int | None = None,
    goal_x_margin_ratio: float = 0.08,
    field_margin_px: float = 8.0,
) -> list[Event]:
    detections_list = list(detections)
    frames = group_by_frame(detections_list)
    possession = estimate_possession(detections_list, possession_radius_px)
    events: list[Event] = []

    last_owner: Detection | None = None
    last_ball: Detection | None = None
    collision_cooldown: dict[tuple[int | None, int | None], int] = {}

    for frame_index in sorted(frames):
        owner = possession.get(frame_index)
        if owner and last_owner and owner.track_id != last_owner.track_id:
            old_team = last_owner.team or "unknown"
            new_team = owner.team or "unknown"
            if old_team == new_team:
                events.append(
                    Event(
                        frame_index=frame_index,
                        event_type="pass",
                        description=f"Posesion cambia dentro de {new_team}",
                        confidence=0.65,
                        actors=[last_owner.track_id or -1, owner.track_id or -1],
                    )
                )
            else:
                events.append(
                    Event(
                        frame_index=frame_index,
                        event_type="interception",
                        description=f"Posesion cambia de {old_team} a {new_team}",
                        confidence=0.7,
                        actors=[last_owner.track_id or -1, owner.track_id or -1],
                    )
                )
        if owner:
            last_owner = owner

        robots = [det for det in frames[frame_index] if det.class_name in ROBOT_CLASSES]
        for a, b in combinations(robots, 2):
            pair = tuple(sorted((a.track_id, b.track_id), key=lambda value: value or -1))
            if distance(a.centroid, b.centroid) > collision_radius_px:
                continue
            if frame_index - collision_cooldown.get(pair, -10_000) < 20:
                continue
            collision_cooldown[pair] = frame_index
            events.append(
                Event(
                    frame_index=frame_index,
                    event_type="collision",
                    description="Dos robots estan a distancia de colision",
                    confidence=0.55,
                    actors=[a.track_id or -1, b.track_id or -1],
                )
            )

        balls = [
            det
            for det in frames[frame_index]
            if det.class_name in BALL_CLASSES
            and ball_in_play(
                det,
                frames[frame_index],
                possession_radius_px=possession_radius_px,
                field_margin_px=field_margin_px,
            )
        ]
        ball = max(balls, key=lambda det: det.score) if balls else None
        if ball and last_ball and frame_width:
            dx = ball.centroid[0] - last_ball.centroid[0]
            dy = ball.centroid[1] - last_ball.centroid[1]
            speed = distance(ball.centroid, last_ball.centroid)
            margin = frame_width * goal_x_margin_ratio
            near_goal = ball.centroid[0] <= margin or ball.centroid[0] >= frame_width - margin
            if near_goal and speed > 8:
                events.append(
                    Event(
                        frame_index=frame_index,
                        event_type="shot",
                        description="Balon se desplaza rapido hacia zona de gol",
                        confidence=0.5,
                        actors=[last_owner.track_id if last_owner else -1],
                        metadata={"speed_px_frame": speed},
                    )
                )
        if ball:
            last_ball = ball

    return events
