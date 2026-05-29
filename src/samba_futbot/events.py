from __future__ import annotations

from itertools import combinations
from typing import Iterable

from .play_state import BALL_CLASSES, ROBOT_CLASSES, ball_in_play, distance, group_by_frame
from .types import Detection, Event

GOAL_CLASSES = {"goal", "goal_blue", "goal_yellow", "blue_goal", "yellow_goal"}


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
    goal_cooldown: dict[str, int] = {}

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
        if ball:
            for goal in [det for det in frames[frame_index] if det.class_name in GOAL_CLASSES]:
                if not _ball_inside_goal(ball, goal):
                    continue
                side = _goal_side_from_class(goal.class_name)
                if frame_index - goal_cooldown.get(side, -10_000) < 45:
                    continue
                goal_cooldown[side] = frame_index
                scoring_team = _scoring_team_for_goal(side)
                events.append(
                    Event(
                        frame_index=frame_index,
                        event_type="goal_candidate",
                        description=f"Balon entra en porteria {side}",
                        confidence=0.6,
                        actors=[ball.track_id or -1],
                        metadata={
                            "goal_side": side,
                            "scoring_team": scoring_team,
                            "goal_track_id": goal.track_id,
                        },
                    )
                )
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


def summarize_events(events: Iterable[Event | dict]) -> dict:
    records = [_event_record(event) for event in events]
    counts: dict[str, int] = {}
    scoreboard: dict[str, int] = {}
    goals_by_side: dict[str, int] = {}
    first_frame: int | None = None
    last_frame: int | None = None
    timeline = []

    for record in records:
        event_type = str(record.get("event_type", "unknown"))
        counts[event_type] = counts.get(event_type, 0) + 1
        frame_index = int(record.get("frame_index", 0))
        first_frame = frame_index if first_frame is None else min(first_frame, frame_index)
        last_frame = frame_index if last_frame is None else max(last_frame, frame_index)
        metadata = record.get("metadata", {}) if isinstance(record.get("metadata", {}), dict) else {}
        if event_type == "goal_candidate":
            scoring_team = str(metadata.get("scoring_team", "unknown"))
            goal_side = str(metadata.get("goal_side", "unknown"))
            scoreboard[scoring_team] = scoreboard.get(scoring_team, 0) + 1
            goals_by_side[goal_side] = goals_by_side.get(goal_side, 0) + 1
        if event_type in {"goal_candidate", "shot", "pass", "interception", "collision"}:
            timeline.append(
                {
                    "frame_index": frame_index,
                    "event_type": event_type,
                    "description": record.get("description", ""),
                    "confidence": float(record.get("confidence", 0.0)),
                    "metadata": metadata,
                }
            )

    return {
        "total_events": len(records),
        "first_frame": first_frame,
        "last_frame": last_frame,
        "counts": dict(sorted(counts.items())),
        "scoreboard": {
            team: scoreboard.get(team, 0)
            for team in sorted(set(scoreboard) | {"blue", "yellow"})
        },
        "goals": {
            "total": counts.get("goal_candidate", 0),
            "by_goal_side": dict(sorted(goals_by_side.items())),
        },
        "possession_changes": {
            "passes": counts.get("pass", 0),
            "interceptions": counts.get("interception", 0),
        },
        "discipline": {
            "collisions": counts.get("collision", 0),
            "shots": counts.get("shot", 0),
        },
        "timeline": timeline[:25],
    }


def _ball_inside_goal(ball: Detection, goal: Detection) -> bool:
    x, y = ball.centroid
    x1, y1, x2, y2 = goal.box
    margin = max(8.0, (ball.box[2] - ball.box[0]) * 0.5)
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


def _goal_side_from_class(class_name: str) -> str:
    lowered = class_name.lower()
    if "blue" in lowered:
        return "blue"
    if "yellow" in lowered:
        return "yellow"
    return "unknown"


def _scoring_team_for_goal(goal_side: str) -> str:
    if goal_side == "blue":
        return "yellow"
    if goal_side == "yellow":
        return "blue"
    return "unknown"


def _event_record(event: Event | dict) -> dict:
    if isinstance(event, Event):
        return event.to_record()
    return event
