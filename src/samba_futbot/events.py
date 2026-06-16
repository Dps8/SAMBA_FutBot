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
            speed = distance(ball.centroid, last_ball.centroid)
            margin = frame_width * goal_x_margin_ratio
            target_side = _shot_target_side(last_ball, ball, frame_width, margin)
            if target_side and speed > 8:
                events.append(
                    Event(
                        frame_index=frame_index,
                        event_type="shot",
                        description=f"Balon se desplaza rapido hacia porteria {target_side}",
                        confidence=0.5,
                        actors=[last_owner.track_id if last_owner else -1],
                        metadata={
                            "speed_px_frame": speed,
                            "target_side": target_side,
                            "shooting_team": _scoring_team_for_field_side(target_side),
                        },
                    )
                )
        if ball:
            last_ball = ball

    return events


def confirm_goal_candidates(
    detections: Iterable[Detection],
    events: Iterable[Event],
    *,
    lookback_frames: int = 8,
    confirmation_frames: int = 4,
    min_inside_frames: int = 2,
    min_entry_motion_px: float = 3.0,
    min_goal_score: float = 0.45,
) -> list[Event]:
    """Promote goal candidates only when temporal entry evidence is strong."""
    detections_list = list(detections)
    frames = group_by_frame(detections_list)
    events_list = list(events)
    confirmed: list[Event] = []
    existing_confirmations = {
        (
            int(event.metadata.get("candidate_frame", event.frame_index)),
            str(event.metadata.get("goal_side", "unknown")),
        )
        for event in events_list
        if event.event_type == "goal_confirmed"
    }

    for candidate in events_list:
        if candidate.event_type != "goal_candidate":
            continue
        side = str(candidate.metadata.get("goal_side", "unknown"))
        frame_index = candidate.frame_index
        if (frame_index, side) in existing_confirmations:
            continue
        goal = _goal_for_candidate(frames.get(frame_index, []), side)
        ball = _ball_for_candidate(frames.get(frame_index, []), candidate)
        if not goal or not ball:
            continue
        if goal.score < min_goal_score or goal.extra.get("source") == "goal_geometry":
            continue
        previous = _previous_ball(
            frames,
            ball,
            frame_index=frame_index,
            lookback_frames=lookback_frames,
        )
        if not previous or _ball_inside_goal(previous, goal):
            continue
        entry_motion = distance(previous.centroid, ball.centroid)
        if entry_motion < min_entry_motion_px:
            continue
        if distance(ball.centroid, goal.centroid) >= distance(previous.centroid, goal.centroid):
            continue
        inside_frames = _inside_goal_frames(
            frames,
            ball,
            goal,
            start_frame=frame_index,
            confirmation_frames=confirmation_frames,
        )
        if len(inside_frames) < min_inside_frames:
            continue
        confirmed.append(
            Event(
                frame_index=frame_index,
                event_type="goal_confirmed",
                description=f"Gol confirmado en porteria {side}",
                confidence=min(0.95, max(0.7, candidate.confidence + 0.2)),
                actors=list(candidate.actors),
                metadata={
                    **candidate.metadata,
                    "candidate_frame": frame_index,
                    "previous_ball_frame": previous.frame_index,
                    "entry_motion_px": round(entry_motion, 6),
                    "inside_frames": inside_frames,
                    "validation": "temporal_goal_entry",
                },
            )
        )
        existing_confirmations.add((frame_index, side))
    return sorted(events_list + confirmed, key=lambda event: (event.frame_index, event.event_type))


def summarize_events(events: Iterable[Event | dict]) -> dict:
    records = [_event_record(event) for event in events]
    counts: dict[str, int] = {}
    scoreboard: dict[str, int] = {}
    confirmed_scoreboard: dict[str, int] = {}
    goals_by_side: dict[str, int] = {}
    shots_by_team: dict[str, int] = {}
    shots_by_target_side: dict[str, int] = {}
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
        if event_type == "goal_confirmed":
            scoring_team = str(metadata.get("scoring_team", "unknown"))
            confirmed_scoreboard[scoring_team] = confirmed_scoreboard.get(scoring_team, 0) + 1
        if event_type == "shot":
            shooting_team = str(metadata.get("shooting_team", "unknown"))
            target_side = str(metadata.get("target_side", "unknown"))
            shots_by_team[shooting_team] = shots_by_team.get(shooting_team, 0) + 1
            shots_by_target_side[target_side] = shots_by_target_side.get(target_side, 0) + 1
        if event_type in {
            "goal_candidate",
            "goal_confirmed",
            "shot",
            "pass",
            "interception",
            "collision",
        }:
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
        "confirmed_scoreboard": {
            team: confirmed_scoreboard.get(team, 0)
            for team in sorted(set(confirmed_scoreboard) | {"blue", "yellow"})
        },
        "goals": {
            "total": counts.get("goal_candidate", 0),
            "confirmed": counts.get("goal_confirmed", 0),
            "by_goal_side": dict(sorted(goals_by_side.items())),
        },
        "shots": {
            "total": counts.get("shot", 0),
            "by_team": {
                team: shots_by_team.get(team, 0)
                for team in sorted(set(shots_by_team) | {"blue", "yellow"})
            },
            "by_target_side": dict(sorted(shots_by_target_side.items())),
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


def _goal_for_candidate(frame_detections: list[Detection], side: str) -> Detection | None:
    candidates = [
        det
        for det in frame_detections
        if det.class_name in GOAL_CLASSES and _goal_side_from_class(det.class_name) == side
    ]
    return max(candidates, key=lambda det: det.score) if candidates else None


def _ball_for_candidate(
    frame_detections: list[Detection],
    candidate: Event,
) -> Detection | None:
    actor_ids = set(candidate.actors)
    balls = [det for det in frame_detections if det.class_name in BALL_CLASSES]
    matching = [det for det in balls if det.track_id in actor_ids]
    pool = matching or balls
    return max(pool, key=lambda det: det.score) if pool else None


def _previous_ball(
    frames: dict[int, list[Detection]],
    ball: Detection,
    *,
    frame_index: int,
    lookback_frames: int,
) -> Detection | None:
    for previous_frame in range(frame_index - 1, max(-1, frame_index - lookback_frames - 1), -1):
        balls = [det for det in frames.get(previous_frame, []) if det.class_name in BALL_CLASSES]
        if ball.track_id is not None:
            balls = [det for det in balls if det.track_id == ball.track_id]
        if balls:
            return max(balls, key=lambda det: det.score)
    return None


def _inside_goal_frames(
    frames: dict[int, list[Detection]],
    ball: Detection,
    goal: Detection,
    *,
    start_frame: int,
    confirmation_frames: int,
) -> list[int]:
    inside = []
    for frame_index in range(start_frame, start_frame + confirmation_frames + 1):
        balls = [det for det in frames.get(frame_index, []) if det.class_name in BALL_CLASSES]
        if ball.track_id is not None:
            balls = [det for det in balls if det.track_id == ball.track_id]
        if not any(_ball_inside_goal(item, goal) for item in balls):
            break
        inside.append(frame_index)
    return inside


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


def _shot_target_side(
    previous: Detection,
    current: Detection,
    frame_width: int,
    margin: float,
) -> str | None:
    dx = current.centroid[0] - previous.centroid[0]
    if current.centroid[0] <= margin and dx < 0:
        return "left"
    if current.centroid[0] >= frame_width - margin and dx > 0:
        return "right"
    return None


def _scoring_team_for_field_side(side: str) -> str:
    if side == "left":
        return "blue"
    if side == "right":
        return "yellow"
    return "unknown"


def _event_record(event: Event | dict) -> dict:
    if isinstance(event, Event):
        return event.to_record()
    return event
