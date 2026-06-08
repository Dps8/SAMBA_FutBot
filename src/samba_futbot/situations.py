from __future__ import annotations

from typing import Any, Iterable

from .play_state import BALL_CLASSES, FIELD_CLASSES, ROBOT_CLASSES, distance, group_by_frame
from .types import Detection

POSSESSION_STATES = ("controlled", "disputed", "free")
SCHEMA_VERSION = "situations.v1"


def analyze_situations(
    detections: Iterable[Detection],
    *,
    possession_radius_px: float = 90.0,
    dispute_margin_px: float = 22.0,
    frame_width: float | None = None,
) -> dict[str, Any]:
    """Compute advanced per-frame soccer signals from already-loaded detections."""
    detections_list = list(detections)
    frames = group_by_frame(detections_list)
    records = [
        _analyze_frame(
            frame_index,
            frames[frame_index],
            possession_radius_px=possession_radius_px,
            dispute_margin_px=dispute_margin_px,
            frame_width=frame_width,
        )
        for frame_index in sorted(frames)
    ]
    return {
        "schema": {
            "name": "samba_futbot.situations",
            "version": SCHEMA_VERSION,
            "possession_states": list(POSSESSION_STATES),
            "frame_fields": [
                "frame_index",
                "ball",
                "robot_ball_distances",
                "possession",
                "loss_risk",
                "action_probabilities",
            ],
        },
        "summary": _summarize_frames(records),
        "frames": records,
    }


def _analyze_frame(
    frame_index: int,
    detections: list[Detection],
    *,
    possession_radius_px: float,
    dispute_margin_px: float,
    frame_width: float | None,
) -> dict[str, Any]:
    balls = [det for det in detections if det.class_name in BALL_CLASSES]
    robots = [det for det in detections if det.class_name in ROBOT_CLASSES]
    ball = max(balls, key=lambda det: det.score) if balls else None

    if ball is None:
        return {
            "frame_index": frame_index,
            "ball": None,
            "robot_ball_distances": [],
            "possession": {
                "state": "free",
                "owner_track_id": None,
                "owner_object_id": None,
                "owner_team": None,
                "owner_distance_px": None,
                "contenders": [],
                "nearest_margin_px": None,
            },
            "loss_risk": 0.0,
            "action_probabilities": {"pass": 0.0, "shot": 0.0, "hold": 0.0},
        }

    distances = sorted(
        (_robot_distance_record(robot, ball) for robot in robots),
        key=lambda record: (
            record["distance_px"],
            record["track_id"] if record["track_id"] is not None else 10**9,
            str(record["object_id"]),
        ),
    )
    possession = _estimate_frame_possession(
        distances,
        possession_radius_px=possession_radius_px,
        dispute_margin_px=dispute_margin_px,
    )
    loss_risk = _loss_risk(
        possession["state"],
        distances,
        possession_radius_px=possession_radius_px,
        dispute_margin_px=dispute_margin_px,
    )
    probabilities = _action_probabilities(
        possession,
        distances,
        detections,
        ball,
        loss_risk=loss_risk,
        frame_width=frame_width,
    )

    return {
        "frame_index": frame_index,
        "ball": _detection_ref(ball),
        "robot_ball_distances": distances,
        "possession": possession,
        "loss_risk": loss_risk,
        "action_probabilities": probabilities,
    }


def _estimate_frame_possession(
    distances: list[dict[str, Any]],
    *,
    possession_radius_px: float,
    dispute_margin_px: float,
) -> dict[str, Any]:
    nearest = distances[0] if distances else None
    second = distances[1] if len(distances) > 1 else None
    state = "free"
    contenders: list[dict[str, Any]] = []
    nearest_margin = None

    if nearest:
        nearest_distance = float(nearest["distance_px"])
        second_distance = float(second["distance_px"]) if second else None
        nearest_margin = second_distance - nearest_distance if second_distance is not None else None
        contest_radius = possession_radius_px + dispute_margin_px
        contenders = [
            _contender_record(record)
            for record in distances
            if float(record["distance_px"]) <= contest_radius
        ]
        close_duel = (
            second_distance is not None
            and second_distance <= contest_radius
            and nearest_margin <= dispute_margin_px
        )
        if close_duel:
            state = "disputed"
        elif nearest_distance <= possession_radius_px:
            state = "controlled"

    owner = nearest if state == "controlled" and nearest else None
    return {
        "state": state,
        "owner_track_id": owner["track_id"] if owner else None,
        "owner_object_id": owner["object_id"] if owner else None,
        "owner_team": owner["team"] if owner else None,
        "owner_distance_px": owner["distance_px"] if owner else None,
        "contenders": contenders if state in {"controlled", "disputed"} else [],
        "nearest_margin_px": _round_float(nearest_margin),
    }


def _loss_risk(
    state: str,
    distances: list[dict[str, Any]],
    *,
    possession_radius_px: float,
    dispute_margin_px: float,
) -> float:
    if state == "disputed":
        return 0.9
    if state != "controlled" or not distances:
        return 0.0

    nearest = float(distances[0]["distance_px"])
    second = float(distances[1]["distance_px"]) if len(distances) > 1 else None
    touch_pressure = _clamp(nearest / max(possession_radius_px, 1.0))
    if second is None:
        opponent_pressure = 0.0
    else:
        gap = max(second - nearest, 0.0)
        opponent_pressure = 1.0 - _clamp(gap / max(dispute_margin_px * 3.0, 1.0))
    return _round_float(_clamp(0.12 + 0.42 * touch_pressure + 0.46 * opponent_pressure))


def _action_probabilities(
    possession: dict[str, Any],
    distances: list[dict[str, Any]],
    detections: list[Detection],
    ball: Detection,
    *,
    loss_risk: float,
    frame_width: float | None,
) -> dict[str, float]:
    state = str(possession["state"])
    owner_team = possession.get("owner_team")
    owner_track_id = possession.get("owner_track_id")
    teammate_signal = _teammate_signal(distances, owner_team, owner_track_id)
    goal_proximity = _goal_proximity(ball, detections, frame_width)

    if state == "controlled":
        pass_probability = 0.18 + 0.32 * teammate_signal + 0.34 * loss_risk
        shot_probability = 0.08 + 0.52 * goal_proximity + 0.08 * (1.0 - loss_risk)
        hold_probability = (
            0.86 - 0.52 * loss_risk - 0.14 * pass_probability - 0.18 * shot_probability
        )
    elif state == "disputed":
        pass_probability = 0.18 + 0.18 * teammate_signal
        shot_probability = 0.05 + 0.28 * goal_proximity
        hold_probability = 0.2
    else:
        pass_probability = 0.04 + 0.08 * teammate_signal
        shot_probability = 0.03 + 0.12 * goal_proximity
        hold_probability = 0.08

    return {
        "pass": _round_float(_clamp(pass_probability)),
        "shot": _round_float(_clamp(shot_probability)),
        "hold": _round_float(_clamp(hold_probability)),
    }


def _teammate_signal(
    distances: list[dict[str, Any]],
    owner_team: str | None,
    owner_track_id: int | None,
) -> float:
    if not distances:
        return 0.0
    if owner_team:
        candidates = [
            record
            for record in distances
            if record["team"] == owner_team and record["track_id"] != owner_track_id
        ]
    else:
        candidates = [record for record in distances if record["track_id"] != owner_track_id]
    if not candidates:
        return 0.0
    nearest_teammate = min(float(record["distance_px"]) for record in candidates)
    return _clamp(1.0 - nearest_teammate / 240.0)


def _goal_proximity(
    ball: Detection,
    detections: list[Detection],
    frame_width: float | None,
) -> float:
    x = ball.centroid[0]
    left = 0.0
    right = frame_width
    fields = [det for det in detections if det.class_name in FIELD_CLASSES]
    if fields:
        field = max(fields, key=lambda det: (det.box[2] - det.box[0]) * (det.box[3] - det.box[1]))
        left, _, right, _ = field.box
    elif right is None:
        max_x = max((det.box[2] for det in detections), default=x)
        right = max(max_x, x)
    width = max(float(right) - left, 1.0)
    edge_distance = min(abs(x - left), abs(float(right) - x))
    return _clamp(1.0 - edge_distance / max(width * 0.25, 1.0))


def _summarize_frames(frames: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(frames)
    state_counts = {state: 0 for state in POSSESSION_STATES}
    risk_total = 0.0
    probability_totals = {"pass": 0.0, "shot": 0.0, "hold": 0.0}
    team_counts: dict[str, int] = {}
    frames_with_ball = 0

    for frame in frames:
        if frame["ball"] is not None:
            frames_with_ball += 1
        state = str(frame["possession"]["state"])
        state_counts[state] = state_counts.get(state, 0) + 1
        team = frame["possession"].get("owner_team") or "unknown"
        if state == "controlled":
            team_counts[team] = team_counts.get(team, 0) + 1
        risk_total += float(frame["loss_risk"])
        for key in probability_totals:
            probability_totals[key] += float(frame["action_probabilities"][key])

    return {
        "total_frames": total,
        "frames_with_ball": frames_with_ball,
        "possession_states": {
            state: {
                "frames": state_counts.get(state, 0),
                "ratio": state_counts.get(state, 0) / total if total else 0.0,
            }
            for state in POSSESSION_STATES
        },
        "controlled_by_team": dict(sorted(team_counts.items())),
        "average_loss_risk": _round_float(risk_total / total) if total else 0.0,
        "average_action_probabilities": {
            key: _round_float(value / total) if total else 0.0
            for key, value in probability_totals.items()
        },
    }


def _robot_distance_record(robot: Detection, ball: Detection) -> dict[str, Any]:
    return {
        **_detection_ref(robot),
        "distance_px": _round_float(distance(robot.centroid, ball.centroid)),
    }


def _detection_ref(detection: Detection) -> dict[str, Any]:
    cx, cy = detection.centroid
    return {
        "track_id": detection.track_id,
        "object_id": detection.object_id,
        "class_name": detection.class_name,
        "team": detection.team,
        "score": float(detection.score),
        "centroid": [_round_float(cx), _round_float(cy)],
    }


def _contender_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "track_id": record["track_id"],
        "object_id": record["object_id"],
        "team": record["team"],
        "distance_px": record["distance_px"],
    }


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _round_float(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


__all__ = ["POSSESSION_STATES", "SCHEMA_VERSION", "analyze_situations"]
