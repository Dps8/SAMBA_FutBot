from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .io_utils import read_json
from .play_state import BALL_CLASSES, FIELD_CLASSES, ROBOT_CLASSES, ball_in_play, distance, group_by_frame
from .types import Detection, Event


HUMAN_CLASSES = {
    "person",
    "human",
    "referee",
    "arbitro",
    "player",
    "jugador",
    "hand",
    "mano",
    "arm",
    "brazo",
}


@dataclass(slots=True)
class FrameState:
    frame_index: int
    state: str
    confidence: float
    reasons: list[str] = field(default_factory=list)
    ball_in_play: bool = False
    human_intervention: bool = False
    robot_removed: list[int] = field(default_factory=list)
    robot_disabled: list[int] = field(default_factory=list)

    def to_record(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class GameSegment:
    state: str
    start_frame: int
    end_frame: int
    frames: int
    confidence: float
    reasons: list[str] = field(default_factory=list)

    def to_record(self) -> dict:
        return asdict(self)


def classify_frame_states(
    detections: Iterable[Detection],
    *,
    possession_radius_px: float = 90.0,
    field_margin_px: float = 8.0,
    missing_ball_frames: int = 12,
    robot_removed_after_frames: int = 18,
    robot_disabled_after_frames: int = 45,
    stationary_threshold_px: float = 2.0,
    human_field_margin_px: float = 12.0,
    field_polygon: list[tuple[float, float]] | None = None,
) -> list[FrameState]:
    detections_list = list(detections)
    frames = group_by_frame(detections_list)
    if not frames:
        return []

    frame_indices = list(range(min(frames), max(frames) + 1))
    last_ball_frame: int | None = None
    last_robot_seen: dict[int, tuple[int, Detection]] = {}
    robot_stationary: dict[int, tuple[int, Detection]] = {}
    states: list[FrameState] = []

    for frame_index in frame_indices:
        frame_dets = frames.get(frame_index, [])
        balls = [det for det in frame_dets if det.class_name in BALL_CLASSES]
        robots = [det for det in frame_dets if det.class_name in ROBOT_CLASSES]
        fields = [det for det in frame_dets if det.class_name in FIELD_CLASSES]
        humans = [det for det in frame_dets if det.class_name in HUMAN_CLASSES]
        in_play = any(
            ball_in_play(
                ball,
                frame_dets,
                possession_radius_px=possession_radius_px,
                field_margin_px=field_margin_px,
            )
            for ball in balls
        )
        if field_polygon and not in_play:
            in_play = any(_point_in_polygon(ball.centroid, field_polygon) for ball in balls)
        if balls:
            last_ball_frame = frame_index

        reasons = []
        human_intervention = _human_intervention(
            humans,
            fields=fields,
            balls=balls,
            robots=robots,
            field_margin_px=human_field_margin_px,
        )
        if human_intervention:
            reasons.append("human_on_or_near_field")

        if not balls and last_ball_frame is not None and frame_index - last_ball_frame >= missing_ball_frames:
            reasons.append("ball_missing")
        if balls and not in_play:
            reasons.append("ball_out_of_play")
        if human_intervention:
            reasons.append("dead_ball_by_intervention")

        removed = _removed_robot_ids(
            frame_index,
            last_robot_seen,
            current_robots=robots,
            removed_after_frames=robot_removed_after_frames,
        )
        if removed:
            reasons.append("robot_removed_candidate")

        disabled = _disabled_robot_ids(
            frame_index,
            robots,
            robot_stationary,
            disabled_after_frames=robot_disabled_after_frames,
            stationary_threshold_px=stationary_threshold_px,
        )
        if disabled:
            reasons.append("robot_disabled_candidate")

        _update_robot_history(frame_index, robots, last_robot_seen)

        state = "in_play"
        confidence = 0.55 if in_play else 0.45
        if human_intervention:
            state = "human_intervention"
            confidence = 0.8
        elif "ball_missing" in reasons or "ball_out_of_play" in reasons:
            state = "dead_ball"
            confidence = 0.7

        states.append(
            FrameState(
                frame_index=frame_index,
                state=state,
                confidence=confidence,
                reasons=sorted(set(reasons)),
                ball_in_play=in_play,
                human_intervention=human_intervention,
                robot_removed=removed,
                robot_disabled=disabled,
            )
        )
    return states


def detect_game_segments(
    frame_states: Iterable[FrameState | dict],
    *,
    min_segment_frames: int = 1,
) -> list[GameSegment]:
    states = [_state_from_record(item) for item in frame_states]
    if not states:
        return []
    segments: list[GameSegment] = []
    start = states[0]
    previous = states[0]
    reasons: set[str] = set(start.reasons)
    confidences = [start.confidence]

    for current in states[1:]:
        contiguous = current.frame_index == previous.frame_index + 1
        if current.state != previous.state or not contiguous:
            _append_segment(
                segments,
                start=start,
                previous=previous,
                reasons=reasons,
                confidences=confidences,
                min_segment_frames=min_segment_frames,
            )
            start = current
            reasons = set(current.reasons)
            confidences = [current.confidence]
        else:
            reasons.update(current.reasons)
            confidences.append(current.confidence)
        previous = current
    _append_segment(
        segments,
        start=start,
        previous=previous,
        reasons=reasons,
        confidences=confidences,
        min_segment_frames=min_segment_frames,
    )
    return segments


def detect_external_events(frame_states: Iterable[FrameState | dict]) -> list[Event]:
    events = []
    previous_state: str | None = None
    emitted_removed: set[tuple[int, int]] = set()
    emitted_disabled: set[tuple[int, int]] = set()
    for state in [_state_from_record(item) for item in frame_states]:
        if state.state != previous_state and state.state in {"dead_ball", "human_intervention"}:
            events.append(
                Event(
                    frame_index=state.frame_index,
                    event_type=state.state,
                    description=_state_description(state),
                    confidence=state.confidence,
                    metadata={"reasons": state.reasons},
                )
            )
        for track_id in state.robot_removed:
            key = (state.frame_index, track_id)
            if key not in emitted_removed:
                emitted_removed.add(key)
                events.append(
                    Event(
                        frame_index=state.frame_index,
                        event_type="robot_removed",
                        description=f"Robot #{track_id} desaparece del juego",
                        confidence=0.55,
                        actors=[track_id],
                        metadata={"reasons": state.reasons},
                    )
                )
        for track_id in state.robot_disabled:
            key = (state.frame_index, track_id)
            if key not in emitted_disabled:
                emitted_disabled.add(key)
                events.append(
                    Event(
                        frame_index=state.frame_index,
                        event_type="robot_disabled",
                        description=f"Robot #{track_id} permanece inmovil",
                        confidence=0.5,
                        actors=[track_id],
                        metadata={"reasons": state.reasons},
                    )
                )
        previous_state = state.state
    return events


def play_mask_from_segments(segments: Iterable[GameSegment | dict]) -> set[int]:
    playable = set()
    for segment in segments:
        record = segment.to_record() if isinstance(segment, GameSegment) else segment
        if record.get("state") != "in_play":
            continue
        playable.update(range(int(record["start_frame"]), int(record["end_frame"]) + 1))
    return playable


def playable_frames_from_game_state(path: str | Path) -> set[int]:
    data = read_json(path)
    if isinstance(data, dict):
        if isinstance(data.get("segments"), list):
            return play_mask_from_segments(data["segments"])
        if isinstance(data.get("states"), list):
            return {
                int(state["frame_index"])
                for state in data["states"]
                if isinstance(state, dict) and state.get("state") == "in_play"
            }
    if isinstance(data, list):
        return play_mask_from_segments(data)
    raise ValueError(f"Expected game-state JSON object or segment list: {path}")


def filter_detections_to_playable_frames(
    detections: Iterable[Detection],
    playable_frames: set[int] | None,
) -> list[Detection]:
    detections_list = list(detections)
    if playable_frames is None:
        return detections_list
    return [det for det in detections_list if det.frame_index in playable_frames]


def _human_intervention(
    humans: list[Detection],
    *,
    fields: list[Detection],
    balls: list[Detection],
    robots: list[Detection],
    field_margin_px: float,
) -> bool:
    if not humans:
        return False
    context = fields + balls + robots
    if not context:
        return True
    return any(
        _boxes_overlap(human.box, det.box, margin_px=field_margin_px)
        for human in humans
        for det in context
    )


def _removed_robot_ids(
    frame_index: int,
    last_robot_seen: dict[int, tuple[int, Detection]],
    *,
    current_robots: list[Detection],
    removed_after_frames: int,
) -> list[int]:
    current_ids = {det.track_id for det in current_robots if det.track_id is not None}
    removed = []
    for track_id, (last_frame, _) in last_robot_seen.items():
        if track_id in current_ids:
            continue
        if frame_index - last_frame == removed_after_frames:
            removed.append(track_id)
    return sorted(removed)


def _disabled_robot_ids(
    frame_index: int,
    robots: list[Detection],
    stationary: dict[int, tuple[int, Detection]],
    *,
    disabled_after_frames: int,
    stationary_threshold_px: float,
) -> list[int]:
    disabled = []
    for robot in robots:
        if robot.track_id is None:
            continue
        start_frame, anchor = stationary.get(robot.track_id, (frame_index, robot))
        if distance(robot.centroid, anchor.centroid) > stationary_threshold_px:
            stationary[robot.track_id] = (frame_index, robot)
            continue
        stationary[robot.track_id] = (start_frame, anchor)
        if frame_index - start_frame == disabled_after_frames:
            disabled.append(robot.track_id)
    return sorted(disabled)


def _update_robot_history(
    frame_index: int,
    robots: list[Detection],
    last_robot_seen: dict[int, tuple[int, Detection]],
) -> None:
    for robot in robots:
        if robot.track_id is not None:
            last_robot_seen[robot.track_id] = (frame_index, robot)


def _boxes_overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
    *,
    margin_px: float = 0.0,
) -> bool:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    return not (
        ax2 + margin_px < bx1
        or bx2 + margin_px < ax1
        or ay2 + margin_px < by1
        or by2 + margin_px < ay1
    )


def _point_in_polygon(
    point: tuple[float, float], polygon: list[tuple[float, float]]
) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1
        ):
            inside = not inside
        previous = current
    return inside


def _append_segment(
    segments: list[GameSegment],
    *,
    start: FrameState,
    previous: FrameState,
    reasons: set[str],
    confidences: list[float],
    min_segment_frames: int,
) -> None:
    frames = previous.frame_index - start.frame_index + 1
    if frames < min_segment_frames:
        return
    segments.append(
        GameSegment(
            state=start.state,
            start_frame=start.frame_index,
            end_frame=previous.frame_index,
            frames=frames,
            confidence=sum(confidences) / len(confidences),
            reasons=sorted(reasons),
        )
    )


def _state_from_record(item: FrameState | dict) -> FrameState:
    if isinstance(item, FrameState):
        return item
    return FrameState(
        frame_index=int(item["frame_index"]),
        state=str(item["state"]),
        confidence=float(item.get("confidence", 0.0)),
        reasons=list(item.get("reasons", [])),
        ball_in_play=bool(item.get("ball_in_play", False)),
        human_intervention=bool(item.get("human_intervention", False)),
        robot_removed=[int(value) for value in item.get("robot_removed", [])],
        robot_disabled=[int(value) for value in item.get("robot_disabled", [])],
    )


def _state_description(state: FrameState) -> str:
    if state.state == "human_intervention":
        return "Intervencion humana cerca del campo"
    if state.state == "dead_ball":
        return "Balon fuera de juego o pausa candidata"
    return state.state
