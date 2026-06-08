from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from .events import estimate_possession
from .io_utils import read_detections, read_json
from .play_state import BALL_CLASSES, ROBOT_CLASSES, distance
from .types import Detection
from .video import require_cv2


COLORS = {
    "field": (60, 170, 60),
    "robots": (255, 180, 50),
    "robot": (255, 180, 50),
    "ball": (40, 140, 255),
    "goal_blue": (40, 90, 255),
    "goal_yellow": (255, 220, 50),
    "robot_allied": (255, 80, 80),
    "robot_rival": (80, 180, 255),
}


def class_color(class_name: str, team: str | None = None) -> tuple[int, int, int]:
    if team == "blue":
        return (60, 110, 255)
    if team == "yellow":
        return (255, 220, 50)
    if team == "allied":
        return (255, 80, 80)
    if team == "rival":
        return (80, 180, 255)
    return COLORS.get(class_name, (230, 230, 230))


def render_demo_video(
    video_path: str | Path,
    tracks_path: str | Path,
    out_path: str | Path,
    *,
    events_path: str | Path | None = None,
    max_seconds: float | None = 120,
    trail_length: int = 45,
    style: str = "narrative",
) -> Path:
    if style not in {"narrative", "analysis"}:
        raise ValueError("style must be 'narrative' or 'analysis'.")
    cv2 = require_cv2()
    detections = read_detections(tracks_path)
    by_frame: dict[int, list[Detection]] = defaultdict(list)
    for det in detections:
        by_frame[det.frame_index].append(det)
    possession = estimate_possession(detections)
    events_by_frame = _events_by_frame(events_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    max_frames = int(max_seconds * fps) if max_seconds else None

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * 2, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output}")

    trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=trail_length))
    frame_index = 0
    previous_ball: Detection | None = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_index >= max_frames:
            break

        annotated = frame.copy()
        frame_dets = by_frame.get(frame_index, [])
        ball = _best_ball(frame_dets)
        for det in frame_dets:
            if _should_draw_detection(det, style=style):
                _draw_detection(cv2, annotated, det, trails, style=style)
        if style == "analysis":
            _draw_robot_ball_distances(cv2, annotated, frame_dets, ball)
            _draw_ball_analysis(cv2, annotated, ball, previous_ball, width)
        _draw_header(cv2, frame, "Original")
        event = _recent_event(events_by_frame, frame_index)
        _draw_header(
            cv2,
            annotated,
            _frame_header(frame_index, possession.get(frame_index), event, style=style),
        )
        writer.write(np.hstack([frame, annotated]))
        if ball:
            previous_ball = ball
        frame_index += 1

    cap.release()
    writer.release()
    return output


def _draw_detection(cv2, frame: np.ndarray, det: Detection, trails, *, style: str) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in det.box]
    color_rgb = class_color(det.class_name, det.team)
    color_bgr = tuple(reversed(color_rgb))
    thickness = 2 if style == "analysis" else 1
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, thickness)

    cx, cy = [int(round(v)) for v in det.centroid]
    if det.track_id is not None:
        trails[det.track_id].append((cx, cy))
        pts = list(trails[det.track_id])
        if det.class_name in BALL_CLASSES or style == "analysis":
            for a, b in zip(pts, pts[1:]):
                cv2.line(frame, a, b, color_bgr, 2)

    label = _detection_label(det, style=style)
    cv2.putText(frame, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)


def _detection_label(det: Detection, *, style: str) -> str:
    label = det.class_name
    if det.track_id is not None:
        label += f" #{det.track_id}"
    if det.team:
        label += f" {det.team}"
    if style == "analysis":
        label += f" {det.score:.2f}"
    return label


def _should_draw_detection(det: Detection, *, style: str) -> bool:
    if style == "analysis":
        return True
    return det.class_name in BALL_CLASSES or det.class_name in ROBOT_CLASSES or det.class_name.startswith("goal_")


def _draw_header(cv2, frame: np.ndarray, text: str) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(frame, text, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def _frame_header(
    frame_index: int,
    owner: Detection | None,
    event: dict | None = None,
    *,
    style: str = "narrative",
) -> str:
    prefix = "SAMBA FutBot: analysis" if style == "analysis" else "SAMBA FutBot: match"
    if owner is None:
        header = f"{prefix} | possession: none"
    else:
        team = owner.team or "unknown"
        header = f"{prefix} | possession: {team} #{owner.track_id}"
    if event:
        header += f" | event: {_event_label(event)}"
    return f"{header} | frame {frame_index}"


def _events_by_frame(events_path: str | Path | None) -> dict[int, list[dict]]:
    if not events_path:
        return {}
    data = read_json(events_path)
    if not isinstance(data, list):
        return {}
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for event in data:
        if not isinstance(event, dict):
            continue
        by_frame[int(event.get("frame_index", 0))].append(event)
    return by_frame


def _recent_event(
    events_by_frame: dict[int, list[dict]],
    frame_index: int,
    *,
    hold_frames: int = 45,
) -> dict | None:
    for index in range(frame_index, max(-1, frame_index - hold_frames), -1):
        events = events_by_frame.get(index)
        if events:
            return events[-1]
    return None


def _event_label(event: dict) -> str:
    event_type = str(event.get("event_type", "event"))
    metadata = event.get("metadata", {}) if isinstance(event.get("metadata", {}), dict) else {}
    if event_type == "goal_candidate":
        return f"goal {metadata.get('scoring_team', 'unknown')}"
    if event_type == "shot":
        return f"shot {metadata.get('shooting_team', 'unknown')}"
    return event_type


def robot_ball_distances(frame_dets: list[Detection], ball: Detection | None = None) -> list[dict]:
    ball = ball or _best_ball(frame_dets)
    if ball is None:
        return []
    records = []
    for robot in [det for det in frame_dets if det.class_name in ROBOT_CLASSES]:
        records.append(
            {
                "track_id": robot.track_id,
                "team": robot.team or "unknown",
                "distance_px": distance(robot.centroid, ball.centroid),
                "robot": robot,
                "ball": ball,
            }
        )
    return sorted(records, key=lambda item: (float(item["distance_px"]), item["track_id"] or -1))


def shot_probability(
    ball: Detection | None,
    previous_ball: Detection | None,
    frame_width: int,
) -> dict:
    if ball is None or previous_ball is None or frame_width <= 0:
        return {"target_side": None, "probability": 0.0, "speed_px_frame": 0.0}
    speed = distance(previous_ball.centroid, ball.centroid)
    dx = ball.centroid[0] - previous_ball.centroid[0]
    if abs(dx) < 1e-6:
        return {"target_side": None, "probability": 0.0, "speed_px_frame": speed}
    target_side = "right" if dx > 0 else "left"
    if target_side == "right":
        proximity = ball.centroid[0] / frame_width
    else:
        proximity = 1.0 - (ball.centroid[0] / frame_width)
    speed_score = min(1.0, speed / 35.0)
    probability = max(0.0, min(1.0, 0.15 + 0.45 * proximity + 0.40 * speed_score))
    return {
        "target_side": target_side,
        "probability": probability,
        "speed_px_frame": speed,
    }


def _best_ball(frame_dets: list[Detection]) -> Detection | None:
    balls = [det for det in frame_dets if det.class_name in BALL_CLASSES]
    return max(balls, key=lambda det: det.score, default=None)


def _draw_robot_ball_distances(cv2, frame: np.ndarray, frame_dets: list[Detection], ball: Detection | None) -> None:
    for record in robot_ball_distances(frame_dets, ball)[:8]:
        robot = record["robot"]
        ball_det = record["ball"]
        color_bgr = tuple(reversed(class_color(robot.class_name, robot.team)))
        robot_center = tuple(int(round(v)) for v in robot.centroid)
        ball_center = tuple(int(round(v)) for v in ball_det.centroid)
        cv2.line(frame, robot_center, ball_center, color_bgr, 1)
        midpoint = (
            int((robot_center[0] + ball_center[0]) / 2),
            int((robot_center[1] + ball_center[1]) / 2),
        )
        cv2.putText(
            frame,
            f"{record['distance_px']:.0f}px",
            midpoint,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color_bgr,
            1,
        )


def _draw_ball_analysis(
    cv2,
    frame: np.ndarray,
    ball: Detection | None,
    previous_ball: Detection | None,
    frame_width: int,
) -> None:
    if ball is None:
        return
    pressure = shot_probability(ball, previous_ball, frame_width)
    x, y = [int(round(v)) for v in ball.centroid]
    text = f"ball v={pressure['speed_px_frame']:.1f}px/f"
    if pressure["target_side"]:
        text += f" | goal {pressure['target_side']} p={pressure['probability']:.0%}"
    cv2.putText(
        frame,
        text,
        (max(6, x + 8), max(52, y - 14)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )
