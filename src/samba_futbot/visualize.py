from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from .events import estimate_possession
from .io_utils import read_detections, read_json
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
) -> Path:
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
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_index >= max_frames:
            break

        annotated = frame.copy()
        for det in by_frame.get(frame_index, []):
            _draw_detection(cv2, annotated, det, trails)
        _draw_header(cv2, frame, "Original")
        event = _recent_event(events_by_frame, frame_index)
        _draw_header(cv2, annotated, _frame_header(frame_index, possession.get(frame_index), event))
        writer.write(np.hstack([frame, annotated]))
        frame_index += 1

    cap.release()
    writer.release()
    return output


def _draw_detection(cv2, frame: np.ndarray, det: Detection, trails) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in det.box]
    color_rgb = class_color(det.class_name, det.team)
    color_bgr = tuple(reversed(color_rgb))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)

    cx, cy = [int(round(v)) for v in det.centroid]
    if det.track_id is not None:
        trails[det.track_id].append((cx, cy))
        pts = list(trails[det.track_id])
        for a, b in zip(pts, pts[1:]):
            cv2.line(frame, a, b, color_bgr, 2)

    label = det.class_name
    if det.track_id is not None:
        label += f" #{det.track_id}"
    if det.team:
        label += f" {det.team}"
    label += f" {det.score:.2f}"
    cv2.putText(frame, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)


def _draw_header(cv2, frame: np.ndarray, text: str) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(frame, text, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def _frame_header(frame_index: int, owner: Detection | None, event: dict | None = None) -> str:
    if owner is None:
        header = "SAMBA FutBot: tracking | possession: none"
    else:
        team = owner.team or "unknown"
        header = f"SAMBA FutBot: tracking | possession: {team} #{owner.track_id}"
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
