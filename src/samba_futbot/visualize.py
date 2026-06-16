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
    "robots": (230, 60, 60),
    "robot": (230, 60, 60),
    "ball": (255, 130, 20),
    "goal_blue": (40, 90, 255),
    "goal_yellow": (255, 220, 50),
    "robot_allied": (255, 80, 80),
    "robot_rival": (245, 245, 245),
}

DEFAULT_FREEZE_EVENT_TYPES = {
    "goal_confirmed",
    "goal_candidate",
    "shot",
    "pass",
    "interception",
    "collision",
    "shot_pressure",
    "pass_option",
    "possession_risk",
    "recovery_window",
    "ball_out_of_play",
}

FREEZE_EVENT_PRIORITY = {
    "goal_confirmed": 120,
    "goal_candidate": 100,
    "shot_pressure": 90,
    "shot": 80,
    "possession_risk": 70,
    "pass_option": 65,
    "interception": 60,
    "pass": 55,
    "recovery_window": 50,
    "ball_out_of_play": 45,
    "collision": 40,
}


def class_color(
    class_name: str,
    team: str | None = None,
    track_id: int | None = None,
) -> tuple[int, int, int]:
    if class_name in BALL_CLASSES:
        return COLORS["ball"]
    if class_name == "goal_blue":
        return COLORS["goal_blue"]
    if class_name == "goal_yellow":
        return COLORS["goal_yellow"]
    if class_name in ROBOT_CLASSES:
        if team in {"rival", "white"}:
            return COLORS["robot_rival"]
        if team in {"allied", "red"}:
            return COLORS["robot_allied"]
        if track_id is not None and int(track_id) % 2 == 0:
            return COLORS["robot_rival"]
        return COLORS["robot_allied"]
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
    analysis_freeze: bool = False,
    freeze_seconds: float = 1.5,
    freeze_min_confidence: float = 0.45,
    freeze_cooldown_frames: int = 60,
    freeze_max_events: int = 20,
    freeze_event_types: str | list[str] | tuple[str, ...] | set[str] | None = None,
    mask_overlay: bool = True,
    mask_alpha: float = 0.35,
    label_scale: float = 1.05,
    box_thickness: int = 3,
    visual_hold_frames: int = 12,
    show_team_labels: bool = False,
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
    freeze_types = _parse_freeze_event_types(freeze_event_types)

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
    freeze_count = 0
    last_freeze_frame = -10_000
    mask_base_dir = Path(tracks_path).resolve().parent
    recent_detections: dict[tuple[str, int], tuple[Detection, int]] = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_index >= max_frames:
            break

        annotated = frame.copy()
        frame_dets = by_frame.get(frame_index, [])
        ball = _best_ball(frame_dets)
        distances = robot_ball_distances(frame_dets, ball)
        current_keys: set[tuple[str, int]] = set()
        label_boxes: list[tuple[int, int, int, int]] = []
        for det in frame_dets:
            if _should_draw_detection(det, style=style):
                key = _visual_track_key(det)
                if key is not None:
                    current_keys.add(key)
                _draw_detection(
                    cv2,
                    annotated,
                    det,
                    trails,
                    style=style,
                    mask_base_dir=mask_base_dir,
                    mask_overlay=mask_overlay,
                    mask_alpha=mask_alpha,
                    label_scale=label_scale,
                    box_thickness=box_thickness,
                    label_boxes=label_boxes,
                    show_team_labels=show_team_labels,
                )
        _draw_held_detections(
            cv2,
            annotated,
            recent_detections,
            current_keys=current_keys,
            frame_index=frame_index,
            hold_frames=visual_hold_frames,
            label_scale=label_scale,
            box_thickness=box_thickness,
        )
        for det in frame_dets:
            key = _visual_track_key(det)
            if key is not None and _should_overlay_detection(det):
                recent_detections[key] = (det, frame_index)
        if style == "analysis":
            _draw_robot_ball_distances(cv2, annotated, distances, label_boxes=label_boxes)
            _draw_ball_analysis(cv2, annotated, ball, previous_ball, width, label_boxes=label_boxes)
        _draw_header(cv2, frame, "Original")
        event = _recent_event(events_by_frame, frame_index)
        _draw_header(
            cv2,
            annotated,
            _frame_header(
                frame_index,
                possession.get(frame_index),
                event,
                nearest_distance=distances[0] if distances else None,
                style=style,
                show_team_labels=show_team_labels,
            ),
        )
        writer.write(np.hstack([frame, annotated]))
        freeze_event = _freeze_event_for_frame(
            events_by_frame,
            frame_index,
            style=style,
            analysis_freeze=analysis_freeze,
            event_types=freeze_types,
            min_confidence=freeze_min_confidence,
        )
        if (
            freeze_event
            and freeze_count < freeze_max_events
            and frame_index - last_freeze_frame >= freeze_cooldown_frames
        ):
            freeze_frames = _freeze_frame_count(fps, freeze_seconds)
            if freeze_frames:
                freeze_frame = annotated.copy()
                _draw_freeze_overlay(
                    cv2,
                    freeze_frame,
                    freeze_event,
                    distances,
                    ball,
                    previous_ball,
                    width,
                    show_team_labels=show_team_labels,
                )
                stacked_freeze = np.hstack([frame, freeze_frame])
                for _ in range(freeze_frames):
                    writer.write(stacked_freeze)
                freeze_count += 1
                last_freeze_frame = frame_index
        if ball:
            previous_ball = ball
        frame_index += 1

    cap.release()
    writer.release()
    return output


def _draw_detection(
    cv2,
    frame: np.ndarray,
    det: Detection,
    trails,
    *,
    style: str,
    mask_base_dir: Path | None = None,
    mask_overlay: bool = True,
    mask_alpha: float = 0.35,
    label_scale: float = 0.75,
    box_thickness: int = 3,
    label_boxes: list[tuple[int, int, int, int]] | None = None,
    show_team_labels: bool = False,
) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in det.box]
    color_rgb = class_color(det.class_name, det.team, det.track_id)
    color_bgr = tuple(reversed(color_rgb))
    if mask_overlay and _should_overlay_detection(det):
        mask = _load_detection_mask(det, mask_base_dir=mask_base_dir, frame_shape=frame.shape)
        if mask is not None:
            _blend_mask(frame, mask, color_bgr, alpha=mask_alpha)
        else:
            _blend_box(frame, (x1, y1, x2, y2), color_bgr, alpha=min(mask_alpha, 0.28))
    thickness = max(1, int(box_thickness + (1 if style == "analysis" else 0)))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, thickness)

    cx, cy = [int(round(v)) for v in det.centroid]
    if det.track_id is not None:
        trails[det.track_id].append((cx, cy))
        pts = list(trails[det.track_id])
        if det.class_name in BALL_CLASSES or style == "analysis":
            for a, b in zip(pts, pts[1:]):
                cv2.line(frame, a, b, color_bgr, 2)

    if det.class_name != "field":
        label = _detection_label(det, style=style, show_team=show_team_labels)
        _draw_label(
            cv2,
            frame,
            label,
            (x1, max(24, y1 - 8)),
            color_bgr,
            scale=label_scale,
            anchor_box=(x1, y1, x2, y2),
            occupied_boxes=label_boxes,
        )


def _visual_track_key(det: Detection) -> tuple[str, int] | None:
    if det.track_id is None:
        return None
    if not _should_overlay_detection(det):
        return None
    return (det.class_name, int(det.track_id))


def _draw_held_detections(
    cv2,
    frame: np.ndarray,
    recent_detections: dict[tuple[str, int], tuple[Detection, int]],
    *,
    current_keys: set[tuple[str, int]],
    frame_index: int,
    hold_frames: int,
    label_scale: float,
    box_thickness: int,
) -> None:
    if hold_frames <= 0:
        return
    expired = []
    for key, (det, last_seen) in recent_detections.items():
        age = frame_index - last_seen
        if age > hold_frames:
            expired.append(key)
            continue
        if key in current_keys or age <= 0:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in det.box]
        color_bgr = tuple(reversed(class_color(det.class_name, det.team, det.track_id)))
        alpha = max(0.10, 0.24 * (1.0 - age / max(1, hold_frames + 1)))
        _blend_box(frame, (x1, y1, x2, y2), color_bgr, alpha=alpha)
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color_bgr,
            max(1, int(box_thickness) - 1),
            lineType=cv2.LINE_AA,
        )
    for key in expired:
        recent_detections.pop(key, None)


def _should_overlay_detection(det: Detection) -> bool:
    if _is_inferred_goal(det):
        return False
    return det.class_name in BALL_CLASSES or det.class_name in ROBOT_CLASSES or det.class_name.startswith("goal_")


def _blend_box(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    color_bgr: tuple[int, int, int],
    *,
    alpha: float,
) -> None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    color = np.asarray(color_bgr, dtype=np.float32)
    roi[:] = np.clip(roi.astype(np.float32) * (1.0 - alpha) + color * alpha, 0, 255).astype(np.uint8)


def _blend_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple[int, int, int],
    *,
    alpha: float,
) -> None:
    mask_bool = np.asarray(mask).astype(bool)
    if mask_bool.shape != frame.shape[:2] or not np.any(mask_bool):
        return
    color = np.asarray(color_bgr, dtype=np.float32)
    pixels = frame[mask_bool].astype(np.float32)
    frame[mask_bool] = np.clip(pixels * (1.0 - alpha) + color * alpha, 0, 255).astype(np.uint8)


def _load_detection_mask(
    det: Detection,
    *,
    mask_base_dir: Path | None,
    frame_shape: tuple[int, ...],
) -> np.ndarray | None:
    if not det.mask_path:
        return None
    mask_index = det.extra.get("mask_index") if isinstance(det.extra, dict) else None
    if isinstance(mask_index, bool) or not isinstance(mask_index, int) or mask_index < 0:
        return None
    path = Path(det.mask_path)
    candidates = [path]
    if not path.is_absolute() and mask_base_dir is not None:
        candidates.insert(0, mask_base_dir / path)
    mask_path = next((candidate for candidate in candidates if candidate.exists()), None)
    if mask_path is None or mask_path.suffix.lower() != ".npz":
        return None
    try:
        with np.load(mask_path) as archive:
            if "masks" not in archive.files:
                return None
            masks = archive["masks"]
    except (OSError, ValueError):
        return None
    if masks.ndim == 2:
        if mask_index != 0:
            return None
        mask = masks
    elif masks.ndim >= 3:
        if mask_index >= masks.shape[0]:
            return None
        mask = np.squeeze(masks[mask_index])
    else:
        return None
    if mask.ndim != 2 or mask.shape != frame_shape[:2]:
        return None
    return np.asarray(mask != 0, dtype=np.uint8)


def _draw_label(
    cv2,
    frame: np.ndarray,
    label: str,
    origin: tuple[int, int],
    color_bgr: tuple[int, int, int],
    *,
    scale: float,
    anchor_box: tuple[int, int, int, int] | None = None,
    occupied_boxes: list[tuple[int, int, int, int]] | None = None,
) -> None:
    thickness = max(2, int(round(scale * 2.0)))
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y, rect = _choose_label_origin(
        frame.shape,
        origin,
        text_w=text_w,
        text_h=text_h,
        baseline=baseline,
        anchor_box=anchor_box,
        occupied_boxes=occupied_boxes or [],
    )
    if occupied_boxes is not None:
        occupied_boxes.append(rect)
    cv2.rectangle(
        frame,
        (rect[0], rect[1]),
        (rect[2], rect[3]),
        (0, 0, 0),
        -1,
    )
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color_bgr, thickness)


def _choose_label_origin(
    frame_shape: tuple[int, ...],
    preferred_origin: tuple[int, int],
    *,
    text_w: int,
    text_h: int,
    baseline: int,
    anchor_box: tuple[int, int, int, int] | None = None,
    occupied_boxes: list[tuple[int, int, int, int]] | None = None,
) -> tuple[int, int, tuple[int, int, int, int]]:
    occupied_boxes = occupied_boxes or []
    candidates = [preferred_origin]
    if anchor_box is not None:
        x1, y1, x2, y2 = anchor_box
        candidates = [
            (x1, y1 - 8),
            (x1, y2 + text_h + baseline + 10),
            (x2 - text_w, y1 - 8),
            (x2 - text_w, y2 + text_h + baseline + 10),
            (x2 + 8, y1 + text_h + 4),
            (x1 - text_w - 8, y1 + text_h + 4),
            (x1 + 4, y1 + text_h + 8),
            preferred_origin,
        ]
    fallback: tuple[int, int, tuple[int, int, int, int]] | None = None
    for candidate in candidates:
        placed = _label_rect_for_origin(frame_shape, candidate, text_w, text_h, baseline)
        if fallback is None:
            fallback = placed
        if not any(_boxes_overlap(placed[2], occupied) for occupied in occupied_boxes):
            return placed
    return fallback or _label_rect_for_origin(frame_shape, preferred_origin, text_w, text_h, baseline)


def _label_rect_for_origin(
    frame_shape: tuple[int, ...],
    origin: tuple[int, int],
    text_w: int,
    text_h: int,
    baseline: int,
) -> tuple[int, int, tuple[int, int, int, int]]:
    frame_h, frame_w = frame_shape[:2]
    x, y = origin
    x = max(2, min(frame_w - text_w - 8, x))
    y = max(text_h + 6, min(frame_h - baseline - 5, y))
    rect = (x - 3, y - text_h - 5, x + text_w + 5, y + baseline + 4)
    return x, y, rect


def _boxes_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    *,
    padding: int = 2,
) -> bool:
    return not (
        a[2] + padding <= b[0]
        or b[2] + padding <= a[0]
        or a[3] + padding <= b[1]
        or b[3] + padding <= a[1]
    )


def _detection_label(det: Detection, *, style: str, show_team: bool = False) -> str:
    label = det.class_name
    if det.track_id is not None:
        label += f" #{det.track_id}"
    if show_team and det.team:
        label += f" {det.team}"
    if style == "analysis":
        label += f" {det.score:.2f}"
    return label


def _should_draw_detection(det: Detection, *, style: str) -> bool:
    if _is_inferred_goal(det):
        return False
    if style == "analysis":
        return True
    return det.class_name in BALL_CLASSES or det.class_name in ROBOT_CLASSES or det.class_name.startswith("goal_")


def _is_inferred_goal(det: Detection) -> bool:
    if not det.class_name.startswith("goal_"):
        return False
    if not isinstance(det.extra, dict):
        return False
    return det.extra.get("source") == "goal_geometry"


def _draw_header(cv2, frame: np.ndarray, text: str) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(frame, text, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def _frame_header(
    frame_index: int,
    owner: Detection | None,
    event: dict | None = None,
    *,
    nearest_distance: dict | None = None,
    style: str = "narrative",
    show_team_labels: bool = False,
) -> str:
    prefix = "SAMBA FutBot: analysis" if style == "analysis" else "SAMBA FutBot: match"
    if owner is None:
        header = f"{prefix} | possession: none"
    else:
        owner_label = _actor_label(owner.team, owner.track_id, show_team=show_team_labels)
        header = f"{prefix} | possession: {owner_label}"
    if nearest_distance and style == "narrative":
        header += f" | nearest ball: {_distance_label(nearest_distance, show_team=show_team_labels)}"
    if event:
        header += f" | event: {_event_label(event)}"
    return f"{header} | frame {frame_index}"


def _actor_label(team: str | None, track_id: int | str | None, *, show_team: bool = False) -> str:
    if show_team:
        prefix = team or "unknown"
        return f"{prefix} #{track_id}" if track_id is not None else prefix
    return f"robot #{track_id}" if track_id is not None else "robot"


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


def _freeze_frame_count(fps: float, freeze_seconds: float) -> int:
    if fps <= 0 or freeze_seconds <= 0:
        return 0
    return max(1, int(round(fps * freeze_seconds)))


def _freeze_event_for_frame(
    events_by_frame: dict[int, list[dict]],
    frame_index: int,
    *,
    style: str,
    analysis_freeze: bool,
    event_types: set[str],
    min_confidence: float,
) -> dict | None:
    if style != "analysis" or not analysis_freeze:
        return None
    candidates = _freeze_event_candidates(
        events_by_frame.get(frame_index, []),
        event_types=event_types,
        min_confidence=min_confidence,
    )
    return candidates[0] if candidates else None


def _freeze_event_candidates(
    events: list[dict],
    *,
    event_types: set[str] | None = None,
    min_confidence: float = 0.45,
) -> list[dict]:
    allowed = event_types or DEFAULT_FREEZE_EVENT_TYPES
    candidates = []
    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("event_type", ""))
        confidence = float(event.get("confidence", 1.0))
        if event_type not in allowed or confidence < min_confidence:
            continue
        candidates.append(event)
    return sorted(
        candidates,
        key=lambda event: (
            -int(event.get("metadata", {}).get("priority", FREEZE_EVENT_PRIORITY.get(str(event.get("event_type", "")), 0)))
            if isinstance(event.get("metadata", {}), dict)
            else -FREEZE_EVENT_PRIORITY.get(str(event.get("event_type", "")), 0),
            -float(event.get("confidence", 1.0)),
            int(event.get("frame_index", 0)),
        ),
    )


def _parse_freeze_event_types(
    value: str | list[str] | tuple[str, ...] | set[str] | None,
) -> set[str]:
    if value is None:
        return set(DEFAULT_FREEZE_EVENT_TYPES)
    if isinstance(value, str):
        return {part.strip() for part in value.split(",") if part.strip()}
    return {str(part).strip() for part in value if str(part).strip()}


def _event_label(event: dict) -> str:
    event_type = str(event.get("event_type", "event"))
    metadata = event.get("metadata", {}) if isinstance(event.get("metadata", {}), dict) else {}
    if event_type == "goal_confirmed":
        return f"confirmed goal {metadata.get('scoring_team', 'unknown')}"
    if event_type == "goal_candidate":
        return f"goal candidate {metadata.get('scoring_team', 'unknown')}"
    if event_type == "shot":
        return f"shot {metadata.get('shooting_team', 'unknown')}"
    return event_type


def _freeze_overlay_summary(
    event: dict,
    distances: list[dict],
    *,
    show_team_labels: bool = False,
) -> list[str]:
    metadata = event.get("metadata", {}) if isinstance(event.get("metadata", {}), dict) else {}
    lines = []
    probability = metadata.get("goal_probability", metadata.get("probability"))
    if isinstance(probability, int | float):
        lines.append(f"probability {float(probability):.0%}")
    if metadata.get("target_side"):
        lines.append(f"target {metadata.get('target_side')}")
    speed = metadata.get("ball_speed_px_frame", metadata.get("speed_px_frame"))
    if isinstance(speed, int | float):
        lines.append(f"ball speed {float(speed):.1f} px/f")
    if distances:
        lines.append(f"nearest {_distance_label(distances[0], show_team=show_team_labels)}")
    description = str(event.get("description", "")).strip()
    if description and show_team_labels:
        lines.append(description)
    return lines[:4]


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


def _distance_label(record: dict, *, show_team: bool = True) -> str:
    team = str(record.get("team") or "unknown")
    track_id = record.get("track_id")
    robot_label = _actor_label(team, track_id, show_team=show_team)
    return f"{robot_label} {float(record.get('distance_px', 0.0)):.0f}px"


def _draw_robot_ball_distances(
    cv2,
    frame: np.ndarray,
    distances: list[dict],
    *,
    label_boxes: list[tuple[int, int, int, int]] | None = None,
) -> None:
    for record in distances[:8]:
        robot = record["robot"]
        ball_det = record["ball"]
        color_bgr = tuple(reversed(class_color(robot.class_name, robot.team, robot.track_id)))
        robot_center = tuple(int(round(v)) for v in robot.centroid)
        ball_center = tuple(int(round(v)) for v in ball_det.centroid)
        cv2.line(frame, robot_center, ball_center, color_bgr, 1)
        midpoint = (
            int((robot_center[0] + ball_center[0]) / 2),
            int((robot_center[1] + ball_center[1]) / 2),
        )
        _draw_label(
            cv2,
            frame,
            f"{record['distance_px']:.0f}px",
            midpoint,
            color_bgr,
            scale=0.48,
            occupied_boxes=label_boxes,
        )


def _draw_freeze_overlay(
    cv2,
    frame: np.ndarray,
    event: dict,
    distances: list[dict],
    ball: Detection | None,
    previous_ball: Detection | None,
    frame_width: int,
    show_team_labels: bool = False,
) -> None:
    overlay = frame.copy()
    height, width = frame.shape[:2]
    panel_h = min(128, max(92, height // 5))
    cv2.rectangle(overlay, (0, height - panel_h), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

    event_type = str(event.get("event_type", "event")).upper().replace("_", " ")
    confidence = float(event.get("confidence", 0.0))
    title = f"{event_type} | confidence {confidence:.0%}"
    cv2.putText(
        frame,
        title,
        (14, height - panel_h + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
    )
    for index, line in enumerate(
        _freeze_overlay_summary(event, distances, show_team_labels=show_team_labels)
    ):
        cv2.putText(
            frame,
            line,
            (16, height - panel_h + 56 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (230, 240, 255),
            1,
        )
    if ball and previous_ball:
        start = tuple(int(round(v)) for v in previous_ball.centroid)
        end = tuple(int(round(v)) for v in ball.centroid)
        cv2.arrowedLine(frame, start, end, (255, 255, 255), 3, tipLength=0.25)
    elif ball:
        center = tuple(int(round(v)) for v in ball.centroid)
        cv2.circle(frame, center, 14, (255, 255, 255), 3)

    pressure = shot_probability(ball, previous_ball, frame_width)
    if ball and pressure["target_side"]:
        center = tuple(int(round(v)) for v in ball.centroid)
        target_x = frame_width - 10 if pressure["target_side"] == "right" else 10
        target = (target_x, center[1])
        cv2.arrowedLine(frame, center, target, (80, 220, 255), 2, tipLength=0.08)


def _draw_ball_analysis(
    cv2,
    frame: np.ndarray,
    ball: Detection | None,
    previous_ball: Detection | None,
    frame_width: int,
    *,
    label_boxes: list[tuple[int, int, int, int]] | None = None,
) -> None:
    if ball is None:
        return
    pressure = shot_probability(ball, previous_ball, frame_width)
    x, y = [int(round(v)) for v in ball.centroid]
    text = f"ball v={pressure['speed_px_frame']:.1f}px/f"
    if pressure["target_side"]:
        text += f" | goal {pressure['target_side']} p={pressure['probability']:.0%}"
    _draw_label(
        cv2,
        frame,
        text,
        (max(6, x + 8), max(52, y - 14)),
        (255, 255, 255),
        scale=0.5,
        anchor_box=tuple(int(round(v)) for v in ball.box),
        occupied_boxes=label_boxes,
    )
