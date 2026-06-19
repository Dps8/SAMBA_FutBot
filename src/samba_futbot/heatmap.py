from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from .play_state import ROBOT_CLASSES
from .types import Detection
from .video import require_cv2


def render_activity_heatmap(
    video_path: str | Path,
    detections: Iterable[Detection],
    out_video: str | Path,
    out_image: str | Path,
    *,
    class_name: str = "robots",
    team: str | None = None,
    radius_px: int = 28,
    decay: float = 0.997,
    alpha: float = 0.48,
    max_seconds: float | None = None,
) -> dict:
    cv2 = require_cv2()
    if radius_px <= 0:
        raise ValueError("radius_px must be positive")
    if not 0 < decay <= 1:
        raise ValueError("decay must be in (0, 1]")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")

    selected = [
        detection
        for detection in detections
        if _matches(detection, class_name=class_name, team=team)
    ]
    by_frame: dict[int, list[Detection]] = defaultdict(list)
    for detection in selected:
        by_frame[detection.frame_index].append(detection)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(max_seconds * fps) if max_seconds else None
    output = Path(out_video)
    output.parent.mkdir(parents=True, exist_ok=True)
    image_output = Path(out_image)
    image_output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create heatmap video: {output}")

    dynamic_heat = np.zeros((height, width), dtype=np.float32)
    total_heat = np.zeros((height, width), dtype=np.float32)
    last_frame = None
    frame_index = 0
    samples = 0
    while max_frames is None or frame_index < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        dynamic_heat *= decay
        for detection in by_frame.get(frame_index, []):
            point = _anchor_point(detection, width=width, height=height)
            if point is None:
                continue
            cv2.circle(dynamic_heat, point, radius_px, 1.0, -1)
            cv2.circle(total_heat, point, radius_px, 1.0, -1)
            samples += 1
        annotated = _overlay_heat(frame, dynamic_heat, alpha=alpha)
        _draw_title(
            annotated,
            title=_title(class_name, team),
            elapsed_seconds=frame_index / fps,
            samples=samples,
        )
        writer.write(annotated)
        last_frame = frame
        frame_index += 1

    cap.release()
    writer.release()
    if last_frame is None:
        raise RuntimeError(f"Video contains no readable frames: {video_path}")
    static_heat = cv2.GaussianBlur(total_heat, (0, 0), sigmaX=max(3.0, radius_px / 2))
    static = _overlay_heat(last_frame, static_heat, alpha=alpha)
    _draw_title(static, title=f"{_title(class_name, team)} - acumulado", samples=samples)
    cv2.imwrite(str(image_output), static)
    return {
        "video": str(output),
        "image": str(image_output),
        "frames": frame_index,
        "samples": samples,
        "fps": fps,
        "class_name": class_name,
        "team": team,
    }


def _matches(detection: Detection, *, class_name: str, team: str | None) -> bool:
    if class_name == "robots":
        class_match = detection.class_name in ROBOT_CLASSES
    else:
        class_match = detection.class_name == class_name
    return class_match and (team is None or detection.team == team)


def _anchor_point(
    detection: Detection,
    *,
    width: int,
    height: int,
) -> tuple[int, int] | None:
    x1, y1, x2, y2 = detection.box
    x = int(round((x1 + x2) / 2))
    y = int(round((y1 + y2) / 2))
    if not 0 <= x < width or not 0 <= y < height:
        return None
    return x, y


def _overlay_heat(frame: np.ndarray, heat: np.ndarray, *, alpha: float) -> np.ndarray:
    cv2 = require_cv2()
    peak = float(np.max(heat))
    if peak <= 0:
        return frame.copy()
    normalized = np.clip(heat / peak * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    mask = normalized > 4
    blended = cv2.addWeighted(frame, 1.0 - alpha, colored, alpha, 0)
    output = frame.copy()
    output[mask] = blended[mask]
    return output


def _draw_title(
    frame: np.ndarray,
    *,
    title: str,
    elapsed_seconds: float | None = None,
    samples: int,
) -> None:
    cv2 = require_cv2()
    height, width = frame.shape[:2]
    scale = max(0.62, min(1.25, width / 1250.0))
    if elapsed_seconds is None:
        stats = f"muestras={samples}"
    else:
        stats = f"t={elapsed_seconds:05.1f}s | muestras={samples}"
    max_text_width = max(120, width - 56)
    scale = min(scale, _fitting_text_scale(title, max_text_width, scale))
    scale = min(scale, _fitting_text_scale(stats, max_text_width, scale))
    thickness = max(1, int(round(scale * 2)))
    (title_width, title_height), title_baseline = cv2.getTextSize(
        title, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    (stats_width, stats_height), stats_baseline = cv2.getTextSize(
        stats, cv2.FONT_HERSHEY_SIMPLEX, scale * 0.9, thickness
    )
    x = 18
    title_y = max(38, int(height * 0.055))
    stats_y = title_y + stats_height + 12
    cv2.rectangle(
        frame,
        (x - 10, title_y - title_height - 12),
        (
            min(width - 8, x + max(title_width, stats_width) + 10),
            stats_y + stats_baseline + 8,
        ),
        (12, 18, 22),
        -1,
    )
    cv2.putText(
        frame,
        title,
        (x, title_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        stats,
        (x, stats_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale * 0.9,
        (230, 235, 238),
        thickness,
        cv2.LINE_AA,
    )


def _fitting_text_scale(text: str, max_width: int, preferred: float) -> float:
    cv2 = require_cv2()
    scale = preferred
    while scale > 0.42:
        thickness = max(1, int(round(scale * 2)))
        (text_width, _), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )
        if text_width <= max_width:
            break
        scale -= 0.05
    return max(0.42, scale)


def _title(class_name: str, team: str | None) -> str:
    subject = "Actividad de robots" if class_name == "robots" else f"Actividad: {class_name}"
    return f"Mapa de calor dinamico - {subject}" + (f" - {team}" if team else "")
