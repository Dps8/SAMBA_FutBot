from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .io_utils import read_detections, write_detections
from .play_state import ROBOT_CLASSES, point_in_box
from .types import Detection
from .video import require_cv2


def detect_orange_ball(
    video_path: str | Path,
    out_path: str | Path,
    *,
    max_frames: int | None = None,
    min_area: float = 80.0,
    max_area: float = 2200.0,
    min_circularity: float = 0.45,
    hsv_lower: tuple[int, int, int] = (0, 90, 90),
    hsv_upper: tuple[int, int, int] = (25, 255, 255),
    color_profile: str = "orange",
    context_detections_path: str | Path | None = None,
    robot_margin_px: float = 8.0,
    border_margin_px: float = 4.0,
    max_per_frame: int = 1,
) -> list[Detection]:
    cv2 = require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    detections: list[Detection] = []
    frame_index = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    while True:
        if max_frames is not None and frame_index >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area or area > max_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = 4.0 * 3.141592653589793 * area / (perimeter * perimeter)
            if circularity < min_circularity:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if _touches_border(x, y, x + w, y + h, frame.shape[1], frame.shape[0], border_margin_px):
                continue
            aspect = w / h if h else 0.0
            if aspect < 0.55 or aspect > 1.8:
                continue
            detections.append(
                Detection(
                    frame_index=frame_index,
                    class_name="ball",
                    score=min(0.99, 0.55 + circularity * 0.4),
                    box=(float(x), float(y), float(x + w), float(y + h)),
                    prompt=f"hsv_{color_profile}_ball_fallback",
                    area=area,
                    extra={
                        "source": "color_ball",
                        "color_profile": color_profile,
                        "circularity": circularity,
                    },
                )
            )
        frame_index += 1

    cap.release()
    if context_detections_path:
        detections = filter_robot_color_blobs(
            detections,
            read_detections(context_detections_path),
            robot_margin_px=robot_margin_px,
        )
    detections = _keep_best_per_frame(detections, max_per_frame=max_per_frame)
    write_detections(out_path, detections)
    return detections


def filter_robot_color_blobs(
    ball_detections: Iterable[Detection],
    context_detections: Iterable[Detection],
    *,
    robot_margin_px: float = 8.0,
) -> list[Detection]:
    robots_by_frame: dict[int, list[Detection]] = {}
    for det in context_detections:
        if det.class_name in ROBOT_CLASSES:
            robots_by_frame.setdefault(det.frame_index, []).append(det)

    filtered: list[Detection] = []
    for ball in ball_detections:
        robots = robots_by_frame.get(ball.frame_index, [])
        if any(point_in_box(ball.centroid, robot.box, margin_px=robot_margin_px) for robot in robots):
            continue
        filtered.append(ball)
    return filtered


def _touches_border(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_width: int,
    frame_height: int,
    border_margin_px: float,
) -> bool:
    if border_margin_px <= 0:
        return False
    return (
        x1 <= border_margin_px
        or y1 <= border_margin_px
        or x2 >= frame_width - border_margin_px
        or y2 >= frame_height - border_margin_px
    )


def _keep_best_per_frame(detections: list[Detection], *, max_per_frame: int) -> list[Detection]:
    if max_per_frame <= 0:
        return detections
    grouped: dict[int, list[Detection]] = {}
    for det in detections:
        grouped.setdefault(det.frame_index, []).append(det)

    kept: list[Detection] = []
    for frame_index in sorted(grouped):
        candidates = sorted(
            grouped[frame_index],
            key=lambda det: (det.area or 0.0, det.score),
            reverse=True,
        )
        kept.extend(candidates[:max_per_frame])
    return kept
