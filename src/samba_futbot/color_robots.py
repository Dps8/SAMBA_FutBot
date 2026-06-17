from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .io_utils import read_detections, write_detections
from .types import Detection
from .video import require_cv2


def detect_dark_robots(
    video_path: str | Path,
    out_path: str | Path,
    *,
    max_frames: int | None = None,
    min_area: float = 350.0,
    max_area: float = 18000.0,
    min_extent: float = 0.18,
    max_extent: float = 0.92,
    min_circularity: float = 0.12,
    min_aspect: float = 0.45,
    max_aspect: float = 2.5,
    hsv_lower: tuple[int, int, int] = (0, 0, 0),
    hsv_upper: tuple[int, int, int] = (179, 255, 105),
    field_detections_path: str | Path | None = None,
    field_margin_px: float = 8.0,
    border_margin_px: float = 4.0,
    min_center_y_ratio: float = 0.0,
    max_center_y_ratio: float = 1.0,
    merge_distance_px: float = 32.0,
    max_per_frame: int = 6,
    box_expand_x_px: float = 0.0,
    box_expand_top_px: float = 0.0,
    box_expand_bottom_px: float = 0.0,
) -> list[Detection]:
    """Detect dark robot bodies from a top-camera video using color/shape cues."""
    cv2 = require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fields_by_frame = _fields_by_frame(field_detections_path)
    detections: list[Detection] = []
    frame_index = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
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
        frame_detections: list[Detection] = []
        for contour in contours:
            candidate = _robot_candidate_from_contour(
                cv2,
                contour,
                frame_index=frame_index,
                frame_width=frame.shape[1],
                frame_height=frame.shape[0],
                min_area=min_area,
                max_area=max_area,
                min_extent=min_extent,
                max_extent=max_extent,
                min_circularity=min_circularity,
                min_aspect=min_aspect,
                max_aspect=max_aspect,
                border_margin_px=border_margin_px,
                min_center_y_ratio=min_center_y_ratio,
                max_center_y_ratio=max_center_y_ratio,
                box_expand_x_px=box_expand_x_px,
                box_expand_top_px=box_expand_top_px,
                box_expand_bottom_px=box_expand_bottom_px,
            )
            if candidate is None:
                continue
            if field_detections_path and not _box_inside_any_field(
                candidate.box,
                fields_by_frame.get(frame_index, []),
                margin_px=field_margin_px,
            ):
                continue
            frame_detections.append(candidate)
        frame_detections = _merge_nearby_candidates(
            frame_detections,
            distance_px=merge_distance_px,
        )
        detections.extend(_keep_best_per_frame(frame_detections, max_per_frame=max_per_frame))
        frame_index += 1

    cap.release()
    write_detections(out_path, detections)
    return detections


def _robot_candidate_from_contour(
    cv2,
    contour,
    *,
    frame_index: int,
    frame_width: int,
    frame_height: int,
    min_area: float,
    max_area: float,
    min_extent: float,
    max_extent: float,
    min_circularity: float,
    min_aspect: float,
    max_aspect: float,
    border_margin_px: float,
    min_center_y_ratio: float,
    max_center_y_ratio: float,
    box_expand_x_px: float,
    box_expand_top_px: float,
    box_expand_bottom_px: float,
) -> Detection | None:
    area = float(cv2.contourArea(contour))
    if area < min_area or area > max_area:
        return None
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None
    circularity = 4.0 * 3.141592653589793 * area / (perimeter * perimeter)
    if circularity < min_circularity:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return None
    if _touches_border(x, y, x + w, y + h, frame_width, frame_height, border_margin_px):
        return None
    center_y_ratio = (y + h / 2.0) / float(frame_height)
    if center_y_ratio < min_center_y_ratio or center_y_ratio > max_center_y_ratio:
        return None
    aspect = w / h
    if aspect < min_aspect or aspect > max_aspect:
        return None
    extent = area / float(w * h)
    if extent < min_extent or extent > max_extent:
        return None
    original_box = (float(x), float(y), float(x + w), float(y + h))
    box = _expand_box(
        original_box,
        frame_width=frame_width,
        frame_height=frame_height,
        expand_x_px=box_expand_x_px,
        expand_top_px=box_expand_top_px,
        expand_bottom_px=box_expand_bottom_px,
    )
    score = min(0.92, 0.40 + min(0.25, area / max_area) + min(0.22, circularity * 0.22))
    return Detection(
        frame_index=frame_index,
        class_name="robots",
        score=score,
        box=box,
        prompt="hsv_dark_robot_fallback",
        area=area,
        extra={
            "source": "color_robots",
            "original_color_robot_box": list(original_box),
            "extent": extent,
            "circularity": circularity,
            "center_y_ratio": center_y_ratio,
            "box_expand_x_px": box_expand_x_px,
            "box_expand_top_px": box_expand_top_px,
            "box_expand_bottom_px": box_expand_bottom_px,
        },
    )


def _expand_box(
    box: tuple[float, float, float, float],
    *,
    frame_width: int,
    frame_height: int,
    expand_x_px: float,
    expand_top_px: float,
    expand_bottom_px: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (
        max(0.0, x1 - max(0.0, expand_x_px)),
        max(0.0, y1 - max(0.0, expand_top_px)),
        min(float(frame_width), x2 + max(0.0, expand_x_px)),
        min(float(frame_height), y2 + max(0.0, expand_bottom_px)),
    )


def _fields_by_frame(path: str | Path | None) -> dict[int, list[Detection]]:
    if path is None:
        return {}
    fields: dict[int, list[Detection]] = defaultdict(list)
    for det in read_detections(path):
        if det.class_name == "field":
            fields[det.frame_index].append(det)
    return fields


def _box_inside_any_field(
    box: tuple[float, float, float, float],
    fields: Iterable[Detection],
    *,
    margin_px: float,
) -> bool:
    if not fields:
        return False
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    for field in fields:
        fx1, fy1, fx2, fy2 = field.box
        if fx1 - margin_px <= cx <= fx2 + margin_px and fy1 - margin_px <= cy <= fy2 + margin_px:
            return True
    return False


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
    return sorted(
        detections,
        key=lambda det: (det.score, det.area or 0.0),
        reverse=True,
    )[:max_per_frame]


def _merge_nearby_candidates(
    detections: list[Detection],
    *,
    distance_px: float,
) -> list[Detection]:
    if distance_px <= 0 or len(detections) <= 1:
        return detections
    clusters: list[list[Detection]] = []
    for detection in sorted(detections, key=lambda det: det.score, reverse=True):
        assigned = False
        for cluster in clusters:
            if any(_centroid_distance(detection, other) <= distance_px for other in cluster):
                cluster.append(detection)
                assigned = True
                break
        if not assigned:
            clusters.append([detection])
    return [_merge_cluster(cluster) for cluster in clusters]


def _centroid_distance(first: Detection, second: Detection) -> float:
    dx = first.centroid[0] - second.centroid[0]
    dy = first.centroid[1] - second.centroid[1]
    return (dx * dx + dy * dy) ** 0.5


def _merge_cluster(cluster: list[Detection]) -> Detection:
    if len(cluster) == 1:
        return cluster[0]
    x1 = min(det.box[0] for det in cluster)
    y1 = min(det.box[1] for det in cluster)
    x2 = max(det.box[2] for det in cluster)
    y2 = max(det.box[3] for det in cluster)
    best = max(cluster, key=lambda det: det.score)
    area = sum(det.area or 0.0 for det in cluster)
    merged = Detection(
        frame_index=best.frame_index,
        class_name=best.class_name,
        score=max(det.score for det in cluster),
        box=(x1, y1, x2, y2),
        prompt=best.prompt,
        area=area,
        extra=dict(best.extra),
    )
    merged.extra["merged_color_robot_parts"] = len(cluster)
    return merged
