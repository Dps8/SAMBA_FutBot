from __future__ import annotations

from pathlib import Path

from .io_utils import write_detections
from .types import Detection
from .video import require_cv2


DEFAULT_GOAL_COLOR_PROFILES = {
    "goal_blue": {
        "hsv_lower": (90, 70, 50),
        "hsv_upper": (135, 255, 255),
    },
    "goal_yellow": {
        "hsv_lower": (18, 70, 80),
        "hsv_upper": (42, 255, 255),
    },
}

GOAL_CLASSES = frozenset(DEFAULT_GOAL_COLOR_PROFILES)


def adapt_goal_color_profiles_from_detections(
    video_path: str | Path,
    seed_detections: list[Detection],
    *,
    profiles: dict[str, dict[str, tuple[int, int, int]]] | None = None,
    broad_profiles: dict[str, dict[str, tuple[int, int, int]]] | None = None,
    hsv_margin: tuple[int, int, int] = (12, 45, 45),
    min_pixels: int = 120,
) -> dict[str, dict[str, tuple[int, int, int]]]:
    cv2 = require_cv2()
    resolved_profiles = dict(profiles or DEFAULT_GOAL_COLOR_PROFILES)
    broad = broad_profiles or _broad_goal_profiles()
    goal_detections = [
        det for det in seed_detections if det.class_name in resolved_profiles and det.class_name in broad
    ]
    if not goal_detections:
        return resolved_profiles

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    samples_by_class: dict[str, list] = {class_name: [] for class_name in resolved_profiles}
    current_frame_index: int | None = None
    current_frame = None
    for det in sorted(goal_detections, key=lambda item: item.frame_index):
        if current_frame_index != det.frame_index:
            cap.set(cv2.CAP_PROP_POS_FRAMES, det.frame_index)
            ok, current_frame = cap.read()
            current_frame_index = det.frame_index
            if not ok:
                current_frame = None
        if current_frame is None:
            continue
        hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
        height, width = hsv.shape[:2]
        x1, y1, x2, y2 = _clip_box(det.box, width, height, inset_ratio=0.08)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = hsv[y1:y2, x1:x2]
        lower = broad[det.class_name]["hsv_lower"]
        upper = broad[det.class_name]["hsv_upper"]
        mask = cv2.inRange(roi, lower, upper)
        pixels = roi[mask > 0]
        if pixels.size:
            samples_by_class[det.class_name].append(pixels)
    cap.release()

    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is already required by cv2 workflows.
        return resolved_profiles

    adapted = dict(resolved_profiles)
    for class_name, sample_chunks in samples_by_class.items():
        if not sample_chunks:
            continue
        pixels = np.concatenate(sample_chunks, axis=0)
        if len(pixels) < min_pixels:
            continue
        low = np.percentile(pixels, 8, axis=0)
        high = np.percentile(pixels, 92, axis=0)
        margin = np.array(hsv_margin)
        lower = np.maximum([0, 0, 0], low - margin).astype(int)
        upper = np.minimum([179, 255, 255], high + margin).astype(int)
        adapted[class_name] = {
            "hsv_lower": tuple(int(value) for value in lower),
            "hsv_upper": tuple(int(value) for value in upper),
        }
    return adapted


def detect_colored_goals(
    video_path: str | Path,
    out_path: str | Path,
    *,
    max_frames: int | None = None,
    profiles: dict[str, dict[str, tuple[int, int, int]]] | None = None,
    seed_detections: list[Detection] | None = None,
    adaptive_color: bool = False,
    broad_profiles: dict[str, dict[str, tuple[int, int, int]]] | None = None,
    adaptive_hsv_margin: tuple[int, int, int] = (12, 45, 45),
    adaptive_min_pixels: int = 120,
    spatial_gate_from_seeds: bool = True,
    seed_spatial_margin_px: float = 90.0,
    require_seed_for_color: bool = False,
    require_field_overlap: bool = False,
    field_margin_px: float = 18.0,
    min_area: float = 180.0,
    max_area: float = 80_000.0,
    min_extent: float = 0.18,
    max_per_frame_per_class: int = 1,
) -> list[Detection]:
    cv2 = require_cv2()
    resolved_profiles = profiles or DEFAULT_GOAL_COLOR_PROFILES
    if adaptive_color and seed_detections:
        resolved_profiles = adapt_goal_color_profiles_from_detections(
            video_path,
            seed_detections,
            profiles=resolved_profiles,
            broad_profiles=broad_profiles,
            hsv_margin=adaptive_hsv_margin,
            min_pixels=adaptive_min_pixels,
        )
    seed_windows = (
        _seed_windows_by_class(seed_detections, seed_spatial_margin_px)
        if adaptive_color and spatial_gate_from_seeds and seed_detections
        else {}
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    detections: list[Detection] = []
    frame_index = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    while True:
        if max_frames is not None and frame_index >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for class_name, profile in resolved_profiles.items():
            if (
                require_seed_for_color
                and seed_detections is not None
                and class_name not in seed_windows
            ):
                continue
            lower = tuple(int(value) for value in profile["hsv_lower"])
            upper = tuple(int(value) for value in profile["hsv_upper"])
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            frame_detections: list[Detection] = []
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < min_area or area > max_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                if w <= 0 or h <= 0:
                    continue
                box = (float(x), float(y), float(x + w), float(y + h))
                class_windows = seed_windows.get(class_name)
                if class_windows and not any(_boxes_intersect(box, window) for window in class_windows):
                    continue
                extent = area / float(w * h)
                if extent < min_extent:
                    continue
                frame_detections.append(
                    Detection(
                        frame_index=frame_index,
                        class_name=class_name,
                        score=min(0.99, 0.45 + min(0.45, area / max_area)),
                        box=box,
                        prompt=f"hsv_{class_name}",
                        area=area,
                        extra={
                            "source": "color_goals",
                            "extent": extent,
                            "hsv_lower": list(lower),
                            "hsv_upper": list(upper),
                            "adaptive_color": bool(adaptive_color and seed_detections),
                            "spatial_gate_from_seeds": bool(class_windows),
                        },
                    )
                )
            detections.extend(
                sorted(
                    frame_detections,
                    key=lambda det: (det.area or 0.0, det.score),
                    reverse=True,
                )
            )
        frame_index += 1

    cap.release()
    constrained = enforce_goal_frame_constraints(
        detections,
        field_detections=seed_detections,
        max_per_frame_per_class=max_per_frame_per_class,
        require_field_overlap=require_field_overlap,
        field_margin_px=field_margin_px,
    )
    write_detections(out_path, constrained)
    return constrained


def enforce_goal_frame_constraints(
    detections: list[Detection],
    *,
    field_detections: list[Detection] | None = None,
    max_per_frame_per_class: int = 1,
    require_field_overlap: bool = False,
    infer_missing_opposite: bool = False,
    inferred_goal_score: float = 0.28,
    field_margin_px: float = 18.0,
) -> list[Detection]:
    if max_per_frame_per_class <= 0:
        raise ValueError("max_per_frame_per_class must be positive.")

    fields_by_frame: dict[int, list[Detection]] = {}
    for det in field_detections or []:
        if det.class_name == "field":
            fields_by_frame.setdefault(det.frame_index, []).append(det)

    passthrough: list[Detection] = []
    goals_by_key: dict[tuple[int, str], list[Detection]] = {}
    for det in detections:
        if det.class_name not in GOAL_CLASSES:
            passthrough.append(det)
            continue
        fields = fields_by_frame.get(det.frame_index, [])
        if require_field_overlap and not _goal_is_on_field(det, fields, field_margin_px):
            continue
        goals_by_key.setdefault((det.frame_index, det.class_name), []).append(det)

    kept_goals: list[Detection] = []
    for candidates in goals_by_key.values():
        kept_goals.extend(
            sorted(
                candidates,
                key=lambda det: (det.score, det.area or 0.0),
                reverse=True,
            )[:max_per_frame_per_class]
        )
    if infer_missing_opposite:
        kept_goals.extend(
            _infer_missing_opposite_goals(
                kept_goals,
                fields_by_frame,
                score=inferred_goal_score,
            )
        )
    return sorted(
        passthrough + kept_goals,
        key=lambda det: (det.frame_index, det.class_name, det.track_id or -1, det.score),
    )


def _infer_missing_opposite_goals(
    goals: list[Detection],
    fields_by_frame: dict[int, list[Detection]],
    *,
    score: float,
) -> list[Detection]:
    inferred: list[Detection] = []
    goals_by_frame: dict[int, list[Detection]] = {}
    for goal in goals:
        goals_by_frame.setdefault(goal.frame_index, []).append(goal)

    for frame_index, frame_goals in goals_by_frame.items():
        classes = {goal.class_name for goal in frame_goals}
        missing_classes = GOAL_CLASSES - classes
        if len(missing_classes) != 1 or len(classes) != 1:
            continue
        fields = fields_by_frame.get(frame_index, [])
        if not fields:
            continue
        source = max(frame_goals, key=lambda det: (det.score, det.area or 0.0))
        field = max(fields, key=lambda det: det.area or _box_area(det.box))
        inferred_class = next(iter(missing_classes))
        inferred_box = _mirror_box_across_field(source.box, field.box)
        inferred.append(
            Detection(
                frame_index=frame_index,
                class_name=inferred_class,
                score=min(float(score), source.score),
                box=inferred_box,
                prompt="geometry_inferred_opposite_goal",
                area=_box_area(inferred_box),
                extra={
                    "source": "goal_geometry",
                    "inferred_from_class": source.class_name,
                    "inferred_from_box": list(source.box),
                    "field_box": list(field.box),
                },
            )
        )
    return inferred


def _mirror_box_across_field(
    box: tuple[float, float, float, float],
    field_box: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    field_x1, _, field_x2, _ = field_box
    return (field_x1 + field_x2 - x2, y1, field_x1 + field_x2 - x1, y2)


def _box_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _broad_goal_profiles() -> dict[str, dict[str, tuple[int, int, int]]]:
    return {
        "goal_blue": {
            "hsv_lower": (80, 35, 35),
            "hsv_upper": (145, 255, 255),
        },
        "goal_yellow": {
            "hsv_lower": (10, 35, 45),
            "hsv_upper": (55, 255, 255),
        },
    }


def _clip_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    *,
    inset_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    inset_x = max(0.0, (x2 - x1) * inset_ratio)
    inset_y = max(0.0, (y2 - y1) * inset_ratio)
    return (
        max(0, int(round(x1 + inset_x))),
        max(0, int(round(y1 + inset_y))),
        min(width, int(round(x2 - inset_x))),
        min(height, int(round(y2 - inset_y))),
    )


def _seed_windows_by_class(
    seed_detections: list[Detection] | None,
    margin_px: float,
) -> dict[str, list[tuple[float, float, float, float]]]:
    windows: dict[str, list[tuple[float, float, float, float]]] = {}
    if not seed_detections:
        return windows
    for det in seed_detections:
        if det.class_name not in DEFAULT_GOAL_COLOR_PROFILES:
            continue
        x1, y1, x2, y2 = det.box
        margin = max(0.0, margin_px)
        windows.setdefault(det.class_name, []).append(
            (x1 - margin, y1 - margin, x2 + margin, y2 + margin)
        )
    return windows


def _goal_is_on_field(
    goal: Detection,
    fields: list[Detection],
    margin_px: float,
) -> bool:
    if not fields:
        return False
    expanded_fields = [_expand_box(field.box, margin_px) for field in fields]
    bottom_center = ((goal.box[0] + goal.box[2]) / 2.0, goal.box[3])
    centroid = goal.centroid
    return any(
        _point_in_box(bottom_center, field_box)
        or _point_in_box(centroid, field_box)
        or _boxes_intersect(goal.box, field_box)
        for field_box in expanded_fields
    )


def _expand_box(
    box: tuple[float, float, float, float],
    margin_px: float,
) -> tuple[float, float, float, float]:
    margin = max(0.0, margin_px)
    x1, y1, x2, y2 = box
    return (x1 - margin, y1 - margin, x2 + margin, y2 + margin)


def _point_in_box(
    point: tuple[float, float],
    box: tuple[float, float, float, float],
) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def _boxes_intersect(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1
