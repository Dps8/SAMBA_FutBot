from __future__ import annotations

from pathlib import Path

import numpy as np

from .field_analysis import FieldCalibration
from .io_utils import write_json
from .video import require_cv2


MIN_FRAME_COVERAGE_WARNING = 0.05
MIN_FRAME_COVERAGE_ERROR = 0.01
MAX_EDGE_RATIO_WARNING = 6.0
MAX_EDGE_RATIO_ERROR = 12.0
MIN_CORNER_SINE_WARNING = 0.08
MIN_CORNER_SINE_ERROR = 0.02
MIN_BOUNDING_BOX_FILL_WARNING = 0.15
MIN_BOUNDING_BOX_FILL_ERROR = 0.05


def render_calibration_frame(
    video_path: str | Path,
    out_path: str | Path,
    *,
    frame_index: int = 0,
    calibration: FieldCalibration | None = None,
) -> Path:
    cv2 = require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read frame {frame_index} from {video_path}")

    annotated = frame.copy()
    _draw_header(cv2, annotated, frame_index)
    if calibration:
        _draw_calibration_points(cv2, annotated, calibration)
    else:
        _draw_instruction_band(cv2, annotated)

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), annotated)
    return output


def calibration_quality_report(
    calibration: FieldCalibration,
    *,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> dict:
    image_polygon = calibration.image_points[:4]
    field_polygon = calibration.field_points[:4]
    errors = []
    projection_failure = None
    for image_point, field_point in zip(
        calibration.image_points,
        calibration.field_points,
        strict=False,
    ):
        try:
            projected = calibration.transform_point(image_point)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            projection_failure = str(exc)
            break
        errors.append(float(np.linalg.norm(np.array(projected) - np.array(field_point))))

    signed_area_px = _signed_polygon_area(image_polygon)
    polygon_area_px = abs(signed_area_px)
    edge_lengths_px = _edge_lengths(image_polygon)
    shortest_edge_px = min(edge_lengths_px, default=0.0)
    longest_edge_px = max(edge_lengths_px, default=0.0)
    edge_ratio = (
        longest_edge_px / shortest_edge_px if shortest_edge_px > 1e-9 else None
    )
    corner_sines = _corner_sines(image_polygon)
    min_corner_sine = min(corner_sines, default=0.0)
    bounding_box = _bounding_box_metrics(image_polygon)
    frame_area_px = (
        float(frame_width * frame_height)
        if frame_width is not None
        and frame_height is not None
        and frame_width > 0
        and frame_height > 0
        else None
    )
    frame_coverage = polygon_area_px / frame_area_px if frame_area_px else None
    is_convex = _is_strictly_convex(image_polygon)
    has_self_intersection = _has_self_intersection(image_polygon)
    orientation = _orientation(signed_area_px)
    field_orientation = _orientation(_signed_polygon_area(field_polygon))
    orientation_matches_field = (
        orientation == field_orientation
        if orientation != "degenerate" and field_orientation != "degenerate"
        else False
    )
    outside_points = _outside_image_points(
        calibration.image_points,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    issues = []
    max_error = max(errors) if errors else 0.0
    if projection_failure is not None:
        issues.append(
            {
                "severity": "error",
                "code": "invalid_homography",
                "message": f"Calibration homography could not project all points: {projection_failure}",
            }
        )
    if polygon_area_px <= 1.0:
        issues.append(
            {
                "severity": "error",
                "code": "degenerate_polygon",
                "message": "Calibration image points do not form a usable field polygon.",
            }
        )
    if shortest_edge_px <= 1e-9:
        issues.append(
            {
                "severity": "error",
                "code": "zero_length_edge",
                "message": "At least two consecutive calibration corners overlap.",
            }
        )
    if not is_convex or has_self_intersection:
        issues.append(
            {
                "severity": "error",
                "code": "invalid_corner_order",
                "message": (
                    "The first four image points must form a strictly convex polygon "
                    "in perimeter order."
                ),
            }
        )
    if not orientation_matches_field:
        issues.append(
            {
                "severity": "error",
                "code": "orientation_mismatch",
                "message": (
                    "Image corner orientation does not match field corner orientation; "
                    "check the TL, TR, BR, BL order."
                ),
            }
        )
    if frame_coverage is not None and frame_coverage < MIN_FRAME_COVERAGE_ERROR:
        issues.append(
            {
                "severity": "error",
                "code": "insufficient_frame_coverage",
                "message": (
                    f"Field polygon covers only {frame_coverage:.2%} of the frame; "
                    f"minimum usable coverage is {MIN_FRAME_COVERAGE_ERROR:.0%}."
                ),
            }
        )
    elif frame_coverage is not None and frame_coverage < MIN_FRAME_COVERAGE_WARNING:
        issues.append(
            {
                "severity": "warning",
                "code": "low_frame_coverage",
                "message": (
                    f"Field polygon covers only {frame_coverage:.2%} of the frame; "
                    f"review calibrations below {MIN_FRAME_COVERAGE_WARNING:.0%}."
                ),
            }
        )
    if edge_ratio is not None and edge_ratio > MAX_EDGE_RATIO_ERROR:
        issues.append(
            {
                "severity": "error",
                "code": "extreme_edge_ratio",
                "message": (
                    f"Longest polygon edge is {edge_ratio:.2f} times the shortest; "
                    "the quadrilateral is extremely skewed."
                ),
            }
        )
    elif edge_ratio is not None and edge_ratio > MAX_EDGE_RATIO_WARNING:
        issues.append(
            {
                "severity": "warning",
                "code": "high_edge_ratio",
                "message": (
                    f"Longest polygon edge is {edge_ratio:.2f} times the shortest; "
                    "verify the selected corners."
                ),
            }
        )
    if min_corner_sine < MIN_CORNER_SINE_ERROR:
        issues.append(
            {
                "severity": "error",
                "code": "near_collinear_corner",
                "message": (
                    f"Minimum normalized corner sine is {min_corner_sine:.4f}; "
                    "at least one corner is nearly collinear."
                ),
            }
        )
    elif min_corner_sine < MIN_CORNER_SINE_WARNING:
        issues.append(
            {
                "severity": "warning",
                "code": "shallow_corner_angle",
                "message": (
                    f"Minimum normalized corner sine is {min_corner_sine:.4f}; "
                    "verify the strongly compressed corner."
                ),
            }
        )
    bounding_box_fill = bounding_box["fill_ratio"]
    if bounding_box_fill < MIN_BOUNDING_BOX_FILL_ERROR:
        issues.append(
            {
                "severity": "error",
                "code": "extreme_polygon_skew",
                "message": (
                    f"Polygon fills only {bounding_box_fill:.2%} of its bounding box; "
                    "the calibration geometry is unstable."
                ),
            }
        )
    elif bounding_box_fill < MIN_BOUNDING_BOX_FILL_WARNING:
        issues.append(
            {
                "severity": "warning",
                "code": "high_polygon_skew",
                "message": (
                    f"Polygon fills only {bounding_box_fill:.2%} of its bounding box; "
                    "verify the calibration geometry."
                ),
            }
        )
    if max_error > 0.05:
        issues.append(
            {
                "severity": "warning",
                "code": "high_reprojection_error",
                "message": f"Max reprojection error is {max_error:.3f} m.",
            }
        )
    if outside_points:
        issues.append(
            {
                "severity": "error",
                "code": "points_outside_frame",
                "message": f"{len(outside_points)} calibration points are outside the frame.",
            }
        )

    return {
        "status": _quality_status(issues),
        "points": len(calibration.image_points),
        "frame": {
            "width": frame_width,
            "height": frame_height,
        },
        "field": {
            "length_m": calibration.field_length_m,
            "width_m": calibration.field_width_m,
            "aspect_ratio": calibration.field_length_m / calibration.field_width_m,
        },
        "image_polygon": {
            "area_px": polygon_area_px,
            "signed_area_px": signed_area_px,
            "orientation": orientation,
            "field_orientation": field_orientation,
            "orientation_matches_field": orientation_matches_field,
            "is_strictly_convex": is_convex,
            "has_self_intersection": has_self_intersection,
            "edge_lengths_px": edge_lengths_px,
            "shortest_edge_px": shortest_edge_px,
            "longest_edge_px": longest_edge_px,
            "max_to_min_edge_ratio": edge_ratio,
            "corner_sines": corner_sines,
            "min_corner_sine": min_corner_sine,
            "bounding_box": bounding_box,
            "frame_coverage_ratio": frame_coverage,
        },
        "geometry_thresholds": {
            "min_frame_coverage_warning": MIN_FRAME_COVERAGE_WARNING,
            "min_frame_coverage_error": MIN_FRAME_COVERAGE_ERROR,
            "max_edge_ratio_warning": MAX_EDGE_RATIO_WARNING,
            "max_edge_ratio_error": MAX_EDGE_RATIO_ERROR,
            "min_corner_sine_warning": MIN_CORNER_SINE_WARNING,
            "min_corner_sine_error": MIN_CORNER_SINE_ERROR,
            "min_bounding_box_fill_warning": MIN_BOUNDING_BOX_FILL_WARNING,
            "min_bounding_box_fill_error": MIN_BOUNDING_BOX_FILL_ERROR,
        },
        "reprojection_error_m": {
            "valid": projection_failure is None,
            "mean": float(np.mean(errors)) if errors else 0.0,
            "max": max_error,
            "samples": len(errors),
            "failure": projection_failure,
        },
        "outside_image_points": outside_points,
        "issues": issues,
    }


def write_calibration_quality(path: str | Path, report: dict) -> Path:
    write_json(path, report)
    return Path(path)


def _draw_header(cv2, frame, frame_index: int) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 58), (20, 20, 20), -1)
    cv2.putText(
        frame,
        f"Calibration frame {frame_index}",
        (18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )


def _draw_instruction_band(cv2, frame) -> None:
    text = "Click or record field corners in order: top_left, top_right, bottom_right, bottom_left"
    cv2.rectangle(
        frame,
        (0, frame.shape[0] - 52),
        (frame.shape[1], frame.shape[0]),
        (20, 20, 20),
        -1,
    )
    cv2.putText(
        frame,
        text,
        (18, frame.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
    )


def _draw_calibration_points(cv2, frame, calibration: FieldCalibration) -> None:
    labels = ["TL", "TR", "BR", "BL"]
    colors = [(0, 230, 255), (255, 190, 0), (70, 90, 255), (90, 220, 90)]
    points = [(int(round(x)), int(round(y))) for x, y in calibration.image_points[:4]]
    for idx, (point, label, color) in enumerate(zip(points, labels, colors, strict=False)):
        cv2.circle(frame, point, 18, color, -1)
        cv2.circle(frame, point, 22, (255, 255, 255), 2)
        cv2.putText(
            frame,
            f"{idx + 1}:{label}",
            (point[0] + 26, point[1] + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
    )
    if len(points) >= 4:
        cv2.polylines(frame, [np.array(points, dtype="int32")], True, (255, 255, 255), 2)


def _polygon_area(points: list[tuple[float, float]]) -> float:
    return abs(_signed_polygon_area(points))


def _signed_polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for first, second in zip(points, points[1:] + points[:1]):
        area += first[0] * second[1] - second[0] * first[1]
    return area / 2.0


def _edge_lengths(points: list[tuple[float, float]]) -> list[float]:
    if len(points) < 2:
        return []
    return [
        float(np.linalg.norm(np.array(first) - np.array(second)))
        for first, second in zip(points, points[1:] + points[:1])
    ]


def _corner_sines(points: list[tuple[float, float]]) -> list[float]:
    if len(points) < 3:
        return []
    values = []
    for index, current in enumerate(points):
        previous = np.array(points[index - 1], dtype=float) - np.array(current, dtype=float)
        following = np.array(points[(index + 1) % len(points)], dtype=float) - np.array(
            current,
            dtype=float,
        )
        denominator = float(np.linalg.norm(previous) * np.linalg.norm(following))
        if denominator <= 1e-9:
            values.append(0.0)
            continue
        values.append(abs(_cross_2d(previous, following)) / denominator)
    return values


def _is_strictly_convex(points: list[tuple[float, float]]) -> bool:
    if len(points) < 4:
        return False
    cross_products = []
    for index in range(len(points)):
        first = np.array(points[(index + 1) % len(points)], dtype=float) - np.array(
            points[index],
            dtype=float,
        )
        second = np.array(points[(index + 2) % len(points)], dtype=float) - np.array(
            points[(index + 1) % len(points)],
            dtype=float,
        )
        cross_products.append(_cross_2d(first, second))
    if any(abs(value) <= 1e-9 for value in cross_products):
        return False
    return all(value > 0 for value in cross_products) or all(
        value < 0 for value in cross_products
    )


def _has_self_intersection(points: list[tuple[float, float]]) -> bool:
    if len(points) != 4:
        return False
    return _segments_intersect(points[0], points[1], points[2], points[3]) or (
        _segments_intersect(points[1], points[2], points[3], points[0])
    )


def _segments_intersect(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> bool:
    def side(
        start: tuple[float, float],
        end: tuple[float, float],
        point: tuple[float, float],
    ) -> float:
        return (end[0] - start[0]) * (point[1] - start[1]) - (
            end[1] - start[1]
        ) * (point[0] - start[0])

    first_side = side(first_start, first_end, second_start)
    second_side = side(first_start, first_end, second_end)
    third_side = side(second_start, second_end, first_start)
    fourth_side = side(second_start, second_end, first_end)
    return first_side * second_side < 0 and third_side * fourth_side < 0


def _bounding_box_metrics(points: list[tuple[float, float]]) -> dict:
    if not points:
        return {
            "width_px": 0.0,
            "height_px": 0.0,
            "area_px": 0.0,
            "fill_ratio": 0.0,
        }
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    width = float(max(xs) - min(xs))
    height = float(max(ys) - min(ys))
    area = width * height
    polygon_area = _polygon_area(points)
    return {
        "width_px": width,
        "height_px": height,
        "area_px": area,
        "fill_ratio": polygon_area / area if area > 1e-9 else 0.0,
    }


def _orientation(signed_area: float) -> str:
    if signed_area > 1e-9:
        return "positive"
    if signed_area < -1e-9:
        return "negative"
    return "degenerate"


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _outside_image_points(
    points: list[tuple[float, float]],
    *,
    frame_width: int | None,
    frame_height: int | None,
) -> list[dict]:
    if frame_width is None or frame_height is None:
        return []
    outside = []
    for index, (x, y) in enumerate(points):
        if x < 0 or y < 0 or x > frame_width or y > frame_height:
            outside.append({"index": index, "x": x, "y": y})
    return outside


def _quality_status(issues: list[dict]) -> str:
    severities = {issue["severity"] for issue in issues}
    if "error" in severities:
        return "fail"
    if "warning" in severities:
        return "review"
    return "good"
