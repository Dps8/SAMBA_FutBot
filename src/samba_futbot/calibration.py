from __future__ import annotations

from pathlib import Path

import numpy as np

from .field_analysis import FieldCalibration
from .io_utils import write_json
from .video import require_cv2


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
    errors = []
    for image_point, field_point in zip(
        calibration.image_points,
        calibration.field_points,
        strict=False,
    ):
        projected = calibration.transform_point(image_point)
        errors.append(float(np.linalg.norm(np.array(projected) - np.array(field_point))))

    polygon_area_px = _polygon_area(calibration.image_points[:4])
    edge_lengths_px = _edge_lengths(calibration.image_points[:4])
    outside_points = _outside_image_points(
        calibration.image_points,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    issues = []
    max_error = max(errors) if errors else 0.0
    if polygon_area_px <= 1.0:
        issues.append(
            {
                "severity": "error",
                "code": "degenerate_polygon",
                "message": "Calibration image points do not form a usable field polygon.",
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
            "edge_lengths_px": edge_lengths_px,
        },
        "reprojection_error_m": {
            "mean": float(np.mean(errors)) if errors else 0.0,
            "max": max_error,
            "samples": len(errors),
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
    if len(points) < 3:
        return 0.0
    area = 0.0
    for first, second in zip(points, points[1:] + points[:1]):
        area += first[0] * second[1] - second[0] * first[1]
    return abs(area) / 2.0


def _edge_lengths(points: list[tuple[float, float]]) -> list[float]:
    if len(points) < 2:
        return []
    return [
        float(np.linalg.norm(np.array(first) - np.array(second)))
        for first, second in zip(points, points[1:] + points[:1])
    ]


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
