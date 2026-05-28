from __future__ import annotations

from pathlib import Path

import numpy as np

from .field_analysis import FieldCalibration
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
