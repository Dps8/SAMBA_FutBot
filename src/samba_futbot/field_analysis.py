from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

import numpy as np

from .config import load_config
from .io_utils import ensure_parent
from .play_state import in_play_balls
from .types import Detection, Point


DEFAULT_FIELD_LENGTH_M = 1.82
DEFAULT_FIELD_WIDTH_M = 1.22


@dataclass(slots=True)
class FieldCalibration:
    image_points: list[Point]
    field_points: list[Point]
    field_length_m: float = DEFAULT_FIELD_LENGTH_M
    field_width_m: float = DEFAULT_FIELD_WIDTH_M

    @classmethod
    def from_mapping(cls, data: dict) -> "FieldCalibration":
        field = data.get("field", {})
        length_m = float(field.get("length_m", data.get("field_length_m", DEFAULT_FIELD_LENGTH_M)))
        width_m = float(field.get("width_m", data.get("field_width_m", DEFAULT_FIELD_WIDTH_M)))
        image_points = _parse_points(data.get("image_points"), name="image_points")
        field_points = _parse_points(
            data.get("field_points"),
            name="field_points",
            default=[
                (0.0, 0.0),
                (length_m, 0.0),
                (length_m, width_m),
                (0.0, width_m),
            ],
        )
        if len(image_points) != len(field_points):
            raise ValueError("image_points and field_points must have the same length.")
        if len(image_points) < 4:
            raise ValueError("At least 4 point correspondences are required.")
        return cls(
            image_points=image_points,
            field_points=field_points,
            field_length_m=length_m,
            field_width_m=width_m,
        )

    @property
    def matrix(self) -> np.ndarray:
        return _homography_matrix(self.image_points, self.field_points)

    def transform_point(self, point: Point) -> Point:
        x, y = point
        vec = self.matrix @ np.array([x, y, 1.0], dtype=float)
        if abs(vec[2]) < 1e-9:
            raise ValueError(f"Point cannot be projected with this homography: {point}")
        return (float(vec[0] / vec[2]), float(vec[1] / vec[2]))

    def zone_for_point(self, point: Point, *, grid_cols: int, grid_rows: int) -> dict:
        x, y = point
        col = _bucket(x, self.field_length_m, grid_cols)
        row = _bucket(y, self.field_width_m, grid_rows)
        return {
            "row": row,
            "col": col,
            "zone": f"r{row + 1}c{col + 1}",
            "label": f"{_third_label(x, self.field_length_m)}_{_lane_label(y, self.field_width_m)}",
            "inside_field": 0.0 <= x <= self.field_length_m and 0.0 <= y <= self.field_width_m,
        }

    def to_record(self) -> dict:
        return {
            "field": {
                "length_m": self.field_length_m,
                "width_m": self.field_width_m,
            },
            "image_points": [list(point) for point in self.image_points],
            "field_points": [list(point) for point in self.field_points],
        }


def load_field_calibration(path: str | Path) -> FieldCalibration:
    data = load_config(path)
    if not isinstance(data, dict):
        raise ValueError(f"Calibration must be a mapping: {path}")
    return FieldCalibration.from_mapping(data)


def analyze_field_tracks(
    detections: Iterable[Detection],
    calibration: FieldCalibration,
    *,
    fps: float | None = None,
    possession_radius_px: float = 90.0,
    field_margin_px: float = 8.0,
    grid_cols: int = 6,
    grid_rows: int = 4,
) -> dict:
    if grid_cols <= 0 or grid_rows <= 0:
        raise ValueError("grid_cols and grid_rows must be positive.")

    balls = in_play_balls(
        detections,
        possession_radius_px=possession_radius_px,
        field_margin_px=field_margin_px,
    )
    path = _field_path_records(balls, calibration, grid_cols=grid_cols, grid_rows=grid_rows)
    speeds_m_s = _field_speeds(path, fps=fps)
    distance_m = _field_distance(path)
    zone_counts = Counter(record["zone"] for record in path)
    grid_counts = [[0 for _ in range(grid_cols)] for _ in range(grid_rows)]
    for record in path:
        grid_counts[int(record["row"])][int(record["col"])] += 1

    return {
        "trajectory_scope": "in_play_ball_field_coordinates",
        "calibration": calibration.to_record(),
        "grid": {
            "cols": grid_cols,
            "rows": grid_rows,
            "sample_counts": grid_counts,
            "seconds": _grid_seconds(grid_counts, fps=fps),
        },
        "summary": {
            "path_samples": len(path),
            "unique_zones": len(zone_counts),
            "distance_m": distance_m,
            "speed_samples": len(speeds_m_s),
            "mean_speed_m_s": mean(speeds_m_s) if speeds_m_s else 0.0,
            "max_speed_m_s": max(speeds_m_s) if speeds_m_s else 0.0,
        },
        "zones": [
            {
                "zone": zone,
                "samples": count,
                "seconds": count / fps if fps and fps > 0 else None,
            }
            for zone, count in sorted(zone_counts.items())
        ],
        "path": path,
    }


def write_field_trajectory_csv(path: str | Path, analysis: dict) -> None:
    output = ensure_parent(path)
    fields = [
        "frame_index",
        "track_id",
        "image_x",
        "image_y",
        "field_x_m",
        "field_y_m",
        "row",
        "col",
        "zone",
        "zone_label",
        "inside_field",
    ]
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for record in analysis.get("path", []):
            writer.writerow({field: record.get(field) for field in fields})


def _field_path_records(
    balls: Iterable[Detection],
    calibration: FieldCalibration,
    *,
    grid_cols: int,
    grid_rows: int,
) -> list[dict]:
    records: list[dict] = []
    for ball in sorted(balls, key=lambda det: (det.frame_index, det.track_id or -1, det.score)):
        field_point = calibration.transform_point(ball.centroid)
        zone = calibration.zone_for_point(field_point, grid_cols=grid_cols, grid_rows=grid_rows)
        records.append(
            {
                "frame_index": ball.frame_index,
                "track_id": ball.track_id,
                "image_x": ball.centroid[0],
                "image_y": ball.centroid[1],
                "field_x_m": field_point[0],
                "field_y_m": field_point[1],
                "row": zone["row"],
                "col": zone["col"],
                "zone": zone["zone"],
                "zone_label": zone["label"],
                "inside_field": zone["inside_field"],
            }
        )
    return records


def _field_speeds(path: list[dict], *, fps: float | None) -> list[float]:
    if not fps or fps <= 0:
        return []
    speeds: list[float] = []
    by_track: dict[int | None, list[dict]] = {}
    for record in path:
        by_track.setdefault(record["track_id"], []).append(record)
    for records in by_track.values():
        ordered = sorted(records, key=lambda item: item["frame_index"])
        for prev, current in zip(ordered, ordered[1:]):
            frame_delta = int(current["frame_index"]) - int(prev["frame_index"])
            if frame_delta <= 0:
                continue
            dist = math.hypot(
                current["field_x_m"] - prev["field_x_m"],
                current["field_y_m"] - prev["field_y_m"],
            )
            speeds.append(dist / (frame_delta / fps))
    return speeds


def _field_distance(path: list[dict]) -> float:
    distance_m = 0.0
    by_track: dict[int | None, list[dict]] = {}
    for record in path:
        by_track.setdefault(record["track_id"], []).append(record)
    for records in by_track.values():
        ordered = sorted(records, key=lambda item: item["frame_index"])
        for prev, current in zip(ordered, ordered[1:]):
            if int(current["frame_index"]) <= int(prev["frame_index"]):
                continue
            distance_m += math.hypot(
                current["field_x_m"] - prev["field_x_m"],
                current["field_y_m"] - prev["field_y_m"],
            )
    return distance_m


def _grid_seconds(grid_counts: list[list[int]], *, fps: float | None) -> list[list[float | None]]:
    if not fps or fps <= 0:
        return [[None for _ in row] for row in grid_counts]
    return [[count / fps for count in row] for row in grid_counts]


def _parse_points(
    value: object,
    *,
    name: str,
    default: list[Point] | None = None,
) -> list[Point]:
    if value is None:
        if default is None:
            raise ValueError(f"Missing {name}.")
        return default
    if isinstance(value, dict):
        order = ["top_left", "top_right", "bottom_right", "bottom_left"]
        try:
            value = [value[key] for key in order]
        except KeyError as exc:
            raise ValueError(f"{name} dictionary must contain {order}.") from exc
    points = []
    for point in value:
        if len(point) != 2:
            raise ValueError(f"{name} point must have exactly two values: {point}")
        points.append((float(point[0]), float(point[1])))
    return points


def _homography_matrix(image_points: list[Point], field_points: list[Point]) -> np.ndarray:
    rows: list[list[float]] = []
    values: list[float] = []
    for (x, y), (u, v) in zip(image_points, field_points):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        values.append(u)
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        values.append(v)
    solution, *_ = np.linalg.lstsq(
        np.array(rows, dtype=float),
        np.array(values, dtype=float),
        rcond=None,
    )
    h11, h12, h13, h21, h22, h23, h31, h32 = solution
    return np.array(
        [
            [h11, h12, h13],
            [h21, h22, h23],
            [h31, h32, 1.0],
        ],
        dtype=float,
    )


def _bucket(value: float, limit: float, buckets: int) -> int:
    if limit <= 0:
        return 0
    index = int((value / limit) * buckets)
    return max(0, min(buckets - 1, index))


def _third_label(value: float, limit: float) -> str:
    ratio = value / limit if limit > 0 else 0.0
    if ratio < 1 / 3:
        return "defensive"
    if ratio < 2 / 3:
        return "middle"
    return "attacking"


def _lane_label(value: float, limit: float) -> str:
    ratio = value / limit if limit > 0 else 0.0
    if ratio < 1 / 3:
        return "left"
    if ratio < 2 / 3:
        return "center"
    return "right"
