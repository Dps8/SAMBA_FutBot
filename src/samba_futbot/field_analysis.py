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
from .play_state import BALL_CLASSES, ROBOT_CLASSES, in_play_balls
from .types import Detection, Point


DEFAULT_FIELD_LENGTH_M = 2.43
DEFAULT_FIELD_WIDTH_M = 1.82
DEFAULT_CENTER_CIRCLE_DIAMETER_M = 0.60
DEFAULT_PENALTY_AREA_DEPTH_M = 0.25
DEFAULT_PENALTY_AREA_WIDTH_M = 0.80
DEFAULT_GOAL_WIDTH_M = 0.60
DEFAULT_GOAL_DEPTH_M = 0.10


@dataclass(slots=True)
class FieldCalibration:
    image_points: list[Point]
    field_points: list[Point]
    field_length_m: float = DEFAULT_FIELD_LENGTH_M
    field_width_m: float = DEFAULT_FIELD_WIDTH_M
    center_circle_diameter_m: float = DEFAULT_CENTER_CIRCLE_DIAMETER_M
    penalty_area_depth_m: float = DEFAULT_PENALTY_AREA_DEPTH_M
    penalty_area_width_m: float = DEFAULT_PENALTY_AREA_WIDTH_M
    goal_width_m: float = DEFAULT_GOAL_WIDTH_M
    goal_depth_m: float = DEFAULT_GOAL_DEPTH_M

    @classmethod
    def from_mapping(cls, data: dict) -> "FieldCalibration":
        field = data.get("field", {})
        length_m = float(field.get("length_m", data.get("field_length_m", DEFAULT_FIELD_LENGTH_M)))
        width_m = float(field.get("width_m", data.get("field_width_m", DEFAULT_FIELD_WIDTH_M)))
        center_circle_diameter_m = float(
            field.get("center_circle_diameter_m", DEFAULT_CENTER_CIRCLE_DIAMETER_M)
        )
        penalty_area_depth_m = float(
            field.get("penalty_area_depth_m", DEFAULT_PENALTY_AREA_DEPTH_M)
        )
        penalty_area_width_m = float(
            field.get("penalty_area_width_m", DEFAULT_PENALTY_AREA_WIDTH_M)
        )
        goal_width_m = float(field.get("goal_width_m", DEFAULT_GOAL_WIDTH_M))
        goal_depth_m = float(field.get("goal_depth_m", DEFAULT_GOAL_DEPTH_M))
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
            center_circle_diameter_m=center_circle_diameter_m,
            penalty_area_depth_m=penalty_area_depth_m,
            penalty_area_width_m=penalty_area_width_m,
            goal_width_m=goal_width_m,
            goal_depth_m=goal_depth_m,
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

    def neutral_points(self) -> list[Point]:
        half = self.field_length_m / 2
        offset = 0.45
        return [
            (half, offset),
            (half, self.field_width_m - offset),
        ]

    def penalty_areas(self) -> dict[str, tuple[float, float, float, float]]:
        y1 = (self.field_width_m - self.penalty_area_width_m) / 2
        y2 = y1 + self.penalty_area_width_m
        return {
            "left": (0.0, y1, self.penalty_area_depth_m, y2),
            "right": (
                self.field_length_m - self.penalty_area_depth_m,
                y1,
                self.field_length_m,
                y2,
            ),
        }

    def goal_mouths(self) -> dict[str, tuple[float, float, float, float]]:
        y1 = (self.field_width_m - self.goal_width_m) / 2
        y2 = y1 + self.goal_width_m
        return {
            "left": (-self.goal_depth_m, y1, 0.0, y2),
            "right": (self.field_length_m, y1, self.field_length_m + self.goal_depth_m, y2),
        }

    def rule_geometry(self) -> dict:
        return {
            "neutral_points": [list(point) for point in self.neutral_points()],
            "penalty_areas": {
                name: list(box) for name, box in self.penalty_areas().items()
            },
            "goal_mouths": {
                name: list(box) for name, box in self.goal_mouths().items()
            },
        }

    def to_record(self) -> dict:
        return {
            "field": {
                "length_m": self.field_length_m,
                "width_m": self.field_width_m,
                "center_circle_diameter_m": self.center_circle_diameter_m,
                "penalty_area_depth_m": self.penalty_area_depth_m,
                "penalty_area_width_m": self.penalty_area_width_m,
                "goal_width_m": self.goal_width_m,
                "goal_depth_m": self.goal_depth_m,
            },
            "rule_geometry": self.rule_geometry(),
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
    robot_anchor: str = "bottom_center",
) -> dict:
    if grid_cols <= 0 or grid_rows <= 0:
        raise ValueError("grid_cols and grid_rows must be positive.")

    detections_list = list(detections)
    balls = in_play_balls(
        detections_list,
        possession_radius_px=possession_radius_px,
        field_margin_px=field_margin_px,
    )
    path = _field_path_records(balls, calibration, grid_cols=grid_cols, grid_rows=grid_rows)
    robot_path = _robot_path_records(
        detections_list,
        calibration,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        anchor=robot_anchor,
    )
    speeds_m_s = _field_speeds(path, fps=fps)
    distance_m = _field_distance(path)
    zone_counts = Counter(record["zone"] for record in path)
    grid_counts = [[0 for _ in range(grid_cols)] for _ in range(grid_rows)]
    for record in path:
        grid_counts[int(record["row"])][int(record["col"])] += 1
    robot_zone_control = _robot_zone_control(robot_path)

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
            "goal_zone_entries": _count_goal_zone_entries(path, calibration),
            "ball_out_of_bounds_samples": sum(
                1 for record in path if not record["inside_field"]
            ),
            "robot_penalty_area_samples": _count_robot_penalty_samples(
                robot_path,
                calibration,
            ),
        },
        "robot_summary": _robot_summary(robot_path, calibration),
        "robot_zone_control": robot_zone_control,
        "zones": [
            {
                "zone": zone,
                "samples": count,
                "seconds": count / fps if fps and fps > 0 else None,
            }
            for zone, count in sorted(zone_counts.items())
        ],
        "path": path,
        "robot_path": robot_path,
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


def write_field_robot_csv(path: str | Path, analysis: dict) -> None:
    output = ensure_parent(path)
    fields = [
        "frame_index",
        "track_id",
        "class_name",
        "team",
        "image_x",
        "image_y",
        "field_x_m",
        "field_y_m",
        "row",
        "col",
        "zone",
        "zone_label",
        "inside_field",
        "in_penalty_area",
        "penalty_side",
    ]
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for record in analysis.get("robot_path", []):
            writer.writerow({field: record.get(field) for field in fields})


def write_field_zone_control_csv(path: str | Path, analysis: dict) -> None:
    output = ensure_parent(path)
    fields = [
        "zone",
        "zone_label",
        "row",
        "col",
        "samples",
        "leader",
        "leader_margin",
        "leader_ratio",
        "samples_by_team",
    ]
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for record in analysis.get("robot_zone_control", []):
            row = {field: record.get(field) for field in fields}
            row["samples_by_team"] = _compact_counter(record.get("samples_by_team", {}))
            writer.writerow(row)


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


def _robot_path_records(
    detections: Iterable[Detection],
    calibration: FieldCalibration,
    *,
    grid_cols: int,
    grid_rows: int,
    anchor: str,
) -> list[dict]:
    if anchor not in {"centroid", "bottom_center"}:
        raise ValueError("robot_anchor must be 'centroid' or 'bottom_center'.")
    records: list[dict] = []
    for robot in sorted(
        (det for det in detections if det.class_name in ROBOT_CLASSES),
        key=lambda det: (det.frame_index, det.track_id or -1, det.score),
    ):
        image_point = _robot_anchor_point(robot, anchor)
        field_point = calibration.transform_point(image_point)
        zone = calibration.zone_for_point(field_point, grid_cols=grid_cols, grid_rows=grid_rows)
        penalty_side = _penalty_side(field_point, calibration)
        records.append(
            {
                "frame_index": robot.frame_index,
                "track_id": robot.track_id,
                "class_name": robot.class_name,
                "team": robot.team,
                "image_x": image_point[0],
                "image_y": image_point[1],
                "field_x_m": field_point[0],
                "field_y_m": field_point[1],
                "row": zone["row"],
                "col": zone["col"],
                "zone": zone["zone"],
                "zone_label": zone["label"],
                "inside_field": zone["inside_field"],
                "in_penalty_area": penalty_side is not None,
                "penalty_side": penalty_side,
            }
        )
    return records


def _robot_anchor_point(robot: Detection, anchor: str) -> Point:
    if anchor == "centroid":
        return robot.centroid
    x1, _, x2, y2 = robot.box
    return ((x1 + x2) / 2.0, y2)


def _penalty_side(point: Point, calibration: FieldCalibration) -> str | None:
    for side, box in calibration.penalty_areas().items():
        if _point_in_field_box(point, box):
            return side
    return None


def _point_in_field_box(point: Point, box: tuple[float, float, float, float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def _count_goal_zone_entries(path: list[dict], calibration: FieldCalibration) -> int:
    entries = 0
    prev_side_by_track: dict[int | None, str | None] = {}
    for record in sorted(path, key=lambda item: (item["track_id"] or -1, item["frame_index"])):
        point = (record["field_x_m"], record["field_y_m"])
        side = _goal_side(point, calibration)
        key = record["track_id"]
        if side and prev_side_by_track.get(key) != side:
            entries += 1
        prev_side_by_track[key] = side
    return entries


def _goal_side(point: Point, calibration: FieldCalibration) -> str | None:
    for side, box in calibration.goal_mouths().items():
        if _point_in_field_box(point, box):
            return side
    return None


def _count_robot_penalty_samples(
    robot_path: list[dict],
    calibration: FieldCalibration,
) -> int:
    return sum(
        1
        for record in robot_path
        if _penalty_side((record["field_x_m"], record["field_y_m"]), calibration)
    )


def _robot_summary(robot_path: list[dict], calibration: FieldCalibration) -> dict:
    by_side = Counter(record["penalty_side"] for record in robot_path if record["penalty_side"])
    by_team = Counter(record["team"] or "unknown" for record in robot_path)
    team_zone_counts: dict[str, Counter] = {}
    team_penalty_counts: dict[str, Counter] = {}
    team_phase_counts: dict[str, Counter] = {}
    for record in robot_path:
        team = record["team"] or "unknown"
        team_zone_counts.setdefault(team, Counter())[record["zone"]] += 1
        phase = _team_phase_label(record["field_x_m"], calibration.field_length_m, team)
        team_phase_counts.setdefault(team, Counter())[phase] += 1
        if record["penalty_side"]:
            team_penalty_counts.setdefault(team, Counter())[record["penalty_side"]] += 1
    return {
        "path_samples": len(robot_path),
        "inside_field_samples": sum(1 for record in robot_path if record["inside_field"]),
        "penalty_area_samples": sum(by_side.values()),
        "penalty_area_samples_by_side": dict(sorted(by_side.items())),
        "samples_by_team": dict(sorted(by_team.items())),
        "zone_samples_by_team": {
            team: dict(sorted(counts.items())) for team, counts in sorted(team_zone_counts.items())
        },
        "phase_samples_by_team": {
            team: dict(sorted(counts.items())) for team, counts in sorted(team_phase_counts.items())
        },
        "phase_ratios_by_team": {
            team: _phase_ratios(counts) for team, counts in sorted(team_phase_counts.items())
        },
        "attacking_pressure_by_team": {
            team: _phase_ratio(counts, "attacking")
            for team, counts in sorted(team_phase_counts.items())
        },
        "penalty_area_samples_by_team": {
            team: dict(sorted(counts.items())) for team, counts in sorted(team_penalty_counts.items())
        },
        "neutral_points": [list(point) for point in calibration.neutral_points()],
    }


def _robot_zone_control(robot_path: list[dict]) -> list[dict]:
    by_zone: dict[str, list[dict]] = {}
    for record in robot_path:
        if not record["inside_field"]:
            continue
        by_zone.setdefault(record["zone"], []).append(record)

    zones = []
    for zone, records in sorted(by_zone.items()):
        counts = Counter(record["team"] or "unknown" for record in records)
        total = sum(counts.values())
        ranked = counts.most_common()
        leader = ranked[0][0] if ranked else "none"
        leader_count = ranked[0][1] if ranked else 0
        runner_up = ranked[1][1] if len(ranked) > 1 else 0
        first = records[0]
        zones.append(
            {
                "zone": zone,
                "zone_label": first["zone_label"],
                "row": first["row"],
                "col": first["col"],
                "samples": total,
                "samples_by_team": dict(sorted(counts.items())),
                "leader": leader,
                "leader_margin": leader_count - runner_up,
                "leader_ratio": leader_count / total if total else 0.0,
            }
        )
    return zones


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


def _team_phase_label(value: float, limit: float, team: str) -> str:
    label = _third_label(value, limit)
    if team == "blue":
        if label == "defensive":
            return "attacking"
        if label == "attacking":
            return "defensive"
    return label


def _phase_ratios(counts: Counter) -> dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {
        phase: counts.get(phase, 0) / total
        for phase in ("defensive", "middle", "attacking")
    }


def _phase_ratio(counts: Counter, phase: str) -> float:
    ratios = _phase_ratios(counts)
    return ratios.get(phase, 0.0)


def _compact_counter(values: object) -> str:
    if not isinstance(values, dict) or not values:
        return ""
    return ";".join(f"{key}:{value}" for key, value in sorted(values.items()))


def _lane_label(value: float, limit: float) -> str:
    ratio = value / limit if limit > 0 else 0.0
    if ratio < 1 / 3:
        return "left"
    if ratio < 2 / 3:
        return "center"
    return "right"
