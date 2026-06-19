from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable


def predict_robot_trajectories(
    robot_path: Iterable[dict],
    *,
    fps: float,
    field_length_m: float,
    field_width_m: float,
    reference_frame: int | None = None,
    horizon_s: float = 1.5,
    step_s: float = 0.25,
    history_frames: int = 18,
    max_age_frames: int = 12,
    min_samples: int = 6,
    turn_rate_deg_s: float = 38.0,
    residual_scale_m: float = 0.08,
    max_robots: int = 4,
    min_robot_separation_m: float = 0.18,
) -> list[dict]:
    """Forecast plausible robot motion branches from measured field coordinates.

    Branch probabilities are normalized heuristic weights. They express relative
    plausibility under this kinematic model; they are not calibrated frequencies.
    """
    if fps <= 0:
        raise ValueError("fps must be positive.")
    if field_length_m <= 0 or field_width_m <= 0:
        raise ValueError("field dimensions must be positive.")
    if horizon_s <= 0 or step_s <= 0:
        raise ValueError("horizon_s and step_s must be positive.")
    if min_samples < 2 or history_frames < 1 or max_age_frames < 0:
        raise ValueError("invalid history or age limits.")

    records = [record for record in robot_path if record.get("track_id") is not None]
    if not records:
        return []
    resolved_frame = (
        max(int(record["frame_index"]) for record in records)
        if reference_frame is None
        else int(reference_frame)
    )
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for record in records:
        frame_index = int(record["frame_index"])
        if frame_index <= resolved_frame:
            grouped[(int(record["track_id"]), str(record.get("team") or "unknown"))].append(
                record
            )

    forecasts = []
    for (track_id, team), track_records in grouped.items():
        history = _recent_unique_records(
            track_records,
            reference_frame=resolved_frame,
            history_frames=history_frames,
        )
        if len(history) < min_samples:
            continue
        latest_frame = int(history[-1]["frame_index"])
        if resolved_frame - latest_frame > max_age_frames:
            continue
        fit = _linear_motion_fit(history, fps=fps)
        branches = _motion_branches(
            fit,
            horizon_s=horizon_s,
            step_s=step_s,
            turn_rate_deg_s=turn_rate_deg_s,
            field_length_m=field_length_m,
            field_width_m=field_width_m,
            residual_scale_m=residual_scale_m,
        )
        forecasts.append(
            {
                "track_id": track_id,
                "team": team,
                "reference_frame": latest_frame,
                "history": {
                    "samples": len(history),
                    "duration_s": fit["duration_s"],
                    "fit_rmse_m": fit["fit_rmse_m"],
                },
                "state": {
                    "field_x_m": fit["x"],
                    "field_y_m": fit["y"],
                    "velocity_x_m_s": fit["vx"],
                    "velocity_y_m_s": fit["vy"],
                    "speed_m_s": fit["speed"],
                },
                "horizon_s": horizon_s,
                "probability_model": "heuristic_kinematic_v1",
                "probability_note": (
                    "Relative branch weights from motion-fit residual and field constraints; "
                    "not statistically calibrated."
                ),
                "trajectories": branches,
            }
        )
    forecasts.sort(
        key=lambda forecast: (
            float(forecast["state"]["speed_m_s"]),
            int(forecast["history"]["samples"]),
        ),
        reverse=True,
    )
    selected = []
    for forecast in forecasts:
        state = forecast["state"]
        if any(
            math.hypot(
                float(state["field_x_m"]) - float(other["state"]["field_x_m"]),
                float(state["field_y_m"]) - float(other["state"]["field_y_m"]),
            )
            < min_robot_separation_m
            for other in selected
        ):
            continue
        selected.append(forecast)
        if len(selected) >= max_robots:
            break
    return selected


def select_robot_prediction_snapshot(
    robot_path: Iterable[dict],
    *,
    fps: float,
    field_length_m: float,
    field_width_m: float,
    max_robots: int = 2,
    sample_stride_frames: int = 5,
) -> dict:
    """Select an observed moment with informative, spatially distinct robot motion."""
    records = list(robot_path)
    frames = sorted({int(record["frame_index"]) for record in records})
    if not frames:
        return {
            "reference_frame": None,
            "selection": "maximum_sum_speed_with_spatial_deduplication",
            "forecasts": [],
        }
    eligible_frames = [frame for frame in frames if frame - frames[0] >= 18]
    best_frame = None
    best_forecasts: list[dict] = []
    best_score = -1.0
    for reference_frame in eligible_frames[:: max(1, sample_stride_frames)]:
        forecasts = predict_robot_trajectories(
            records,
            fps=fps,
            field_length_m=field_length_m,
            field_width_m=field_width_m,
            reference_frame=reference_frame,
            max_age_frames=max(3, sample_stride_frames),
            max_robots=max_robots,
        )
        if not forecasts:
            continue
        score = sum(float(forecast["state"]["speed_m_s"]) for forecast in forecasts)
        score *= 0.8 + 0.2 * min(1.0, len(forecasts) / max(1, max_robots))
        if score > best_score:
            best_frame = reference_frame
            best_forecasts = forecasts
            best_score = score
    return {
        "reference_frame": best_frame,
        "selection": "maximum_sum_speed_with_spatial_deduplication",
        "forecasts": best_forecasts,
    }


def _recent_unique_records(
    records: Iterable[dict],
    *,
    reference_frame: int,
    history_frames: int,
) -> list[dict]:
    by_frame: dict[int, dict] = {}
    for record in records:
        frame_index = int(record["frame_index"])
        if reference_frame - history_frames <= frame_index <= reference_frame:
            by_frame[frame_index] = record
    return [by_frame[frame] for frame in sorted(by_frame)]


def _linear_motion_fit(records: list[dict], *, fps: float) -> dict[str, float]:
    final_frame = int(records[-1]["frame_index"])
    times = [(int(record["frame_index"]) - final_frame) / fps for record in records]
    xs = [float(record["field_x_m"]) for record in records]
    ys = [float(record["field_y_m"]) for record in records]
    mean_t = sum(times) / len(times)
    denominator = sum((time - mean_t) ** 2 for time in times)
    if denominator <= 1e-12:
        vx = vy = 0.0
    else:
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        vx = sum((time - mean_t) * (x - mean_x) for time, x in zip(times, xs)) / denominator
        vy = sum((time - mean_t) * (y - mean_y) for time, y in zip(times, ys)) / denominator
    x0 = sum(x - vx * time for time, x in zip(times, xs)) / len(xs)
    y0 = sum(y - vy * time for time, y in zip(times, ys)) / len(ys)
    residuals = [
        math.hypot(x - (x0 + vx * time), y - (y0 + vy * time))
        for time, x, y in zip(times, xs, ys)
    ]
    return {
        "x": xs[-1],
        "y": ys[-1],
        "vx": vx,
        "vy": vy,
        "speed": math.hypot(vx, vy),
        "fit_rmse_m": math.sqrt(sum(value * value for value in residuals) / len(residuals)),
        "duration_s": max(times) - min(times),
    }


def _motion_branches(
    fit: dict[str, float],
    *,
    horizon_s: float,
    step_s: float,
    turn_rate_deg_s: float,
    field_length_m: float,
    field_width_m: float,
    residual_scale_m: float,
) -> list[dict]:
    consistency = math.exp(-fit["fit_rmse_m"] / max(1e-6, residual_scale_m))
    straight_weight = 0.48 + 0.32 * consistency
    turn_weight = (1.0 - straight_weight) / 2.0
    candidates = [
        ("continue", 0.0, straight_weight),
        ("turn_left", abs(turn_rate_deg_s), turn_weight),
        ("turn_right", -abs(turn_rate_deg_s), turn_weight),
    ]
    branches = []
    weighted_total = 0.0
    for mode, turn_rate, base_weight in candidates:
        points = _integrate_branch(
            fit,
            horizon_s=horizon_s,
            step_s=step_s,
            turn_rate_deg_s=turn_rate,
            field_length_m=field_length_m,
            field_width_m=field_width_m,
        )
        inside_ratio = sum(point["inside_field"] for point in points) / max(1, len(points))
        weight = base_weight * max(0.08, inside_ratio**2)
        weighted_total += weight
        branches.append(
            {
                "mode": mode,
                "probability": weight,
                "turn_rate_deg_s": turn_rate,
                "points": points,
            }
        )
    for branch in branches:
        branch["probability"] = branch["probability"] / max(1e-12, weighted_total)
    return branches


def _integrate_branch(
    fit: dict[str, float],
    *,
    horizon_s: float,
    step_s: float,
    turn_rate_deg_s: float,
    field_length_m: float,
    field_width_m: float,
) -> list[dict]:
    speed = fit["speed"]
    heading = math.atan2(fit["vy"], fit["vx"]) if speed > 1e-6 else 0.0
    omega = math.radians(turn_rate_deg_s)
    steps = max(1, int(math.ceil(horizon_s / step_s)))
    times = [min(horizon_s, index * step_s) for index in range(steps + 1)]
    points = []
    for time_s in times:
        if abs(omega) < 1e-9 or speed <= 1e-6:
            x = fit["x"] + fit["vx"] * time_s
            y = fit["y"] + fit["vy"] * time_s
        else:
            radius = speed / omega
            x = fit["x"] + radius * (math.sin(heading + omega * time_s) - math.sin(heading))
            y = fit["y"] - radius * (math.cos(heading + omega * time_s) - math.cos(heading))
        inside = 0.0 <= x <= field_length_m and 0.0 <= y <= field_width_m
        points.append(
            {
                "t_s": time_s,
                "field_x_m": min(field_length_m, max(0.0, x)),
                "field_y_m": min(field_width_m, max(0.0, y)),
                "inside_field": inside,
            }
        )
    return points
