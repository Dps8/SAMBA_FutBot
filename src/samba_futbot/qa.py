from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import ensure_parent, read_json, write_json


DEFAULT_THRESHOLDS = {
    "min_ball_coverage": 0.75,
    "fail_ball_coverage": 0.30,
    "min_field_coverage": 0.90,
    "fail_field_coverage": 0.50,
    "min_robot_coverage": 0.50,
    "max_ball_jump_px_frame": 45.0,
    "fail_ball_jump_px_frame": 120.0,
    "max_out_of_bounds_ratio": 0.05,
    "fail_out_of_bounds_ratio": 0.20,
    "max_unknown_team_ratio": 0.35,
    "fail_unknown_team_ratio": 0.75,
    "min_possession_coverage": 0.30,
    "min_field_path_samples": 10,
}


def evaluate_run_quality(
    *,
    metrics_path: str | Path | None = None,
    events_path: str | Path | None = None,
    field_analysis_path: str | Path | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict:
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    metrics = _read_mapping(metrics_path)
    events = _read_list(events_path)
    field = _read_mapping(field_analysis_path)

    summary = _summary(metrics, events, field)
    issues = _issues(summary, limits)
    status = _status(issues)
    score = _quality_score(issues)
    claim_readiness = _claim_readiness(summary, limits)

    return {
        "status": status,
        "quality_score": score,
        "summary": summary,
        "claim_readiness": claim_readiness,
        "issues": issues,
        "thresholds": limits,
        "inputs": {
            "metrics": str(metrics_path) if metrics_path else None,
            "events": str(events_path) if events_path else None,
            "field_analysis": str(field_analysis_path) if field_analysis_path else None,
        },
    }


def write_quality_json(path: str | Path, report: dict) -> Path:
    write_json(path, report)
    return Path(path)


def write_quality_markdown(path: str | Path, report: dict) -> Path:
    summary = report.get("summary", {})
    issues = report.get("issues", [])
    lines = [
        "# Run QA",
        "",
        f"- Status: `{report.get('status', 'unknown')}`",
        f"- Quality score: `{report.get('quality_score', 0)}`",
        f"- Frames observed: `{summary.get('frames_observed', 0)}`",
        f"- Ball coverage: `{summary.get('ball_in_play_coverage_ratio', 0.0):.1%}`",
        f"- Max ball jump: `{summary.get('max_ball_speed_px_frame', 0.0):.1f} px/frame`",
        f"- Field coverage: `{summary.get('field_coverage_ratio', 0.0):.1%}`",
        f"- Robot coverage: `{summary.get('robot_coverage_ratio', 0.0):.1%}`",
        f"- Field path samples: `{summary.get('field_path_samples', 0)}`",
        f"- Ball out-of-bounds ratio: `{summary.get('ball_out_of_bounds_ratio', 0.0):.1%}`",
        f"- Robot penalty-area samples: `{summary.get('robot_penalty_area_samples', 0)}`",
        f"- Unknown-team robot ratio: `{summary.get('unknown_team_ratio', 0.0):.1%}`",
        f"- Possession coverage: `{summary.get('possession_coverage_ratio', 0.0):.1%}`",
        "",
        "## Claim Readiness",
        "",
    ]
    readiness = report.get("claim_readiness", {})
    if readiness:
        for claim, values in sorted(readiness.items()):
            status = values.get("status", "unknown") if isinstance(values, dict) else "unknown"
            reason = values.get("reason", "") if isinstance(values, dict) else ""
            lines.append(f"- `{claim}`: `{status}` - {reason}")
    else:
        lines.append("- No claim-readiness data was generated.")
    lines.extend(
        [
            "",
            "## Issues",
            "",
        ]
    )
    if issues:
        for issue in issues:
            lines.append(
                f"- `{issue.get('severity', 'info')}` `{issue.get('code', 'unknown')}`: "
                f"{issue.get('message', '')}"
            )
    else:
        lines.append("- No automatic QA issues were detected.")
    lines.extend(["", "## Event Counts", ""])
    event_counts = summary.get("event_counts", {})
    if event_counts:
        for event_type, count in sorted(event_counts.items()):
            lines.append(f"- `{event_type}`: `{count}`")
    else:
        lines.append("- No events file was provided or no events were detected.")

    output = ensure_parent(path)
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output


def collect_quality_reports(root: str | Path, *, pattern: str = "*.json") -> list[dict]:
    base = Path(root)
    reports = []
    for path in sorted(base.rglob(pattern)):
        if not path.is_file():
            continue
        try:
            data = read_json(path)
        except Exception:
            continue
        if not isinstance(data, dict) or "quality_score" not in data or "status" not in data:
            continue
        reports.append(_quality_index_record(path, data, base))
    return sorted(
        reports,
        key=lambda item: (
            _status_rank(str(item.get("status", "unknown"))),
            -int(item.get("quality_score", 0)),
            str(item.get("path", "")),
        ),
    )


def write_quality_index_json(path: str | Path, reports: list[dict]) -> Path:
    write_json(path, {"runs": reports, "total": len(reports)})
    return Path(path)


def write_quality_index_markdown(path: str | Path, reports: list[dict]) -> Path:
    lines = [
        "# QA Run Index",
        "",
        "| Rank | Status | Score | Ball coverage | Max jump | Unknown teams | Ready claims | Path |",
        "|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for index, report in enumerate(reports, start=1):
        summary = report.get("summary", {})
        lines.append(
            "| "
            f"{index} | "
            f"`{report.get('status', 'unknown')}` | "
            f"{int(report.get('quality_score', 0))} | "
            f"{float(summary.get('ball_in_play_coverage_ratio', 0.0)):.1%} | "
            f"{float(summary.get('max_ball_speed_px_frame', 0.0)):.1f} | "
            f"{float(summary.get('unknown_team_ratio', 0.0)):.1%} | "
            f"`{_ready_claims(report.get('claim_readiness', {}))}` | "
            f"`{report.get('path', '')}` |"
        )
    if not reports:
        lines.append("| 0 | `none` | 0 | 0.0% | 0.0 | 0.0% | `none` | `No QA reports found` |")
    output = ensure_parent(path)
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output


def _summary(metrics: dict[str, Any], events: list[Any], field: dict[str, Any]) -> dict:
    classes = metrics.get("classes", {})
    ball = classes.get("ball", {})
    field_class = classes.get("field", {})
    robots = classes.get("robots", {})
    motion = metrics.get("motion", {}).get("ball", {})
    field_summary = field.get("summary", {})
    robot_summary = field.get("robot_summary", {})
    robot_samples_by_team = (
        robot_summary.get("samples_by_team", {})
        if isinstance(robot_summary.get("samples_by_team", {}), dict)
        else {}
    )
    robot_sample_total = sum(int(_number(value)) for value in robot_samples_by_team.values())
    unknown_team_samples = int(_number(robot_samples_by_team.get("unknown")))
    path_samples = _number(field_summary.get("path_samples"))
    out_of_bounds = _number(field_summary.get("ball_out_of_bounds_samples"))
    event_counts: dict[str, int] = {}
    for event in events:
        if isinstance(event, dict):
            event_type = str(event.get("event_type", "unknown"))
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

    return {
        "frames_observed": int(_number(metrics.get("frames_observed"))),
        "detections": int(_number(metrics.get("detections"))),
        "tracks": int(_number(metrics.get("tracks"))),
        "ball_in_play_coverage_ratio": _number(ball.get("in_play_coverage_ratio")),
        "ball_frame_coverage_ratio": _number(ball.get("frame_coverage_ratio")),
        "field_coverage_ratio": _number(field_class.get("frame_coverage_ratio")),
        "robot_coverage_ratio": _number(robots.get("frame_coverage_ratio")),
        "ball_track_fragmentation_gaps": int(_number(ball.get("track_fragmentation_gaps"))),
        "max_ball_speed_px_frame": _number(motion.get("max_speed_px_frame")),
        "max_ball_speed_px_second": _optional_number(motion.get("max_speed_px_second")),
        "field_path_samples": int(path_samples),
        "field_distance_m": _number(field_summary.get("distance_m")),
        "mean_speed_m_s": _number(field_summary.get("mean_speed_m_s")),
        "max_speed_m_s": _number(field_summary.get("max_speed_m_s")),
        "goal_zone_entries": int(_number(field_summary.get("goal_zone_entries"))),
        "ball_out_of_bounds_samples": int(out_of_bounds),
        "ball_out_of_bounds_ratio": out_of_bounds / path_samples if path_samples else 0.0,
        "robot_penalty_area_samples": int(_number(robot_summary.get("penalty_area_samples"))),
        "robot_samples_by_team": robot_samples_by_team,
        "unknown_team_ratio": (
            unknown_team_samples / robot_sample_total if robot_sample_total else 0.0
        ),
        "possession_coverage_ratio": _number(
            metrics.get("possession", {}).get("coverage_ratio", 0.0)
            if isinstance(metrics.get("possession", {}), dict)
            else 0.0
        ),
        "possession_frames": int(
            _number(
                metrics.get("possession", {}).get("frames_with_possession", 0)
                if isinstance(metrics.get("possession", {}), dict)
                else 0
            )
        ),
        "event_counts": event_counts,
    }


def _claim_readiness(summary: dict[str, Any], limits: dict[str, float]) -> dict:
    ball_ok = (
        summary["ball_in_play_coverage_ratio"] >= limits["min_ball_coverage"]
        and summary["max_ball_speed_px_frame"] <= limits["max_ball_jump_px_frame"]
    )
    field_ok = (
        summary["field_path_samples"] >= limits["min_field_path_samples"]
        and summary["ball_out_of_bounds_ratio"] <= limits["max_out_of_bounds_ratio"]
    )
    teams_ok = (
        summary["unknown_team_ratio"] <= limits["max_unknown_team_ratio"]
        and summary["possession_coverage_ratio"] >= limits["min_possession_coverage"]
    )
    goals = int(summary.get("event_counts", {}).get("goal_candidate", 0))
    shots = int(summary.get("event_counts", {}).get("shot", 0))
    return {
        "ball_tracking": _claim(
            ball_ok,
            "ball coverage and jump checks support trajectory claims",
            "ball coverage or jump checks need review before claiming robust tracking",
        ),
        "metric_speed_trajectory": _claim(
            field_ok,
            "calibrated field samples support metric speed and distance claims",
            "homography/path evidence is too thin for final metric speed claims",
        ),
        "team_possession": _claim(
            teams_ok,
            "team-color assignment and possession coverage support team possession claims",
            "team assignment or possession coverage is not strong enough yet",
        ),
        "goal_scoring": _claim(
            goals > 0,
            f"{goals} goal candidate events were detected",
            "no goal candidate events were detected in this run",
        ),
        "shot_pressure": _claim(
            shots > 0,
            f"{shots} shot candidate events were detected",
            "no shot candidate events were detected in this run",
        ),
    }


def _claim(ready: bool, ready_reason: str, review_reason: str) -> dict:
    return {
        "status": "ready" if ready else "review",
        "reason": ready_reason if ready else review_reason,
    }


def _ready_claims(readiness: Any) -> str:
    if not isinstance(readiness, dict):
        return "none"
    ready = [
        str(claim)
        for claim, values in sorted(readiness.items())
        if isinstance(values, dict) and values.get("status") == "ready"
    ]
    return ", ".join(ready) if ready else "none"


def _issues(summary: dict[str, Any], limits: dict[str, float]) -> list[dict]:
    issues = []
    _coverage_issue(
        issues,
        code="low_ball_coverage",
        label="Ball in-play coverage",
        value=summary["ball_in_play_coverage_ratio"],
        warn_limit=limits["min_ball_coverage"],
        fail_limit=limits["fail_ball_coverage"],
    )
    _coverage_issue(
        issues,
        code="low_field_coverage",
        label="Field coverage",
        value=summary["field_coverage_ratio"],
        warn_limit=limits["min_field_coverage"],
        fail_limit=limits["fail_field_coverage"],
    )
    if summary["robot_coverage_ratio"] < limits["min_robot_coverage"]:
        issues.append(
            _issue(
                "warning",
                "low_robot_coverage",
                f"Robot coverage is {summary['robot_coverage_ratio']:.1%}.",
                summary["robot_coverage_ratio"],
                limits["min_robot_coverage"],
            )
        )
    _upper_issue(
        issues,
        code="large_ball_jump",
        label="Max ball jump",
        value=summary["max_ball_speed_px_frame"],
        warn_limit=limits["max_ball_jump_px_frame"],
        fail_limit=limits["fail_ball_jump_px_frame"],
        unit="px/frame",
    )
    _upper_issue(
        issues,
        code="ball_out_of_bounds",
        label="Ball out-of-bounds ratio",
        value=summary["ball_out_of_bounds_ratio"],
        warn_limit=limits["max_out_of_bounds_ratio"],
        fail_limit=limits["fail_out_of_bounds_ratio"],
        unit="ratio",
    )
    if summary["robot_penalty_area_samples"] > 0:
        issues.append(
            _issue(
                "warning",
                "robot_penalty_area",
                f"Robot projected into penalty area in {summary['robot_penalty_area_samples']} samples.",
                summary["robot_penalty_area_samples"],
                0,
            )
        )
    _upper_issue(
        issues,
        code="unknown_robot_teams",
        label="Unknown-team robot ratio",
        value=summary["unknown_team_ratio"],
        warn_limit=limits["max_unknown_team_ratio"],
        fail_limit=limits["fail_unknown_team_ratio"],
        unit="ratio",
    )
    return issues


def _coverage_issue(
    issues: list[dict],
    *,
    code: str,
    label: str,
    value: float,
    warn_limit: float,
    fail_limit: float,
) -> None:
    if value < fail_limit:
        issues.append(
            _issue("error", code, f"{label} is only {value:.1%}.", value, fail_limit)
        )
    elif value < warn_limit:
        issues.append(
            _issue("warning", code, f"{label} is {value:.1%}.", value, warn_limit)
        )


def _upper_issue(
    issues: list[dict],
    *,
    code: str,
    label: str,
    value: float,
    warn_limit: float,
    fail_limit: float,
    unit: str,
) -> None:
    if value > fail_limit:
        issues.append(
            _issue("error", code, f"{label} is {value:.1f} {unit}.", value, fail_limit)
        )
    elif value > warn_limit:
        issues.append(
            _issue("warning", code, f"{label} is {value:.1f} {unit}.", value, warn_limit)
        )


def _issue(severity: str, code: str, message: str, value: float, threshold: float) -> dict:
    return {
        "severity": severity,
        "code": code,
        "message": message,
        "value": value,
        "threshold": threshold,
    }


def _status(issues: list[dict]) -> str:
    severities = {issue["severity"] for issue in issues}
    if "error" in severities:
        return "fail"
    if "warning" in severities:
        return "review"
    return "good"


def _quality_score(issues: list[dict]) -> int:
    score = 100
    for issue in issues:
        score -= 25 if issue["severity"] == "error" else 10
    return max(0, score)


def _quality_index_record(path: Path, report: dict, root: Path) -> dict:
    try:
        rel_path = path.relative_to(root)
    except ValueError:
        rel_path = path
    summary = report.get("summary", {}) if isinstance(report.get("summary", {}), dict) else {}
    return {
        "path": str(rel_path).replace("\\", "/"),
        "status": str(report.get("status", "unknown")),
        "quality_score": int(_number(report.get("quality_score"))),
        "summary": {
            "frames_observed": int(_number(summary.get("frames_observed"))),
            "ball_in_play_coverage_ratio": _number(
                summary.get("ball_in_play_coverage_ratio")
            ),
            "max_ball_speed_px_frame": _number(summary.get("max_ball_speed_px_frame")),
            "field_path_samples": int(_number(summary.get("field_path_samples"))),
            "robot_penalty_area_samples": int(
                _number(summary.get("robot_penalty_area_samples"))
            ),
            "unknown_team_ratio": _number(summary.get("unknown_team_ratio")),
        },
        "claim_readiness": (
            report.get("claim_readiness", {})
            if isinstance(report.get("claim_readiness", {}), dict)
            else {}
        ),
        "issues": report.get("issues", []) if isinstance(report.get("issues", []), list) else [],
        "inputs": report.get("inputs", {}) if isinstance(report.get("inputs", {}), dict) else {},
    }


def _status_rank(status: str) -> int:
    return {"good": 0, "review": 1, "fail": 2}.get(status, 3)


def _read_mapping(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _read_list(path: str | Path | None) -> list[Any]:
    if not path:
        return []
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array: {path}")
    return data


def _number(value: object) -> float:
    numeric = _optional_number(value)
    return numeric if numeric is not None else 0.0


def _optional_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None
