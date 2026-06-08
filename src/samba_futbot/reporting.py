from __future__ import annotations

from pathlib import Path

from .events import summarize_events
from .io_utils import ensure_parent, read_json


def write_run_report(
    out_path: str | Path,
    *,
    title: str,
    metrics_path: str | Path | None = None,
    events_path: str | Path | None = None,
    field_analysis_path: str | Path | None = None,
    qa_path: str | Path | None = None,
    demo_path: str | Path | None = None,
    field_map_path: str | Path | None = None,
) -> Path:
    lines = [f"# {title}", ""]
    if demo_path:
        lines.extend(["## Demo", "", f"`{demo_path}`", ""])
    if metrics_path:
        lines.extend(_metrics_section(metrics_path))
    if events_path:
        lines.extend(_events_section(events_path))
    if field_analysis_path:
        lines.extend(_field_section(field_analysis_path, field_map_path=field_map_path))
    if qa_path:
        lines.extend(_qa_section(qa_path))

    output = ensure_parent(out_path)
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output


def _metrics_section(path: str | Path) -> list[str]:
    metrics = read_json(path)
    ball = metrics.get("classes", {}).get("ball", {})
    motion = metrics.get("motion", {}).get("ball", {})
    possession = metrics.get("possession", {})
    return [
        "## Tracking Metrics",
        "",
        f"- Frames observed: `{metrics.get('frames_observed', 0)}`",
        f"- Detections: `{metrics.get('detections', 0)}`",
        f"- Tracks: `{metrics.get('tracks', 0)}`",
        f"- Ball in-play coverage: `{ball.get('in_play_coverage_ratio', 0.0):.1%}`",
        f"- Possession coverage: `{possession.get('coverage_ratio', 0.0):.1%}`",
        f"- Possession by team: `{_format_possession_by_team(possession)}`",
        f"- Possession dominance: `{_format_possession_dominance(possession)}`",
        f"- Longest possession: `{_format_longest_possession(possession)}`",
        f"- Mean ball speed: `{motion.get('mean_speed_px_second', 0.0):.1f} px/s`",
        f"- Max ball speed: `{motion.get('max_speed_px_second', 0.0):.1f} px/s`",
        "",
    ]


def _events_section(path: str | Path) -> list[str]:
    events = read_json(path)
    summary = summarize_events(events)
    counts: dict[str, int] = summary.get("counts", {})
    scoreboard = summary.get("scoreboard", {})
    lines = ["## Event Candidates", "", f"- Total events: `{len(events)}`"]
    lines.append(
        "- Candidate score: "
        f"`blue {scoreboard.get('blue', 0)} - {scoreboard.get('yellow', 0)} yellow`"
    )
    lines.append(
        "- Possession changes: "
        f"`{summary.get('possession_changes', {}).get('passes', 0)} passes, "
        f"{summary.get('possession_changes', {}).get('interceptions', 0)} interceptions`"
    )
    lines.append(
        "- Shots by team: "
        f"`{_format_counter(summary.get('shots', {}).get('by_team', {}))}`"
    )
    for event_type, count in sorted(counts.items()):
        lines.append(f"- `{event_type}`: `{count}`")
    lines.append("")
    return lines


def _format_possession_by_team(possession: dict) -> str:
    by_team = possession.get("by_team", {})
    if not by_team:
        return "none"
    return ", ".join(
        f"{team}: {values.get('ratio', 0.0):.1%}" for team, values in sorted(by_team.items())
    )


def _format_longest_possession(possession: dict) -> str:
    longest = possession.get("longest_streak")
    if not longest:
        return "none"
    seconds = longest.get("seconds")
    suffix = f", {seconds:.2f}s" if isinstance(seconds, int | float) else ""
    return (
        f"{longest.get('team', 'unknown')} #{longest.get('track_id', 'unknown')}: "
        f"{longest.get('frames', 0)} frames{suffix}"
    )


def _format_possession_dominance(possession: dict) -> str:
    dominance = possession.get("dominance")
    if not isinstance(dominance, dict) or dominance.get("team") in {None, "none"}:
        return "none"
    return (
        f"{dominance.get('team', 'unknown')}: "
        f"{float(dominance.get('ratio', 0.0)):.1%}, "
        f"margin {float(dominance.get('margin_ratio', 0.0)):.1%}"
    )


def _field_section(path: str | Path, *, field_map_path: str | Path | None) -> list[str]:
    analysis = read_json(path)
    summary = analysis.get("summary", {})
    robot_summary = analysis.get("robot_summary", {})
    zone_control = _zone_control_summary(analysis.get("robot_zone_control", []))
    field = analysis.get("calibration", {}).get("field", {})
    lines = [
        "## Field Analysis",
        "",
        f"- Field model: `{field.get('length_m', 0.0):.2f} m x {field.get('width_m', 0.0):.2f} m`",
        f"- Ball path samples: `{summary.get('path_samples', 0)}`",
        f"- Ball distance: `{summary.get('distance_m', 0.0):.2f} m`",
        f"- Mean metric speed: `{summary.get('mean_speed_m_s', 0.0):.2f} m/s`",
        f"- Max metric speed: `{summary.get('max_speed_m_s', 0.0):.2f} m/s`",
        f"- Goal-zone entries: `{summary.get('goal_zone_entries', 0)}`",
        f"- Robot penalty-area samples: `{robot_summary.get('penalty_area_samples', 0)}`",
        f"- Robot samples by team: `{_format_counter(robot_summary.get('samples_by_team', {}))}`",
        f"- Robot phases by team: `{_format_nested_counter(robot_summary.get('phase_samples_by_team', {}))}`",
        f"- Attacking pressure by team: `{_format_ratio_counter(robot_summary.get('attacking_pressure_by_team', {}))}`",
        f"- Territorial control by leader: `{_format_counter(zone_control)}`",
    ]
    if field_map_path:
        lines.append(f"- Tactical map: `{field_map_path}`")
    lines.extend(
        [
            "",
            "> Metric distances require calibrated image corners for the analyzed camera.",
            "",
        ]
    )
    return lines


def _qa_section(path: str | Path) -> list[str]:
    report = read_json(path)
    summary = report.get("summary", {})
    lines = [
        "## Run QA",
        "",
        f"- Status: `{report.get('status', 'unknown')}`",
        f"- Quality score: `{report.get('quality_score', 0)}`",
        f"- Ball coverage: `{summary.get('ball_in_play_coverage_ratio', 0.0):.1%}`",
        f"- Max ball jump: `{summary.get('max_ball_speed_px_frame', 0.0):.1f} px/frame`",
        f"- Unknown-team robot ratio: `{summary.get('unknown_team_ratio', 0.0):.1%}`",
        f"- Possession coverage: `{summary.get('possession_coverage_ratio', 0.0):.1%}`",
        "",
        "### Claim Readiness",
        "",
    ]
    readiness = report.get("claim_readiness", {})
    if isinstance(readiness, dict) and readiness:
        for claim, values in sorted(readiness.items()):
            if not isinstance(values, dict):
                continue
            lines.append(
                f"- `{claim}`: `{values.get('status', 'unknown')}` - "
                f"{values.get('reason', '')}"
            )
    else:
        lines.append("- No claim-readiness data was generated.")
    lines.extend(
        [
            "",
            "### QA Issues",
            "",
        ]
    )
    issues = report.get("issues", [])
    if issues:
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            lines.append(
                f"- `{issue.get('severity', 'info')}` `{issue.get('code', 'unknown')}`: "
                f"{issue.get('message', '')}"
            )
    else:
        lines.append("- No automatic QA issues were detected.")
    lines.append("")
    return lines


def _format_counter(values: dict) -> str:
    if not values:
        return "none"
    return ", ".join(f"{key}: {value}" for key, value in sorted(values.items()))


def _format_nested_counter(values: dict) -> str:
    if not values:
        return "none"
    parts = []
    for key, inner in sorted(values.items()):
        if isinstance(inner, dict):
            parts.append(f"{key} ({_format_counter(inner)})")
        else:
            parts.append(f"{key}: {inner}")
    return "; ".join(parts)


def _format_ratio_counter(values: dict) -> str:
    if not values:
        return "none"
    return ", ".join(
        f"{key}: {float(value):.1%}"
        for key, value in sorted(values.items())
        if isinstance(value, int | float)
    )


def _zone_control_summary(zones: list) -> dict:
    counts: dict[str, int] = {}
    for zone in zones:
        if not isinstance(zone, dict):
            continue
        leader = str(zone.get("leader", "unknown"))
        counts[leader] = counts.get(leader, 0) + 1
    return counts
