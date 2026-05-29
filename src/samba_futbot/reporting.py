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


def _field_section(path: str | Path, *, field_map_path: str | Path | None) -> list[str]:
    analysis = read_json(path)
    summary = analysis.get("summary", {})
    robot_summary = analysis.get("robot_summary", {})
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
