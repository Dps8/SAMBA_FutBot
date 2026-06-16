from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable

from .io_utils import ensure_parent, read_detections
from .play_state import ROBOT_CLASSES
from .types import Detection


DEFAULT_UNKNOWN_RATIO_THRESHOLD = 0.20
DEFAULT_AMBIGUOUS_TRACK_DOMINANCE = 0.75
DEFAULT_MIN_AMBIGUOUS_TRACK_SAMPLES = 2
DEFAULT_MIN_FRAME_TEAM_COVERAGE = 0.80
DEFAULT_MAX_DOMINANT_TEAM_RATIO = 0.85
DEFAULT_MAX_REVIEW_CANDIDATES = 100
UNKNOWN_TEAM = "unknown"
_UNKNOWN_LABELS = {"", "none", "null", "unassigned", "unknown"}


def analyze_team_quality(
    detections: Iterable[Detection],
    *,
    unknown_ratio_threshold: float = DEFAULT_UNKNOWN_RATIO_THRESHOLD,
    ambiguous_track_dominance_threshold: float = DEFAULT_AMBIGUOUS_TRACK_DOMINANCE,
    min_ambiguous_track_samples: int = DEFAULT_MIN_AMBIGUOUS_TRACK_SAMPLES,
    min_frame_team_coverage: float = DEFAULT_MIN_FRAME_TEAM_COVERAGE,
    max_dominant_team_ratio: float = DEFAULT_MAX_DOMINANT_TEAM_RATIO,
    max_review_candidates: int = DEFAULT_MAX_REVIEW_CANDIDATES,
    robot_classes: Iterable[str] = tuple(sorted(ROBOT_CLASSES)),
) -> dict:
    """Measure temporal consistency and team-label coverage for robot detections."""
    _validate_thresholds(
        unknown_ratio_threshold=unknown_ratio_threshold,
        ambiguous_track_dominance_threshold=ambiguous_track_dominance_threshold,
        min_ambiguous_track_samples=min_ambiguous_track_samples,
        min_frame_team_coverage=min_frame_team_coverage,
        max_dominant_team_ratio=max_dominant_team_ratio,
        max_review_candidates=max_review_candidates,
    )
    robot_class_names = {str(name).strip().casefold() for name in robot_classes}
    robots = [
        detection
        for detection in detections
        if detection.class_name.strip().casefold() in robot_class_names
    ]
    robots.sort(key=lambda detection: detection.frame_index)

    by_frame: dict[int, list[Detection]] = defaultdict(list)
    by_track: dict[int, list[Detection]] = defaultdict(list)
    for detection in robots:
        by_frame[detection.frame_index].append(detection)
        if detection.track_id is not None:
            by_track[detection.track_id].append(detection)

    frame_coverage = _frame_coverage(
        by_frame,
        min_frame_team_coverage=min_frame_team_coverage,
    )
    track_analysis = [
        _analyze_track(
            track_id,
            track_detections,
            ambiguous_track_dominance_threshold=ambiguous_track_dominance_threshold,
            min_ambiguous_track_samples=min_ambiguous_track_samples,
        )
        for track_id, track_detections in sorted(by_track.items())
    ]

    assigned_samples = sum(_team(det.team) != UNKNOWN_TEAM for det in robots)
    unknown_samples = len(robots) - assigned_samples
    unknown_ratio = _ratio(unknown_samples, len(robots))
    total_team_changes = sum(track["team_changes"] for track in track_analysis)
    ambiguous_tracks = [track["track_id"] for track in track_analysis if track["ambiguous"]]
    by_team = _by_team(robots, track_analysis)
    known_team_samples = {
        team: values["samples"]
        for team, values in by_team.items()
        if team != UNKNOWN_TEAM
    }
    dominant_team, dominant_team_samples = _dominant_team(Counter(known_team_samples))
    dominant_team_ratio = _ratio(dominant_team_samples, sum(known_team_samples.values()))
    team_imbalance = (
        len(known_team_samples) < 2
        or dominant_team_ratio > max_dominant_team_ratio
    ) if known_team_samples else False
    review_candidates = _review_candidates(
        robots,
        track_analysis,
        frame_coverage,
        max_review_candidates=max_review_candidates,
    )
    if team_imbalance and max_review_candidates > 0:
        review_candidates.insert(
            0,
            {
                "reason": "team_imbalance",
                "dominant_team": dominant_team,
                "dominant_ratio": dominant_team_ratio,
                "samples_by_team": known_team_samples,
            },
        )
        review_candidates = review_candidates[:max_review_candidates]

    return {
        "schema": "samba_futbot.team_quality.v1",
        "summary": {
            "robot_samples": len(robots),
            "assigned_samples": assigned_samples,
            "unknown_samples": unknown_samples,
            "unknown_ratio": unknown_ratio,
            "unknown_ratio_above_threshold": unknown_ratio > unknown_ratio_threshold,
            "robot_frames": len(by_frame),
            "tracked_samples": sum(det.track_id is not None for det in robots),
            "untracked_samples": sum(det.track_id is None for det in robots),
            "tracks": len(track_analysis),
            "team_changes": total_team_changes,
            "tracks_with_team_changes": sum(
                track["team_changes"] > 0 for track in track_analysis
            ),
            "ambiguous_tracks": len(ambiguous_tracks),
            "frames_below_team_coverage": sum(
                frame["below_threshold"] for frame in frame_coverage
            ),
            "review_candidates": len(review_candidates),
            "dominant_team": dominant_team,
            "dominant_team_ratio": dominant_team_ratio,
            "team_imbalance_above_threshold": team_imbalance,
        },
        "thresholds": {
            "unknown_ratio": unknown_ratio_threshold,
            "ambiguous_track_dominance": ambiguous_track_dominance_threshold,
            "min_ambiguous_track_samples": min_ambiguous_track_samples,
            "min_frame_team_coverage": min_frame_team_coverage,
            "max_dominant_team_ratio": max_dominant_team_ratio,
            "max_review_candidates": max_review_candidates,
        },
        "by_team": by_team,
        "ambiguous_track_ids": ambiguous_tracks,
        "tracks": track_analysis,
        "frame_coverage": {
            "mean_ratio": mean(
                frame["coverage_ratio"] for frame in frame_coverage
            )
            if frame_coverage
            else 0.0,
            "overall_ratio": _ratio(assigned_samples, len(robots)),
            "frames": frame_coverage,
        },
        "review_candidates": review_candidates,
    }


def analyze_team_quality_file(
    detections_path: str | Path,
    **kwargs,
) -> dict:
    """Read detections from JSONL through io_utils and analyze their team quality."""
    report = analyze_team_quality(read_detections(detections_path), **kwargs)
    report["inputs"] = {"detections": str(detections_path)}
    return report


def analyze_team_quality_jsonl(
    detections_path: str | Path,
    **kwargs,
) -> dict:
    """Alias with an explicit JSONL name for callers that prefer format-specific APIs."""
    return analyze_team_quality_file(detections_path, **kwargs)


def write_team_quality_markdown(report: dict, out_path: str | Path) -> Path:
    output = ensure_parent(out_path)
    summary = report.get("summary", {})
    lines = [
        "# Team Quality Report",
        "",
        f"- Robot samples: `{summary.get('robot_samples', 0)}`",
        f"- Assigned samples: `{summary.get('assigned_samples', 0)}`",
        f"- Unknown ratio: `{float(summary.get('unknown_ratio', 0.0)):.1%}`",
        f"- Tracks: `{summary.get('tracks', 0)}`",
        f"- Team changes: `{summary.get('team_changes', 0)}`",
        f"- Ambiguous tracks: `{summary.get('ambiguous_tracks', 0)}`",
        f"- Frames below team coverage: `{summary.get('frames_below_team_coverage', 0)}`",
        f"- Dominant team ratio: `{float(summary.get('dominant_team_ratio', 0.0)):.1%}`",
        f"- Team imbalance: `{str(bool(summary.get('team_imbalance_above_threshold'))).lower()}`",
        "",
        "## Tracks",
        "",
        "| Track | Samples | Dominant team | Dominance | Changes | Ambiguous |",
        "|---:|---:|---|---:|---:|---|",
    ]
    tracks = report.get("tracks", [])
    if not tracks:
        lines.append("| 0 | 0 | `none` | 0.0% | 0 | `false` |")
    for track in tracks:
        lines.append(
            "| "
            f"{track.get('track_id', 0)} | "
            f"{track.get('samples', 0)} | "
            f"`{track.get('dominant_team') or 'unknown'}` | "
            f"{float(track.get('dominant_ratio', 0.0)):.1%} | "
            f"{track.get('team_changes', 0)} | "
            f"`{str(bool(track.get('ambiguous'))).lower()}` |"
        )
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _analyze_track(
    track_id: int,
    detections: list[Detection],
    *,
    ambiguous_track_dominance_threshold: float,
    min_ambiguous_track_samples: int,
) -> dict:
    ordered = sorted(detections, key=lambda detection: detection.frame_index)
    teams = [_team(detection.team) for detection in ordered]
    known_teams = [team for team in teams if team != UNKNOWN_TEAM]
    counts = Counter(known_teams)
    dominant_team, dominant_samples = _dominant_team(counts)
    dominant_ratio = _ratio(dominant_samples, len(known_teams))
    ambiguous = (
        len(ordered) >= min_ambiguous_track_samples
        and len(counts) > 1
        and dominant_ratio < ambiguous_track_dominance_threshold
    )
    resolved_team = (
        dominant_team
        if dominant_team is not None
        and dominant_ratio >= ambiguous_track_dominance_threshold
        else UNKNOWN_TEAM
    )
    changes = _team_change_events(ordered)
    return {
        "track_id": track_id,
        "first_frame": ordered[0].frame_index,
        "last_frame": ordered[-1].frame_index,
        "samples": len(ordered),
        "assigned_samples": len(known_teams),
        "unknown_samples": len(ordered) - len(known_teams),
        "unknown_ratio": _ratio(len(ordered) - len(known_teams), len(ordered)),
        "team_counts": dict(sorted(counts.items())),
        "dominant_team": dominant_team,
        "dominant_ratio": dominant_ratio,
        "resolved_team": resolved_team,
        "team_changes": len(changes),
        "change_events": changes,
        "ambiguous": ambiguous,
    }


def _team_change_events(detections: list[Detection]) -> list[dict]:
    frame_teams: dict[int, Counter[str]] = defaultdict(Counter)
    for detection in detections:
        team = _team(detection.team)
        if team != UNKNOWN_TEAM:
            frame_teams[detection.frame_index][team] += 1

    timeline: list[tuple[int, str]] = []
    for frame_index, counts in sorted(frame_teams.items()):
        ordered = counts.most_common()
        if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
            continue
        timeline.append((frame_index, ordered[0][0]))

    changes: list[dict] = []
    previous: tuple[int, str] | None = None
    for current in timeline:
        if previous is not None and current[1] != previous[1]:
            changes.append(
                {
                    "from_frame": previous[0],
                    "to_frame": current[0],
                    "from_team": previous[1],
                    "to_team": current[1],
                }
            )
        previous = current
    return changes


def _frame_coverage(
    by_frame: dict[int, list[Detection]],
    *,
    min_frame_team_coverage: float,
) -> list[dict]:
    coverage = []
    for frame_index, detections in sorted(by_frame.items()):
        assigned = sum(_team(detection.team) != UNKNOWN_TEAM for detection in detections)
        ratio = _ratio(assigned, len(detections))
        coverage.append(
            {
                "frame_index": frame_index,
                "robot_samples": len(detections),
                "assigned_samples": assigned,
                "unknown_samples": len(detections) - assigned,
                "coverage_ratio": ratio,
                "below_threshold": ratio < min_frame_team_coverage,
            }
        )
    return coverage


def _by_team(robots: list[Detection], tracks: list[dict]) -> dict[str, dict[str, int]]:
    samples = Counter(_team(detection.team) for detection in robots)
    track_counts = Counter(track["resolved_team"] for track in tracks)
    teams = sorted(set(samples) | set(track_counts))
    return {
        team: {
            "samples": samples[team],
            "tracks": track_counts[team],
        }
        for team in teams
    }


def _review_candidates(
    robots: list[Detection],
    tracks: list[dict],
    frame_coverage: list[dict],
    *,
    max_review_candidates: int,
) -> list[dict]:
    candidates: list[dict] = []
    for track in tracks:
        if track["team_changes"]:
            candidates.append(
                {
                    "reason": "team_change",
                    "track_id": track["track_id"],
                    "first_frame": track["first_frame"],
                    "last_frame": track["last_frame"],
                    "team_changes": track["team_changes"],
                    "change_events": track["change_events"],
                }
            )
        if track["ambiguous"]:
            candidates.append(
                {
                    "reason": "ambiguous_track",
                    "track_id": track["track_id"],
                    "first_frame": track["first_frame"],
                    "last_frame": track["last_frame"],
                    "team_counts": track["team_counts"],
                    "dominant_ratio": track["dominant_ratio"],
                }
            )
    for frame in frame_coverage:
        if frame["below_threshold"]:
            candidates.append(
                {
                    "reason": "low_frame_team_coverage",
                    "frame_index": frame["frame_index"],
                    "robot_samples": frame["robot_samples"],
                    "unknown_samples": frame["unknown_samples"],
                    "coverage_ratio": frame["coverage_ratio"],
                }
            )
    for detection in robots:
        if _team(detection.team) == UNKNOWN_TEAM:
            candidates.append(
                {
                    "reason": "unknown_team",
                    "frame_index": detection.frame_index,
                    "track_id": detection.track_id,
                    "score": detection.score,
                    "box": list(detection.box),
                }
            )
    return candidates[:max_review_candidates]


def _dominant_team(counts: Counter[str]) -> tuple[str | None, int]:
    if not counts:
        return None, 0
    maximum = max(counts.values())
    return sorted(team for team, count in counts.items() if count == maximum)[0], maximum


def _team(value: str | None) -> str:
    if value is None:
        return UNKNOWN_TEAM
    normalized = str(value).strip().casefold()
    return UNKNOWN_TEAM if normalized in _UNKNOWN_LABELS else normalized


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _validate_thresholds(
    *,
    unknown_ratio_threshold: float,
    ambiguous_track_dominance_threshold: float,
    min_ambiguous_track_samples: int,
    min_frame_team_coverage: float,
    max_dominant_team_ratio: float,
    max_review_candidates: int,
) -> None:
    for name, value in (
        ("unknown_ratio_threshold", unknown_ratio_threshold),
        ("ambiguous_track_dominance_threshold", ambiguous_track_dominance_threshold),
        ("min_frame_team_coverage", min_frame_team_coverage),
        ("max_dominant_team_ratio", max_dominant_team_ratio),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if min_ambiguous_track_samples < 1:
        raise ValueError("min_ambiguous_track_samples must be at least 1")
    if max_review_candidates < 0:
        raise ValueError("max_review_candidates must be non-negative")
