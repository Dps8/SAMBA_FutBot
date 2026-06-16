from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import ensure_parent, read_json, write_json


METRICS = {
    "quality_score": ("quality_score", "higher"),
    "ball_coverage": ("summary.ball_in_play_coverage_ratio", "higher"),
    "field_coverage": ("summary.field_coverage_ratio", "higher"),
    "robot_coverage": ("summary.robot_coverage_ratio", "higher"),
    "possession_coverage": ("summary.possession_coverage_ratio", "higher"),
    "unknown_team_ratio": ("summary.unknown_team_ratio", "lower"),
    "max_ball_jump_px_frame": ("summary.max_ball_speed_px_frame", "lower"),
    "out_of_bounds_ratio": ("summary.ball_out_of_bounds_ratio", "lower"),
    "field_path_samples": ("summary.field_path_samples", "higher"),
}


def compare_qa_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    metrics = {}
    improvements = 0
    regressions = 0
    for name, (path, direction) in METRICS.items():
        before = _number(_get_path(baseline, path))
        after = _number(_get_path(candidate, path))
        delta = after - before
        signed_improvement = delta if direction == "higher" else -delta
        status = (
            "improved"
            if signed_improvement > tolerance
            else "regressed"
            if signed_improvement < -tolerance
            else "unchanged"
        )
        improvements += int(status == "improved")
        regressions += int(status == "regressed")
        metrics[name] = {
            "baseline": before,
            "candidate": after,
            "delta": delta,
            "direction": direction,
            "status": status,
        }

    baseline_claims = _ready_claims(baseline)
    candidate_claims = _ready_claims(candidate)
    gained_claims = sorted(candidate_claims - baseline_claims)
    lost_claims = sorted(baseline_claims - candidate_claims)
    verdict = _verdict(
        improvements=improvements,
        regressions=regressions,
        gained_claims=gained_claims,
        lost_claims=lost_claims,
        score_delta=metrics["quality_score"]["delta"],
    )
    return {
        "schema": "samba_futbot.qa_comparison.v1",
        "verdict": verdict,
        "baseline": {
            "status": baseline.get("status", "unknown"),
            "quality_score": int(_number(baseline.get("quality_score"))),
            "ready_claims": sorted(baseline_claims),
        },
        "candidate": {
            "status": candidate.get("status", "unknown"),
            "quality_score": int(_number(candidate.get("quality_score"))),
            "ready_claims": sorted(candidate_claims),
        },
        "metrics": metrics,
        "claims": {
            "gained": gained_claims,
            "lost": lost_claims,
            "unchanged_ready": sorted(baseline_claims & candidate_claims),
        },
        "summary": {
            "improved_metrics": improvements,
            "regressed_metrics": regressions,
            "unchanged_metrics": len(metrics) - improvements - regressions,
        },
    }


def compare_qa_files(
    baseline_path: str | Path,
    candidate_path: str | Path,
) -> dict[str, Any]:
    baseline = _read_mapping(baseline_path)
    candidate = _read_mapping(candidate_path)
    result = compare_qa_reports(baseline, candidate)
    result["inputs"] = {
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
    }
    return result


def write_qa_comparison_json(path: str | Path, comparison: dict[str, Any]) -> Path:
    write_json(path, comparison)
    return Path(path)


def write_qa_comparison_markdown(
    path: str | Path,
    comparison: dict[str, Any],
) -> Path:
    output = ensure_parent(path)
    lines = [
        "# QA Comparison",
        "",
        f"- Verdict: `{comparison.get('verdict', 'unknown')}`",
        f"- Baseline score: `{comparison.get('baseline', {}).get('quality_score', 0)}`",
        f"- Candidate score: `{comparison.get('candidate', {}).get('quality_score', 0)}`",
        f"- Gained claims: `{_format_list(comparison.get('claims', {}).get('gained', []))}`",
        f"- Lost claims: `{_format_list(comparison.get('claims', {}).get('lost', []))}`",
        "",
        "## Metrics",
        "",
        "| Metric | Baseline | Candidate | Delta | Direction | Result |",
        "|---|---:|---:|---:|---|---|",
    ]
    for name, values in comparison.get("metrics", {}).items():
        lines.append(
            "| "
            f"`{name}` | "
            f"{float(values.get('baseline', 0.0)):.4f} | "
            f"{float(values.get('candidate', 0.0)):.4f} | "
            f"{float(values.get('delta', 0.0)):+.4f} | "
            f"`{values.get('direction', 'unknown')}` | "
            f"`{values.get('status', 'unknown')}` |"
        )
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _verdict(
    *,
    improvements: int,
    regressions: int,
    gained_claims: list[str],
    lost_claims: list[str],
    score_delta: float,
) -> str:
    if lost_claims or score_delta < 0:
        return "regressed" if not gained_claims else "mixed"
    if gained_claims or score_delta > 0:
        return "improved" if regressions == 0 else "mixed"
    if improvements and regressions:
        return "mixed"
    if improvements:
        return "improved"
    if regressions:
        return "regressed"
    return "unchanged"


def _ready_claims(report: dict[str, Any]) -> set[str]:
    readiness = report.get("claim_readiness", {})
    if not isinstance(readiness, dict):
        return set()
    return {
        str(name)
        for name, values in readiness.items()
        if isinstance(values, dict) and values.get("status") == "ready"
    }


def _get_path(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, dict):
            return 0.0
        value = value.get(part, 0.0)
    return value


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _read_mapping(path: str | Path) -> dict[str, Any]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected QA JSON object: {path}")
    return data


def _format_list(values: Any) -> str:
    return ", ".join(str(value) for value in values) if values else "none"
