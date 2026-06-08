from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .io_utils import ensure_parent, write_json
from .qa import collect_quality_reports


DEFAULT_REQUIRED_CLAIMS = ("ball_tracking", "team_possession")


def collect_showcase_candidates(
    root: str | Path,
    *,
    limit: int = 12,
    required_claims: Iterable[str] = DEFAULT_REQUIRED_CLAIMS,
) -> list[dict]:
    reports = collect_quality_reports(root)
    candidates = [showcase_record(report, required_claims=required_claims) for report in reports]
    candidates.sort(
        key=lambda item: (
            -int(item["ready_claims"]),
            item["status_rank"],
            -int(item["quality_score"]),
            str(item["path"]),
        )
    )
    return candidates[:limit]


def showcase_record(report: dict, *, required_claims: Iterable[str]) -> dict:
    readiness = report.get("claim_readiness", {})
    ready = ready_claims(readiness)
    required = list(required_claims)
    missing_required = [claim for claim in required if claim not in ready]
    summary = report.get("summary", {}) if isinstance(report.get("summary", {}), dict) else {}
    return {
        "path": report.get("path", ""),
        "status": report.get("status", "unknown"),
        "status_rank": _status_rank(str(report.get("status", "unknown"))),
        "quality_score": int(report.get("quality_score", 0)),
        "ready_claims": len(ready),
        "ready_claim_names": ready,
        "missing_required_claims": missing_required,
        "showcase_ready": not missing_required and str(report.get("status")) in {"good", "review"},
        "ball_coverage": float(summary.get("ball_in_play_coverage_ratio", 0.0)),
        "unknown_team_ratio": float(summary.get("unknown_team_ratio", 0.0)),
        "max_ball_jump_px_frame": float(summary.get("max_ball_speed_px_frame", 0.0)),
        "field_path_samples": int(summary.get("field_path_samples", 0)),
    }


def write_showcase_json(path: str | Path, candidates: list[dict]) -> Path:
    write_json(path, {"schema": "samba_futbot.showcase.v1", "runs": candidates})
    return Path(path)


def write_showcase_markdown(path: str | Path, candidates: list[dict]) -> Path:
    lines = [
        "# Showcase Candidates",
        "",
        "| Rank | Ready | Status | Score | Claims | Ball | Unknown teams | Path |",
        "|---:|---|---|---:|---|---:|---:|---|",
    ]
    for index, candidate in enumerate(candidates, start=1):
        lines.append(
            "| "
            f"{index} | "
            f"`{'yes' if candidate.get('showcase_ready') else 'review'}` | "
            f"`{candidate.get('status', 'unknown')}` | "
            f"{int(candidate.get('quality_score', 0))} | "
            f"`{_format_claims(candidate.get('ready_claim_names', []))}` | "
            f"{float(candidate.get('ball_coverage', 0.0)):.1%} | "
            f"{float(candidate.get('unknown_team_ratio', 0.0)):.1%} | "
            f"`{candidate.get('path', '')}` |"
        )
    if not candidates:
        lines.append("| 0 | `review` | `none` | 0 | `none` | 0.0% | 0.0% | `No QA reports found` |")
    output = ensure_parent(path)
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output


def ready_claims(readiness: object) -> list[str]:
    if not isinstance(readiness, dict):
        return []
    return [
        str(claim)
        for claim, values in sorted(readiness.items())
        if isinstance(values, dict) and values.get("status") == "ready"
    ]


def _format_claims(claims: object) -> str:
    if not isinstance(claims, list) or not claims:
        return "none"
    return ", ".join(str(claim) for claim in claims)


def _status_rank(status: str) -> int:
    return {"good": 0, "review": 1, "fail": 2}.get(status, 3)
