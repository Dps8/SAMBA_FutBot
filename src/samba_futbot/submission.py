from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import ensure_parent, read_json


def write_submission_report(
    out_path: str | Path,
    *,
    batch_root: str | Path,
    training_root: str | Path | None = None,
    title: str = "SAMBA FutBot Submission Evidence",
    top: int = 4,
) -> Path:
    batch = Path(batch_root)
    training = Path(training_root) if training_root else None
    showcase = _read_mapping(batch / "showcase-index.json")
    qa_index = _read_mapping(batch / "qa-index.json")
    videos = _read_list(batch / "VIDEO_RENDER_SUMMARY.json")
    batch_summary = _read_list(batch / "BATCH_SUMMARY.json")
    situations = _read_situations(batch / "situations")
    training_summary = _read_list(training / "TRAINING_DATASET_SUMMARY.json") if training else []
    merged_summary = (
        _read_mapping(training / "merged_top_camera_balanced_manifest.json").get("summary", {})
        if training
        else {}
    )
    dataset_quality = (
        _read_mapping(training / "merged_top_camera_balanced_quality.json").get("summary", {})
        if training
        else {}
    )

    video_by_stem = {str(item.get("stem", "")): item for item in videos if isinstance(item, dict)}
    lines = [f"# {title}", ""]
    lines.extend(_overview_lines(batch, training))
    candidates = showcase.get("runs", [])[:top]
    candidate_stems = [_stem_from_qa_path(str(item.get("path", ""))) for item in candidates if isinstance(item, dict)]
    lines.extend(_candidate_lines(candidates, video_by_stem))
    lines.extend(_qa_lines(qa_index.get("runs", [])))
    lines.extend(_batch_lines(batch_summary))
    lines.extend(_situation_lines(situations, candidate_stems))
    lines.extend(_training_lines(training_summary, merged_summary, dataset_quality))
    lines.extend(_rule_alignment_lines())

    output = ensure_parent(out_path)
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output


def _overview_lines(batch: Path, training: Path | None) -> list[str]:
    lines = [
        "## Overview",
        "",
        f"- Batch root: `{batch}`",
    ]
    if training:
        lines.append(f"- Dataset root: `{training}`")
    lines.extend(
        [
            "- Detector path: `SAM3/SAM 3.1 + prompt/context engineering + color/geometry cues`",
            "- Training preparation: `pseudo-label manifests, frame/crop exports and COCO boxes`",
            "",
        ]
    )
    return lines


def _candidate_lines(candidates: list[Any], video_by_stem: dict[str, dict]) -> list[str]:
    lines = [
        "## Showcase Candidates",
        "",
        "| Rank | Status | Score | Ball | Claims | Narrative | Analysis |",
        "|---:|---|---:|---:|---|---|---|",
    ]
    if not candidates:
        lines.append("| 0 | `none` | 0 | 0.0% | `none` | `missing` | `missing` |")
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            continue
        stem = _stem_from_qa_path(str(candidate.get("path", "")))
        video = video_by_stem.get(stem, {})
        lines.append(
            "| "
            f"{index} | "
            f"`{candidate.get('status', 'unknown')}` | "
            f"{int(candidate.get('quality_score', 0))} | "
            f"{float(candidate.get('ball_coverage', 0.0)):.1%} | "
            f"`{_format_list(candidate.get('ready_claim_names', []))}` | "
            f"`{video.get('narrative', 'missing')}` | "
            f"`{video.get('analysis', 'missing')}` |"
        )
    lines.append("")
    return lines


def _qa_lines(runs: list[Any]) -> list[str]:
    counts: dict[str, int] = {}
    for run in runs:
        if isinstance(run, dict):
            status = str(run.get("status", "unknown"))
            counts[status] = counts.get(status, 0) + 1
    return [
        "## QA Summary",
        "",
        f"- Runs indexed: `{len(runs)}`",
        f"- Status counts: `{_format_counter(counts)}`",
        "",
    ]


def _batch_lines(batch_summary: list[Any]) -> list[str]:
    top = sorted(
        [item for item in batch_summary if isinstance(item, dict)],
        key=lambda item: float(item.get("ballCoverage", 0.0)),
        reverse=True,
    )[:6]
    lines = [
        "## Tracking Batch",
        "",
        "| Variant | Frames | Detections | Ball coverage | Possession coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    if not top:
        lines.append("| `none` | 0 | 0 | 0.0% | 0.0% |")
    for item in top:
        lines.append(
            "| "
            f"`{item.get('stem', '')}` | "
            f"{int(item.get('frames', 0))} | "
            f"{int(item.get('detections', 0))} | "
            f"{float(item.get('ballCoverage', 0.0)):.1%} | "
            f"{float(item.get('possessionCoverage', 0.0)):.1%} |"
        )
    lines.append("")
    return lines


def _situation_lines(situations: dict[str, dict], candidate_stems: list[str]) -> list[str]:
    selected = [
        (stem, situations[stem])
        for stem in candidate_stems
        if stem in situations and isinstance(situations[stem].get("summary"), dict)
    ]
    lines = [
        "## Tactical Situation Layer",
        "",
        "| Variant | Ball frames | Controlled | Disputed | Free | Loss risk | Pass | Shot | Hold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if not selected:
        lines.append("| `none` | 0 | 0.0% | 0.0% | 0.0% | 0.00 | 0.00 | 0.00 | 0.00 |")
    for stem, situation in selected:
        summary = situation["summary"]
        states = summary.get("possession_states", {})
        probabilities = summary.get("average_action_probabilities", {})
        lines.append(
            "| "
            f"`{stem}` | "
            f"{int(summary.get('frames_with_ball', 0))} | "
            f"{_state_ratio(states, 'controlled'):.1%} | "
            f"{_state_ratio(states, 'disputed'):.1%} | "
            f"{_state_ratio(states, 'free'):.1%} | "
            f"{float(summary.get('average_loss_risk', 0.0)):.2f} | "
            f"{float(probabilities.get('pass', 0.0)):.2f} | "
            f"{float(probabilities.get('shot', 0.0)):.2f} | "
            f"{float(probabilities.get('hold', 0.0)):.2f} |"
        )
    lines.extend(
        [
            "",
            "This layer is the source for robot-ball distance overlays, tactical freeze frames and pass/shot/hold heuristics.",
            "",
        ]
    )
    return lines


def _training_lines(training_summary: list[Any], merged_summary: dict, dataset_quality: dict) -> list[str]:
    lines = [
        "## Dataset Preparation",
        "",
        f"- Source datasets: `{len(training_summary)}`",
    ]
    if merged_summary:
        lines.extend(
            [
                f"- Merged frames: `{int(merged_summary.get('frames', 0))}`",
                f"- Merged detections/crops: `{int(merged_summary.get('detections', 0))}`",
                f"- Samples by class: `{_format_counter(merged_summary.get('detections_by_class', {}))}`",
                f"- Frames by split: `{_format_counter(merged_summary.get('frames_by_split', {}))}`",
            ]
        )
    if dataset_quality:
        lines.extend(
            [
                f"- Dataset QA invalid boxes: `{int(dataset_quality.get('invalid_boxes', 0))}`",
                f"- Dataset QA low-score detections: `{int(dataset_quality.get('low_scores', 0))}`",
                f"- Dataset QA review candidates: `{int(dataset_quality.get('review_candidates', 0))}`",
                f"- Dataset QA videos in multiple splits: `{int(dataset_quality.get('videos_in_multiple_splits', 0))}`",
                f"- Dataset QA duplicate image paths: `{int(dataset_quality.get('duplicate_image_paths', 0))}`",
            ]
        )
    lines.append("")
    return lines


def _rule_alignment_lines() -> list[str]:
    return [
        "## Rule Alignment",
        "",
        "- The evidence path stays centered on SAM3/SAM 3.1 and project-owned post-processing.",
        "- Color/geometry cues are used as contextual recovery signals for the ball and goals.",
        "- Dataset export is neutral: manifests, crops and COCO boxes for audit or SAM-compatible adaptation.",
        "",
    ]


def _stem_from_qa_path(path: str) -> str:
    name = Path(path).name
    return name.removesuffix("-qa.json")


def _format_list(values: object) -> str:
    if not isinstance(values, list) or not values:
        return "none"
    return ", ".join(str(value) for value in values)


def _format_counter(values: object) -> str:
    if not isinstance(values, dict) or not values:
        return "none"
    return ", ".join(f"{key}: {value}" for key, value in sorted(values.items()))


def _state_ratio(states: object, name: str) -> float:
    if not isinstance(states, dict):
        return 0.0
    item = states.get(name, {})
    if not isinstance(item, dict):
        return 0.0
    return float(item.get("ratio", 0.0))


def _read_situations(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    situations = {}
    for item in sorted(path.glob("*-situations.json")):
        data = _read_mapping(item)
        situations[item.name.removesuffix("-situations.json")] = data
    return situations


def _read_mapping(path: Path) -> dict:
    if not path.exists():
        return {}
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _read_list(path: Path) -> list:
    if not path.exists():
        return []
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array: {path}")
    return data
