from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .io_utils import read_json, write_json

READY_STATUSES = {"accepted", "approved", "complete", "completed", "reviewed"}
REJECTED_STATUSES = {"rejected", "skip", "skipped", "ignore", "ignored"}


def audit_ball_review(review: Mapping[str, Any], *, class_name: str = "ball") -> dict:
    """Audit a ball-review package before it is converted into training data."""
    if not isinstance(review, Mapping):
        raise ValueError("review must be a JSON object")
    images = review.get("images")
    if not isinstance(images, list):
        raise ValueError("review images must be a list")
    if not isinstance(class_name, str) or not class_name.strip():
        raise ValueError("class_name must be a non-empty string")
    class_name = class_name.strip()

    by_task: Counter[str] = Counter()
    by_status: Counter[str] = Counter()
    by_source_group: dict[str, Counter[str]] = defaultdict(Counter)
    issues = []
    positive_annotations = 0
    positive_mask_annotations = 0
    positive_bbox_only_annotations = 0
    positive_frames = 0
    verified_absence_frames = 0
    pending_frames = 0
    rejected_frames = 0

    for index, image in enumerate(images):
        if not isinstance(image, Mapping):
            issues.append(_issue(index, "malformed_image", "review image must be an object"))
            continue
        task = str(image.get("review_task", "unknown"))
        source_group = str(image.get("source_group") or image.get("video") or "unknown")
        by_task[task] += 1
        status = _review_status(image)
        by_status[status] += 1
        by_source_group[source_group][task] += 1

        if task == "verify_mask":
            annotations = _annotation_records(image)
            accepted = [
                _normalize_annotation(annotation, class_name=class_name)
                for annotation in annotations
            ]
            accepted = [annotation for annotation in accepted if annotation is not None]
            if accepted:
                positive_frames += 1
                positive_annotations += len(accepted)
                positive_mask_annotations += sum(1 for item in accepted if _has_mask_evidence(item))
                positive_bbox_only_annotations += sum(
                    1 for item in accepted if not _has_mask_evidence(item)
                )
            elif status in REJECTED_STATUSES:
                rejected_frames += 1
            else:
                pending_frames += 1
                issues.append(
                    _issue(
                        index,
                        "positive_without_annotation",
                        "verify_mask frame needs a reviewed annotation or rejected status",
                    )
                )
        elif task == "verify_absence":
            if image.get("ball_absent_verified") is True:
                verified_absence_frames += 1
            elif image.get("ball_absent_verified") is False or status in REJECTED_STATUSES:
                rejected_frames += 1
            else:
                pending_frames += 1
                issues.append(
                    _issue(
                        index,
                        "absence_not_verified",
                        "verify_absence frame needs ball_absent_verified=true or a rejected status",
                    )
                )
        else:
            pending_frames += 1
            issues.append(_issue(index, "unknown_review_task", f"unknown review task: {task}"))

    ready = not issues and pending_frames == 0 and positive_frames > 0
    return {
        "schema": "samba_futbot.ball_review_audit.v1",
        "class_name": class_name,
        "ready_for_training": ready,
        "summary": {
            "frames": len(images),
            "positive_frames": positive_frames,
            "positive_annotations": positive_annotations,
            "positive_mask_annotations": positive_mask_annotations,
            "positive_bbox_only_annotations": positive_bbox_only_annotations,
            "verified_absence_frames": verified_absence_frames,
            "pending_frames": pending_frames,
            "rejected_frames": rejected_frames,
            "issues": len(issues),
        },
        "by_task": dict(sorted(by_task.items())),
        "by_status": dict(sorted(by_status.items())),
        "by_source_group": {
            key: dict(sorted(values.items()))
            for key, values in sorted(by_source_group.items())
        },
        "issues": issues,
    }


def audit_ball_review_file(
    review_path: str | Path,
    out_path: str | Path | None = None,
    *,
    class_name: str = "ball",
    report_path: str | Path | None = None,
) -> dict:
    review = _read_review(review_path)
    audit = audit_ball_review(review, class_name=class_name)
    audit["inputs"] = {"review": str(review_path)}
    if out_path is not None:
        write_json(out_path, audit)
    if report_path is not None:
        write_ball_review_audit_markdown(audit, report_path)
    return audit


def write_ball_review_audit_markdown(audit: Mapping[str, Any], path: str | Path) -> Path:
    """Write a compact human-readable checklist for a ball-review audit."""
    summary = audit.get("summary", {})
    ready = bool(audit.get("ready_for_training"))
    lines = [
        "# Ball Review Audit",
        "",
        f"Ready for training: {'yes' if ready else 'no'}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in (
        "frames",
        "positive_frames",
        "positive_annotations",
        "positive_mask_annotations",
        "positive_bbox_only_annotations",
        "verified_absence_frames",
        "pending_frames",
        "rejected_frames",
        "issues",
    ):
        lines.append(f"| `{key}` | {summary.get(key, 0)} |")

    lines.extend(["", "## Review Tasks", "", "| Task | Frames |", "|---|---:|"])
    for key, value in sorted(_mapping(audit.get("by_task")).items()):
        lines.append(f"| `{key}` | {value} |")

    lines.extend(["", "## Status", "", "| Status | Frames |", "|---|---:|"])
    for key, value in sorted(_mapping(audit.get("by_status")).items()):
        lines.append(f"| `{key}` | {value} |")

    source_groups = _mapping(audit.get("by_source_group"))
    if source_groups:
        lines.extend(
            [
                "",
                "## Source Groups",
                "",
                "| Source group | verify_mask | verify_absence | Other |",
                "|---|---:|---:|---:|",
            ]
        )
        for source_group, values in sorted(source_groups.items()):
            group_counts = _mapping(values)
            verify_mask = int(group_counts.get("verify_mask", 0))
            verify_absence = int(group_counts.get("verify_absence", 0))
            other = sum(
                int(count)
                for task, count in group_counts.items()
                if task not in {"verify_mask", "verify_absence"}
            )
            lines.append(
                f"| `{source_group}` | {verify_mask} | {verify_absence} | {other} |"
            )

    issues = audit.get("issues", [])
    lines.extend(["", "## First Issues", ""])
    if isinstance(issues, list) and issues:
        lines.extend(["| Image index | Code | Message |", "|---:|---|---|"])
        for issue in issues[:20]:
            if not isinstance(issue, Mapping):
                continue
            image_index = issue.get("image_index", "")
            code = _markdown_cell(issue.get("code", ""))
            message = _markdown_cell(issue.get("message", ""))
            lines.append(f"| {image_index} | `{code}` | {message} |")
        if len(issues) > 20:
            lines.append("")
            lines.append(f"Showing 20 of {len(issues)} issues.")
    else:
        lines.append("No issues found.")

    lines.extend(
        [
            "",
            "## Next Actions",
            "",
            "- For `verify_mask`, add one reviewed `annotations` entry with a valid `box` and, for SAM3 segmentation, mask evidence.",
            "- For `verify_absence`, set `ball_absent_verified` to `true` only when the ball is genuinely absent.",
            "- Re-run `audit-ball-review` until `ready_for_training` is `yes` before exporting.",
            "",
        ]
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def export_reviewed_ball_manifest(
    review: Mapping[str, Any],
    *,
    class_name: str = "ball",
    require_complete: bool = True,
    include_verified_absence: bool = True,
    split_strategy: str = "preserve",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[dict, dict]:
    """Convert reviewed ball annotations into a frame-dataset manifest."""
    if split_strategy not in {"preserve", "by-source-balanced"}:
        raise ValueError("split_strategy must be 'preserve' or 'by-source-balanced'")
    _validate_split_ratios(train_ratio=train_ratio, val_ratio=val_ratio)
    audit = audit_ball_review(review, class_name=class_name)
    if require_complete and not audit["ready_for_training"]:
        raise ValueError("ball review is not ready for training")

    output_images = []
    detections_by_class: Counter[str] = Counter()
    mask_ready_annotations = 0
    bbox_only_annotations = 0
    verified_absence_frames = 0

    for image in review.get("images", []):
        if not isinstance(image, Mapping):
            continue
        task = str(image.get("review_task", "unknown"))
        detections = []
        if task == "verify_mask":
            for annotation in _annotation_records(image):
                normalized = _normalize_annotation(annotation, class_name=class_name)
                if normalized is None:
                    continue
                detections.append(normalized)
                detections_by_class[class_name] += 1
                if _has_mask_evidence(normalized):
                    mask_ready_annotations += 1
                else:
                    bbox_only_annotations += 1
        elif task == "verify_absence":
            if image.get("ball_absent_verified") is True and include_verified_absence:
                verified_absence_frames += 1
            else:
                continue
        else:
            continue

        if not detections and task != "verify_absence":
            continue
        record = _manifest_image_record(image, detections=detections)
        output_images.append(record)

    source_group_splits: dict[str, str] = {}
    if split_strategy == "by-source-balanced":
        source_group_splits = _balanced_source_splits(
            [_record_source_group(record) for record in output_images],
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        for record in output_images:
            record["split"] = source_group_splits[_record_source_group(record)]

    frames_by_split: Counter[str] = Counter()
    source_groups: Counter[str] = Counter()
    for record in output_images:
        frames_by_split[str(record.get("split") or "unknown")] += 1
        source_groups[_record_source_group(record)] += 1

    manifest = {
        "schema": "samba_futbot.reviewed_ball_dataset.v1",
        "source_schema": review.get("schema"),
        "selection_fingerprint": deepcopy(review.get("selection_fingerprint")),
        "review_policy": {
            "class_name": class_name,
            "ground_truth_source": "human_review",
            "include_verified_absence": include_verified_absence,
            "require_complete": require_complete,
            "split_strategy": split_strategy,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
        },
        "summary": {
            "frames": len(output_images),
            "positive_frames": sum(1 for image in output_images if image.get("detections")),
            "verified_absence_frames": verified_absence_frames,
            "detections": sum(detections_by_class.values()),
            "detections_by_class": dict(sorted(detections_by_class.items())),
            "mask_ready_annotations": mask_ready_annotations,
            "bbox_only_annotations": bbox_only_annotations,
            "frames_by_split": dict(sorted(frames_by_split.items())),
            "source_groups": len(source_groups),
        },
        "images": output_images,
    }
    report = {
        "schema": "samba_futbot.reviewed_ball_dataset_report.v1",
        "audit": audit,
        "summary": deepcopy(manifest["summary"]),
        "source_groups": dict(sorted(source_groups.items())),
        "source_group_splits": dict(sorted(source_group_splits.items())),
    }
    return manifest, report


def export_reviewed_ball_manifest_file(
    review_path: str | Path,
    out_path: str | Path,
    report_path: str | Path,
    *,
    class_name: str = "ball",
    require_complete: bool = True,
    include_verified_absence: bool = True,
    split_strategy: str = "preserve",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[dict, dict]:
    review = _read_review(review_path)
    manifest, report = export_reviewed_ball_manifest(
        review,
        class_name=class_name,
        require_complete=require_complete,
        include_verified_absence=include_verified_absence,
        split_strategy=split_strategy,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    report = deepcopy(report)
    report["inputs"] = {"review": str(review_path)}
    report["outputs"] = {"manifest": str(out_path), "report": str(report_path)}
    write_json(out_path, manifest)
    write_json(report_path, report)
    return manifest, report


def _read_review(path: str | Path) -> dict:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected ball review JSON object: {path}")
    return data


def _review_status(image: Mapping[str, Any]) -> str:
    value = image.get("annotation_status")
    return str(value).strip().casefold() if isinstance(value, str) and value.strip() else "pending"


def _annotation_records(image: Mapping[str, Any]) -> list[dict]:
    annotations = image.get("annotations", [])
    if not isinstance(annotations, list):
        return []
    return [dict(annotation) for annotation in annotations if isinstance(annotation, Mapping)]


def _normalize_annotation(annotation: Mapping[str, Any], *, class_name: str) -> dict | None:
    label = annotation.get("class_name") or annotation.get("label") or annotation.get("category")
    if label is not None and str(label) != class_name:
        return None
    box = _box(annotation.get("box") or annotation.get("bbox"))
    if box is None:
        return None
    normalized = {
        "class_name": class_name,
        "score": 1.0,
        "box": box,
        "source": "human_review",
    }
    for key in (
        "mask_path",
        "mask_index",
        "segmentation",
        "area",
        "track_id",
        "team",
        "notes",
        "reviewer",
    ):
        if key in annotation:
            normalized[key] = deepcopy(annotation[key])
    return normalized


def _has_mask_evidence(annotation: Mapping[str, Any]) -> bool:
    if annotation.get("segmentation") is not None:
        return True
    return annotation.get("mask_path") is not None and annotation.get("mask_index") is not None


def _mapping(value: Any) -> dict:
    return dict(value) if isinstance(value, Mapping) else {}


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _box(value: Any) -> list[float] | None:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return None
    try:
        box = [float(item) for item in value]
    except (TypeError, ValueError, OverflowError):
        return None
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    return box


def _manifest_image_record(image: Mapping[str, Any], *, detections: list[dict]) -> dict:
    record = {
        "video": image.get("video"),
        "source_group": image.get("source_group"),
        "frame_index": image.get("frame_index"),
        "split": image.get("split"),
        "image_path": image.get("image_path"),
        "width": image.get("width"),
        "height": image.get("height"),
        "detections": detections,
    }
    if image.get("review_task") == "verify_absence":
        record["ball_absent_verified"] = True
    return record


def _record_source_group(record: Mapping[str, Any]) -> str:
    value = record.get("source_group") or record.get("video") or "unknown"
    return str(value)


def _validate_split_ratios(*, train_ratio: float, val_ratio: float) -> None:
    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if val_ratio < 0 or train_ratio + val_ratio > 1:
        raise ValueError("train_ratio + val_ratio must be 1 or below")


def _balanced_source_splits(
    source_keys: list[str],
    *,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, str]:
    _validate_split_ratios(train_ratio=train_ratio, val_ratio=val_ratio)
    if not source_keys:
        return {}
    counts = Counter(source_keys)
    total_frames = sum(counts.values())
    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
    reserve_for_test = 1 if test_ratio > 0 and len(counts) >= 3 else 0
    val_keys = _select_weighted_groups(
        counts,
        target_frames=total_frames * val_ratio,
        reserve_groups=1 + reserve_for_test,
    )
    remaining = Counter({key: count for key, count in counts.items() if key not in val_keys})
    test_keys = _select_weighted_groups(
        remaining,
        target_frames=total_frames * test_ratio,
        reserve_groups=1,
    )
    return {
        source_key: (
            "val"
            if source_key in val_keys
            else "test"
            if source_key in test_keys
            else "train"
        )
        for source_key in sorted(counts)
    }


def _select_weighted_groups(
    counts: Counter[str],
    *,
    target_frames: float,
    reserve_groups: int,
) -> set[str]:
    if target_frames <= 0 or len(counts) <= reserve_groups:
        return set()
    selected: set[str] = set()
    selected_frames = 0
    candidates = sorted(counts.items(), key=lambda item: (item[1], item[0]))
    max_selected = max(0, len(candidates) - reserve_groups)
    for source_key, frame_count in candidates:
        if len(selected) >= max_selected:
            break
        current_error = abs(target_frames - selected_frames)
        next_error = abs(target_frames - (selected_frames + frame_count))
        if next_error <= current_error or not selected:
            selected.add(source_key)
            selected_frames += frame_count
        if selected_frames == target_frames:
            break
    return selected


def _issue(image_index: int, code: str, message: str) -> dict:
    return {"image_index": image_index, "code": code, "message": message}
