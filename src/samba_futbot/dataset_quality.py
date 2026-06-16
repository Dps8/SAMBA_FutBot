from __future__ import annotations

from collections import defaultdict
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from .io_utils import read_json
from .io_utils import ensure_parent

DEFAULT_LOW_SCORE_THRESHOLD = 0.60
DEFAULT_REVIEW_LIMIT = 25
UNKNOWN = "unknown"


def analyze_dataset_quality(
    manifest: Mapping[str, Any],
    *,
    low_score_threshold: float = DEFAULT_LOW_SCORE_THRESHOLD,
    max_review_examples: int = DEFAULT_REVIEW_LIMIT,
) -> dict:
    """Summarize a SAM-compatible frame dataset manifest for adaptation review."""
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a JSON object")
    if max_review_examples < 0:
        raise ValueError("max_review_examples must be non-negative")

    frames = _frame_records(manifest)
    by_class: dict[str, dict[str, int]] = defaultdict(_bucket)
    by_split: dict[str, dict[str, int]] = defaultdict(_bucket)
    by_video: dict[str, dict[str, int]] = defaultdict(_bucket)
    by_detection_source: dict[str, dict[str, int]] = defaultdict(_bucket)
    review_candidates: list[dict] = []
    splits_by_video: dict[str, set[str]] = defaultdict(set)
    image_path_counts: dict[str, int] = defaultdict(int)
    images_by_source_frame: dict[tuple[str, int], list[dict]] = defaultdict(list)

    total_detections = 0
    total_crops = 0
    detections_with_crops = 0
    invalid_boxes = 0
    low_scores = 0
    frames_without_detections = 0

    for image_index, image in enumerate(frames):
        split = _text(image.get("split"), UNKNOWN)
        video = _image_video(manifest, image)
        detections = _detection_records(image)
        image_crop_count = _image_crop_count(image, detections)
        total_crops += image_crop_count
        frames_without_detections += int(not detections)

        by_split[split]["frames"] += 1
        by_split[split]["crops"] += image_crop_count
        by_video[video]["frames"] += 1
        by_video[video]["crops"] += image_crop_count
        splits_by_video[video].add(split)
        image_path = _text(image.get("image_path"), "")
        if image_path:
            image_path_counts[image_path] += 1
        frame_index = _optional_integer(image.get("frame_index"))
        if video != UNKNOWN and frame_index is not None:
            images_by_source_frame[(_path_key(video), frame_index)].append(
                {
                    "video": video,
                    "frame_index": frame_index,
                    "image_path": image.get("image_path"),
                    "split": split,
                    "detections": len(detections),
                }
            )

        classes_in_frame: set[str] = set()
        sources_in_frame: set[str] = set()
        for detection_index, detection in enumerate(detections):
            total_detections += 1
            class_name = _class_name(detection)
            source = _text(detection.get("source"), UNKNOWN)
            score = _optional_number(detection.get("score"))
            has_crop = bool(detection.get("crop_path"))
            box_issue = invalid_box_reason(
                detection.get("box"),
                width=_optional_number(image.get("width")),
                height=_optional_number(image.get("height")),
            )
            is_low_score = score is not None and score < low_score_threshold

            classes_in_frame.add(class_name)
            sources_in_frame.add(source)
            detections_with_crops += int(has_crop)
            invalid_boxes += int(box_issue is not None)
            low_scores += int(is_low_score)

            for bucket in (
                by_class[class_name],
                by_split[split],
                by_video[video],
                by_detection_source[source],
            ):
                bucket["detections"] += 1
                bucket["invalid_boxes"] += int(box_issue is not None)
                bucket["low_scores"] += int(is_low_score)
            by_class[class_name]["crops"] += int(has_crop)
            by_detection_source[source]["crops"] += int(has_crop)

            if box_issue:
                _add_review_candidate(
                    review_candidates,
                    max_review_examples=max_review_examples,
                    reason="invalid_box",
                    detail=box_issue,
                    image=image,
                    image_index=image_index,
                    detection=detection,
                    detection_index=detection_index,
                    video=video,
                    split=split,
                    score=score,
                    low_score_threshold=low_score_threshold,
                )
            if is_low_score:
                _add_review_candidate(
                    review_candidates,
                    max_review_examples=max_review_examples,
                    reason="low_score",
                    detail=f"score below {low_score_threshold:g}",
                    image=image,
                    image_index=image_index,
                    detection=detection,
                    detection_index=detection_index,
                    video=video,
                    split=split,
                    score=score,
                    low_score_threshold=low_score_threshold,
                )

        for class_name in classes_in_frame:
            by_class[class_name]["frames"] += 1
        for source in sources_in_frame:
            by_detection_source[source]["frames"] += 1

    split_leakage_videos = {
        video: sorted(splits)
        for video, splits in sorted(splits_by_video.items())
        if len(splits) > 1
    }
    duplicate_image_paths = sorted(
        path for path, count in image_path_counts.items() if count > 1
    )
    duplicate_source_frames = [
        {
            "video": records[0]["video"],
            "frame_index": frame_index,
            "copies": len(records),
            "images": records,
        }
        for (_, frame_index), records in sorted(images_by_source_frame.items())
        if len(records) > 1
    ]
    duplicate_source_frame_extras = sum(
        group["copies"] - 1 for group in duplicate_source_frames
    )
    summary = {
        "frames": len(frames),
        "frames_without_detections": frames_without_detections,
        "detections": total_detections,
        "crops": total_crops,
        "detections_with_crops": detections_with_crops,
        "classes": len(by_class),
        "splits": len(by_split),
        "videos": len(by_video),
        "invalid_boxes": invalid_boxes,
        "low_scores": low_scores,
        "review_candidates": len(review_candidates),
        "videos_in_multiple_splits": len(split_leakage_videos),
        "duplicate_image_paths": len(duplicate_image_paths),
        "duplicate_source_frame_groups": len(duplicate_source_frames),
        "duplicate_source_frame_extras": duplicate_source_frame_extras,
    }
    return {
        "schema": "samba_futbot.dataset_quality.v1",
        "summary": summary,
        "thresholds": {"low_score": low_score_threshold},
        "by_class": _sorted_buckets(by_class),
        "by_split": _sorted_buckets(by_split),
        "by_video": _sorted_buckets(by_video),
        "by_detection_source": _sorted_buckets(by_detection_source),
        "split_leakage_videos": split_leakage_videos,
        "duplicate_image_paths": duplicate_image_paths,
        "duplicate_source_frames": duplicate_source_frames,
        "review_candidates": review_candidates,
    }


def analyze_dataset_quality_file(
    manifest_path: str | Path,
    *,
    low_score_threshold: float = DEFAULT_LOW_SCORE_THRESHOLD,
    max_review_examples: int = DEFAULT_REVIEW_LIMIT,
) -> dict:
    """Read a dataset manifest JSON file and return its quality summary."""
    data = read_json(manifest_path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dataset manifest object: {manifest_path}")
    report = analyze_dataset_quality(
        data,
        low_score_threshold=low_score_threshold,
        max_review_examples=max_review_examples,
    )
    report["inputs"] = {"manifest": str(manifest_path)}
    return report


def write_dataset_quality_markdown(report: Mapping[str, Any], out_path: str | Path) -> Path:
    """Write a compact Markdown review for a dataset quality report."""
    out = ensure_parent(out_path)
    summary = report.get("summary", {})
    lines = [
        "# Dataset Quality Report",
        "",
        "## Summary",
        "",
        f"- Frames: `{int(summary.get('frames', 0))}`",
        f"- Detections: `{int(summary.get('detections', 0))}`",
        f"- Crops: `{int(summary.get('crops', 0))}`",
        f"- Invalid boxes: `{int(summary.get('invalid_boxes', 0))}`",
        f"- Low-score detections: `{int(summary.get('low_scores', 0))}`",
        f"- Frames without detections: `{int(summary.get('frames_without_detections', 0))}`",
        f"- Videos in multiple splits: `{int(summary.get('videos_in_multiple_splits', 0))}`",
        f"- Duplicate image paths: `{int(summary.get('duplicate_image_paths', 0))}`",
        (
            "- Duplicate source-frame groups: "
            f"`{int(summary.get('duplicate_source_frame_groups', 0))}`"
        ),
        (
            "- Extra source-frame copies: "
            f"`{int(summary.get('duplicate_source_frame_extras', 0))}`"
        ),
        "",
    ]
    lines.extend(_bucket_table("By Class", report.get("by_class", {})))
    lines.extend(_bucket_table("By Split", report.get("by_split", {})))
    lines.extend(_review_table(report.get("review_candidates", [])))
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return out


def invalid_box_reason(
    box: Any,
    *,
    width: float | None = None,
    height: float | None = None,
) -> str | None:
    """Return a compact reason for an unusable xyxy box, or None when it is valid."""
    values = _box_values(box)
    if values is None:
        return "malformed_box"
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        return "non_positive_box"
    if width is not None and width > 0 and (x1 < 0 or x2 > width):
        return "box_outside_width"
    if height is not None and height > 0 and (y1 < 0 or y2 > height):
        return "box_outside_height"
    return None


def _frame_records(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_frames = manifest.get("images")
    if raw_frames is None:
        raw_frames = manifest.get("frames", [])
    if not isinstance(raw_frames, list):
        return []
    return [frame for frame in raw_frames if isinstance(frame, Mapping)]


def _detection_records(image: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_detections = image.get("detections", [])
    if not isinstance(raw_detections, list):
        return []
    return [detection for detection in raw_detections if isinstance(detection, Mapping)]


def _image_crop_count(
    image: Mapping[str, Any],
    detections: list[Mapping[str, Any]],
) -> int:
    raw_crops = image.get("crops")
    if isinstance(raw_crops, list):
        return len(raw_crops)
    return sum(1 for detection in detections if detection.get("crop_path"))


def _image_video(manifest: Mapping[str, Any], image: Mapping[str, Any]) -> str:
    return _text(
        image.get("video")
        or image.get("source_video")
        or image.get("source")
        or manifest.get("source_video"),
        UNKNOWN,
    )


def _class_name(detection: Mapping[str, Any]) -> str:
    return _text(detection.get("class_name") or detection.get("label"), UNKNOWN)


def _add_review_candidate(
    candidates: list[dict],
    *,
    max_review_examples: int,
    reason: str,
    detail: str,
    image: Mapping[str, Any],
    image_index: int,
    detection: Mapping[str, Any],
    detection_index: int,
    video: str,
    split: str,
    score: float | None,
    low_score_threshold: float,
) -> None:
    if len(candidates) >= max_review_examples:
        return
    candidates.append(
        {
            "reason": reason,
            "detail": detail,
            "image_index": image_index,
            "detection_index": detection_index,
            "image_path": image.get("image_path"),
            "frame_index": image.get("frame_index"),
            "split": split,
            "video": video,
            "class_name": _class_name(detection),
            "score": score,
            "low_score_threshold": low_score_threshold,
            "box": detection.get("box"),
            "crop_path": detection.get("crop_path"),
        }
    )


def _bucket() -> dict[str, int]:
    return {
        "frames": 0,
        "detections": 0,
        "crops": 0,
        "invalid_boxes": 0,
        "low_scores": 0,
    }


def _sorted_buckets(buckets: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, int]]:
    return {key: dict(buckets[key]) for key in sorted(buckets)}


def _box_values(box: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(box, list | tuple) or len(box) != 4:
        return None
    values = tuple(_optional_number(value) for value in box)
    if any(value is None for value in values):
        return None
    x1, y1, x2, y2 = values
    return x1, y1, x2, y2


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        number = float(value)
        return number if isfinite(number) else None
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
        return number if isfinite(number) else None
    return None


def _optional_integer(value: Any) -> int | None:
    number = _optional_number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _path_key(value: Any) -> str:
    return str(value).strip().replace("\\", "/").casefold()


def _text(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _bucket_table(title: str, buckets: Any) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Name | Frames | Detections | Crops | Invalid boxes | Low scores |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if not isinstance(buckets, Mapping) or not buckets:
        lines.append("| `none` | 0 | 0 | 0 | 0 | 0 |")
    else:
        for name, bucket in sorted(buckets.items()):
            if not isinstance(bucket, Mapping):
                continue
            lines.append(
                "| "
                f"`{name}` | "
                f"{int(bucket.get('frames', 0))} | "
                f"{int(bucket.get('detections', 0))} | "
                f"{int(bucket.get('crops', 0))} | "
                f"{int(bucket.get('invalid_boxes', 0))} | "
                f"{int(bucket.get('low_scores', 0))} |"
            )
    lines.append("")
    return lines


def _review_table(candidates: Any) -> list[str]:
    lines = [
        "## Review Candidates",
        "",
        "| Reason | Class | Split | Frame | Score | Detail | Image |",
        "|---|---|---|---:|---:|---|---|",
    ]
    if not isinstance(candidates, list) or not candidates:
        lines.append("| `none` | `none` | `none` | 0 | 0.00 | `none` | `none` |")
    else:
        for item in candidates:
            if not isinstance(item, Mapping):
                continue
            score = item.get("score")
            score_text = f"{float(score):.2f}" if score is not None else "0.00"
            lines.append(
                "| "
                f"`{item.get('reason', 'unknown')}` | "
                f"`{item.get('class_name', 'unknown')}` | "
                f"`{item.get('split', 'unknown')}` | "
                f"{int(item.get('frame_index') or 0)} | "
                f"{score_text} | "
                f"`{item.get('detail', '')}` | "
                f"`{item.get('image_path', '')}` |"
            )
    lines.append("")
    return lines
