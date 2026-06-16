from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Mapping

from .io_utils import read_json, write_json

UNKNOWN_CLASS = "unknown"


def curate_dataset_manifest(
    manifest: Mapping[str, Any],
    *,
    classes: Iterable[str] | None = None,
    min_score: float = 0.0,
    review_exclusions: Iterable[Mapping[str, Any] | tuple[Any, Any, Any]] | None = None,
    drop_empty_frames: bool = True,
    deduplicate_source_frames: bool = False,
) -> tuple[dict, dict]:
    """Return a curated frame-dataset manifest and its rejection report."""
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a JSON object")
    threshold = _number(min_score)
    if threshold is None:
        raise ValueError("min_score must be a finite number")

    raw_images = manifest.get("images", [])
    if not isinstance(raw_images, list):
        raise ValueError("manifest images must be a list")

    selected_classes = _selected_classes(classes)
    exclusions = _review_exclusion_keys(review_exclusions)
    accepted_by_class: Counter[str] = Counter()
    rejected_by_class: Counter[str] = Counter()
    rejected_by_reason: Counter[str] = Counter()
    rejected_by_reason_and_class: dict[str, Counter[str]] = defaultdict(Counter)
    invalid_box_details: Counter[str] = Counter()
    output_images: list[dict] = []
    dropped_empty_frames = 0
    dropped_duplicate_frames = 0
    input_detections = 0
    selected_image_indices, duplicate_groups = _source_frame_selection(
        raw_images,
        enabled=deduplicate_source_frames,
    )

    for image_index, raw_image in enumerate(raw_images):
        if not isinstance(raw_image, Mapping):
            raise ValueError(f"manifest image at index {image_index} must be an object")
        image = deepcopy(dict(raw_image))
        raw_detections = raw_image.get("detections", [])
        if not isinstance(raw_detections, list):
            raw_detections = []
        if image_index not in selected_image_indices:
            dropped_duplicate_frames += 1
            input_detections += len(raw_detections)
            for raw_detection in raw_detections:
                class_name = _class_name(raw_detection)
                rejected_by_reason["duplicate_source_frame"] += 1
                rejected_by_class[class_name] += 1
                rejected_by_reason_and_class["duplicate_source_frame"][class_name] += 1
            continue

        accepted_detections: list[dict] = []
        for detection_index, raw_detection in enumerate(raw_detections):
            input_detections += 1
            class_name = _class_name(raw_detection)
            reason, detail = _rejection_reason(
                raw_detection,
                image=raw_image,
                detection_index=detection_index,
                selected_classes=selected_classes,
                min_score=threshold,
                exclusions=exclusions,
            )
            if reason is not None:
                rejected_by_reason[reason] += 1
                rejected_by_class[class_name] += 1
                rejected_by_reason_and_class[reason][class_name] += 1
                if reason == "invalid_box" and detail is not None:
                    invalid_box_details[detail] += 1
                continue

            accepted_detections.append(deepcopy(dict(raw_detection)))
            accepted_by_class[class_name] += 1

        image["detections"] = accepted_detections
        if "crops" in image:
            image["crops"] = [
                detection["crop_path"]
                for detection in accepted_detections
                if detection.get("crop_path")
            ]
        if drop_empty_frames and not accepted_detections:
            dropped_empty_frames += 1
            continue
        output_images.append(image)

    summary = _manifest_summary(output_images)
    curated_manifest = deepcopy(dict(manifest))
    curated_manifest["curation"] = {
        "classes": sorted(selected_classes) if selected_classes is not None else "all",
        "min_score": threshold,
        "drop_empty_frames": bool(drop_empty_frames),
        "deduplicate_source_frames": bool(deduplicate_source_frames),
        "review_exclusions": len(exclusions),
    }
    curated_manifest["summary"] = summary
    curated_manifest["images"] = output_images

    accepted_detections = sum(accepted_by_class.values())
    rejected_detections = sum(rejected_by_reason.values())
    by_class = _class_report(accepted_by_class, rejected_by_class, rejected_by_reason_and_class)
    report = {
        "schema": "samba_futbot.dataset_curation_report.v1",
        "filters": deepcopy(curated_manifest["curation"]),
        "reason_precedence": [
            "duplicate_source_frame",
            "review_exclusion",
            "malformed_detection",
            "class_not_selected",
            "invalid_score",
            "score_below_minimum",
            "invalid_box",
        ],
        "summary": {
            "input_frames": len(raw_images),
            "output_frames": len(output_images),
            "dropped_empty_frames": dropped_empty_frames,
            "dropped_duplicate_frames": dropped_duplicate_frames,
            "input_detections": input_detections,
            "accepted_detections": accepted_detections,
            "rejected_detections": rejected_detections,
            "accepted_crops": summary["crops"],
        },
        "accepted_by_class": _sorted_counter(accepted_by_class),
        "rejected_by_class": _sorted_counter(rejected_by_class),
        "rejected_by_reason": _sorted_counter(rejected_by_reason),
        "rejected_by_reason_and_class": {
            reason: _sorted_counter(counts)
            for reason, counts in sorted(rejected_by_reason_and_class.items())
        },
        "invalid_box_details": _sorted_counter(invalid_box_details),
        "by_class": by_class,
        "frames_by_split": deepcopy(summary["frames_by_split"]),
        "duplicate_source_frame_groups": duplicate_groups,
    }
    return curated_manifest, report


def curate_dataset_manifest_file(
    manifest_path: str | Path,
    curated_manifest_path: str | Path,
    report_path: str | Path,
    *,
    classes: Iterable[str] | None = None,
    min_score: float = 0.0,
    review_exclusions: Iterable[Mapping[str, Any] | tuple[Any, Any, Any]] | None = None,
    drop_empty_frames: bool = True,
    deduplicate_source_frames: bool = False,
) -> tuple[dict, dict]:
    """Curate a JSON manifest and write both the curated manifest and report."""
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected dataset manifest object: {manifest_path}")
    curated, report = curate_dataset_manifest(
        manifest,
        classes=classes,
        min_score=min_score,
        review_exclusions=review_exclusions,
        drop_empty_frames=drop_empty_frames,
        deduplicate_source_frames=deduplicate_source_frames,
    )
    report = deepcopy(report)
    report["inputs"] = {"manifest": str(manifest_path)}
    report["outputs"] = {
        "manifest": str(curated_manifest_path),
        "report": str(report_path),
    }
    write_json(curated_manifest_path, curated)
    write_json(report_path, report)
    return curated, report


curate_frame_dataset_manifest = curate_dataset_manifest
curate_frame_dataset_manifest_file = curate_dataset_manifest_file


def _rejection_reason(
    detection: Any,
    *,
    image: Mapping[str, Any],
    detection_index: int,
    selected_classes: set[str] | None,
    min_score: float,
    exclusions: set[tuple[str, int, int]],
) -> tuple[str | None, str | None]:
    exclusion_key = (
        _path_key(image.get("image_path")),
        _integer(image.get("frame_index"), default=-1),
        detection_index,
    )
    if exclusion_key in exclusions:
        return "review_exclusion", None
    if not isinstance(detection, Mapping):
        return "malformed_detection", None

    class_name = _class_name(detection)
    if selected_classes is not None and class_name not in selected_classes:
        return "class_not_selected", None

    score = _number(detection.get("score"))
    if score is None:
        return "invalid_score", None
    if score < min_score:
        return "score_below_minimum", None

    box_detail = _invalid_box_detail(
        detection.get("box"),
        width=_number(image.get("width")),
        height=_number(image.get("height")),
    )
    if box_detail is not None:
        return "invalid_box", box_detail
    return None, None


def _source_frame_selection(
    images: list[Any],
    *,
    enabled: bool,
) -> tuple[set[int], list[dict]]:
    if not enabled:
        return set(range(len(images))), []

    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    ungrouped: set[int] = set()
    for image_index, image in enumerate(images):
        if not isinstance(image, Mapping):
            ungrouped.add(image_index)
            continue
        video = _path_key(
            image.get("video")
            or image.get("source_video")
            or image.get("source")
        )
        frame_index = _integer(image.get("frame_index"))
        if not video or frame_index is None:
            ungrouped.add(image_index)
            continue
        groups[(video, frame_index)].append(image_index)

    selected = set(ungrouped)
    duplicate_groups = []
    for (_, frame_index), indices in sorted(groups.items()):
        best_index = max(indices, key=lambda index: _image_quality(images[index]))
        selected.add(best_index)
        if len(indices) > 1:
            best_image = images[best_index]
            duplicate_groups.append(
                {
                    "video": (
                        best_image.get("video")
                        or best_image.get("source_video")
                        or best_image.get("source")
                    ),
                    "frame_index": frame_index,
                    "kept_image_path": best_image.get("image_path"),
                    "dropped_image_paths": [
                        images[index].get("image_path")
                        for index in indices
                        if index != best_index
                    ],
                }
            )
    return selected, duplicate_groups


def _image_quality(image: Any) -> tuple[int, float, int]:
    if not isinstance(image, Mapping):
        return 0, 0.0, 0
    detections = image.get("detections", [])
    if not isinstance(detections, list):
        return 0, 0.0, 0
    valid_boxes = 0
    score_sum = 0.0
    mapped = 0
    for detection in detections:
        if not isinstance(detection, Mapping):
            continue
        mapped += 1
        if _invalid_box_detail(
            detection.get("box"),
            width=_number(image.get("width")),
            height=_number(image.get("height")),
        ) is None:
            valid_boxes += 1
        score_sum += _number(detection.get("score")) or 0.0
    return valid_boxes, score_sum, mapped


def _manifest_summary(images: list[dict]) -> dict:
    class_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    crop_count = 0
    for image in images:
        split_counts[str(image.get("split", "unknown"))] += 1
        for detection in image.get("detections", []):
            class_counts[_class_name(detection)] += 1
            crop_count += int(bool(detection.get("crop_path")))
    return {
        "frames": len(images),
        "detections": sum(class_counts.values()),
        "crops": crop_count,
        "detections_by_class": _sorted_counter(class_counts),
        "frames_by_split": _sorted_counter(split_counts),
    }


def _class_report(
    accepted: Counter[str],
    rejected: Counter[str],
    rejected_by_reason_and_class: Mapping[str, Counter[str]],
) -> dict[str, dict]:
    report = {}
    for class_name in sorted(set(accepted) | set(rejected)):
        report[class_name] = {
            "accepted": accepted[class_name],
            "rejected": rejected[class_name],
            "rejected_by_reason": {
                reason: counts[class_name]
                for reason, counts in sorted(rejected_by_reason_and_class.items())
                if counts[class_name]
            },
        }
    return report


def _review_exclusion_keys(
    exclusions: Iterable[Mapping[str, Any] | tuple[Any, Any, Any]] | None,
) -> set[tuple[str, int, int]]:
    keys: set[tuple[str, int, int]] = set()
    for index, exclusion in enumerate(exclusions or []):
        if isinstance(exclusion, Mapping):
            values = (
                exclusion.get("image_path"),
                exclusion.get("frame_index"),
                exclusion.get("detection_index"),
            )
        elif isinstance(exclusion, tuple) and len(exclusion) == 3:
            values = exclusion
        else:
            raise ValueError(f"review exclusion at index {index} must contain three fields")
        image_path, frame_index, detection_index = values
        normalized_frame = _integer(frame_index)
        normalized_detection = _integer(detection_index)
        if not _path_key(image_path) or normalized_frame is None or normalized_detection is None:
            raise ValueError(f"review exclusion at index {index} has invalid values")
        if normalized_detection < 0:
            raise ValueError(f"review exclusion at index {index} has invalid detection_index")
        keys.add((_path_key(image_path), normalized_frame, normalized_detection))
    return keys


def _selected_classes(classes: Iterable[str] | None) -> set[str] | None:
    if classes is None:
        return None
    selected = {str(class_name).strip() for class_name in classes if str(class_name).strip()}
    return selected or None


def _invalid_box_detail(
    box: Any,
    *,
    width: float | None,
    height: float | None,
) -> str | None:
    if not isinstance(box, list | tuple) or len(box) != 4:
        return "malformed_box"
    values = tuple(_number(value) for value in box)
    if any(value is None for value in values):
        return "malformed_box"
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        return "non_positive_box"
    if width is not None and width > 0 and (x1 < 0 or x2 > width):
        return "box_outside_width"
    if height is not None and height > 0 and (y1 < 0 or y2 > height):
        return "box_outside_height"
    return None


def _class_name(detection: Any) -> str:
    if not isinstance(detection, Mapping):
        return UNKNOWN_CLASS
    value = detection.get("class_name") or detection.get("label")
    text = str(value).strip() if value is not None else ""
    return text or UNKNOWN_CLASS


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
    else:
        return None
    return number if isfinite(number) else None


def _integer(value: Any, *, default: int | None = None) -> int | None:
    number = _number(value)
    if number is None or not number.is_integer():
        return default
    return int(number)


def _path_key(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().replace("\\", "/").casefold()


def _sorted_counter(counter: Mapping[str, int]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}
