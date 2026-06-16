from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from copy import deepcopy
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Mapping

from .io_utils import read_json, write_json

DEFAULT_HOLDOUT_SPLIT = "val"
DEFAULT_HOLDOUT_SEED = 2026
DEFAULT_BALL_REVIEW_SEED = 2027


def select_ball_review_set(
    manifest: Mapping[str, Any],
    *,
    positive_frames: int,
    negative_frames: int,
    seed: int = DEFAULT_BALL_REVIEW_SEED,
    class_name: str = "ball",
    source_group_mode: str = "original-video",
    min_frame_gap: int = 5,
) -> tuple[dict, dict]:
    """Select a review set for ball masks and human-verified ball absence."""
    _validate_ball_review_arguments(
        manifest,
        positive_frames=positive_frames,
        negative_frames=negative_frames,
        seed=seed,
        class_name=class_name,
        source_group_mode=source_group_mode,
        min_frame_gap=min_frame_gap,
    )
    raw_images = manifest.get("images")
    if not isinstance(raw_images, list):
        raise ValueError("manifest images must be a list")

    candidates = [
        _ball_review_candidate(manifest, image, image_index, class_name=class_name)
        for image_index, image in enumerate(raw_images)
    ]
    positives = [candidate for candidate in candidates if candidate["has_focus_class"]]
    negatives = [candidate for candidate in candidates if not candidate["has_focus_class"]]
    selected_positives = _balanced_take_with_gap(
        positives,
        limit=positive_frames,
        seed=seed,
        tier="positive",
        source_group_mode=source_group_mode,
        min_frame_gap=min_frame_gap,
    )
    selected_negatives = _balanced_take_with_gap(
        negatives,
        limit=negative_frames,
        seed=seed,
        tier="negative",
        source_group_mode=source_group_mode,
        min_frame_gap=min_frame_gap,
    )
    selected = [
        *_review_records(selected_positives, review_task="verify_mask"),
        *_review_records(selected_negatives, review_task="verify_absence"),
    ]
    selected.sort(
        key=lambda item: (
            item["review_task"],
            _path_key(item["source_group"]),
            item["frame_index"],
            _path_key(item["image_path"]),
        )
    )
    config = {
        "strategy": "ball-positive-negative-source-balanced",
        "class_name": class_name,
        "positive_frames": positive_frames,
        "negative_frames": negative_frames,
        "seed": seed,
        "source_group_mode": source_group_mode,
        "min_frame_gap": min_frame_gap,
    }
    fingerprint = _selection_fingerprint(config, selected)
    review = {
        "schema": "samba_futbot.ball_review_set.v1",
        "source_schema": manifest.get("schema"),
        "selection": deepcopy(config),
        "selection_fingerprint": deepcopy(fingerprint),
        "annotation_policy": {
            "ground_truth_source": "human",
            "pseudo_detections_copied": True,
            "positive_review_task": "verify_or_correct_ball_mask",
            "negative_review_task": "confirm_ball_absent",
        },
        "summary": {
            "frames": len(selected),
            "positive_frames": len(selected_positives),
            "negative_frames": len(selected_negatives),
            "source_groups": len({record["source_group"] for record in selected}),
        },
        "images": selected,
    }
    report = _ball_review_report(
        raw_count=len(raw_images),
        positives=positives,
        negatives=negatives,
        selected_positives=selected_positives,
        selected_negatives=selected_negatives,
        config=config,
        fingerprint=fingerprint,
    )
    return review, report


def select_ball_review_set_file(
    manifest_path: str | Path,
    annotation_path: str | Path,
    report_path: str | Path,
    *,
    positive_frames: int,
    negative_frames: int,
    seed: int = DEFAULT_BALL_REVIEW_SEED,
    class_name: str = "ball",
    source_group_mode: str = "original-video",
    min_frame_gap: int = 5,
) -> tuple[dict, dict]:
    """Read a manifest and write a ball-specific human review template."""
    try:
        manifest = read_json(manifest_path)
    except JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected dataset manifest object: {manifest_path}")

    review, report = select_ball_review_set(
        manifest,
        positive_frames=positive_frames,
        negative_frames=negative_frames,
        seed=seed,
        class_name=class_name,
        source_group_mode=source_group_mode,
        min_frame_gap=min_frame_gap,
    )
    report = deepcopy(report)
    report["inputs"] = {"manifest": str(manifest_path)}
    report["outputs"] = {
        "annotations": str(annotation_path),
        "report": str(report_path),
    }
    write_json(annotation_path, review)
    write_json(report_path, report)
    return review, report


def select_human_holdout(
    manifest: Mapping[str, Any],
    *,
    max_frames: int,
    preferred_split: str = DEFAULT_HOLDOUT_SPLIT,
    seed: int = DEFAULT_HOLDOUT_SEED,
) -> tuple[dict, dict]:
    """Select a deterministic, video-balanced set for independent human labels."""
    _validate_arguments(
        manifest,
        max_frames=max_frames,
        preferred_split=preferred_split,
        seed=seed,
    )
    raw_images = manifest.get("images")
    if not isinstance(raw_images, list):
        raise ValueError("manifest images must be a list")

    candidates = [
        _candidate_from_image(manifest, image, image_index)
        for image_index, image in enumerate(raw_images)
    ]
    unique_candidates, duplicate_source_frames = _deduplicate_source_frames(
        candidates,
        preferred_split=preferred_split,
        seed=seed,
    )
    preferred = [
        candidate
        for candidate in unique_candidates
        if candidate["split"] == preferred_split
    ]
    fallback = [
        candidate
        for candidate in unique_candidates
        if candidate["split"] != preferred_split
    ]

    selected = _balanced_take(
        preferred,
        limit=max_frames,
        seed=seed,
        tier="preferred",
    )
    preferred_selected = len(selected)
    if len(selected) < max_frames:
        selected.extend(
            _balanced_take(
                fallback,
                limit=max_frames - len(selected),
                seed=seed,
                tier="fallback",
            )
        )

    selected.sort(
        key=lambda item: (
            _path_key(item["video"]),
            item["frame_index"],
            _path_key(item["image_path"]),
        )
    )
    config = {
        "max_frames": max_frames,
        "preferred_split": preferred_split,
        "seed": seed,
        "strategy": "preferred-split-video-balanced-round-robin",
        "deduplicate_by": ["video", "frame_index"],
    }
    fingerprint = _selection_fingerprint(config, selected)
    annotation_images = [_annotation_record(candidate) for candidate in selected]
    holdout = {
        "schema": "samba_futbot.human_holdout.v1",
        "source_schema": manifest.get("schema"),
        "selection": deepcopy(config),
        "selection_fingerprint": deepcopy(fingerprint),
        "annotation_policy": {
            "ground_truth_source": "human",
            "pseudo_detections_copied": False,
            "expected_classes_are_stratification_metadata": True,
        },
        "summary": {
            "frames": len(annotation_images),
            "videos": len({record["video"] for record in annotation_images}),
            "preferred_split_frames": preferred_selected,
            "fallback_split_frames": len(annotation_images) - preferred_selected,
        },
        "images": annotation_images,
    }
    report = _selection_report(
        raw_count=len(raw_images),
        unique_candidates=unique_candidates,
        duplicate_source_frames=duplicate_source_frames,
        selected=selected,
        preferred_selected=preferred_selected,
        config=config,
        fingerprint=fingerprint,
    )
    return holdout, report


def select_human_holdout_file(
    manifest_path: str | Path,
    annotation_path: str | Path,
    report_path: str | Path,
    *,
    max_frames: int,
    preferred_split: str = DEFAULT_HOLDOUT_SPLIT,
    seed: int = DEFAULT_HOLDOUT_SEED,
) -> tuple[dict, dict]:
    """Read a manifest and write the human annotation template and report."""
    try:
        manifest = read_json(manifest_path)
    except JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected dataset manifest object: {manifest_path}")

    holdout, report = select_human_holdout(
        manifest,
        max_frames=max_frames,
        preferred_split=preferred_split,
        seed=seed,
    )
    report = deepcopy(report)
    report["inputs"] = {"manifest": str(manifest_path)}
    report["outputs"] = {
        "annotations": str(annotation_path),
        "report": str(report_path),
    }
    write_json(annotation_path, holdout)
    write_json(report_path, report)
    return holdout, report


build_human_holdout = select_human_holdout
build_human_holdout_file = select_human_holdout_file


def _validate_arguments(
    manifest: Mapping[str, Any],
    *,
    max_frames: int,
    preferred_split: str,
    seed: int,
) -> None:
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a JSON object")
    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0:
        raise ValueError("max_frames must be a positive integer")
    if not isinstance(preferred_split, str) or not preferred_split.strip():
        raise ValueError("preferred_split must be a non-empty string")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")


def _validate_ball_review_arguments(
    manifest: Mapping[str, Any],
    *,
    positive_frames: int,
    negative_frames: int,
    seed: int,
    class_name: str,
    source_group_mode: str,
    min_frame_gap: int,
) -> None:
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a JSON object")
    for field_name, value in {
        "positive_frames": positive_frames,
        "negative_frames": negative_frames,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field_name} must be a non-negative integer")
    if positive_frames + negative_frames <= 0:
        raise ValueError("at least one review frame must be requested")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not isinstance(class_name, str) or not class_name.strip():
        raise ValueError("class_name must be a non-empty string")
    if source_group_mode not in {"video", "original-video"}:
        raise ValueError("source_group_mode must be 'video' or 'original-video'")
    if isinstance(min_frame_gap, bool) or not isinstance(min_frame_gap, int) or min_frame_gap < 0:
        raise ValueError("min_frame_gap must be a non-negative integer")


def _candidate_from_image(
    manifest: Mapping[str, Any],
    image: Any,
    image_index: int,
) -> dict:
    if not isinstance(image, Mapping):
        raise ValueError(f"manifest image at index {image_index} must be an object")

    video = _required_text(
        image.get("video")
        or image.get("source_video")
        or image.get("source")
        or manifest.get("source_video"),
        field="video",
        image_index=image_index,
    )
    image_path = _required_text(
        image.get("image_path"),
        field="image_path",
        image_index=image_index,
    )
    split = _required_text(
        image.get("split"),
        field="split",
        image_index=image_index,
    )
    frame_index = _integer(image.get("frame_index"))
    if frame_index is None or frame_index < 0:
        raise ValueError(
            f"manifest image at index {image_index} has invalid frame_index"
        )
    width = _positive_number(image.get("width"))
    height = _positive_number(image.get("height"))
    if width is None:
        raise ValueError(f"manifest image at index {image_index} has invalid width")
    if height is None:
        raise ValueError(f"manifest image at index {image_index} has invalid height")

    detections = image.get("detections", [])
    if detections is None:
        detections = []
    if not isinstance(detections, list):
        raise ValueError(
            f"manifest image at index {image_index} detections must be a list"
        )
    expected_classes = sorted(
        {
            class_name
            for detection in detections
            if isinstance(detection, Mapping)
            for class_name in [_detection_class(detection)]
            if class_name
        }
    )
    return {
        "video": video,
        "frame_index": frame_index,
        "image_path": image_path,
        "width": image.get("width"),
        "height": image.get("height"),
        "split": split,
        "expected_classes": expected_classes,
    }


def _ball_review_candidate(
    manifest: Mapping[str, Any],
    image: Any,
    image_index: int,
    *,
    class_name: str,
) -> dict:
    base = _candidate_from_image(manifest, image, image_index)
    assert isinstance(image, Mapping)
    detections = image.get("detections", [])
    if not isinstance(detections, list):
        detections = []
    focus_detections = [
        deepcopy(dict(detection))
        for detection in detections
        if isinstance(detection, Mapping)
        and _detection_class(detection) == class_name
    ]
    video = base["video"]
    base.update(
        {
            "source_group": _source_group(video, mode="original-video"),
            "has_focus_class": bool(focus_detections),
            "focus_detections": focus_detections,
            "candidate_kind": "positive" if focus_detections else "negative_candidate",
        }
    )
    return base


def _balanced_take_with_gap(
    candidates: list[dict],
    *,
    limit: int,
    seed: int,
    tier: str,
    source_group_mode: str,
    min_frame_gap: int,
) -> list[dict]:
    if limit <= 0 or not candidates:
        return []
    by_group: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        group = _source_group(candidate["video"], mode=source_group_mode)
        item = deepcopy(candidate)
        item["source_group"] = group
        by_group[group].append(item)
    for group, records in by_group.items():
        records.sort(
            key=lambda item: _stable_digest(seed, tier, group, _candidate_identity(item))
        )
    group_order = sorted(
        by_group,
        key=lambda group: _stable_digest(seed, tier, "group", group),
    )

    selected = []
    selected_by_group: dict[str, list[int]] = defaultdict(list)
    while len(selected) < limit:
        added = False
        for group in group_order:
            records = by_group[group]
            chosen_index = _first_gap_compatible_index(
                records,
                selected_by_group[group],
                min_frame_gap=min_frame_gap,
            )
            if chosen_index is None:
                continue
            chosen = records.pop(chosen_index)
            selected.append(chosen)
            selected_by_group[group].append(int(chosen["frame_index"]))
            added = True
            if len(selected) == limit:
                break
        if not added:
            break
    return selected


def _first_gap_compatible_index(
    records: list[dict],
    selected_frames: list[int],
    *,
    min_frame_gap: int,
) -> int | None:
    for index, record in enumerate(records):
        frame_index = int(record["frame_index"])
        if all(abs(frame_index - selected) >= min_frame_gap for selected in selected_frames):
            return index
    return None


def _review_records(candidates: list[dict], *, review_task: str) -> list[dict]:
    return [_ball_review_record(candidate, review_task=review_task) for candidate in candidates]


def _ball_review_record(candidate: Mapping[str, Any], *, review_task: str) -> dict:
    return {
        "video": candidate["video"],
        "source_group": candidate["source_group"],
        "frame_index": candidate["frame_index"],
        "image_path": candidate["image_path"],
        "width": candidate["width"],
        "height": candidate["height"],
        "split": candidate["split"],
        "expected_classes": list(candidate["expected_classes"]),
        "review_task": review_task,
        "candidate_kind": candidate["candidate_kind"],
        "candidate_detections": deepcopy(list(candidate["focus_detections"])),
        "annotation_status": "pending",
        "annotations": [],
        "ball_absent_verified": None if review_task == "verify_absence" else False,
    }


def _ball_review_report(
    *,
    raw_count: int,
    positives: list[dict],
    negatives: list[dict],
    selected_positives: list[dict],
    selected_negatives: list[dict],
    config: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
) -> dict:
    selected = [*selected_positives, *selected_negatives]
    return {
        "schema": "samba_futbot.ball_review_report.v1",
        "selection": deepcopy(dict(config)),
        "selection_fingerprint": deepcopy(dict(fingerprint)),
        "summary": {
            "input_frames": raw_count,
            "candidate_positive_frames": len(positives),
            "candidate_negative_frames": len(negatives),
            "selected_positive_frames": len(selected_positives),
            "selected_negative_frames": len(selected_negatives),
            "selected_frames": len(selected),
            "selected_source_groups": len({record["source_group"] for record in selected}),
        },
        "selected_by_source_group": dict(
            sorted(Counter(record["source_group"] for record in selected).items())
        ),
        "selected_by_task": dict(
            sorted(
                Counter(
                    "verify_mask" if record in selected_positives else "verify_absence"
                    for record in selected
                ).items()
            )
        ),
        "selected_frames": [
            {
                "video": record["video"],
                "source_group": record["source_group"],
                "frame_index": record["frame_index"],
                "image_path": record["image_path"],
                "split": record["split"],
                "candidate_kind": record["candidate_kind"],
            }
            for record in selected
        ],
    }


def _deduplicate_source_frames(
    candidates: list[dict],
    *,
    preferred_split: str,
    seed: int,
) -> tuple[list[dict], int]:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for candidate in candidates:
        groups[(_path_key(candidate["video"]), candidate["frame_index"])].append(
            candidate
        )

    unique = []
    duplicates = 0
    for source_key, records in sorted(groups.items()):
        ordered = sorted(
            records,
            key=lambda item: (
                item["split"] != preferred_split,
                _stable_digest(seed, "duplicate", _candidate_identity(item)),
            ),
        )
        selected = deepcopy(ordered[0])
        selected["expected_classes"] = sorted(
            {
                class_name
                for record in records
                for class_name in record["expected_classes"]
            }
        )
        unique.append(selected)
        duplicates += len(records) - 1
    return unique, duplicates


def _balanced_take(
    candidates: list[dict],
    *,
    limit: int,
    seed: int,
    tier: str,
) -> list[dict]:
    if limit <= 0 or not candidates:
        return []

    by_video: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        by_video[_path_key(candidate["video"])].append(candidate)
    for video, records in by_video.items():
        records.sort(
            key=lambda item: _stable_digest(
                seed,
                tier,
                video,
                _candidate_identity(item),
            )
        )
    video_order = sorted(
        by_video,
        key=lambda video: _stable_digest(seed, tier, "video", video),
    )

    selected = []
    while len(selected) < limit:
        added = False
        for video in video_order:
            records = by_video[video]
            if not records:
                continue
            selected.append(records.pop(0))
            added = True
            if len(selected) == limit:
                break
        if not added:
            break
    return selected


def _annotation_record(candidate: Mapping[str, Any]) -> dict:
    return {
        "video": candidate["video"],
        "frame_index": candidate["frame_index"],
        "image_path": candidate["image_path"],
        "width": candidate["width"],
        "height": candidate["height"],
        "split": candidate["split"],
        "expected_classes": list(candidate["expected_classes"]),
        "annotation_status": "pending",
        "annotations": [],
    }


def _selection_fingerprint(config: Mapping[str, Any], selected: list[dict]) -> dict:
    payload = {
        "config": config,
        "selected": [
            {
                "video": record["video"],
                "frame_index": record["frame_index"],
                "image_path": record["image_path"],
                "split": record["split"],
                "expected_classes": record["expected_classes"],
            }
            for record in selected
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "algorithm": "sha256",
        "digest": hashlib.sha256(encoded).hexdigest(),
    }


def _selection_report(
    *,
    raw_count: int,
    unique_candidates: list[dict],
    duplicate_source_frames: int,
    selected: list[dict],
    preferred_selected: int,
    config: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
) -> dict:
    by_video: Counter[str] = Counter()
    by_class: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    for record in selected:
        by_video[record["video"]] += 1
        by_split[record["split"]] += 1
        for class_name in record["expected_classes"]:
            by_class[class_name] += 1
    return {
        "schema": "samba_futbot.human_holdout_report.v1",
        "selection": deepcopy(dict(config)),
        "selection_fingerprint": deepcopy(dict(fingerprint)),
        "summary": {
            "input_frames": raw_count,
            "unique_source_frames": len(unique_candidates),
            "duplicate_source_frames_removed": duplicate_source_frames,
            "selected_frames": len(selected),
            "selected_videos": len(by_video),
            "preferred_split_frames": preferred_selected,
            "fallback_split_frames": len(selected) - preferred_selected,
        },
        "selected_by_video": dict(sorted(by_video.items())),
        "selected_by_expected_class": dict(sorted(by_class.items())),
        "selected_by_split": dict(sorted(by_split.items())),
        "selected_frames": [
            {
                "video": record["video"],
                "frame_index": record["frame_index"],
                "image_path": record["image_path"],
                "split": record["split"],
                "expected_classes": list(record["expected_classes"]),
            }
            for record in selected
        ],
    }


def _candidate_identity(candidate: Mapping[str, Any]) -> str:
    return "|".join(
        (
            _path_key(candidate["video"]),
            str(candidate["frame_index"]),
            _path_key(candidate["image_path"]),
        )
    )


def _source_group(video: str, *, mode: str) -> str:
    normalized = str(video).strip().replace("\\", "/")
    if mode == "video":
        return normalized
    stem = Path(normalized).stem
    marker = "_f"
    marker_index = stem.find(marker)
    if marker_index > 0:
        stem = stem[:marker_index]
    suffix = Path(normalized).suffix
    parent = str(Path(normalized).parent).replace("\\", "/")
    name = f"{stem}{suffix}" if suffix else stem
    return f"{parent}/{name}" if parent not in {"", "."} else name


def _stable_digest(seed: int, *parts: str) -> str:
    payload = "\0".join((str(seed), *parts))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _detection_class(detection: Mapping[str, Any]) -> str:
    value = detection.get("class_name") or detection.get("label")
    if not isinstance(value, str):
        return ""
    return value.strip()


def _required_text(value: Any, *, field: str, image_index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest image at index {image_index} has invalid {field}")
    return value.strip()


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    try:
        if float(value) != numeric:
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric


def _positive_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not numeric > 0 or numeric == float("inf"):
        return None
    return numeric


def _path_key(value: Any) -> str:
    return str(value).strip().replace("\\", "/").casefold()
