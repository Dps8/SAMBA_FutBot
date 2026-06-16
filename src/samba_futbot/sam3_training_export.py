from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .io_utils import read_json, write_json

DEFAULT_CLASS_PROMPTS = {
    "ball": "small orange soccer ball",
    "robots": "robot soccer player",
    "goal_blue": "dark blue soccer goal",
    "goal_yellow": "yellow soccer goal",
}
KNOWN_SPLITS = ("train", "val", "test")


def manifest_to_sam3_training(
    manifest: Mapping[str, Any] | str | Path,
    *,
    split: str | None = None,
    class_prompts: Mapping[str, str] | None = None,
    include_negatives: bool = False,
    negative_classes: Iterable[str] | None = None,
    max_negative_classes_per_image: int = 1,
    max_negative_pairs_per_class: int = 100,
    manifest_base: str | Path | None = None,
    image_root: str | Path | None = None,
    is_instance_exhaustive: int | bool = 1,
    is_pixel_exhaustive: int | bool = 1,
) -> tuple[dict, dict]:
    """Convert a frame manifest into SAM 3 SA-Co image/noun-phrase records."""
    data, inferred_base = _load_manifest(manifest)
    base = Path(manifest_base) if manifest_base is not None else inferred_base
    resolved_image_root = Path(image_root).resolve() if image_root is not None else None
    prompts = _class_prompts(class_prompts)
    max_per_image = _non_negative_int(
        max_negative_classes_per_image,
        field="max_negative_classes_per_image",
    )
    max_per_class = _non_negative_int(
        max_negative_pairs_per_class,
        field="max_negative_pairs_per_class",
    )
    default_instance_exhaustive = _binary_flag(
        is_instance_exhaustive,
        field="is_instance_exhaustive",
    )
    default_pixel_exhaustive = _binary_flag(
        is_pixel_exhaustive,
        field="is_pixel_exhaustive",
    )
    allowed_negative_classes = _negative_class_names(negative_classes, prompts)
    source_images = _filtered_images(data, split=split)

    images: list[dict] = []
    annotations: list[dict] = []
    failures: list[dict] = []
    mask_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    annotations_by_class: Counter[str] = Counter()
    pairs_by_class: Counter[str] = Counter()
    negative_pairs_by_class: Counter[str] = Counter()
    negative_budget: Counter[str] = Counter()
    image_np_id = 1
    annotation_id = 1

    for source_index, image in enumerate(source_images):
        image_path = _training_image_path(
            image,
            image_root=resolved_image_root,
        )
        width, height = _image_dimensions(image)
        detections = image.get("detections", [])
        if not isinstance(detections, list):
            detections = []
        detections_by_class: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        present_classes: set[str] = set()
        for detection in detections:
            if not isinstance(detection, Mapping):
                continue
            class_name = _clean_text(detection.get("class_name"))
            if not class_name:
                continue
            present_classes.add(class_name)
            if class_name in prompts:
                detections_by_class[class_name].append(detection)

        for class_name in sorted(detections_by_class):
            valid_annotations: list[dict] = []
            pair_failures = 0
            for detection_index, detection in enumerate(detections_by_class[class_name]):
                mask_counts["referenced"] += 1
                mask, issue = _load_detection_mask(
                    detection,
                    manifest_dir=base,
                    width=width,
                    height=height,
                )
                if mask is None:
                    pair_failures += 1
                    mask_counts["failed"] += 1
                    failures.append(
                        _failure_record(
                            image=image,
                            source_index=source_index,
                            class_name=class_name,
                            detection_index=detection_index,
                            detection=detection,
                            reason=issue or "unknown_mask_error",
                        )
                    )
                    continue
                mask_counts["loaded"] += 1
                if not np.any(mask):
                    pair_failures += 1
                    mask_counts["failed"] += 1
                    failures.append(
                        _failure_record(
                            image=image,
                            source_index=source_index,
                            class_name=class_name,
                            detection_index=detection_index,
                            detection=detection,
                            reason="empty_mask",
                        )
                    )
                    continue

                bbox = _normalized_bbox(
                    detection.get("box"),
                    width=width,
                    height=height,
                )
                if bbox is None:
                    pair_failures += 1
                    failures.append(
                        _failure_record(
                            image=image,
                            source_index=source_index,
                            class_name=class_name,
                            detection_index=detection_index,
                            detection=detection,
                            reason="invalid_box",
                        )
                    )
                    continue

                mask_counts["exported"] += 1
                valid_annotations.append(
                    {
                        "bbox": bbox,
                        "area": float(mask.sum()) / float(width * height),
                        "segmentation": _encode_uncompressed_rle(mask),
                        "category_id": 1,
                        "iscrowd": 0,
                    }
                )

            if not valid_annotations:
                pair_counts["dropped_positive"] += 1
                continue

            instance_flag = _image_flag(
                image,
                "is_instance_exhaustive",
                default_instance_exhaustive,
            )
            pixel_flag = _image_flag(
                image,
                "is_pixel_exhaustive",
                default_pixel_exhaustive,
            )
            if pair_failures:
                instance_flag = 0
                pixel_flag = 0
            images.append(
                _image_np_record(
                    image_np_id=image_np_id,
                    image_path=image_path,
                    width=width,
                    height=height,
                    class_name=class_name,
                        prompt=prompts[class_name],
                        instance_exhaustive=instance_flag,
                        pixel_exhaustive=pixel_flag,
                    )
                )
            for annotation in valid_annotations:
                annotation.update({"id": annotation_id, "image_id": image_np_id})
                annotations.append(annotation)
                annotation_id += 1
                annotations_by_class[class_name] += 1
            pairs_by_class[class_name] += 1
            pair_counts["positive"] += 1
            image_np_id += 1

        if include_negatives and max_per_image and max_per_class:
            absent = [
                class_name
                for class_name in allowed_negative_classes
                if class_name not in present_classes
                and negative_budget[class_name] < max_per_class
            ]
            for class_name in _ordered_negative_classes(absent, image_path)[:max_per_image]:
                images.append(
                    _image_np_record(
                        image_np_id=image_np_id,
                        image_path=image_path,
                        width=width,
                        height=height,
                        class_name=class_name,
                        prompt=prompts[class_name],
                        instance_exhaustive=0,
                        pixel_exhaustive=0,
                    )
                )
                negative_budget[class_name] += 1
                negative_pairs_by_class[class_name] += 1
                pairs_by_class[class_name] += 1
                pair_counts["negative"] += 1
                image_np_id += 1

    output = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "object"}],
    }
    report = {
        "schema": "samba_futbot.sam3_training_export_report.v1",
        "split": split or "all",
        "summary": {
            "source_images": len(source_images),
            "image_np_pairs": len(images),
            "positive_pairs": pair_counts["positive"],
            "negative_pairs": pair_counts["negative"],
            "dropped_positive_pairs": pair_counts["dropped_positive"],
            "annotations": len(annotations),
        },
        "masks": {
            "referenced": mask_counts["referenced"],
            "loaded": mask_counts["loaded"],
            "exported": mask_counts["exported"],
            "failed": mask_counts["failed"],
        },
        "pairs_by_class": _sorted_counter(pairs_by_class),
        "positive_annotations_by_class": _sorted_counter(annotations_by_class),
        "negative_pairs_by_class": _sorted_counter(negative_pairs_by_class),
        "failures": failures,
        "settings": {
            "include_negatives": bool(include_negatives),
            "negative_classes": list(allowed_negative_classes),
            "max_negative_classes_per_image": max_per_image,
            "max_negative_pairs_per_class": max_per_class,
            "class_prompts": dict(prompts),
            "image_root": str(resolved_image_root) if resolved_image_root else None,
        },
    }
    return output, report


def export_sam3_training(
    manifest: Mapping[str, Any] | str | Path,
    out_dir: str | Path,
    *,
    class_prompts: Mapping[str, str] | None = None,
    include_negatives: bool = False,
    negative_classes: Iterable[str] | None = None,
    max_negative_classes_per_image: int = 1,
    max_negative_pairs_per_class: int = 100,
    manifest_base: str | Path | None = None,
    image_root: str | Path | None = None,
    is_instance_exhaustive: int | bool = 1,
    is_pixel_exhaustive: int | bool = 1,
) -> dict:
    """Write official-style SAM 3 annotations per split plus an audit report."""
    data, inferred_base = _load_manifest(manifest)
    base = Path(manifest_base) if manifest_base is not None else inferred_base
    resolved_image_root = Path(image_root).resolve() if image_root is not None else None
    output = Path(out_dir)
    annotation_dir = output / "annotations"
    splits = _manifest_splits(data)
    paths: dict[str, str] = {}
    reports: dict[str, dict] = {}

    for split in splits:
        split_filter = None if split == "all" else split
        annotations, report = manifest_to_sam3_training(
            data,
            split=split_filter,
            class_prompts=class_prompts,
            include_negatives=include_negatives,
            negative_classes=negative_classes,
            max_negative_classes_per_image=max_negative_classes_per_image,
            max_negative_pairs_per_class=max_negative_pairs_per_class,
            manifest_base=base,
            image_root=resolved_image_root,
            is_instance_exhaustive=is_instance_exhaustive,
            is_pixel_exhaustive=is_pixel_exhaustive,
        )
        path = annotation_dir / f"{split}.json"
        write_json(path, annotations)
        paths[split] = str(path)
        reports[split] = report

    if splits != ["all"]:
        annotations, report = manifest_to_sam3_training(
            data,
            class_prompts=class_prompts,
            include_negatives=include_negatives,
            negative_classes=negative_classes,
            max_negative_classes_per_image=max_negative_classes_per_image,
            max_negative_pairs_per_class=max_negative_pairs_per_class,
            manifest_base=base,
            image_root=resolved_image_root,
            is_instance_exhaustive=is_instance_exhaustive,
            is_pixel_exhaustive=is_pixel_exhaustive,
        )
        path = annotation_dir / "all.json"
        write_json(path, annotations)
        paths["all"] = str(path)
        reports["all"] = report

    aggregate = _aggregate_reports(reports)
    aggregate["outputs"] = {"annotations": dict(paths)}
    report_path = output / "report.json"
    aggregate["outputs"]["report"] = str(report_path)
    write_json(report_path, aggregate)
    return {
        "annotations": {key: Path(value) for key, value in paths.items()},
        "report": report_path,
        "summary": aggregate["summary"],
    }


def _load_manifest(
    manifest: Mapping[str, Any] | str | Path,
) -> tuple[dict, Path | None]:
    if isinstance(manifest, (str, Path)):
        path = Path(manifest)
        data = read_json(path)
        if not isinstance(data, dict):
            raise ValueError("manifest JSON must be an object")
        return data, path.parent
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a mapping or JSON path")
    return dict(manifest), None


def _class_prompts(overrides: Mapping[str, str] | None) -> dict[str, str]:
    prompts = dict(DEFAULT_CLASS_PROMPTS)
    for class_name, prompt in (overrides or {}).items():
        clean_class = _clean_text(class_name)
        clean_prompt = _clean_text(prompt)
        if not clean_class or not clean_prompt:
            raise ValueError("class prompt names and phrases must be non-empty")
        prompts[clean_class] = clean_prompt
    return dict(sorted(prompts.items()))


def _negative_class_names(
    requested: Iterable[str] | None,
    prompts: Mapping[str, str],
) -> tuple[str, ...]:
    if requested is None:
        return tuple(sorted(prompts))
    classes = sorted({_clean_text(item) for item in requested if _clean_text(item)})
    unknown = [class_name for class_name in classes if class_name not in prompts]
    if unknown:
        raise ValueError(f"negative classes have no prompt mapping: {', '.join(unknown)}")
    return tuple(classes)


def _filtered_images(manifest: Mapping[str, Any], *, split: str | None) -> list[Mapping[str, Any]]:
    raw_images = manifest.get("images", [])
    if not isinstance(raw_images, list):
        raise ValueError("manifest images must be a list")
    images = []
    for index, image in enumerate(raw_images):
        if not isinstance(image, Mapping):
            raise ValueError(f"manifest image at index {index} must be an object")
        if split is None or image.get("split") == split:
            images.append(image)
    return images


def _manifest_splits(manifest: Mapping[str, Any]) -> list[str]:
    raw_images = manifest.get("images", [])
    if not isinstance(raw_images, list):
        raise ValueError("manifest images must be a list")
    splits = {
        str(image.get("split"))
        for image in raw_images
        if isinstance(image, Mapping) and image.get("split")
    }
    if not splits:
        return ["all"]
    return [
        *(split for split in KNOWN_SPLITS if split in splits),
        *sorted(split for split in splits if split not in KNOWN_SPLITS),
    ]


def _image_path(image: Mapping[str, Any]) -> str:
    path = _clean_text(image.get("image_path"))
    if not path:
        raise ValueError("manifest image_path must be non-empty")
    return path.replace("\\", "/")


def _training_image_path(
    image: Mapping[str, Any],
    *,
    image_root: Path | None,
) -> str:
    raw = Path(_image_path(image))
    if image_root is None:
        return raw.as_posix()
    candidate = raw.resolve() if raw.is_absolute() else (image_root / raw).resolve()
    try:
        return candidate.relative_to(image_root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"image_path is outside image_root: {candidate}"
        ) from exc


def _image_dimensions(image: Mapping[str, Any]) -> tuple[int, int]:
    width = image.get("width")
    height = image.get("height")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("manifest image width must be a positive integer")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("manifest image height must be a positive integer")
    return width, height


def _image_np_record(
    *,
    image_np_id: int,
    image_path: str,
    width: int,
    height: int,
    class_name: str,
    prompt: str,
    instance_exhaustive: int,
    pixel_exhaustive: int,
) -> dict:
    return {
        "id": image_np_id,
        "file_name": image_path,
        "width": width,
        "height": height,
        "text_input": prompt,
        "queried_category": class_name,
        "is_instance_exhaustive": bool(instance_exhaustive),
        "is_pixel_exhaustive": bool(pixel_exhaustive),
    }


def _load_detection_mask(
    detection: Mapping[str, Any],
    *,
    manifest_dir: Path | None,
    width: int,
    height: int,
) -> tuple[np.ndarray | None, str | None]:
    mask_path = detection.get("mask_path")
    mask_index = detection.get("mask_index")
    if not isinstance(mask_path, str) or not mask_path.strip():
        return None, "missing_mask_path"
    if isinstance(mask_index, bool) or not isinstance(mask_index, int) or mask_index < 0:
        return None, "invalid_mask_index"

    path = Path(mask_path)
    if not path.is_absolute():
        if manifest_dir is None:
            return None, "relative_mask_path_without_manifest_base"
        path = manifest_dir / path
    if path.suffix.lower() != ".npz":
        return None, "unsupported_mask_file"
    if not path.is_file():
        return None, "mask_file_not_found"
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "masks" not in archive.files:
                return None, "missing_masks_array"
            masks = archive["masks"]
    except (OSError, ValueError, EOFError):
        return None, "invalid_mask_archive"

    if masks.ndim == 2:
        if mask_index != 0:
            return None, "mask_index_out_of_range"
        mask = masks
    elif masks.ndim >= 3:
        if mask_index >= masks.shape[0]:
            return None, "mask_index_out_of_range"
        mask = np.squeeze(masks[mask_index])
    else:
        return None, "invalid_mask_shape"
    if mask.ndim != 2:
        return None, "invalid_mask_shape"
    if mask.shape != (height, width):
        return None, "mask_size_mismatch"
    return np.asarray(mask != 0, dtype=np.uint8), None


def _normalized_bbox(box: Any, *, width: int, height: int) -> list[float] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in box)
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [
        x1 / width,
        y1 / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    ]


def _encode_uncompressed_rle(mask: np.ndarray) -> dict[str, Any]:
    pixels = np.asarray(mask != 0, dtype=np.uint8).reshape(-1, order="F")
    if pixels.size == 0:
        counts: list[int] = []
    else:
        changes = np.flatnonzero(pixels[1:] != pixels[:-1]) + 1
        boundaries = np.concatenate(([0], changes, [pixels.size]))
        runs = np.diff(boundaries)
        if pixels[0]:
            runs = np.concatenate(([0], runs))
        counts = runs.astype(int).tolist()
    height, width = mask.shape
    return {"size": [int(height), int(width)], "counts": counts}


def _failure_record(
    *,
    image: Mapping[str, Any],
    source_index: int,
    class_name: str,
    detection_index: int,
    detection: Mapping[str, Any],
    reason: str,
) -> dict:
    return {
        "source_image_index": source_index,
        "image_path": image.get("image_path"),
        "class_name": class_name,
        "detection_index": detection_index,
        "mask_path": detection.get("mask_path"),
        "mask_index": detection.get("mask_index"),
        "reason": reason,
    }


def _ordered_negative_classes(classes: list[str], image_path: str) -> list[str]:
    return sorted(
        classes,
        key=lambda class_name: hashlib.sha256(
            f"{image_path}\0{class_name}".encode("utf-8")
        ).hexdigest(),
    )


def _image_flag(image: Mapping[str, Any], field: str, default: int) -> int:
    value = image.get(field, default)
    return _binary_flag(value, field=field)


def _binary_flag(value: Any, *, field: str) -> int:
    if value in (0, False):
        return 0
    if value in (1, True):
        return 1
    raise ValueError(f"{field} must be 0 or 1")


def _non_negative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _clean_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def _aggregate_reports(reports: Mapping[str, Mapping[str, Any]]) -> dict:
    preferred = reports.get("all")
    if preferred is not None:
        summary = dict(preferred["summary"])
        masks = dict(preferred["masks"])
        failures = list(preferred["failures"])
    else:
        summary_counter: Counter[str] = Counter()
        mask_counter: Counter[str] = Counter()
        failures = []
        for report in reports.values():
            summary_counter.update(report["summary"])
            mask_counter.update(report["masks"])
            failures.extend(report["failures"])
        summary = dict(summary_counter)
        masks = dict(mask_counter)
    return {
        "schema": "samba_futbot.sam3_training_export_report.v1",
        "summary": summary,
        "masks": masks,
        "failures": failures,
        "splits": {key: dict(value["summary"]) for key, value in reports.items()},
    }
