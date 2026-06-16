from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .io_utils import read_json, write_json

KNOWN_SPLITS = ("train", "val", "test")


def manifest_to_coco_detection(
    manifest: Mapping[str, Any] | str | Path,
    *,
    split: str | None = None,
    image_root: str | Path | None = None,
) -> dict:
    """Convert a frame dataset manifest to COCO detection/segmentation JSON data."""
    data, manifest_dir = _load_manifest(manifest)
    return _manifest_to_coco_detection(
        data,
        manifest_dir=manifest_dir,
        split=split,
        image_root=Path(image_root).resolve() if image_root is not None else None,
    )


def _manifest_to_coco_detection(
    data: Mapping[str, Any],
    *,
    manifest_dir: Path | None,
    split: str | None,
    image_root: Path | None,
) -> dict:
    categories = _categories(data)
    category_ids = {category["name"]: category["id"] for category in categories}
    source_images = _filtered_images(data, split=split)

    images = []
    annotations = []
    mask_report = {
        "referenced": 0,
        "exported": 0,
        "failed": 0,
        "issues": [],
    }
    annotation_id = 1
    for image_id, image in enumerate(source_images, start=1):
        images.append(
            {
                "id": image_id,
                "file_name": _coco_image_path(image["image_path"], image_root=image_root),
                "width": image["width"],
                "height": image["height"],
                **({"split": image["split"]} if "split" in image else {}),
            }
        )
        for detection in image.get("detections", []):
            class_name = detection.get("class_name")
            if class_name not in category_ids:
                continue
            bbox = _coco_bbox(detection.get("box", []), width=image["width"], height=image["height"])
            if bbox is None:
                continue
            _, _, box_width, box_height = bbox
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_ids[class_name],
                "bbox": bbox,
                "area": box_width * box_height,
                "iscrowd": 0,
            }
            if detection.get("mask_path") is not None or detection.get("mask_index") is not None:
                mask_report["referenced"] += 1
                mask, issue = _load_detection_mask(
                    detection,
                    manifest_dir=manifest_dir,
                    width=int(image["width"]),
                    height=int(image["height"]),
                )
                if mask is None:
                    mask_report["failed"] += 1
                    mask_report["issues"].append(
                        {
                            "image_path": image["image_path"],
                            "class_name": class_name,
                            "mask_path": detection.get("mask_path"),
                            "mask_index": detection.get("mask_index"),
                            "reason": issue,
                        }
                    )
                else:
                    annotation["segmentation"] = _encode_uncompressed_rle(mask)
                    annotation["area"] = int(mask.sum())
                    mask_report["exported"] += 1
            annotations.append(annotation)
            annotation_id += 1

    result = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    if mask_report["referenced"]:
        result["samba_futbot_export"] = {"masks": mask_report}
    return result


def export_coco_detection(
    manifest: Mapping[str, Any] | str | Path,
    out_dir: str | Path,
    *,
    image_root: str | Path | None = None,
) -> dict[str, Path]:
    """Write COCO detection JSON files, preserving split-specific exports when present."""
    data, manifest_dir = _load_manifest(manifest)
    output = Path(out_dir)
    resolved_image_root = Path(image_root).resolve() if image_root is not None else None
    splits = _splits(data)
    paths: dict[str, Path] = {}

    if splits == ["all"]:
        path = output / "annotations.json"
        write_json(
            path,
            _manifest_to_coco_detection(
                data,
                manifest_dir=manifest_dir,
                split=None,
                image_root=resolved_image_root,
            ),
        )
        paths["all"] = path
        return paths

    annotations_dir = output / "annotations"
    for split in splits:
        path = annotations_dir / f"{split}.json"
        write_json(
            path,
            _manifest_to_coco_detection(
                data,
                manifest_dir=manifest_dir,
                split=split,
                image_root=resolved_image_root,
            ),
        )
        paths[split] = path

    all_path = annotations_dir / "all.json"
    write_json(
        all_path,
        _manifest_to_coco_detection(
            data,
            manifest_dir=manifest_dir,
            split=None,
            image_root=resolved_image_root,
        ),
    )
    paths["all"] = all_path
    return paths


def export_balanced_coco_subset(
    annotations: Mapping[str, Any] | str | Path,
    out_path: str | Path,
    *,
    focus_classes: list[str],
    negative_ratio: float = 1.0,
    max_positive_images: int | None = None,
    seed: int = 123,
    focus_only: bool = False,
) -> dict[str, Any]:
    """Create a deterministic training subset centered on selected classes."""
    if negative_ratio < 0:
        raise ValueError("negative_ratio must be non-negative")
    if max_positive_images is not None and max_positive_images <= 0:
        raise ValueError("max_positive_images must be positive")

    data = _read_coco_mapping(annotations)
    requested = {str(name).strip() for name in focus_classes if str(name).strip()}
    if not requested:
        raise ValueError("focus_classes must contain at least one class")

    categories = {
        int(category["id"]): str(category["name"])
        for category in data.get("categories", [])
    }
    unknown = sorted(requested - set(categories.values()))
    if unknown:
        raise ValueError(f"focus classes missing from COCO categories: {unknown}")
    focus_ids = {
        category_id
        for category_id, category_name in categories.items()
        if category_name in requested
    }
    positive_ids = {
        int(annotation["image_id"])
        for annotation in data.get("annotations", [])
        if int(annotation.get("category_id", -1)) in focus_ids
    }
    all_image_ids = [int(image["id"]) for image in data.get("images", [])]
    positive_ordered = [image_id for image_id in all_image_ids if image_id in positive_ids]
    negative_ordered = [image_id for image_id in all_image_ids if image_id not in positive_ids]

    rng = random.Random(seed)
    if max_positive_images is not None and len(positive_ordered) > max_positive_images:
        positive_ordered = sorted(rng.sample(positive_ordered, max_positive_images))
    negative_count = min(
        len(negative_ordered),
        int(round(len(positive_ordered) * negative_ratio)),
    )
    selected_negatives = (
        sorted(rng.sample(negative_ordered, negative_count))
        if negative_count < len(negative_ordered)
        else negative_ordered
    )
    selected_ids = set(positive_ordered) | set(selected_negatives)

    subset = {
        key: value
        for key, value in data.items()
        if key not in {"images", "annotations", "samba_futbot_subset"}
    }
    subset["images"] = [
        dict(image)
        for image in data.get("images", [])
        if int(image["id"]) in selected_ids
    ]
    subset_annotations = [
        dict(annotation)
        for annotation in data.get("annotations", [])
        if int(annotation["image_id"]) in selected_ids
    ]
    if focus_only:
        subset_annotations = [
            annotation
            for annotation in subset_annotations
            if int(annotation["category_id"]) in focus_ids
        ]
        subset["categories"] = [
            dict(category)
            for category in data.get("categories", [])
            if int(category["id"]) in focus_ids
        ]
    subset["annotations"] = subset_annotations
    class_counts = {
        class_name: sum(
            1
            for annotation in subset["annotations"]
            if categories.get(int(annotation["category_id"])) == class_name
        )
        for class_name in sorted(categories.values())
    }
    subset["samba_futbot_subset"] = {
        "strategy": "focus_class_balanced",
        "focus_classes": sorted(requested),
        "focus_only": focus_only,
        "seed": seed,
        "negative_ratio": negative_ratio,
        "negative_semantics": "no_focus_annotation_not_human_verified",
        "source_images": len(all_image_ids),
        "positive_images": len(positive_ordered),
        "negative_images": len(selected_negatives),
        "selected_images": len(subset["images"]),
        "selected_annotations": len(subset["annotations"]),
        "class_annotations": class_counts,
    }
    write_json(out_path, subset)
    return subset["samba_futbot_subset"]


def _load_manifest(manifest: Mapping[str, Any] | str | Path) -> tuple[dict, Path | None]:
    if isinstance(manifest, (str, Path)):
        path = Path(manifest)
        data = read_json(path)
        if not isinstance(data, dict):
            raise ValueError("manifest JSON must be an object")
        return data, path.parent
    return dict(manifest), None


def _read_coco_mapping(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    data = read_json(value) if isinstance(value, (str, Path)) else dict(value)
    if not isinstance(data, dict):
        raise ValueError("COCO annotations JSON must be an object")
    if not isinstance(data.get("images"), list):
        raise ValueError("COCO annotations must contain an images array")
    if not isinstance(data.get("annotations"), list):
        raise ValueError("COCO annotations must contain an annotations array")
    if not isinstance(data.get("categories"), list):
        raise ValueError("COCO annotations must contain a categories array")
    return data


def _categories(manifest: Mapping[str, Any]) -> list[dict]:
    class_names = sorted(
        {
            detection["class_name"]
            for image in manifest.get("images", [])
            for detection in image.get("detections", [])
            if detection.get("class_name")
        }
    )
    return [{"id": index, "name": name} for index, name in enumerate(class_names, start=1)]


def _splits(manifest: Mapping[str, Any]) -> list[str]:
    splits = {image.get("split") for image in manifest.get("images", []) if image.get("split")}
    if not splits:
        return ["all"]
    known = [split for split in KNOWN_SPLITS if split in splits]
    extras = sorted(split for split in splits if split not in KNOWN_SPLITS)
    return [*known, *extras]


def _filtered_images(manifest: Mapping[str, Any], *, split: str | None) -> list[dict]:
    images = list(manifest.get("images", []))
    if split is None:
        return images
    return [image for image in images if image.get("split") == split]


def _coco_image_path(image_path: str, *, image_root: Path | None) -> str:
    raw = Path(str(image_path))
    if image_root is None:
        return raw.as_posix()
    candidate = raw.resolve() if raw.is_absolute() else (image_root / raw).resolve()
    try:
        return candidate.relative_to(image_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"image_path is outside image_root: {candidate}") from exc


def _coco_bbox(box: list[float], *, width: int, height: int) -> list[float] | None:
    clipped = _clip_xyxy(box, width=float(width), height=float(height))
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0 or box_height <= 0:
        return None
    return [x1, y1, box_width, box_height]


def _clip_xyxy(box: list[float], *, width: float, height: float) -> tuple[float, float, float, float] | None:
    if len(box) != 4:
        return None
    x1, y1, x2, y2 = (float(value) for value in box)
    x1 = max(0.0, min(width, x1))
    y1 = max(0.0, min(height, y1))
    x2 = max(0.0, min(width, x2))
    y2 = max(0.0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


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


def _encode_uncompressed_rle(mask: np.ndarray) -> dict[str, Any]:
    pixels = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
    if pixels.size == 0:
        counts = []
    else:
        pixels = np.asarray(pixels != 0, dtype=np.uint8)
        changes = np.flatnonzero(pixels[1:] != pixels[:-1]) + 1
        boundaries = np.concatenate(([0], changes, [pixels.size]))
        counts_array = np.diff(boundaries)
        if pixels[0]:
            counts_array = np.concatenate(([0], counts_array))
        counts = counts_array.astype(int).tolist()
    height, width = mask.shape
    return {"size": [int(height), int(width)], "counts": counts}
