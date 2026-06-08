from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .io_utils import read_json, write_json

KNOWN_SPLITS = ("train", "val", "test")


def manifest_to_coco_detection(manifest: Mapping[str, Any] | str | Path, *, split: str | None = None) -> dict:
    """Convert a frame dataset manifest to COCO detection JSON data."""
    data, _ = _load_manifest(manifest)
    categories = _categories(data)
    category_ids = {category["name"]: category["id"] for category in categories}
    source_images = _filtered_images(data, split=split)

    images = []
    annotations = []
    annotation_id = 1
    for image_id, image in enumerate(source_images, start=1):
        images.append(
            {
                "id": image_id,
                "file_name": image["image_path"],
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
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_ids[class_name],
                    "bbox": bbox,
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def export_coco_detection(manifest: Mapping[str, Any] | str | Path, out_dir: str | Path) -> dict[str, Path]:
    """Write COCO detection JSON files, preserving split-specific exports when present."""
    data, _ = _load_manifest(manifest)
    output = Path(out_dir)
    splits = _splits(data)
    paths: dict[str, Path] = {}

    if splits == ["all"]:
        path = output / "annotations.json"
        write_json(path, manifest_to_coco_detection(data))
        paths["all"] = path
        return paths

    annotations_dir = output / "annotations"
    for split in splits:
        path = annotations_dir / f"{split}.json"
        write_json(path, manifest_to_coco_detection(data, split=split))
        paths[split] = path

    all_path = annotations_dir / "all.json"
    write_json(all_path, manifest_to_coco_detection(data))
    paths["all"] = all_path
    return paths


def _load_manifest(manifest: Mapping[str, Any] | str | Path) -> tuple[dict, Path | None]:
    if isinstance(manifest, (str, Path)):
        path = Path(manifest)
        data = read_json(path)
        if not isinstance(data, dict):
            raise ValueError("manifest JSON must be an object")
        return data, path.parent
    return dict(manifest), None


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
