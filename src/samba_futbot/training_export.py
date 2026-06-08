from __future__ import annotations

from collections import defaultdict
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


def yolo_lines_for_image(image: Mapping[str, Any], class_ids: Mapping[str, int]) -> list[str]:
    """Return YOLO detection label lines for one manifest image record."""
    width = float(image["width"])
    height = float(image["height"])
    lines = []
    for detection in image.get("detections", []):
        class_name = detection.get("class_name")
        if class_name not in class_ids:
            continue
        values = _yolo_bbox(detection.get("box", []), width=width, height=height)
        if values is None:
            continue
        parts = [str(class_ids[class_name]), *(_format_float(value) for value in values)]
        lines.append(" ".join(parts))
    return lines


def export_yolo_detection(manifest: Mapping[str, Any] | str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Write YOLO label txt files, image lists, classes.txt, and data.yaml."""
    data, manifest_base = _load_manifest(manifest)
    output = Path(out_dir)
    categories = _categories(data)
    class_names = [category["name"] for category in categories]
    class_ids = {name: index for index, name in enumerate(class_names)}
    grouped_images = _images_by_split(data)

    (output / "labels").mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "classes.txt").write_text("\n".join(class_names) + ("\n" if class_names else ""), encoding="utf-8")

    image_lists: dict[str, Path] = {}
    label_paths: list[Path] = []
    label_manifest = []
    for split, images in grouped_images.items():
        labels_dir = output / "labels" / split
        labels_dir.mkdir(parents=True, exist_ok=True)
        list_path = output / "images" / f"{split}.txt"
        list_lines = []
        for index, image in enumerate(images, start=1):
            image_path = _source_image_path(image["image_path"], manifest_base)
            label_path = labels_dir / _label_filename(image, index=index)
            yolo_lines = yolo_lines_for_image(image, class_ids)
            label_text = "\n".join(yolo_lines) + ("\n" if yolo_lines else "")
            label_path.write_text(label_text, encoding="utf-8")
            label_paths.append(label_path)
            list_lines.append(image_path)
            label_manifest.append(
                {
                    "split": split,
                    "image_path": image_path,
                    "label_path": _relative_to_output(label_path, output),
                }
            )
        list_path.write_text("\n".join(list_lines) + ("\n" if list_lines else ""), encoding="utf-8")
        image_lists[split] = list_path

    data_yaml = output / "data.yaml"
    data_yaml.write_text(_data_yaml(class_names, grouped_images.keys()), encoding="utf-8")
    write_json(output / "manifest.json", {"labels": label_manifest})

    return {
        "data_yaml": data_yaml,
        "classes": output / "classes.txt",
        "image_lists": image_lists,
        "labels": label_paths,
        "manifest": output / "manifest.json",
    }


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


def _images_by_split(manifest: Mapping[str, Any]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    split_names = _splits(manifest)
    for image in manifest.get("images", []):
        split = image.get("split") if split_names != ["all"] else "all"
        groups[split or "all"].append(image)
    return {split: groups.get(split, []) for split in split_names}


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


def _yolo_bbox(box: list[float], *, width: float, height: float) -> tuple[float, float, float, float] | None:
    if width <= 0 or height <= 0:
        return None
    clipped = _clip_xyxy(box, width=width, height=height)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0 or box_height <= 0:
        return None
    return (
        ((x1 + x2) / 2.0) / width,
        ((y1 + y2) / 2.0) / height,
        box_width / width,
        box_height / height,
    )


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


def _source_image_path(image_path: str, manifest_base: Path | None) -> str:
    path = Path(image_path)
    if path.is_absolute() or manifest_base is None:
        return str(path)
    return str((manifest_base / path).resolve())


def _label_filename(image: Mapping[str, Any], *, index: int) -> str:
    stem = Path(str(image["image_path"])).with_suffix("").as_posix()
    safe = "".join(char if char.isalnum() else "_" for char in stem).strip("_")
    return f"{index:06d}_{safe or 'image'}.txt"


def _relative_to_output(path: Path, output: Path) -> str:
    return path.resolve().relative_to(output.resolve()).as_posix()


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _data_yaml(class_names: list[str], splits: object) -> str:
    split_lines = []
    for split in splits:
        split_lines.append(f"{split}: images/{split}.txt")
    names = ", ".join(_yaml_quote(name) for name in class_names)
    return "\n".join(
        [
            "path: .",
            *split_lines,
            f"nc: {len(class_names)}",
            f"names: [{names}]",
            "",
        ]
    )


def _yaml_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
