from __future__ import annotations

import importlib
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

TRAIN_SCRIPT_CANDIDATES = ("sam3/train/train.py", "sam3/train.py")
README_NAME = "README_TRAIN.md"
VOCAB_NAME = "bpe_simple_vocab_16e6.txt.gz"


def analyze_finetune_preflight(snapshot: Mapping[str, Any]) -> dict:
    """Evaluate an already-loaded SAM3 fine-tuning preflight snapshot."""
    issues: list[dict[str, Any]] = []
    paths = dict(snapshot.get("paths", {}))
    existence = dict(snapshot.get("exists", {}))

    _require_path(issues, existence, "sam3_root", "missing_sam3_root", "SAM3 repository root")
    _require_path(issues, existence, "train_script", "missing_train_script", "official SAM3 train script")
    _require_path(issues, existence, "training_readme", "missing_training_readme", README_NAME)
    _require_path(issues, existence, "vocab", "missing_bpe_vocab", VOCAB_NAME)
    _require_path(issues, existence, "checkpoint", "missing_checkpoint", "initial checkpoint")

    datasets: dict[str, dict[str, Any]] = {}
    for split in ("train", "val"):
        _require_path(
            issues,
            existence,
            f"{split}_json",
            f"missing_{split}_json",
            f"{split} annotation JSON",
        )
        _require_path(
            issues,
            existence,
            f"{split}_images",
            f"missing_{split}_images",
            f"{split} image directory",
        )
        datasets[split] = _analyze_dataset(
            split,
            snapshot.get("datasets", {}).get(split),
            issues,
            image_checks=snapshot.get("image_checks", {}).get(split),
        )

    overlap = sorted(
        set(datasets["train"]["file_names_normalized"])
        & set(datasets["val"]["file_names_normalized"])
    )
    if overlap:
        _issue(
            issues,
            "error",
            "train_val_file_overlap",
            f"Train and val share {len(overlap)} file_name value(s).",
            examples=overlap[:20],
        )

    cuda = _analyze_cuda(snapshot.get("cuda"), issues)
    severity_counts = Counter(issue["severity"] for issue in issues)
    if severity_counts["error"]:
        status = "fail"
    elif severity_counts["warning"]:
        status = "review"
    else:
        status = "ready"

    train_script = paths.get("train_script")
    command = (
        f'{_quote_command_path(snapshot.get("python_executable", "python"))} '
        f'{_quote_command_path(train_script)} --config-name <official_config>'
        if train_script
        else None
    )
    return {
        "schema": "samba_futbot.sam3_finetune_preflight.v1",
        "status": status,
        "summary": {
            "errors": severity_counts["error"],
            "warnings": severity_counts["warning"],
            "train_images": datasets["train"]["images"],
            "train_annotations": datasets["train"]["annotations"],
            "val_images": datasets["val"]["images"],
            "val_annotations": datasets["val"]["annotations"],
            "train_val_overlap": len(overlap),
        },
        "paths": paths,
        "datasets": {
            split: {
                key: value
                for key, value in report.items()
                if key != "file_names_normalized"
            }
            for split, report in datasets.items()
        },
        "cuda": cuda,
        "issues": issues,
        "suggested_command": command,
        "command_inputs": {
            "train_json": paths.get("train_json"),
            "val_json": paths.get("val_json"),
            "train_images": paths.get("train_images"),
            "val_images": paths.get("val_images"),
            "checkpoint": paths.get("checkpoint"),
            "note": "Replace <official_config> with a config from the checked-out SAM3 version.",
        },
    }


def run_finetune_preflight(
    *,
    sam3_root: str | Path,
    checkpoint: str | Path,
    train_json: str | Path,
    val_json: str | Path,
    train_images: str | Path,
    val_images: str | Path,
    check_cuda: bool = False,
    python_executable: str = "python",
) -> dict:
    """Inspect files and datasets required before invoking official SAM3 training."""
    root = Path(sam3_root).expanduser().resolve()
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    train_json_path = Path(train_json).expanduser().resolve()
    val_json_path = Path(val_json).expanduser().resolve()
    train_images_path = Path(train_images).expanduser().resolve()
    val_images_path = Path(val_images).expanduser().resolve()

    train_script = _first_file(root, TRAIN_SCRIPT_CANDIDATES)
    training_readme = root / README_NAME
    vocab = _find_vocab(root)

    paths = {
        "sam3_root": str(root),
        "train_script": str(train_script) if train_script else None,
        "training_readme": str(training_readme),
        "vocab": str(vocab) if vocab else None,
        "checkpoint": str(checkpoint_path),
        "train_json": str(train_json_path),
        "val_json": str(val_json_path),
        "train_images": str(train_images_path),
        "val_images": str(val_images_path),
    }
    exists = {
        "sam3_root": root.is_dir(),
        "train_script": bool(train_script and train_script.is_file()),
        "training_readme": training_readme.is_file(),
        "vocab": bool(vocab and vocab.is_file()),
        "checkpoint": checkpoint_path.is_file(),
        "train_json": train_json_path.is_file(),
        "val_json": val_json_path.is_file(),
        "train_images": train_images_path.is_dir(),
        "val_images": val_images_path.is_dir(),
    }
    train_data, train_load_issue = _read_json_object(train_json_path)
    val_data, val_load_issue = _read_json_object(val_json_path)
    image_checks = {
        "train": _check_referenced_images(train_data, train_images_path),
        "val": _check_referenced_images(val_data, val_images_path),
    }
    snapshot = {
        "paths": paths,
        "exists": exists,
        "datasets": {"train": train_data, "val": val_data},
        "dataset_load_issues": {
            "train": train_load_issue,
            "val": val_load_issue,
        },
        "image_checks": image_checks,
        "cuda": _probe_cuda() if check_cuda else {"checked": False},
        "python_executable": python_executable,
    }
    report = analyze_finetune_preflight(snapshot)
    for split, load_issue in snapshot["dataset_load_issues"].items():
        if load_issue and exists[f"{split}_json"]:
            _issue(
                report["issues"],
                "error",
                f"invalid_{split}_json",
                load_issue,
                split=split,
            )
    if any(issue["severity"] == "error" for issue in report["issues"]):
        report["status"] = "fail"
        report["summary"]["errors"] = sum(
            issue["severity"] == "error" for issue in report["issues"]
        )
    return report


def write_finetune_preflight(
    out_path: str | Path,
    **kwargs: Any,
) -> dict:
    """Run the preflight and persist its JSON report."""
    report = run_finetune_preflight(**kwargs)
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _analyze_dataset(
    split: str,
    data: Any,
    issues: list[dict[str, Any]],
    *,
    image_checks: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        if data is not None:
            _issue(
                issues,
                "error",
                f"invalid_{split}_dataset",
                f"{split} JSON must contain an object.",
                split=split,
            )
        return _empty_dataset_report()

    images = data.get("images")
    annotations = data.get("annotations")
    if not isinstance(images, list):
        _issue(
            issues,
            "error",
            f"invalid_{split}_images_array",
            f"{split} JSON must contain an images array.",
            split=split,
        )
        images = []
    if not isinstance(annotations, list):
        _issue(
            issues,
            "error",
            f"invalid_{split}_annotations_array",
            f"{split} JSON must contain an annotations array.",
            split=split,
        )
        annotations = []

    image_ids: set[Any] = set()
    file_names: list[str] = []
    phrase_counts: Counter[str] = Counter()
    for index, image in enumerate(images):
        if not isinstance(image, Mapping):
            _issue(
                issues,
                "error",
                "invalid_image_record",
                "Image entry must be an object.",
                split=split,
                index=index,
            )
            continue
        image_id = image.get("id")
        if image_id is None:
            _issue(issues, "error", "missing_image_id", "Image is missing id.", split=split, index=index)
        elif image_id in image_ids:
            _issue(
                issues,
                "error",
                "duplicate_image_id",
                f"Duplicate image id: {image_id!r}.",
                split=split,
                index=index,
            )
        else:
            image_ids.add(image_id)

        file_name = image.get("file_name")
        if not isinstance(file_name, str) or not file_name.strip():
            _issue(
                issues,
                "error",
                "missing_file_name",
                "Image is missing a non-empty file_name.",
                split=split,
                index=index,
            )
        else:
            file_names.append(file_name)

        for field in ("is_instance_exhaustive", "is_pixel_exhaustive"):
            if not isinstance(image.get(field), bool):
                _issue(
                    issues,
                    "error",
                    f"invalid_{field}",
                    f"Image field {field} must be boolean.",
                    split=split,
                    index=index,
                )
        phrases = _text_phrases(image.get("text_input"))
        if not phrases:
            _issue(
                issues,
                "error",
                "invalid_text_input",
                "Image text_input must contain at least one phrase.",
                split=split,
                index=index,
            )
        phrase_counts.update(phrases)

    category_names, category_ids = _category_metadata(data, split, issues)
    annotation_class_counts: Counter[str] = Counter()
    normalized_boxes = 0
    rle_segmentations = 0
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, Mapping):
            _issue(
                issues,
                "error",
                "invalid_annotation_record",
                "Annotation entry must be an object.",
                split=split,
                index=index,
            )
            continue
        if annotation.get("image_id") not in image_ids:
            _issue(
                issues,
                "error",
                "unknown_annotation_image",
                f"Annotation references unknown image_id {annotation.get('image_id')!r}.",
                split=split,
                index=index,
            )
        bbox_issue = _normalized_bbox_issue(annotation.get("bbox"))
        if bbox_issue:
            _issue(
                issues,
                "error",
                "invalid_normalized_bbox",
                bbox_issue,
                split=split,
                index=index,
            )
        else:
            normalized_boxes += 1
        rle_issue = _rle_issue(annotation.get("segmentation"))
        if rle_issue:
            _issue(
                issues,
                "error",
                "invalid_segmentation_rle",
                rle_issue,
                split=split,
                index=index,
            )
        else:
            rle_segmentations += 1

        category_id = annotation.get("category_id")
        if category_ids:
            if category_id not in category_ids:
                _issue(
                    issues,
                    "error",
                    "unknown_category_id",
                    f"Annotation category_id {category_id!r} is not declared.",
                    split=split,
                    index=index,
                )
            else:
                annotation_class_counts[category_ids[category_id]] += 1
        else:
            class_name = annotation.get("class_name")
            if isinstance(class_name, str) and class_name.strip():
                annotation_class_counts[class_name.strip()] += 1

    missing_files = list((image_checks or {}).get("missing", []))
    if missing_files:
        _issue(
            issues,
            "error",
            "missing_referenced_images",
            f"{split} image directory is missing {len(missing_files)} referenced file(s).",
            split=split,
            examples=missing_files[:20],
        )

    return {
        "images": len(images),
        "annotations": len(annotations),
        "normalized_boxes": normalized_boxes,
        "rle_segmentations": rle_segmentations,
        "file_names": len(file_names),
        "file_names_normalized": [_normalize_file_name(name) for name in file_names],
        "phrases": dict(sorted(phrase_counts.items())),
        "classes": sorted(category_names | set(annotation_class_counts)),
        "annotations_by_class": dict(sorted(annotation_class_counts.items())),
        "missing_image_files": len(missing_files),
    }


def _empty_dataset_report() -> dict[str, Any]:
    return {
        "images": 0,
        "annotations": 0,
        "normalized_boxes": 0,
        "rle_segmentations": 0,
        "file_names": 0,
        "file_names_normalized": [],
        "phrases": {},
        "classes": [],
        "annotations_by_class": {},
        "missing_image_files": 0,
    }


def _category_metadata(
    data: Mapping[str, Any],
    split: str,
    issues: list[dict[str, Any]],
) -> tuple[set[str], dict[Any, str]]:
    categories = data.get("categories", [])
    if categories is None:
        categories = []
    if not isinstance(categories, list):
        _issue(
            issues,
            "error",
            "invalid_categories_array",
            "categories must be an array when present.",
            split=split,
        )
        return set(), {}
    names: set[str] = set()
    ids: dict[Any, str] = {}
    for index, category in enumerate(categories):
        if not isinstance(category, Mapping):
            _issue(
                issues,
                "error",
                "invalid_category_record",
                "Category entry must be an object.",
                split=split,
                index=index,
            )
            continue
        category_id = category.get("id")
        name = category.get("name")
        if category_id is None or not isinstance(name, str) or not name.strip():
            _issue(
                issues,
                "error",
                "invalid_category",
                "Category requires id and non-empty name.",
                split=split,
                index=index,
            )
            continue
        ids[category_id] = name.strip()
        names.add(name.strip())
    return names, ids


def _normalized_bbox_issue(value: Any) -> str | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return "bbox must be [x, y, width, height]."
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return "bbox values must be numeric."
    x, y, width, height = (float(item) for item in value)
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        return "bbox origin must be non-negative and dimensions must be positive."
    tolerance = 1e-6
    if x > 1 + tolerance or y > 1 + tolerance or width > 1 + tolerance or height > 1 + tolerance:
        return "bbox values must be normalized to the [0, 1] range."
    if x + width > 1 + tolerance or y + height > 1 + tolerance:
        return "bbox must fit inside normalized image bounds."
    return None


def _rle_issue(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return "segmentation must be an RLE object."
    size = value.get("size")
    counts = value.get("counts")
    if (
        not isinstance(size, (list, tuple))
        or len(size) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in size)
    ):
        return "RLE size must contain positive integer [height, width]."
    if isinstance(counts, str):
        return None if counts else "Compressed RLE counts cannot be empty."
    if not isinstance(counts, (list, tuple)) or not counts:
        return "RLE counts must be a non-empty string or integer array."
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in counts):
        return "Uncompressed RLE counts must contain non-negative integers."
    expected = int(size[0]) * int(size[1])
    if sum(counts) != expected:
        return f"Uncompressed RLE counts sum to {sum(counts)}, expected {expected}."
    return None


def _text_phrases(value: Any) -> list[str]:
    if isinstance(value, str):
        phrase = value.strip()
        return [phrase] if phrase else []
    if isinstance(value, (list, tuple)):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def _analyze_cuda(value: Any, issues: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value.get("checked"):
        return {"checked": False}
    report = {
        "checked": True,
        "torch_available": bool(value.get("torch_available")),
        "cuda_available": bool(value.get("cuda_available")),
        "torch_version": value.get("torch_version"),
        "device_count": int(value.get("device_count", 0) or 0),
        "devices": list(value.get("devices", [])),
    }
    if value.get("probe_error"):
        report["probe_error"] = str(value["probe_error"])
    if not report["torch_available"]:
        _issue(
            issues,
            "warning",
            "torch_unavailable",
            "PyTorch is not importable in the selected environment.",
        )
    elif not report["cuda_available"]:
        _issue(
            issues,
            "warning",
            "cuda_unavailable",
            "PyTorch is available but CUDA is not.",
        )
    elif report["device_count"] < 1:
        _issue(
            issues,
            "warning",
            "cuda_device_missing",
            "CUDA reports available but no devices were enumerated.",
        )
    return report


def _probe_cuda() -> dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {
            "checked": True,
            "torch_available": False,
            "cuda_available": False,
            "device_count": 0,
            "devices": [],
        }
    try:
        torch = importlib.import_module("torch")
        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if cuda_available else 0
        devices = [str(torch.cuda.get_device_name(index)) for index in range(device_count)]
        return {
            "checked": True,
            "torch_available": True,
            "cuda_available": cuda_available,
            "torch_version": str(getattr(torch, "__version__", "unknown")),
            "device_count": device_count,
            "devices": devices,
        }
    except Exception as exc:
        return {
            "checked": True,
            "torch_available": False,
            "cuda_available": False,
            "device_count": 0,
            "devices": [],
            "probe_error": f"{type(exc).__name__}: {exc}",
        }


def _read_json_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"Could not read {path}: {type(exc).__name__}: {exc}"
    if not isinstance(value, dict):
        return None, f"{path} must contain a JSON object."
    return value, None


def _check_referenced_images(data: Any, image_dir: Path) -> dict[str, Any]:
    if not image_dir.is_dir() or not isinstance(data, Mapping):
        return {"checked": False, "missing": []}
    images = data.get("images")
    if not isinstance(images, list):
        return {"checked": False, "missing": []}
    missing: list[str] = []
    for image in images:
        if not isinstance(image, Mapping):
            continue
        file_name = image.get("file_name")
        if isinstance(file_name, str) and file_name.strip():
            candidate = image_dir / Path(file_name.replace("\\", "/"))
            if not candidate.is_file():
                missing.append(file_name)
    return {"checked": True, "missing": missing}


def _first_file(root: Path, candidates: tuple[str, ...]) -> Path | None:
    for relative in candidates:
        candidate = root / relative
        if candidate.is_file():
            return candidate
    return None


def _find_vocab(root: Path) -> Path | None:
    direct_candidates = (
        root / VOCAB_NAME,
        root / "assets" / VOCAB_NAME,
        root / "sam3" / "assets" / VOCAB_NAME,
    )
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate
    if not root.is_dir():
        return None
    try:
        return next(root.rglob(VOCAB_NAME), None)
    except OSError:
        return None


def _require_path(
    issues: list[dict[str, Any]],
    existence: Mapping[str, Any],
    key: str,
    code: str,
    label: str,
) -> None:
    if not existence.get(key):
        _issue(issues, "error", code, f"Required {label} was not found.")


def _normalize_file_name(value: str) -> str:
    return os.path.normpath(value.strip().replace("\\", "/")).replace("\\", "/").casefold()


def _quote_command_path(value: Any) -> str:
    text = str(value or "")
    return f'"{text}"' if any(character.isspace() for character in text) else text


def _issue(
    issues: list[dict[str, Any]],
    severity: str,
    code: str,
    message: str,
    **context: Any,
) -> None:
    issue = {"severity": severity, "code": code, "message": message}
    issue.update({key: value for key, value in context.items() if value is not None})
    issues.append(issue)
