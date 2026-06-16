from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .io_utils import read_detections, read_json, write_json
from .types import Detection
from .video import require_cv2, video_info


def export_frame_dataset(
    *,
    video_path: str | Path,
    detections_path: str | Path,
    out_dir: str | Path,
    classes: Iterable[str] | None = None,
    min_score: float = 0.60,
    frame_stride: int = 1,
    max_frames: int | None = None,
    crop: bool = True,
    crop_padding_px: int = 8,
    max_detections_per_class_per_frame: int = 8,
    split_strategy: str = "by-video",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
) -> dict:
    cv2 = require_cv2()
    video = Path(video_path)
    output = Path(out_dir)
    frames_dir = output / "frames" / video.stem
    crops_dir = output / "crops" / video.stem
    frames_dir.mkdir(parents=True, exist_ok=True)
    if crop:
        crops_dir.mkdir(parents=True, exist_ok=True)

    detections = read_detections(detections_path)
    selected_by_frame = selected_detections_by_frame(
        detections,
        classes=classes,
        min_score=min_score,
        frame_stride=frame_stride,
        max_frames=max_frames,
        max_detections_per_class_per_frame=max_detections_per_class_per_frame,
    )
    info = video_info(video)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video}")

    images = []
    class_counts = Counter()
    crop_count = 0
    video_split = split_for_key(
        video.stem,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    for frame_index, frame_detections in selected_by_frame.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        height, width = frame.shape[:2]
        frame_path = frames_dir / f"{video.stem}_f{frame_index:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        split = (
            video_split
            if split_strategy == "by-video"
            else split_for_key(f"{video.stem}:{frame_index}", train_ratio=train_ratio, val_ratio=val_ratio)
        )
        image_record = {
            "video": str(video),
            "frame_index": frame_index,
            "split": split,
            "image_path": _relative(frame_path, output),
            "width": width,
            "height": height,
            "detections": [],
        }
        crops = []
        for det_index, det in enumerate(frame_detections):
            class_counts[det.class_name] += 1
            box = clip_box(det.box, width=width, height=height, padding_px=0)
            record = {
                "class_name": det.class_name,
                "score": det.score,
                "box": list(box),
                "track_id": det.track_id,
                "team": det.team,
                "prompt": det.prompt,
                "area": det.area,
                "source": det.extra.get("source") if isinstance(det.extra, dict) else None,
            }
            if det.mask_path:
                record["mask_path"] = _source_artifact_path(
                    det.mask_path,
                    detections_path=Path(detections_path),
                )
                if isinstance(det.extra, dict) and det.extra.get("mask_index") is not None:
                    record["mask_index"] = det.extra["mask_index"]
            if crop:
                crop_box = clip_box(det.box, width=width, height=height, padding_px=crop_padding_px)
                crop_path = (
                    crops_dir
                    / det.class_name
                    / f"{video.stem}_f{frame_index:06d}_{det_index:02d}.jpg"
                )
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                x1, y1, x2, y2 = (int(round(value)) for value in crop_box)
                cv2.imwrite(str(crop_path), frame[y1:y2, x1:x2])
                record["crop_path"] = _relative(crop_path, output)
                record["crop_box"] = list(crop_box)
                crop_count += 1
                crops.append(record["crop_path"])
            image_record["detections"].append(record)
        if crops:
            image_record["crops"] = crops
        images.append(image_record)

    cap.release()
    split_counts = Counter(image["split"] for image in images)
    manifest = {
        "schema": "samba_futbot.frame_dataset.v1",
        "source_video": str(video),
        "source_detections": str(detections_path),
        "video_info": info,
        "filters": {
            "classes": sorted({item.strip() for item in classes or [] if item.strip()}) or "all",
            "min_score": min_score,
            "frame_stride": frame_stride,
            "max_frames": max_frames,
            "crop": crop,
            "crop_padding_px": crop_padding_px,
            "max_detections_per_class_per_frame": max_detections_per_class_per_frame,
            "split_strategy": split_strategy,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
        },
        "summary": {
            "frames": len(images),
            "detections": sum(len(image["detections"]) for image in images),
            "crops": crop_count,
            "detections_by_class": dict(sorted(class_counts.items())),
            "frames_by_split": dict(sorted(split_counts.items())),
        },
        "images": images,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def selected_detections_by_frame(
    detections: Iterable[Detection],
    *,
    classes: Iterable[str] | None = None,
    min_score: float,
    frame_stride: int,
    max_frames: int | None,
    max_detections_per_class_per_frame: int,
) -> dict[int, list[Detection]]:
    selected_classes = {item.strip() for item in classes or [] if item.strip()}
    by_frame: dict[int, dict[str, list[Detection]]] = defaultdict(lambda: defaultdict(list))
    for det in detections:
        if selected_classes and det.class_name not in selected_classes:
            continue
        if det.score < min_score:
            continue
        if frame_stride > 1 and det.frame_index % frame_stride != 0:
            continue
        by_frame[det.frame_index][det.class_name].append(det)

    selected: dict[int, list[Detection]] = {}
    for frame_index in sorted(by_frame):
        if max_frames is not None and len(selected) >= max_frames:
            break
        frame_detections = []
        for class_name in sorted(by_frame[frame_index]):
            candidates = sorted(
                by_frame[frame_index][class_name],
                key=lambda det: det.score,
                reverse=True,
            )
            frame_detections.extend(candidates[:max_detections_per_class_per_frame])
        if frame_detections:
            selected[frame_index] = sorted(frame_detections, key=lambda det: (det.class_name, -det.score))
    return selected


def merge_frame_dataset_manifests(
    manifest_paths: Iterable[str | Path],
    out_path: str | Path,
    *,
    split_strategy: str = "preserve",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
) -> dict:
    paths = [Path(path) for path in manifest_paths]
    loaded_manifests = []
    for path in paths:
        manifest = read_json(path)
        if not isinstance(manifest, dict):
            raise ValueError(f"Expected dataset manifest object: {path}")
        loaded_manifests.append((path, manifest))
    source_keys = [
        _dataset_source_key(path, manifest, image)
        for path, manifest in loaded_manifests
        for image in manifest.get("images", [])
        if isinstance(image, dict)
    ]
    source_splits = (
        _balanced_source_splits(source_keys, train_ratio=train_ratio, val_ratio=val_ratio)
        if split_strategy == "by-source-balanced"
        else {}
    )
    manifests = []
    images = []
    class_counts = Counter()
    split_counts = Counter()
    crop_count = 0
    for path, manifest in loaded_manifests:
        manifests.append(str(path))
        base = path.parent
        for image in manifest.get("images", []):
            if not isinstance(image, dict):
                continue
            merged_image = dict(image)
            source_key = _dataset_source_key(path, manifest, image)
            if source_key in source_splits:
                merged_image["split"] = source_splits[source_key]
            merged_image["image_path"] = _absolute_dataset_path(
                str(image.get("image_path", "")),
                base,
            )
            detections = []
            for detection in image.get("detections", []):
                if not isinstance(detection, dict):
                    continue
                merged_detection = dict(detection)
                class_name = str(merged_detection.get("class_name", "unknown"))
                class_counts[class_name] += 1
                if merged_detection.get("crop_path"):
                    merged_detection["crop_path"] = _absolute_dataset_path(
                        str(merged_detection["crop_path"]),
                        base,
                    )
                    crop_count += 1
                if merged_detection.get("mask_path"):
                    merged_detection["mask_path"] = _absolute_dataset_path(
                        str(merged_detection["mask_path"]),
                        base,
                    )
                detections.append(merged_detection)
            merged_image["detections"] = detections
            if merged_image.get("crops"):
                merged_image["crops"] = [
                    _absolute_dataset_path(str(crop), base) for crop in merged_image["crops"]
                ]
            split_counts[str(merged_image.get("split", "unknown"))] += 1
            images.append(merged_image)

    merged = {
        "schema": "samba_futbot.frame_dataset_merged.v1",
        "sources": manifests,
        "merge": {
            "split_strategy": split_strategy,
            "split_group": "source_video" if split_strategy == "by-source-balanced" else "preserve",
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
        },
        "summary": {
            "frames": len(images),
            "detections": sum(len(image.get("detections", [])) for image in images),
            "crops": crop_count,
            "detections_by_class": dict(sorted(class_counts.items())),
            "frames_by_split": dict(sorted(split_counts.items())),
        },
        "images": images,
    }
    write_json(out_path, merged)
    return merged


def split_for_key(
    key: str,
    *,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
) -> str:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be below 1")
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"


def clip_box(
    box: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
    padding_px: int = 0,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(width - 1), x1 - padding_px))
    y1 = max(0.0, min(float(height - 1), y1 - padding_px))
    x2 = max(x1 + 1.0, min(float(width), x2 + padding_px))
    y2 = max(y1 + 1.0, min(float(height), y2 + padding_px))
    return (x1, y1, x2, y2)


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def _absolute_dataset_path(path: str, base: Path) -> str:
    raw = Path(path)
    if raw.is_absolute():
        return str(raw)
    return str((base / raw).resolve())


def _source_artifact_path(path: str, *, detections_path: Path) -> str:
    raw = Path(path)
    if raw.is_absolute():
        return str(raw.resolve())
    for candidate in (raw, detections_path.resolve().parent / raw):
        if candidate.exists():
            return str(candidate.resolve())
    return str((detections_path.resolve().parent / raw).resolve())


def _balanced_source_splits(
    source_keys: list[str],
    *,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, str]:
    if not source_keys:
        return {}
    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if val_ratio < 0 or train_ratio + val_ratio > 1:
        raise ValueError("train_ratio + val_ratio must be 1 or below")
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
    counts: Counter,
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


def _dataset_source_key(path: Path, manifest: dict, image: dict) -> str:
    source = (
        image.get("video")
        or image.get("source_video")
        or image.get("source")
        or manifest.get("source_video")
        or path
    )
    return str(source).replace("\\", "/").casefold()
