from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .io_utils import read_detections, write_detections, write_json
from .tracking import iou
from .types import Detection


def parse_int_list(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return [int(item) for item in value]


def deduplicate_detections(
    detections: Iterable[Detection], *, iou_threshold: float = 0.9
) -> list[Detection]:
    kept: list[Detection] = []
    grouped: dict[tuple[int, str], list[Detection]] = {}
    for det in detections:
        grouped.setdefault((det.frame_index, det.class_name), []).append(det)

    for key in sorted(grouped):
        selected: list[Detection] = []
        candidates = sorted(grouped[key], key=lambda det: det.score, reverse=True)
        for candidate in candidates:
            if any(iou(candidate.box, existing.box) >= iou_threshold for existing in selected):
                continue
            selected.append(candidate)
        kept.extend(sorted(selected, key=lambda det: (det.object_id is None, str(det.object_id))))
    return sorted(kept, key=lambda det: (det.frame_index, det.class_name, det.score))


def merge_detection_files(
    inputs: Iterable[str | Path],
    out_path: str | Path,
    *,
    iou_threshold: float = 0.9,
) -> list[Detection]:
    detections: list[Detection] = []
    for input_path in inputs:
        detections.extend(read_detections(input_path))
    merged = deduplicate_detections(detections, iou_threshold=iou_threshold)
    write_detections(out_path, merged)
    return merged


def write_window_manifest(
    path: str | Path,
    *,
    video: str | Path,
    windows: list[dict],
    detections: int,
) -> None:
    write_json(
        path,
        {
            "video": str(video),
            "windows": windows,
            "detections": detections,
        },
    )
