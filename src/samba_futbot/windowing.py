from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

from .io_utils import read_detections, write_detections, write_json
from .play_state import BALL_CLASSES
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


def filter_edge_ball_detections(
    detections: Iterable[Detection],
    *,
    frame_width: int | None,
    frame_height: int | None,
    border_margin_px: float = 4.0,
) -> list[Detection]:
    if not frame_width or not frame_height or border_margin_px <= 0:
        return list(detections)

    kept: list[Detection] = []
    for det in detections:
        if det.class_name not in BALL_CLASSES:
            kept.append(det)
            continue
        x1, y1, x2, y2 = det.box
        touches_border = (
            x1 <= border_margin_px
            or y1 <= border_margin_px
            or x2 >= frame_width - border_margin_px
            or y2 >= frame_height - border_margin_px
        )
        if not touches_border:
            kept.append(det)
    return kept


def offset_detections(detections: list[Detection], frame_offset: int) -> list[Detection]:
    if frame_offset == 0:
        return detections
    shifted: list[Detection] = []
    for det in detections:
        extra = dict(det.extra)
        extra.setdefault("clip_frame_index", det.frame_index)
        shifted.append(replace(det, frame_index=det.frame_index + frame_offset, extra=extra))
    return shifted


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
