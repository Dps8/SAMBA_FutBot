from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

from .io_utils import read_detections, write_json
from .types import Detection


def export_pseudolabel_candidates(
    detections_path: str | Path,
    out_path: str | Path,
    *,
    classes: Iterable[str] | None = None,
    min_score: float = 0.60,
    min_area: float = 1.0,
    require_mask: bool = True,
    root: str | Path | None = None,
) -> dict:
    detections = read_detections(detections_path)
    selected_classes = {item.strip() for item in classes or [] if item.strip()}
    base = Path(root).resolve() if root else Path(detections_path).resolve().parent
    candidates = []
    rejected = Counter()

    for det in detections:
        reason = _rejection_reason(
            det,
            classes=selected_classes,
            min_score=min_score,
            min_area=min_area,
            require_mask=require_mask,
        )
        if reason:
            rejected[reason] += 1
            continue
        candidates.append(_candidate_record(det, base=base))

    manifest = {
        "schema": "samba_futbot.pseudolabel_candidates.v1",
        "source_detections": str(detections_path),
        "filters": {
            "classes": sorted(selected_classes) if selected_classes else "all",
            "min_score": min_score,
            "min_area": min_area,
            "require_mask": require_mask,
            "root": str(base),
        },
        "summary": {
            "input_detections": len(detections),
            "candidates": len(candidates),
            "rejected": dict(sorted(rejected.items())),
            "candidates_by_class": dict(
                sorted(Counter(item["class_name"] for item in candidates).items())
            ),
        },
        "candidates": candidates,
    }
    write_json(out_path, manifest)
    return manifest


def _rejection_reason(
    det: Detection,
    *,
    classes: set[str],
    min_score: float,
    min_area: float,
    require_mask: bool,
) -> str | None:
    if classes and det.class_name not in classes:
        return "class_filter"
    if det.score < min_score:
        return "low_score"
    if (det.area or 0.0) < min_area:
        return "small_area"
    if require_mask and not det.mask_path:
        return "missing_mask"
    return None


def _candidate_record(det: Detection, *, base: Path) -> dict:
    mask_path = _relative_or_original(det.mask_path, base=base) if det.mask_path else None
    return {
        "frame_index": det.frame_index,
        "class_name": det.class_name,
        "score": det.score,
        "box": list(det.box),
        "area": det.area,
        "track_id": det.track_id,
        "team": det.team,
        "prompt": det.prompt,
        "mask_path": mask_path,
        "mask_index": det.extra.get("mask_index") if isinstance(det.extra, dict) else None,
        "source": det.extra.get("source") if isinstance(det.extra, dict) else None,
    }


def _relative_or_original(path: str, *, base: Path) -> str:
    raw = Path(path)
    try:
        return str(raw.resolve().relative_to(base)).replace("\\", "/")
    except (OSError, ValueError):
        return str(raw).replace("\\", "/")
