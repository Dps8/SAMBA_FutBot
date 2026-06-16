from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io_utils import ensure_parent, read_json, write_json

COCO_METRICS = (
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_1",
    "AR_10",
    "AR_100",
    "AR_small",
    "AR_medium",
    "AR_large",
)


def subset_coco_ground_truth(
    ground_truth: Mapping[str, Any],
    image_ids: Sequence[int],
) -> dict[str, Any]:
    """Restrict COCO ground truth to the images that were actually inferred."""
    selected = {int(image_id) for image_id in image_ids}
    images = [
        dict(image)
        for image in ground_truth.get("images", [])
        if int(image.get("id", -1)) in selected
    ]
    present = {int(image["id"]) for image in images}
    missing = sorted(selected - present)
    if missing:
        raise ValueError(f"prediction image IDs missing from ground truth: {missing}")

    annotations = [
        dict(annotation)
        for annotation in ground_truth.get("annotations", [])
        if int(annotation.get("image_id", -1)) in present
    ]
    result = {
        key: value
        for key, value in ground_truth.items()
        if key not in {"images", "annotations"}
    }
    result["images"] = images
    result["annotations"] = annotations
    return result


def evaluate_coco_predictions(
    ground_truth: Mapping[str, Any] | str | Path,
    predictions: Sequence[Mapping[str, Any]] | str | Path,
    *,
    iou_type: str = "segm",
) -> dict[str, Any]:
    """Evaluate predictions against the exact set of predicted COCO image IDs."""
    gt_data = _read_mapping(ground_truth, label="ground truth")
    pred_data = _read_predictions(predictions)
    image_ids = sorted({int(prediction["image_id"]) for prediction in pred_data})
    if not image_ids:
        raise ValueError("predictions contain no image IDs")

    subset = subset_coco_ground_truth(gt_data, image_ids)
    category_ids = {int(category["id"]) for category in subset.get("categories", [])}
    filtered_predictions = [
        dict(prediction)
        for prediction in pred_data
        if int(prediction.get("image_id", -1)) in image_ids
        and int(prediction.get("category_id", -1)) in category_ids
    ]

    coco_gt, coco_dt = _build_coco_objects(subset, filtered_predictions)
    overall = _run_coco_eval(coco_gt, coco_dt, image_ids=image_ids, iou_type=iou_type)
    categories = {}
    for category in subset.get("categories", []):
        category_id = int(category["id"])
        categories[str(category["name"])] = {
            "id": category_id,
            "metrics": _run_coco_eval(
                coco_gt,
                coco_dt,
                image_ids=image_ids,
                category_ids=[category_id],
                iou_type=iou_type,
            ),
        }

    return {
        "schema": "samba_futbot.sam3_coco_evaluation.v1",
        "iou_type": iou_type,
        "images": len(subset["images"]),
        "annotations": len(subset["annotations"]),
        "predictions": len(filtered_predictions),
        "image_ids": image_ids,
        "overall": overall,
        "categories": categories,
    }


def compare_coco_evaluations(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two exact-subset COCO evaluations."""
    baseline_ids = list(baseline.get("image_ids", []))
    candidate_ids = list(candidate.get("image_ids", []))
    if baseline_ids != candidate_ids:
        raise ValueError("baseline and candidate were not evaluated on the same image IDs")

    overall = _metric_comparison(
        baseline.get("overall", {}),
        candidate.get("overall", {}),
    )
    category_names = sorted(
        set(baseline.get("categories", {})) | set(candidate.get("categories", {}))
    )
    categories = {}
    for name in category_names:
        baseline_metrics = baseline.get("categories", {}).get(name, {}).get("metrics", {})
        candidate_metrics = candidate.get("categories", {}).get(name, {}).get("metrics", {})
        categories[name] = _metric_comparison(baseline_metrics, candidate_metrics)

    ap_delta = overall["AP"]["delta"]
    verdict = "improved" if ap_delta > 0 else "regressed" if ap_delta < 0 else "unchanged"
    return {
        "schema": "samba_futbot.sam3_finetune_comparison.v1",
        "verdict": verdict,
        "images": len(baseline_ids),
        "image_ids": baseline_ids,
        "baseline": dict(baseline),
        "candidate": dict(candidate),
        "overall": overall,
        "categories": categories,
    }


def evaluate_and_compare_coco_files(
    ground_truth_path: str | Path,
    baseline_path: str | Path,
    candidate_path: str | Path,
    *,
    out_path: str | Path | None = None,
    report_path: str | Path | None = None,
    iou_type: str = "segm",
) -> dict[str, Any]:
    baseline = evaluate_coco_predictions(
        ground_truth_path,
        baseline_path,
        iou_type=iou_type,
    )
    candidate = evaluate_coco_predictions(
        ground_truth_path,
        candidate_path,
        iou_type=iou_type,
    )
    comparison = compare_coco_evaluations(baseline, candidate)
    comparison["inputs"] = {
        "ground_truth": str(ground_truth_path),
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
    }
    if out_path is not None:
        write_json(out_path, comparison)
    if report_path is not None:
        write_finetune_comparison_markdown(report_path, comparison)
    return comparison


def write_finetune_comparison_markdown(
    path: str | Path,
    comparison: Mapping[str, Any],
) -> Path:
    output = ensure_parent(path)
    lines = [
        "# SAM3 Fine-tuning Comparison",
        "",
        f"- Verdict: `{comparison.get('verdict', 'unknown')}`",
        f"- Evaluated images: `{comparison.get('images', 0)}`",
        "- Evaluation subset: exact image IDs present in both prediction files",
        "",
        "## Overall Segmentation",
        "",
        "| Metric | Baseline | Candidate | Delta | Relative |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend(_comparison_rows(comparison.get("overall", {})))
    for category, metrics in comparison.get("categories", {}).items():
        lines.extend(
            [
                "",
                f"## Class: {category}",
                "",
                "| Metric | Baseline | Candidate | Delta | Relative |",
                "|---|---:|---:|---:|---:|",
                *_comparison_rows(metrics),
            ]
        )
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _build_coco_objects(
    ground_truth: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
):
    try:
        from pycocotools.coco import COCO
    except ImportError as exc:
        raise RuntimeError(
            "pycocotools is required for SAM3 fine-tuning evaluation"
        ) from exc

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = dict(ground_truth)
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes([dict(prediction) for prediction in predictions])
    return coco_gt, coco_dt


def _run_coco_eval(
    coco_gt,
    coco_dt,
    *,
    image_ids: Sequence[int],
    iou_type: str,
    category_ids: Sequence[int] | None = None,
) -> dict[str, float]:
    try:
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError(
            "pycocotools is required for SAM3 fine-tuning evaluation"
        ) from exc

    evaluator = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    evaluator.params.imgIds = list(image_ids)
    if category_ids is not None:
        evaluator.params.catIds = list(category_ids)
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {
        name: float(value)
        for name, value in zip(COCO_METRICS, evaluator.stats, strict=True)
    }


def _metric_comparison(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, dict[str, float | None]]:
    result = {}
    for metric in COCO_METRICS:
        before = float(baseline.get(metric, -1.0))
        after = float(candidate.get(metric, -1.0))
        delta = after - before
        relative = delta / before if before > 0 else None
        result[metric] = {
            "baseline": before,
            "candidate": after,
            "delta": delta,
            "relative_change": relative,
        }
    return result


def _comparison_rows(metrics: Mapping[str, Mapping[str, Any]]) -> list[str]:
    rows = []
    for metric in COCO_METRICS:
        values = metrics.get(metric, {})
        relative = values.get("relative_change")
        relative_text = f"{float(relative):+.1%}" if relative is not None else "n/a"
        rows.append(
            f"| `{metric}` | "
            f"{float(values.get('baseline', -1.0)):.4f} | "
            f"{float(values.get('candidate', -1.0)):.4f} | "
            f"{float(values.get('delta', 0.0)):+.4f} | "
            f"{relative_text} |"
        )
    return rows


def _read_mapping(value: Mapping[str, Any] | str | Path, *, label: str) -> dict[str, Any]:
    data = read_json(value) if isinstance(value, (str, Path)) else dict(value)
    if not isinstance(data, dict):
        raise ValueError(f"{label} JSON must be an object")
    return data


def _read_predictions(
    value: Sequence[Mapping[str, Any]] | str | Path,
) -> list[dict[str, Any]]:
    data = read_json(value) if isinstance(value, (str, Path)) else list(value)
    if not isinstance(data, list):
        raise ValueError("predictions JSON must be an array")
    if not all(isinstance(prediction, dict) for prediction in data):
        raise ValueError("every prediction must be an object")
    return [dict(prediction) for prediction in data]
