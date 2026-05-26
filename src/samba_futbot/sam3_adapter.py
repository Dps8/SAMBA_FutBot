from __future__ import annotations

import inspect
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .io_utils import write_detections
from .types import Detection


def flatten_prompt_config(prompts: dict[str, Any]) -> list[tuple[str, str]]:
    flattened: list[tuple[str, str]] = []
    for class_name, value in prompts.items():
        if isinstance(value, str):
            flattened.append((class_name, value))
        else:
            flattened.extend((class_name, str(prompt)) for prompt in value)
    return flattened


def run_sam3_video(
    video_path: str | Path,
    out_dir: str | Path,
    *,
    prompts: dict[str, Any],
    backend: str = "official",
    model_id: str = "facebook/sam3.1",
    max_frames: int | None = None,
    stride: int = 1,
    threshold: float = 0.45,
    mask_threshold: float = 0.5,
    use_fa3: bool = False,
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = True,
    prompt_frame_index: int = 0,
) -> list[Detection]:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_pairs = flatten_prompt_config(prompts)
    if backend == "official":
        detections = _run_official_sam3(
            video_path=video_path,
            out_dir=output_dir,
            prompt_pairs=prompt_pairs,
            model_id=model_id,
            max_frames=max_frames,
            threshold=threshold,
            use_fa3=use_fa3,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            prompt_frame_index=prompt_frame_index,
        )
    elif backend == "transformers":
        detections = _run_transformers_sam3(
            video_path=video_path,
            out_dir=output_dir,
            prompt_pairs=prompt_pairs,
            model_id=model_id,
            max_frames=max_frames,
            stride=stride,
            threshold=threshold,
            mask_threshold=mask_threshold,
        )
    else:
        raise ValueError(f"Unknown SAM3 backend: {backend}")

    write_detections(output_dir / "detections.jsonl", detections)
    return detections


def _run_official_sam3(
    *,
    video_path: str | Path,
    out_dir: Path,
    prompt_pairs: list[tuple[str, str]],
    model_id: str,
    max_frames: int | None,
    threshold: float,
    use_fa3: bool,
    offload_video_to_cpu: bool,
    offload_state_to_cpu: bool,
    prompt_frame_index: int,
) -> list[Detection]:
    try:
        from sam3.model_builder import (
            build_sam3_multiplex_video_predictor,
            build_sam3_video_predictor,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Official SAM 3 backend is not installed. Run `pip install -r requirements-sam3.txt`."
        ) from exc

    if "3.1" in model_id:
        predictor = build_sam3_multiplex_video_predictor(use_fa3=use_fa3)
    else:
        predictor = build_sam3_video_predictor()
    _patch_start_session_for_init_signature(predictor)

    detections: list[Detection] = []

    for class_name, prompt in prompt_pairs:
        session_id = None
        try:
            start = predictor.handle_request(
                request={
                    "type": "start_session",
                    "resource_path": str(video_path),
                    "offload_video_to_cpu": offload_video_to_cpu,
                    "offload_state_to_cpu": offload_state_to_cpu,
                }
            )
            session_id = start["session_id"]
            response = predictor.handle_request(
                request={
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": prompt_frame_index,
                    "text": prompt,
                }
            )
            detections.extend(
                _detections_from_processed(
                    processed=response.get("outputs", {}),
                    frame_index=prompt_frame_index,
                    class_name=class_name,
                    prompt=prompt,
                    out_dir=out_dir,
                    threshold=threshold,
                )
            )
            stream_request = {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward",
                "start_frame_index": prompt_frame_index,
                "max_frame_num_to_track": max_frames,
            }
            if hasattr(predictor, "handle_stream_request"):
                for processed in predictor.handle_stream_request(stream_request):
                    frame_index = _frame_index_from_output(processed)
                    if frame_index == prompt_frame_index:
                        continue
                    if max_frames is not None and frame_index >= max_frames:
                        break
                    detections.extend(
                        _detections_from_processed(
                            processed=processed.get("outputs", processed),
                            frame_index=frame_index,
                            class_name=class_name,
                            prompt=prompt,
                            out_dir=out_dir,
                            threshold=threshold,
                        )
                    )
            else:
                for frame_index, processed in _iter_frame_outputs(response.get("outputs", {})):
                    if max_frames is not None and frame_index >= max_frames:
                        continue
                    detections.extend(
                        _detections_from_processed(
                            processed=processed,
                            frame_index=frame_index,
                            class_name=class_name,
                            prompt=prompt,
                            out_dir=out_dir,
                            threshold=threshold,
                        )
                    )
        finally:
            if session_id is not None:
                predictor.handle_request(
                    request={
                        "type": "close_session",
                        "session_id": session_id,
                        "run_gc_collect": True,
                        "clear_cache_threshold": 0,
                    }
                )
    return detections


def _run_transformers_sam3(
    *,
    video_path: str | Path,
    out_dir: Path,
    prompt_pairs: list[tuple[str, str]],
    model_id: str,
    max_frames: int | None,
    stride: int,
    threshold: float,
    mask_threshold: float,
) -> list[Detection]:
    try:
        import torch
        from accelerate import Accelerator
        from transformers import Sam3VideoModel, Sam3VideoProcessor
        from transformers.video_utils import load_video
    except ImportError as exc:
        raise RuntimeError(
            "Transformers SAM 3 backend is not installed. Install project extras and PyTorch."
        ) from exc

    device = Accelerator().device
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = Sam3VideoModel.from_pretrained(model_id).to(device, dtype=dtype)
    processor = Sam3VideoProcessor.from_pretrained(model_id)
    video_frames, _ = load_video(str(video_path))
    if stride > 1:
        video_frames = video_frames[::stride]

    detections: list[Detection] = []
    for class_name, prompt in prompt_pairs:
        session = processor.init_video_session(
            video=video_frames,
            inference_device=device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=dtype,
        )
        session = processor.add_text_prompt(inference_session=session, text=prompt)
        iterator = model.propagate_in_video_iterator(
            inference_session=session,
            max_frame_num_to_track=max_frames,
        )
        for model_outputs in iterator:
            try:
                processed = processor.postprocess_outputs(
                    session,
                    model_outputs,
                    threshold=threshold,
                    mask_threshold=mask_threshold,
                )
            except TypeError:
                processed = processor.postprocess_outputs(session, model_outputs)
            frame_index = int(model_outputs.frame_idx) * max(1, stride)
            detections.extend(
                _detections_from_processed(
                    processed=processed,
                    frame_index=frame_index,
                    class_name=class_name,
                    prompt=prompt,
                    out_dir=out_dir,
                    threshold=threshold,
                )
            )
    return detections


def _iter_frame_outputs(outputs: Any) -> Iterable[tuple[int, Any]]:
    if isinstance(outputs, dict):
        if any(key in outputs for key in ("boxes", "masks", "scores", "object_ids")):
            yield int(outputs.get("frame_index", 0)), outputs
            return
        for key, value in outputs.items():
            try:
                frame_index = int(key)
            except (TypeError, ValueError):
                frame_index = int(value.get("frame_index", 0)) if isinstance(value, dict) else 0
            yield frame_index, value
    elif isinstance(outputs, list):
        for idx, value in enumerate(outputs):
            frame_index = int(value.get("frame_index", idx)) if isinstance(value, dict) else idx
            yield frame_index, value


def _detections_from_processed(
    *,
    processed: Any,
    frame_index: int,
    class_name: str,
    prompt: str,
    out_dir: Path,
    threshold: float,
) -> list[Detection]:
    if not isinstance(processed, dict):
        return []
    processed = _normalize_output_keys(processed)
    boxes = _boxes_from_processed(processed)
    scores = _to_numpy(processed.get("scores"))
    object_ids = _to_numpy(processed.get("object_ids"))
    masks = _normalize_masks(_to_numpy(processed.get("masks")))

    if boxes is None and masks is not None:
        boxes = np.asarray([_mask_to_box(mask) for mask in masks], dtype=np.float32)
    if boxes is None:
        return []

    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return []
    if boxes.ndim == 1:
        if boxes.size != 4:
            return []
        boxes = boxes.reshape(1, 4)
    if boxes.shape[-1] != 4:
        return []
    boxes = _scale_normalized_boxes(boxes, masks)
    count = len(boxes)
    if scores is None:
        scores = np.ones((count,), dtype=np.float32)
    scores = np.asarray(scores).reshape(-1)

    mask_path = None
    if masks is not None and len(masks):
        mask_dir = out_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_path = mask_dir / f"{class_name}_{frame_index:06d}_{_safe_name(prompt)}.npz"
        np.savez_compressed(mask_path, masks=masks.astype(np.uint8))

    detections: list[Detection] = []
    for idx in range(count):
        score = float(scores[idx]) if idx < len(scores) else 1.0
        if score < threshold:
            continue
        object_id = None
        if object_ids is not None and idx < len(object_ids):
            object_id = int(object_ids[idx])
        area = float(masks[idx].sum()) if masks is not None and idx < len(masks) else None
        detections.append(
            Detection(
                frame_index=frame_index,
                class_name=class_name,
                score=score,
                box=tuple(float(v) for v in boxes[idx]),
                prompt=prompt,
                object_id=object_id,
                mask_path=str(mask_path) if mask_path else None,
                area=area,
                extra={"mask_index": idx} if mask_path else {},
            )
        )
    return detections


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _normalize_masks(masks: np.ndarray | None) -> np.ndarray | None:
    if masks is None:
        return None
    masks = np.asarray(masks)
    masks = np.squeeze(masks)
    if masks.ndim == 2:
        masks = masks[None, :, :]
    if masks.ndim != 3:
        return None
    return masks > 0


def _mask_to_box(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def _boxes_from_processed(processed: dict[str, Any]) -> np.ndarray | None:
    boxes = _to_numpy(processed.get("boxes"))
    if boxes is not None:
        return boxes

    for key in ("out_boxes_xywh", "boxes_xywh", "pred_boxes_xywh"):
        xywh = _to_numpy(processed.get(key))
        if xywh is None:
            continue
        xywh = np.asarray(xywh, dtype=np.float32)
        if xywh.size == 0:
            return xywh.reshape(0, 4)
        if xywh.ndim == 1:
            if xywh.size != 4:
                return None
            xywh = xywh.reshape(1, 4)
        boxes_xyxy = xywh.copy()
        boxes_xyxy[..., 2] = xywh[..., 0] + xywh[..., 2]
        boxes_xyxy[..., 3] = xywh[..., 1] + xywh[..., 3]
        return boxes_xyxy
    return None


def _scale_normalized_boxes(
    boxes: np.ndarray, masks: np.ndarray | None
) -> np.ndarray:
    if masks is None or boxes.size == 0:
        return boxes
    if float(np.nanmax(boxes)) > 2.0:
        return boxes
    height, width = masks.shape[-2:]
    scaled = boxes.copy()
    scaled[..., [0, 2]] *= float(width)
    scaled[..., [1, 3]] *= float(height)
    return scaled


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text.lower()).strip("_")[:48]


def _patch_start_session_for_init_signature(predictor: Any) -> None:
    """Adapt SAM 3.1 builds whose init_state does not accept all base kwargs."""

    init_state = getattr(getattr(predictor, "model", None), "init_state", None)
    if init_state is None:
        return
    parameters = inspect.signature(init_state).parameters
    if "offload_state_to_cpu" in parameters:
        return

    def start_session(
        resource_path,
        session_id=None,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
    ):
        init_kwargs = {"resource_path": resource_path}
        if "offload_video_to_cpu" in parameters:
            init_kwargs["offload_video_to_cpu"] = offload_video_to_cpu
        if "offload_state_to_cpu" in parameters:
            init_kwargs["offload_state_to_cpu"] = offload_state_to_cpu
        if hasattr(predictor, "async_loading_frames") and "async_loading_frames" in parameters:
            init_kwargs["async_loading_frames"] = predictor.async_loading_frames
        if hasattr(predictor, "video_loader_type") and "video_loader_type" in parameters:
            init_kwargs["video_loader_type"] = predictor.video_loader_type
        inference_state = predictor.model.init_state(**init_kwargs)

        resolved_session_id = session_id or str(uuid.uuid4())
        predictor._all_inference_states[resolved_session_id] = {
            "state": inference_state,
            "session_id": resolved_session_id,
            "start_time": time.time(),
            "last_use_time": time.time(),
        }
        return {"session_id": resolved_session_id}

    predictor.start_session = start_session


def _frame_index_from_output(output: dict[str, Any]) -> int:
    for key in ("frame_index", "frame_idx", "out_frame_idx"):
        if key in output:
            value = output[key]
            if hasattr(value, "item"):
                value = value.item()
            return int(value)
    return 0


def _normalize_output_keys(output: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "out_binary_masks": "masks",
        "out_mask_logits": "masks",
        "pred_masks": "masks",
        "out_boxes": "boxes",
        "pred_boxes": "boxes",
        "out_scores": "scores",
        "out_probs": "scores",
        "pred_scores": "scores",
        "out_obj_ids": "object_ids",
        "obj_ids": "object_ids",
    }
    normalized = dict(output)
    for source, target in aliases.items():
        if target not in normalized and source in normalized:
            normalized[target] = normalized[source]
    return normalized
