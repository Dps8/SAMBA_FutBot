from __future__ import annotations

from collections.abc import Iterable
from typing import Any

DEFAULT_TRAINABLE_PREFIXES = (
    "segmentation_head.pixel_decoder.",
    "segmentation_head.mask_predictor.",
    "segmentation_head.cross_attend_prompt.",
    "segmentation_head.cross_attn_norm.",
    "segmentation_head.instance_seg_head.",
    "dot_prod_scoring.",
)


def freeze_for_adaptation(
    model: Any,
    trainable_prefixes: Iterable[str] = DEFAULT_TRAINABLE_PREFIXES,
) -> dict[str, int]:
    """Freeze a SAM3 model except for explicitly selected parameter prefixes."""
    prefixes = tuple(str(prefix).strip() for prefix in trainable_prefixes if str(prefix).strip())
    if not prefixes:
        raise ValueError("trainable_prefixes must contain at least one prefix")

    total = 0
    trainable = 0
    matched_prefixes: set[str] = set()
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total += count
        selected = False
        for prefix in prefixes:
            if name.startswith(prefix):
                selected = True
                matched_prefixes.add(prefix)
        parameter.requires_grad_(selected)
        if selected:
            trainable += count

    missing = [prefix for prefix in prefixes if prefix not in matched_prefixes]
    if missing:
        raise ValueError(f"trainable prefixes matched no parameters: {', '.join(missing)}")
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
    }


def build_sam3_adaptation_model(
    *,
    trainable_prefixes: Iterable[str] = DEFAULT_TRAINABLE_PREFIXES,
    **model_kwargs: Any,
):
    """Build the official SAM3 image model and freeze it for head adaptation."""
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(**model_kwargs)
    freeze_for_adaptation(model, trainable_prefixes)
    return model
