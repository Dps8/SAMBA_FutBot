from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from .io_utils import read_json
from .sam3_finetune_model import DEFAULT_TRAINABLE_PREFIXES

DEFAULT_PROMPTS = {
    "ball": "small orange soccer ball",
    "robots": "robot soccer player",
    "goal_blue": "dark blue soccer goal",
    "goal_yellow": "yellow soccer goal",
}


def prepare_sam3_finetune_config(
    template: str | Path,
    out: str | Path,
    *,
    data_root: str | Path,
    train_json: str | Path,
    val_json: str | Path,
    experiment_dir: str | Path,
    bpe_path: str | Path,
    epochs: int = 1,
    train_limit: int | None = 8,
    val_limit: int | None = 8,
    resolution: int = 1008,
    num_workers: int = 0,
    prompts: Mapping[str, str] | None = None,
    trainable_prefixes: tuple[str, ...] = DEFAULT_TRAINABLE_PREFIXES,
    mode: str = "train",
) -> dict[str, Any]:
    """Derive a local segmentation fine-tuning config from Meta's installed YAML."""
    template_path = Path(template)
    config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("SAM3 template must contain a YAML object")

    train_categories = _coco_categories(train_json)
    val_categories = _coco_categories(val_json)
    if train_categories != val_categories:
        raise ValueError("train and validation COCO categories must match exactly")
    prompt_entries = _prompt_entries(train_categories, prompts)

    _require_positive_int(epochs, "epochs")
    _require_positive_int(resolution, "resolution")
    if train_limit is not None:
        _require_positive_int(train_limit, "train_limit")
    if val_limit is not None:
        _require_positive_int(val_limit, "val_limit")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer")
    if mode not in {"train", "val"}:
        raise ValueError("mode must be 'train' or 'val'")

    paths = config["paths"]
    paths["roboflow_vl_100_root"] = str(Path(data_root).resolve())
    paths["experiment_log_dir"] = str(Path(experiment_dir).resolve())
    paths["bpe_path"] = str(Path(bpe_path).resolve())

    dataset_settings = config["roboflow_train"]
    dataset_settings["num_images"] = train_limit
    dataset_settings["supercategory"] = "samba_futbot"
    dataset_settings["loss"] = _segmentation_loss(config)
    _ensure_rle_decode(dataset_settings["val_transforms"])

    scratch = config["scratch"]
    scratch["enable_segmentation"] = True
    scratch["resolution"] = resolution
    scratch["num_train_workers"] = num_workers
    scratch["num_val_workers"] = num_workers
    scratch["mask_postprocessor"] = {
        "_target_": "sam3.eval.postprocessors.PostProcessImage",
        "max_dets_per_img": -1,
        "iou_type": "segm",
        "use_original_ids": True,
        "use_original_sizes_box": True,
        "use_original_sizes_mask": True,
        "convert_mask_to_rle": True,
        "use_presence": "${scratch.use_presence_eval}",
    }

    trainer = config["trainer"]
    trainer["skip_saving_ckpts"] = False
    trainer["skip_first_val"] = True
    trainer["max_epochs"] = epochs
    trainer["val_epoch_freq"] = 1
    trainer["mode"] = mode
    if mode == "val":
        trainer["skip_saving_ckpts"] = True
    trainer["model"]["_target_"] = (
        "samba_futbot.sam3_finetune_model.build_sam3_adaptation_model"
    )
    trainer["model"]["trainable_prefixes"] = list(trainable_prefixes)

    train_dataset = trainer["data"]["train"]["dataset"]
    val_dataset = trainer["data"]["val"]["dataset"]
    loader = {
        "_target_": "sam3.train.data.coco_json_loaders.COCO_FROM_JSON",
        "prompts": repr(prompt_entries),
        "include_negatives": True,
        "category_chunk_size": len(prompt_entries),
        "_partial_": True,
    }
    train_dataset["coco_json_loader"] = dict(loader)
    train_dataset["img_folder"] = str(Path(data_root).resolve())
    train_dataset["ann_file"] = str(Path(train_json).resolve())
    train_dataset["limit_ids"] = train_limit
    val_dataset["coco_json_loader"] = dict(loader)
    val_dataset["img_folder"] = str(Path(data_root).resolve())
    val_dataset["ann_file"] = str(Path(val_json).resolve())
    val_dataset["limit_ids"] = val_limit

    meter = trainer["meters"]["val"]["roboflow100"]["detection"]
    meter["iou_type"] = "segm"
    meter["dump_dir"] = "${launcher.experiment_log_dir}/dumps/samba_futbot"
    meter["postprocessor"] = "${scratch.mask_postprocessor}"
    evaluator = meter["pred_file_evaluators"][0]
    evaluator["gt_path"] = str(Path(val_json).resolve())
    evaluator["iou_type"] = "segm"

    trainer["checkpoint"]["save_dir"] = "${launcher.experiment_log_dir}/checkpoints"
    trainer["logging"]["log_dir"] = "${launcher.experiment_log_dir}/logs/samba_futbot"
    config["launcher"]["num_nodes"] = 1
    config["launcher"]["gpus_per_node"] = 1
    config["submitit"]["use_cluster"] = False
    config["submitit"].pop("job_array", None)

    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(config, sort_keys=False, allow_unicode=False)
    output.write_text(f"# @package _global_\n{rendered}", encoding="utf-8")
    return {
        "template": str(template_path),
        "out": str(output),
        "categories": train_categories,
        "prompts": prompt_entries,
        "epochs": epochs,
        "train_limit": train_limit,
        "val_limit": val_limit,
        "resolution": resolution,
        "trainable_prefixes": list(trainable_prefixes),
        "mode": mode,
    }


def _coco_categories(path: str | Path) -> list[dict[str, Any]]:
    data = read_json(path)
    categories = data.get("categories") if isinstance(data, dict) else None
    if not isinstance(categories, list) or not categories:
        raise ValueError(f"COCO file has no categories: {path}")
    result = []
    for category in categories:
        if not isinstance(category, dict):
            raise ValueError(f"invalid COCO category in {path}")
        category_id = category.get("id")
        name = str(category.get("name", "")).strip()
        if isinstance(category_id, bool) or not isinstance(category_id, int) or not name:
            raise ValueError(f"invalid COCO category in {path}")
        result.append({"id": category_id, "name": name})
    return sorted(result, key=lambda item: item["id"])


def _prompt_entries(
    categories: list[dict[str, Any]],
    overrides: Mapping[str, str] | None,
) -> list[dict[str, Any]]:
    prompt_map = dict(DEFAULT_PROMPTS)
    prompt_map.update(
        {
            str(class_name).strip(): str(prompt).strip()
            for class_name, prompt in (overrides or {}).items()
        }
    )
    entries = []
    for category in categories:
        name = category["name"]
        prompt = prompt_map.get(name, name).strip()
        if not prompt:
            raise ValueError(f"empty prompt for COCO category: {name}")
        entries.append({"id": category["id"], "name": prompt})
    return entries


def _segmentation_loss(config: Mapping[str, Any]) -> dict[str, Any]:
    base = dict(config["roboflow_train"]["loss"])
    loss_fns = list(base["loss_fns_find"])
    loss_fns.append(
        {
            "_target_": "sam3.train.loss.loss_fns.Masks",
            "focal_alpha": 0.25,
            "focal_gamma": 2.0,
            "weight_dict": {"loss_mask": 200.0, "loss_dice": 10.0},
            "compute_aux": False,
        }
    )
    base["loss_fns_find"] = loss_fns
    return base


def _ensure_rle_decode(transforms: list[dict[str, Any]]) -> None:
    compose = transforms[0]
    pipeline = compose["transforms"]
    target = "sam3.train.transforms.segmentation.DecodeRle"
    if not any(transform.get("_target_") == target for transform in pipeline):
        pipeline.insert(0, {"_target_": target})


def _require_positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
