import tempfile
import unittest
from pathlib import Path

import yaml

from samba_futbot.finetune_config import prepare_sam3_finetune_config
from samba_futbot.io_utils import write_json


def template_config() -> dict:
    return {
        "paths": {
            "roboflow_vl_100_root": "old",
            "experiment_log_dir": "old",
            "bpe_path": "old",
        },
        "roboflow_train": {
            "num_images": 100,
            "supercategory": "old",
            "val_transforms": [
                {
                    "_target_": "Compose",
                    "transforms": [{"_target_": "Resize"}],
                }
            ],
            "loss": {
                "_target_": "sam3.train.loss.sam3_loss.Sam3LossWrapper",
                "loss_fns_find": [{"_target_": "Boxes"}],
            },
        },
        "scratch": {
            "enable_segmentation": False,
            "resolution": 1008,
            "num_train_workers": 10,
            "num_val_workers": 0,
            "use_presence_eval": True,
        },
        "trainer": {
            "skip_saving_ckpts": True,
            "skip_first_val": True,
            "max_epochs": 20,
            "val_epoch_freq": 10,
            "data": {
                "train": {"dataset": {}},
                "val": {"dataset": {"coco_json_loader": {}}},
            },
            "meters": {
                "val": {
                    "roboflow100": {
                        "detection": {
                            "pred_file_evaluators": [{}],
                        }
                    }
                }
            },
            "checkpoint": {"save_dir": "old"},
            "logging": {"log_dir": "old"},
            "model": {
                "_target_": "sam3.model_builder.build_sam3_image_model",
                "bpe_path": "old",
            },
        },
        "launcher": {"num_nodes": 1, "gpus_per_node": 2},
        "submitit": {"use_cluster": True, "job_array": {"num_tasks": 100}},
    }


class FinetuneConfigTest(unittest.TestCase):
    def test_prepares_segmentation_config_from_official_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "template.yaml"
            template.write_text(yaml.safe_dump(template_config()), encoding="utf-8")
            categories = [
                {"id": 1, "name": "ball"},
                {"id": 2, "name": "robots"},
            ]
            train_json = root / "train.json"
            val_json = root / "val.json"
            write_json(train_json, {"categories": categories})
            write_json(val_json, {"categories": categories})
            out = root / "generated.yaml"

            report = prepare_sam3_finetune_config(
                template,
                out,
                data_root=root,
                train_json=train_json,
                val_json=val_json,
                experiment_dir=root / "run",
                bpe_path=root / "bpe.gz",
                epochs=2,
                train_limit=12,
                val_limit=6,
                resolution=672,
                mode="val",
            )
            rendered = out.read_text(encoding="utf-8")
            config = yaml.safe_load(rendered)

        self.assertEqual(report["categories"], categories)
        self.assertTrue(rendered.startswith("# @package _global_"))
        self.assertTrue(config["scratch"]["enable_segmentation"])
        self.assertEqual(config["trainer"]["max_epochs"], 2)
        self.assertTrue(config["trainer"]["skip_saving_ckpts"])
        self.assertEqual(config["trainer"]["mode"], "val")
        self.assertEqual(
            config["trainer"]["model"]["_target_"],
            "samba_futbot.sam3_finetune_model.build_sam3_adaptation_model",
        )
        self.assertEqual(
            config["trainer"]["model"]["trainable_prefixes"],
            [
                "segmentation_head.pixel_decoder.",
                "segmentation_head.mask_predictor.",
                "segmentation_head.cross_attend_prompt.",
                "segmentation_head.cross_attn_norm.",
                "segmentation_head.instance_seg_head.",
                "dot_prod_scoring.",
            ],
        )
        self.assertEqual(config["trainer"]["data"]["train"]["dataset"]["limit_ids"], 12)
        self.assertEqual(config["trainer"]["data"]["val"]["dataset"]["limit_ids"], 6)
        self.assertIn("small orange soccer ball", config["trainer"]["data"]["train"]["dataset"]["coco_json_loader"]["prompts"])
        self.assertEqual(config["trainer"]["meters"]["val"]["roboflow100"]["detection"]["iou_type"], "segm")
        self.assertEqual(
            config["roboflow_train"]["loss"]["loss_fns_find"][-1]["_target_"],
            "sam3.train.loss.loss_fns.Masks",
        )
        self.assertEqual(
            config["roboflow_train"]["val_transforms"][0]["transforms"][0]["_target_"],
            "sam3.train.transforms.segmentation.DecodeRle",
        )
        self.assertNotIn("job_array", config["submitit"])

    def test_rejects_mismatched_categories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "template.yaml"
            template.write_text(yaml.safe_dump(template_config()), encoding="utf-8")
            train_json = root / "train.json"
            val_json = root / "val.json"
            write_json(train_json, {"categories": [{"id": 1, "name": "ball"}]})
            write_json(val_json, {"categories": [{"id": 2, "name": "robots"}]})

            with self.assertRaisesRegex(ValueError, "must match"):
                prepare_sam3_finetune_config(
                    template,
                    root / "out.yaml",
                    data_root=root,
                    train_json=train_json,
                    val_json=val_json,
                    experiment_dir=root / "run",
                    bpe_path=root / "bpe.gz",
                )


if __name__ == "__main__":
    unittest.main()
