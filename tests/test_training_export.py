import tempfile
import unittest
from pathlib import Path

import numpy as np

from samba_futbot.io_utils import read_json, write_json
from samba_futbot.training_export import (
    _encode_uncompressed_rle,
    export_balanced_coco_subset,
    export_coco_detection,
    manifest_to_coco_detection,
)


def synthetic_manifest() -> dict:
    return {
        "schema": "samba_futbot.frame_dataset.v1",
        "images": [
            {
                "image_path": "frames/clip/clip_f000000.jpg",
                "width": 100,
                "height": 50,
                "split": "train",
                "detections": [
                    {"class_name": "robots", "box": [10, 5, 40, 25]},
                    {"class_name": "ball", "box": [50, 20, 60, 30]},
                ],
            },
            {
                "image_path": "frames/clip/clip_f000001.jpg",
                "width": 80,
                "height": 80,
                "split": "val",
                "detections": [
                    {"class_name": "robots", "box": [0, 0, 20, 20]},
                ],
            },
            {
                "image_path": "frames/clip/clip_f000002.jpg",
                "width": 40,
                "height": 40,
                "split": "test",
                "detections": [],
            },
        ],
    }


class TrainingExportTest(unittest.TestCase):
    def test_manifest_to_coco_detection_uses_stable_categories_and_xywh_boxes(self):
        coco = manifest_to_coco_detection(synthetic_manifest())

        self.assertEqual(coco["categories"], [{"id": 1, "name": "ball"}, {"id": 2, "name": "robots"}])
        self.assertEqual(coco["images"][0]["split"], "train")
        self.assertEqual(coco["annotations"][0]["category_id"], 2)
        self.assertEqual(coco["annotations"][0]["bbox"], [10.0, 5.0, 30.0, 20.0])
        self.assertEqual(coco["annotations"][0]["area"], 600.0)
        self.assertEqual(coco["annotations"][1]["category_id"], 1)
        self.assertEqual(coco["annotations"][1]["bbox"], [50.0, 20.0, 10.0, 10.0])

    def test_export_coco_detection_writes_split_files_with_consistent_category_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "dataset" / "manifest.json"
            write_json(manifest_path, synthetic_manifest())

            paths = export_coco_detection(manifest_path, root / "coco")
            train = read_json(paths["train"])
            val = read_json(paths["val"])
            test = read_json(paths["test"])

        self.assertEqual(sorted(paths), ["all", "test", "train", "val"])
        self.assertEqual([image["split"] for image in train["images"]], ["train"])
        self.assertEqual([annotation["category_id"] for annotation in val["annotations"]], [2])
        self.assertEqual(test["images"][0]["split"], "test")
        self.assertEqual(test["annotations"], [])
        self.assertEqual(val["categories"], [{"id": 1, "name": "ball"}, {"id": 2, "name": "robots"}])

    def test_export_coco_detection_can_make_image_paths_relative_to_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "frames" / "clip" / "frame.jpg"
            image.parent.mkdir(parents=True)
            image.touch()
            manifest = synthetic_manifest()
            manifest["images"] = [
                {
                    **manifest["images"][0],
                    "image_path": str(image),
                }
            ]

            coco = manifest_to_coco_detection(manifest, image_root=root)

        self.assertEqual(coco["images"][0]["file_name"], "frames/clip/frame.jpg")

    def test_export_coco_detection_rejects_image_outside_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = synthetic_manifest()
            manifest["images"][0]["image_path"] = str(root.parent / "outside.jpg")

            with self.assertRaisesRegex(ValueError, "outside image_root"):
                manifest_to_coco_detection(manifest, image_root=root)

    def test_manifest_file_exports_npz_mask_as_uncompressed_coco_rle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            masks_dir = dataset / "masks"
            masks_dir.mkdir(parents=True)
            masks = np.zeros((2, 4, 5), dtype=np.uint8)
            masks[1, 1:3, 2:5] = 1
            np.savez_compressed(masks_dir / "robots.npz", masks=masks)
            manifest = {
                "images": [
                    {
                        "image_path": "frames/frame.jpg",
                        "width": 5,
                        "height": 4,
                        "detections": [
                            {
                                "class_name": "robots",
                                "box": [2, 1, 5, 3],
                                "mask_path": "masks/robots.npz",
                                "mask_index": 1,
                            }
                        ],
                    }
                ]
            }
            manifest_path = dataset / "manifest.json"
            write_json(manifest_path, manifest)

            coco = manifest_to_coco_detection(manifest_path)

        annotation = coco["annotations"][0]
        self.assertEqual(annotation["segmentation"]["size"], [4, 5])
        self.assertEqual(annotation["segmentation"]["counts"], [9, 2, 2, 2, 2, 2, 1])
        self.assertEqual(sum(annotation["segmentation"]["counts"]), 20)
        self.assertEqual(annotation["area"], 6)
        self.assertEqual(
            coco["samba_futbot_export"]["masks"],
            {"referenced": 1, "exported": 1, "failed": 0, "issues": []},
        )

    def test_invalid_mask_is_audited_and_keeps_box_annotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez_compressed(root / "masks.npz", masks=np.ones((1, 3, 3), dtype=np.uint8))
            manifest = {
                "images": [
                    {
                        "image_path": "frame.jpg",
                        "width": 3,
                        "height": 3,
                        "detections": [
                            {
                                "class_name": "ball",
                                "box": [0, 0, 2, 2],
                                "mask_path": "masks.npz",
                                "mask_index": 4,
                            }
                        ],
                    }
                ]
            }
            manifest_path = root / "manifest.json"
            write_json(manifest_path, manifest)

            paths = export_coco_detection(manifest_path, root / "coco")
            coco = read_json(paths["all"])

        annotation = coco["annotations"][0]
        self.assertNotIn("segmentation", annotation)
        self.assertEqual(annotation["bbox"], [0.0, 0.0, 2.0, 2.0])
        self.assertEqual(annotation["area"], 4.0)
        report = coco["samba_futbot_export"]["masks"]
        self.assertEqual(report["referenced"], 1)
        self.assertEqual(report["exported"], 0)
        self.assertEqual(report["failed"], 1)
        self.assertEqual(report["issues"][0]["reason"], "mask_index_out_of_range")

    def test_mapping_with_relative_mask_path_reports_missing_base(self):
        manifest = {
            "images": [
                {
                    "image_path": "frame.jpg",
                    "width": 2,
                    "height": 2,
                    "detections": [
                        {
                            "class_name": "ball",
                            "box": [0, 0, 2, 2],
                            "mask_path": "masks.npz",
                            "mask_index": 0,
                        }
                    ],
                }
            ]
        }

        coco = manifest_to_coco_detection(manifest)

        self.assertNotIn("segmentation", coco["annotations"][0])
        self.assertEqual(
            coco["samba_futbot_export"]["masks"]["issues"][0]["reason"],
            "relative_mask_path_without_manifest_base",
        )

    def test_balanced_coco_subset_keeps_all_focus_images_and_sampled_negatives(self):
        coco = {
            "images": [{"id": image_id} for image_id in range(1, 6)],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1},
                {"id": 2, "image_id": 1, "category_id": 2},
                {"id": 3, "image_id": 2, "category_id": 1},
                {"id": 4, "image_id": 3, "category_id": 2},
                {"id": 5, "image_id": 4, "category_id": 2},
                {"id": 6, "image_id": 5, "category_id": 2},
            ],
            "categories": [
                {"id": 1, "name": "ball"},
                {"id": 2, "name": "robots"},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "balanced.json"
            summary = export_balanced_coco_subset(
                coco,
                output,
                focus_classes=["ball"],
                negative_ratio=1.0,
                seed=7,
            )
            balanced = read_json(output)

        selected_ids = {image["id"] for image in balanced["images"]}
        self.assertTrue({1, 2}.issubset(selected_ids))
        self.assertEqual(len(selected_ids), 4)
        self.assertEqual(summary["positive_images"], 2)
        self.assertEqual(summary["negative_images"], 2)
        self.assertEqual(summary["class_annotations"]["ball"], 2)
        self.assertGreaterEqual(summary["class_annotations"]["robots"], 2)

    def test_balanced_coco_subset_rejects_unknown_focus_class(self):
        coco = {
            "images": [{"id": 1}],
            "annotations": [],
            "categories": [{"id": 1, "name": "ball"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "missing from COCO categories"):
                export_balanced_coco_subset(
                    coco,
                    Path(tmp) / "balanced.json",
                    focus_classes=["goal"],
                )

    def test_balanced_coco_subset_can_drop_non_focus_categories(self):
        coco = {
            "images": [{"id": 1}, {"id": 2}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1},
                {"id": 2, "image_id": 1, "category_id": 2},
                {"id": 3, "image_id": 2, "category_id": 2},
            ],
            "categories": [
                {"id": 1, "name": "ball"},
                {"id": 2, "name": "robots"},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "ball-only.json"
            summary = export_balanced_coco_subset(
                coco,
                output,
                focus_classes=["ball"],
                negative_ratio=1.0,
                focus_only=True,
            )
            balanced = read_json(output)

        self.assertEqual(balanced["categories"], [{"id": 1, "name": "ball"}])
        self.assertEqual(
            [annotation["category_id"] for annotation in balanced["annotations"]],
            [1],
        )
        self.assertTrue(summary["focus_only"])

    def test_vectorized_rle_handles_background_and_foreground_starts(self):
        background = np.zeros((2, 3), dtype=np.uint8)
        foreground = np.ones((2, 3), dtype=np.uint8)

        self.assertEqual(
            _encode_uncompressed_rle(background),
            {"size": [2, 3], "counts": [6]},
        )
        self.assertEqual(
            _encode_uncompressed_rle(foreground),
            {"size": [2, 3], "counts": [0, 6]},
        )


if __name__ == "__main__":
    unittest.main()
