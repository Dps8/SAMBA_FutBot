import tempfile
import unittest
from pathlib import Path

import numpy as np

from samba_futbot.io_utils import read_json, write_json
from samba_futbot.sam3_training_export import (
    DEFAULT_CLASS_PROMPTS,
    _encode_uncompressed_rle,
    export_sam3_training,
    manifest_to_sam3_training,
)


class Sam3TrainingExportTest(unittest.TestCase):
    def test_positive_pairs_are_per_prompt_with_normalized_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            masks = np.zeros((2, 4, 8), dtype=np.uint8)
            masks[0, 1:3, 2:5] = 1
            masks[1, 0:2, 6:8] = 1
            np.savez_compressed(root / "masks.npz", masks=masks)
            manifest_path = root / "manifest.json"
            write_json(
                manifest_path,
                _manifest(
                    [
                        _image(
                            "frame.jpg",
                            "train",
                            [
                                _detection("robots", [2, 1, 5, 3], "masks.npz", 0),
                                _detection("ball", [6, 0, 8, 2], "masks.npz", 1),
                            ],
                        )
                    ]
                ),
            )

            output, report = manifest_to_sam3_training(manifest_path)

        self.assertEqual(len(output["images"]), 2)
        self.assertEqual(
            [image["queried_category"] for image in output["images"]],
            ["ball", "robots"],
        )
        self.assertEqual(
            [image["text_input"] for image in output["images"]],
            ["small orange soccer ball", "robot soccer player"],
        )
        self.assertEqual(
            set(output["images"][0]),
            {
                "id",
                "file_name",
                "width",
                "height",
                "text_input",
                "queried_category",
                "is_instance_exhaustive",
                "is_pixel_exhaustive",
            },
        )
        self.assertEqual(output["categories"], [{"id": 1, "name": "object"}])
        self.assertEqual({item["category_id"] for item in output["annotations"]}, {1})
        ball = output["annotations"][0]
        self.assertEqual(ball["bbox"], [0.75, 0.0, 0.25, 0.5])
        self.assertEqual(ball["area"], 4 / 32)
        self.assertEqual(ball["segmentation"]["size"], [4, 8])
        self.assertEqual(sum(ball["segmentation"]["counts"]), 32)
        self.assertEqual(
            report["masks"],
            {"referenced": 2, "loaded": 2, "exported": 2, "failed": 0},
        )
        self.assertEqual(report["summary"]["positive_pairs"], 2)

    def test_prompt_override_and_goal_defaults_are_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask = np.ones((1, 4, 8), dtype=np.uint8)
            np.savez_compressed(root / "goal.npz", masks=mask)
            manifest = _manifest(
                [
                    _image(
                        "goal.jpg",
                        "train",
                        [_detection("goal_blue", [0, 0, 8, 4], "goal.npz", 0)],
                    )
                ]
            )

            output, report = manifest_to_sam3_training(
                manifest,
                manifest_base=root,
                class_prompts={"goal_blue": "blue box soccer goal"},
            )

        self.assertEqual(DEFAULT_CLASS_PROMPTS["goal_yellow"], "yellow soccer goal")
        self.assertEqual(output["images"][0]["text_input"], "blue box soccer goal")
        self.assertEqual(output["images"][0]["queried_category"], "goal_blue")
        self.assertEqual(report["settings"]["class_prompts"]["goal_blue"], "blue box soccer goal")

    def test_negative_pairs_are_absent_classes_only_and_conservatively_limited(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez_compressed(root / "ball.npz", masks=np.ones((1, 4, 8), dtype=np.uint8))
            manifest = _manifest(
                [
                    _image(
                        f"frame-{index}.jpg",
                        "train",
                        [_detection("ball", [0, 0, 8, 4], "ball.npz", 0)],
                    )
                    for index in range(4)
                ]
            )

            output, report = manifest_to_sam3_training(
                manifest,
                manifest_base=root,
                include_negatives=True,
                negative_classes=["ball", "robots"],
                max_negative_classes_per_image=1,
                max_negative_pairs_per_class=2,
            )

        negatives = [
            image
            for image in output["images"]
            if image["id"] not in {annotation["image_id"] for annotation in output["annotations"]}
        ]
        self.assertEqual(len(negatives), 2)
        self.assertEqual({image["queried_category"] for image in negatives}, {"robots"})
        self.assertTrue(all(image["is_instance_exhaustive"] is False for image in negatives))
        self.assertEqual(report["negative_pairs_by_class"], {"robots": 2})
        self.assertEqual(report["summary"]["negative_pairs"], 2)

    def test_broken_positive_is_dropped_and_never_recast_as_negative(self):
        manifest = _manifest(
            [
                _image(
                    "frame.jpg",
                    "train",
                    [_detection("ball", [0, 0, 8, 4], "missing.npz", 0)],
                )
            ]
        )

        output, report = manifest_to_sam3_training(
            manifest,
            manifest_base=".",
            include_negatives=True,
            negative_classes=["ball"],
        )

        self.assertEqual(output["images"], [])
        self.assertEqual(output["annotations"], [])
        self.assertEqual(report["summary"]["dropped_positive_pairs"], 1)
        self.assertEqual(report["summary"]["negative_pairs"], 0)
        self.assertEqual(report["failures"][0]["reason"], "mask_file_not_found")

    def test_partial_mask_failure_downgrades_exhaustiveness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez_compressed(root / "masks.npz", masks=np.ones((1, 4, 8), dtype=np.uint8))
            manifest = _manifest(
                [
                    _image(
                        "frame.jpg",
                        "train",
                        [
                            _detection("robots", [0, 0, 8, 4], "masks.npz", 0),
                            _detection("robots", [0, 0, 8, 4], "masks.npz", 9),
                        ],
                    )
                ]
            )

            output, report = manifest_to_sam3_training(manifest, manifest_base=root)

        self.assertEqual(len(output["annotations"]), 1)
        self.assertIs(output["images"][0]["is_instance_exhaustive"], False)
        self.assertIs(output["images"][0]["is_pixel_exhaustive"], False)
        self.assertEqual(
            report["masks"],
            {"referenced": 2, "loaded": 1, "exported": 1, "failed": 1},
        )
        self.assertEqual(report["failures"][0]["reason"], "mask_index_out_of_range")

    def test_empty_mask_and_invalid_box_are_distinguished_in_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            masks = np.zeros((2, 4, 8), dtype=np.uint8)
            masks[1, 1:3, 2:5] = 1
            np.savez_compressed(root / "masks.npz", masks=masks)
            manifest = _manifest(
                [
                    _image(
                        "frame.jpg",
                        "train",
                        [
                            _detection("ball", [0, 0, 8, 4], "masks.npz", 0),
                            _detection("robots", [3, 3, 2, 2], "masks.npz", 1),
                        ],
                    )
                ]
            )

            output, report = manifest_to_sam3_training(manifest, manifest_base=root)

        self.assertEqual(output["images"], [])
        self.assertEqual(
            report["masks"],
            {"referenced": 2, "loaded": 2, "exported": 0, "failed": 1},
        )
        self.assertEqual(
            {failure["reason"] for failure in report["failures"]},
            {"empty_mask", "invalid_box"},
        )

    def test_mask_failure_reasons_cover_safe_loader_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez_compressed(root / "wrong-size.npz", masks=np.ones((1, 2, 2), dtype=np.uint8))
            np.savez_compressed(root / "wrong-key.npz", values=np.ones((1, 4, 8), dtype=np.uint8))
            (root / "bad.npz").write_bytes(b"not an npz")
            cases = [
                ({}, "missing_mask_path"),
                ({"mask_path": "x.npz", "mask_index": -1}, "invalid_mask_index"),
                ({"mask_path": "x.npy", "mask_index": 0}, "unsupported_mask_file"),
                ({"mask_path": "wrong-size.npz", "mask_index": 0}, "mask_size_mismatch"),
                ({"mask_path": "wrong-key.npz", "mask_index": 0}, "missing_masks_array"),
                ({"mask_path": "bad.npz", "mask_index": 0}, "invalid_mask_archive"),
            ]
            for index, (mask_fields, reason) in enumerate(cases):
                with self.subTest(reason=reason):
                    detection = {"class_name": "ball", "box": [0, 0, 8, 4], **mask_fields}
                    output, report = manifest_to_sam3_training(
                        _manifest([_image(f"{index}.jpg", "train", [detection])]),
                        manifest_base=root,
                    )
                    self.assertEqual(output["images"], [])
                    self.assertEqual(report["failures"][0]["reason"], reason)

    def test_export_writes_each_split_all_file_and_aggregate_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez_compressed(root / "masks.npz", masks=np.ones((1, 4, 8), dtype=np.uint8))
            manifest_path = root / "manifest.json"
            write_json(
                manifest_path,
                _manifest(
                    [
                        _image(
                            "train.jpg",
                            "train",
                            [_detection("ball", [0, 0, 8, 4], "masks.npz", 0)],
                        ),
                        _image(
                            "val.jpg",
                            "val",
                            [_detection("robots", [0, 0, 8, 4], "masks.npz", 0)],
                        ),
                    ]
                ),
            )

            result = export_sam3_training(manifest_path, root / "sam3")
            train = read_json(result["annotations"]["train"])
            val = read_json(result["annotations"]["val"])
            all_data = read_json(result["annotations"]["all"])
            report = read_json(result["report"])

        self.assertEqual(sorted(result["annotations"]), ["all", "train", "val"])
        self.assertEqual(len(train["images"]), 1)
        self.assertEqual(train["images"][0]["queried_category"], "ball")
        self.assertEqual(val["images"][0]["queried_category"], "robots")
        self.assertEqual(len(all_data["images"]), 2)
        self.assertEqual(report["summary"]["annotations"], 2)
        self.assertEqual(report["masks"]["failed"], 0)
        self.assertEqual(report["splits"]["train"]["source_images"], 1)
        self.assertTrue(Path(report["outputs"]["report"]).name == "report.json")

    def test_mapping_requires_base_for_relative_masks(self):
        manifest = _manifest(
            [
                _image(
                    "frame.jpg",
                    "train",
                    [_detection("ball", [0, 0, 8, 4], "masks.npz", 0)],
                )
            ]
        )

        output, report = manifest_to_sam3_training(manifest)

        self.assertEqual(output["images"], [])
        self.assertEqual(
            report["failures"][0]["reason"],
            "relative_mask_path_without_manifest_base",
        )

    def test_image_root_makes_file_names_portable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "source" / "frames" / "frame.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"image")
            np.savez_compressed(
                root / "mask.npz",
                masks=np.ones((1, 4, 8), dtype=np.uint8),
            )
            manifest = _manifest(
                [
                    {
                        **_image(
                            str(image),
                            "train",
                            [_detection("ball", [0, 0, 8, 4], "mask.npz", 0)],
                        ),
                    }
                ]
            )

            output, report = manifest_to_sam3_training(
                manifest,
                manifest_base=root,
                image_root=root,
            )

        self.assertEqual(output["images"][0]["file_name"], "source/frames/frame.jpg")
        self.assertEqual(report["settings"]["image_root"], str(root.resolve()))

    def test_image_outside_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root.parent / "outside.jpg"
            manifest = _manifest([_image(str(outside), "train", [])])

            with self.assertRaisesRegex(ValueError, "outside image_root"):
                manifest_to_sam3_training(manifest, image_root=root)

    def test_uncompressed_rle_uses_fortran_order_and_background_first(self):
        mask = np.array([[1, 0, 0], [1, 0, 1]], dtype=np.uint8)

        rle = _encode_uncompressed_rle(mask)

        self.assertEqual(rle, {"size": [2, 3], "counts": [0, 2, 3, 1]})
        self.assertEqual(sum(rle["counts"]), 6)

    def test_invalid_limits_and_unknown_negative_class_are_rejected(self):
        manifest = _manifest([])
        with self.assertRaises(ValueError):
            manifest_to_sam3_training(manifest, max_negative_classes_per_image=-1)
        with self.assertRaises(ValueError):
            manifest_to_sam3_training(
                manifest,
                include_negatives=True,
                negative_classes=["goal_green"],
            )


def _manifest(images: list[dict]) -> dict:
    return {"schema": "samba_futbot.frame_dataset.v1", "images": images}


def _image(path: str, split: str, detections: list[dict]) -> dict:
    return {
        "image_path": path,
        "width": 8,
        "height": 4,
        "split": split,
        "detections": detections,
    }


def _detection(
    class_name: str,
    box: list[float],
    mask_path: str,
    mask_index: int,
) -> dict:
    return {
        "class_name": class_name,
        "score": 0.9,
        "box": box,
        "mask_path": mask_path,
        "mask_index": mask_index,
    }


if __name__ == "__main__":
    unittest.main()
