import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from samba_futbot.dataset_curation import (
    curate_dataset_manifest,
    curate_dataset_manifest_file,
    curate_frame_dataset_manifest,
)
from samba_futbot.io_utils import read_json, write_json


class DatasetCurationTest(unittest.TestCase):
    def test_curates_merged_manifest_without_mutating_input(self):
        manifest = _manifest()
        original = deepcopy(manifest)
        exclusions = [
            {
                "image_path": "C:/dataset/frames/b.jpg",
                "frame_index": 1,
                "detection_index": 0,
            }
        ]

        curated, report = curate_dataset_manifest(
            manifest,
            classes=["ball", "robots"],
            min_score=0.6,
            review_exclusions=exclusions,
        )

        self.assertEqual(manifest, original)
        self.assertEqual(curated["schema"], manifest["schema"])
        self.assertEqual(curated["sources"], manifest["sources"])
        self.assertEqual(curated["merge"], manifest["merge"])
        self.assertEqual(
            [(image["image_path"], image["split"]) for image in curated["images"]],
            [("C:\\dataset\\frames\\a.jpg", "train"), ("C:\\dataset\\frames\\b.jpg", "val")],
        )
        self.assertEqual(curated["images"][0]["crops"], ["C:\\dataset\\crops\\ball-a.jpg"])
        self.assertEqual(curated["images"][1]["crops"], ["C:\\dataset\\crops\\robot-b.jpg"])
        self.assertEqual(
            curated["summary"],
            {
                "frames": 2,
                "detections": 2,
                "crops": 2,
                "detections_by_class": {"ball": 1, "robots": 1},
                "frames_by_split": {"train": 1, "val": 1},
            },
        )
        self.assertEqual(report["summary"]["input_frames"], 3)
        self.assertEqual(report["summary"]["output_frames"], 2)
        self.assertEqual(report["summary"]["dropped_empty_frames"], 1)
        self.assertEqual(report["summary"]["input_detections"], 9)
        self.assertEqual(report["summary"]["accepted_detections"], 2)
        self.assertEqual(report["summary"]["rejected_detections"], 7)
        self.assertEqual(
            report["rejected_by_reason"],
            {
                "class_not_selected": 1,
                "invalid_box": 1,
                "invalid_score": 1,
                "malformed_detection": 1,
                "review_exclusion": 1,
                "score_below_minimum": 2,
            },
        )
        self.assertEqual(report["invalid_box_details"], {"non_positive_box": 1})
        self.assertEqual(report["accepted_by_class"], {"ball": 1, "robots": 1})
        self.assertEqual(report["rejected_by_class"]["ball"], 3)
        self.assertEqual(
            report["by_class"]["robots"]["rejected_by_reason"],
            {"invalid_score": 1, "score_below_minimum": 1},
        )

    def test_can_preserve_empty_frames_and_accept_all_classes(self):
        manifest = {
            "schema": "samba_futbot.frame_dataset_merged.v1",
            "images": [
                {
                    "image_path": "frames/empty.jpg",
                    "frame_index": 0,
                    "split": "test",
                    "width": 20,
                    "height": 10,
                    "detections": [],
                    "crops": [],
                },
                {
                    "image_path": "frames/field.jpg",
                    "frame_index": 1,
                    "split": "test",
                    "width": 20,
                    "height": 10,
                    "detections": [
                        {"class_name": "field", "score": 0.0, "box": [0, 0, 20, 10]}
                    ],
                },
            ],
        }

        curated, report = curate_dataset_manifest(
            manifest,
            classes=[],
            min_score=0.0,
            drop_empty_frames=False,
        )

        self.assertEqual(len(curated["images"]), 2)
        self.assertEqual(curated["summary"]["frames_by_split"], {"test": 2})
        self.assertEqual(curated["summary"]["detections_by_class"], {"field": 1})
        self.assertEqual(report["summary"]["dropped_empty_frames"], 0)
        self.assertEqual(report["rejected_by_reason"], {})

    def test_review_exclusion_uses_original_detection_index_and_normalizes_path(self):
        manifest = {
            "images": [
                {
                    "image_path": "C:\\DATASET\\frames\\a.jpg",
                    "frame_index": "7",
                    "split": "train",
                    "width": 100,
                    "height": 100,
                    "detections": [
                        {"class_name": "ball", "score": 0.1, "box": [1, 1, 2, 2]},
                        {"class_name": "ball", "score": 0.9, "box": [2, 2, 4, 4]},
                    ],
                }
            ]
        }

        curated, report = curate_frame_dataset_manifest(
            manifest,
            min_score=0.5,
            review_exclusions=[("c:/dataset/frames/a.jpg", 7, 1)],
        )

        self.assertEqual(curated["summary"]["frames"], 0)
        self.assertEqual(
            report["rejected_by_reason"],
            {"review_exclusion": 1, "score_below_minimum": 1},
        )

    def test_rejects_invalid_box_shapes_bounds_and_non_finite_values(self):
        detections = [
            {"class_name": "ball", "score": 0.6, "box": [0, 0, 10, 10]},
            {"class_name": "ball", "score": 0.6, "box": [0, 0, 11, 10]},
            {"class_name": "ball", "score": 0.6, "box": [0, -1, 10, 10]},
            {"class_name": "ball", "score": 0.6, "box": [0, 0, "nan", 10]},
            {"class_name": "ball", "score": float("inf"), "box": [0, 0, 10, 10]},
        ]
        manifest = {
            "images": [
                {
                    "image_path": "frame.jpg",
                    "frame_index": 0,
                    "split": "train",
                    "width": 10,
                    "height": 10,
                    "detections": detections,
                }
            ]
        }

        curated, report = curate_dataset_manifest(manifest, min_score=0.6)

        self.assertEqual(curated["summary"]["detections"], 1)
        self.assertEqual(report["rejected_by_reason"], {"invalid_box": 3, "invalid_score": 1})
        self.assertEqual(
            report["invalid_box_details"],
            {"box_outside_height": 1, "box_outside_width": 1, "malformed_box": 1},
        )

    def test_file_wrapper_writes_manifest_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            curated_path = root / "out" / "curated.json"
            report_path = root / "out" / "report.json"
            write_json(source, _manifest())

            curated, report = curate_dataset_manifest_file(
                source,
                curated_path,
                report_path,
                classes=["ball"],
                min_score=0.8,
            )

            self.assertEqual(read_json(curated_path), curated)
            self.assertEqual(read_json(report_path), report)
            self.assertEqual(report["inputs"]["manifest"], str(source))
            self.assertEqual(report["outputs"]["manifest"], str(curated_path))
        self.assertEqual(report["outputs"]["report"], str(report_path))

    def test_deduplicates_source_frames_and_keeps_stronger_variant(self):
        manifest = {
            "images": [
                {
                    "video": "clips/game.mp4",
                    "frame_index": 4,
                    "split": "train",
                    "image_path": "weak.jpg",
                    "width": 100,
                    "height": 100,
                    "detections": [
                        {"class_name": "ball", "score": 0.7, "box": [1, 1, 5, 5]}
                    ],
                },
                {
                    "video": "clips\\GAME.mp4",
                    "frame_index": 4,
                    "split": "train",
                    "image_path": "strong.jpg",
                    "width": 100,
                    "height": 100,
                    "detections": [
                        {"class_name": "ball", "score": 0.9, "box": [1, 1, 5, 5]},
                        {"class_name": "robots", "score": 0.8, "box": [10, 10, 30, 30]},
                    ],
                },
            ]
        }

        curated, report = curate_dataset_manifest(
            manifest,
            min_score=0.6,
            deduplicate_source_frames=True,
        )

        self.assertEqual([image["image_path"] for image in curated["images"]], ["strong.jpg"])
        self.assertEqual(curated["summary"]["detections"], 2)
        self.assertEqual(report["summary"]["dropped_duplicate_frames"], 1)
        self.assertEqual(report["rejected_by_reason"], {"duplicate_source_frame": 1})
        self.assertEqual(
            report["duplicate_source_frame_groups"][0]["dropped_image_paths"],
            ["weak.jpg"],
        )

    def test_validates_manifest_threshold_and_review_exclusions(self):
        with self.assertRaisesRegex(ValueError, "JSON object"):
            curate_dataset_manifest([])
        with self.assertRaisesRegex(ValueError, "finite number"):
            curate_dataset_manifest({"images": []}, min_score=float("nan"))
        with self.assertRaisesRegex(ValueError, "images must be a list"):
            curate_dataset_manifest({"images": {}})
        with self.assertRaisesRegex(ValueError, "three fields"):
            curate_dataset_manifest({"images": []}, review_exclusions=[("a", 1)])
        with self.assertRaisesRegex(ValueError, "invalid values"):
            curate_dataset_manifest(
                {"images": []},
                review_exclusions=[
                    {"image_path": "", "frame_index": 0, "detection_index": 0}
                ],
            )


def _manifest() -> dict:
    return {
        "schema": "samba_futbot.frame_dataset_merged.v1",
        "sources": ["source-a.json", "source-b.json"],
        "merge": {"split_strategy": "preserve"},
        "summary": {"stale": True},
        "images": [
            {
                "video": "clip-a.mp4",
                "frame_index": 0,
                "split": "train",
                "image_path": "C:\\dataset\\frames\\a.jpg",
                "width": 100,
                "height": 80,
                "detections": [
                    {
                        "class_name": "ball",
                        "score": 0.9,
                        "box": [10, 10, 20, 20],
                        "crop_path": "C:\\dataset\\crops\\ball-a.jpg",
                    },
                    {
                        "class_name": "robots",
                        "score": 0.4,
                        "box": [20, 20, 30, 30],
                        "crop_path": "C:\\dataset\\crops\\robot-low.jpg",
                    },
                    {
                        "class_name": "field",
                        "score": 0.95,
                        "box": [0, 0, 100, 80],
                        "crop_path": "C:\\dataset\\crops\\field.jpg",
                    },
                    {
                        "class_name": "ball",
                        "score": 0.8,
                        "box": [20, 20, 10, 30],
                        "crop_path": "C:\\dataset\\crops\\ball-invalid.jpg",
                    },
                    {
                        "class_name": "robots",
                        "score": "bad",
                        "box": [30, 30, 50, 60],
                        "crop_path": "C:\\dataset\\crops\\robot-score.jpg",
                    },
                    "malformed",
                ],
                "crops": [
                    "C:\\dataset\\crops\\ball-a.jpg",
                    "C:\\dataset\\crops\\robot-low.jpg",
                    "C:\\dataset\\crops\\field.jpg",
                    "C:\\dataset\\crops\\ball-invalid.jpg",
                    "C:\\dataset\\crops\\robot-score.jpg",
                ],
            },
            {
                "video": "clip-b.mp4",
                "frame_index": 1,
                "split": "val",
                "image_path": "C:\\dataset\\frames\\b.jpg",
                "width": 100,
                "height": 80,
                "detections": [
                    {
                        "class_name": "ball",
                        "score": 0.95,
                        "box": [1, 1, 5, 5],
                        "crop_path": "C:\\dataset\\crops\\ball-b.jpg",
                    },
                    {
                        "class_name": "robots",
                        "score": 0.8,
                        "box": [40, 20, 60, 60],
                        "crop_path": "C:\\dataset\\crops\\robot-b.jpg",
                    },
                ],
                "crops": [
                    "C:\\dataset\\crops\\ball-b.jpg",
                    "C:\\dataset\\crops\\robot-b.jpg",
                ],
            },
            {
                "video": "clip-c.mp4",
                "frame_index": 2,
                "split": "train",
                "image_path": "C:\\dataset\\frames\\c.jpg",
                "width": 100,
                "height": 80,
                "detections": [
                    {
                        "class_name": "ball",
                        "score": 0.1,
                        "box": [10, 10, 20, 20],
                        "crop_path": "C:\\dataset\\crops\\ball-c.jpg",
                    }
                ],
                "crops": ["C:\\dataset\\crops\\ball-c.jpg"],
            },
        ],
    }


if __name__ == "__main__":
    unittest.main()
