import tempfile
import unittest
from pathlib import Path

from samba_futbot.dataset_quality import (
    analyze_dataset_quality,
    analyze_dataset_quality_file,
    invalid_box_reason,
    write_dataset_quality_markdown,
)
from samba_futbot.io_utils import write_json


class DatasetQualityTest(unittest.TestCase):
    def test_analyze_dataset_quality_summarizes_manifest_dimensions(self):
        report = analyze_dataset_quality(_manifest(), low_score_threshold=0.6)

        self.assertEqual(report["summary"]["frames"], 3)
        self.assertEqual(report["summary"]["detections"], 4)
        self.assertEqual(report["summary"]["crops"], 2)
        self.assertEqual(report["summary"]["detections_with_crops"], 2)
        self.assertEqual(report["summary"]["invalid_boxes"], 1)
        self.assertEqual(report["summary"]["low_scores"], 1)
        self.assertEqual(report["summary"]["frames_without_detections"], 1)
        self.assertEqual(report["summary"]["videos_in_multiple_splits"], 0)
        self.assertEqual(report["summary"]["duplicate_image_paths"], 0)
        self.assertEqual(report["summary"]["duplicate_source_frame_groups"], 0)
        self.assertEqual(report["summary"]["duplicate_source_frame_extras"], 0)
        self.assertEqual(report["summary"]["classes"], 2)
        self.assertEqual(report["by_class"]["ball"]["detections"], 2)
        self.assertEqual(report["by_class"]["ball"]["low_scores"], 1)
        self.assertEqual(report["by_class"]["robots"]["invalid_boxes"], 1)
        self.assertEqual(report["by_split"]["train"]["frames"], 2)
        self.assertEqual(report["by_split"]["train"]["detections"], 4)
        self.assertEqual(report["by_split"]["train"]["crops"], 2)
        self.assertEqual(report["by_split"]["val"]["frames"], 1)
        self.assertEqual(report["by_video"]["clip-a.mp4"]["detections"], 4)
        self.assertEqual(report["by_video"]["clip-a.mp4"]["crops"], 2)
        self.assertEqual(report["by_video"]["clip-b.mp4"]["frames"], 1)
        self.assertEqual(report["by_detection_source"]["tracker"]["detections"], 3)
        self.assertEqual(report["by_detection_source"]["unknown"]["detections"], 1)

    def test_review_candidates_include_low_scores_and_invalid_boxes(self):
        report = analyze_dataset_quality(_manifest(), low_score_threshold=0.6)

        self.assertEqual([candidate["reason"] for candidate in report["review_candidates"]], [
            "low_score",
            "invalid_box",
        ])
        low_score = report["review_candidates"][0]
        invalid_box = report["review_candidates"][1]
        self.assertEqual(low_score["class_name"], "ball")
        self.assertEqual(low_score["frame_index"], 1)
        self.assertEqual(low_score["score"], 0.55)
        self.assertEqual(invalid_box["detail"], "box_outside_width")
        self.assertEqual(invalid_box["class_name"], "robots")
        self.assertEqual(invalid_box["image_path"], "frames/clip-a/000001.jpg")

    def test_review_candidate_limit_is_respected(self):
        manifest = {
            "images": [
                {
                    "image_path": "frames/a.jpg",
                    "width": 20,
                    "height": 20,
                    "split": "train",
                    "detections": [
                        {"class_name": "ball", "score": 0.1, "box": [0, 0, 1, 1]},
                        {"class_name": "ball", "score": 0.2, "box": [2, 2, 3, 3]},
                    ],
                }
            ]
        }

        report = analyze_dataset_quality(
            manifest,
            low_score_threshold=0.6,
            max_review_examples=1,
        )

        self.assertEqual(report["summary"]["low_scores"], 2)
        self.assertEqual(report["summary"]["review_candidates"], 1)
        self.assertEqual(len(report["review_candidates"]), 1)

    def test_split_leakage_and_duplicate_images_are_reported(self):
        manifest = {
            "images": [
                {
                    "video": "clip-a.mp4",
                    "split": "train",
                    "image_path": "frames/shared.jpg",
                    "detections": [],
                },
                {
                    "video": "clip-a.mp4",
                    "split": "val",
                    "image_path": "frames/shared.jpg",
                    "detections": [],
                },
            ]
        }

        report = analyze_dataset_quality(manifest)

        self.assertEqual(report["summary"]["videos_in_multiple_splits"], 1)
        self.assertEqual(report["summary"]["duplicate_image_paths"], 1)
        self.assertEqual(report["split_leakage_videos"], {"clip-a.mp4": ["train", "val"]})
        self.assertEqual(report["duplicate_image_paths"], ["frames/shared.jpg"])

    def test_duplicate_source_frames_are_reported_across_variant_paths(self):
        manifest = {
            "images": [
                {
                    "video": "clips/game.mp4",
                    "frame_index": 12,
                    "split": "train",
                    "image_path": "variant-a/frame.jpg",
                    "detections": [{"class_name": "ball", "box": [1, 1, 3, 3]}],
                },
                {
                    "video": "clips\\GAME.mp4",
                    "frame_index": "12",
                    "split": "train",
                    "image_path": "variant-b/frame.jpg",
                    "detections": [{"class_name": "ball", "box": [1, 1, 3, 3]}],
                },
            ]
        }

        report = analyze_dataset_quality(manifest)

        self.assertEqual(report["summary"]["duplicate_image_paths"], 0)
        self.assertEqual(report["summary"]["duplicate_source_frame_groups"], 1)
        self.assertEqual(report["summary"]["duplicate_source_frame_extras"], 1)
        self.assertEqual(report["duplicate_source_frames"][0]["frame_index"], 12)
        self.assertEqual(report["duplicate_source_frames"][0]["copies"], 2)

    def test_analyze_dataset_quality_file_reads_json_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            write_json(path, _manifest())

            report = analyze_dataset_quality_file(path, low_score_threshold=0.6)

        self.assertEqual(report["inputs"]["manifest"], str(path))
        self.assertEqual(report["summary"]["detections"], 4)
        self.assertEqual(report["by_class"]["robots"]["detections"], 2)

    def test_write_dataset_quality_markdown_summarizes_review_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = analyze_dataset_quality(_manifest(), low_score_threshold=0.6)
            out = write_dataset_quality_markdown(report, Path(tmp) / "quality.md")

            text = out.read_text(encoding="utf-8")

        self.assertIn("# Dataset Quality Report", text)
        self.assertIn("## By Class", text)
        self.assertIn("box_outside_width", text)
        self.assertIn("low_score", text)

    def test_invalid_box_reason_validates_shape_area_and_bounds(self):
        self.assertIsNone(invalid_box_reason([1, 2, 4, 5], width=10, height=10))
        self.assertEqual(invalid_box_reason([1, 2, 1, 5], width=10, height=10), "non_positive_box")
        self.assertEqual(invalid_box_reason([1, 2, 4], width=10, height=10), "malformed_box")
        self.assertEqual(invalid_box_reason([-1, 2, 4, 5], width=10, height=10), "box_outside_width")
        self.assertEqual(invalid_box_reason([1, 2, 4, 12], width=10, height=10), "box_outside_height")


def _manifest() -> dict:
    return {
        "schema": "samba_futbot.frame_dataset.v1",
        "source_video": "clip-a.mp4",
        "images": [
            {
                "video": "clip-a.mp4",
                "frame_index": 0,
                "split": "train",
                "image_path": "frames/clip-a/000000.jpg",
                "width": 100,
                "height": 80,
                "detections": [
                    {
                        "class_name": "ball",
                        "score": 0.92,
                        "box": [10, 10, 20, 20],
                        "crop_path": "crops/ball/000000.jpg",
                        "source": "tracker",
                    },
                    {
                        "class_name": "robots",
                        "score": 0.80,
                        "box": [30, 10, 60, 60],
                        "crop_path": "crops/robots/000000.jpg",
                        "source": "tracker",
                    },
                ],
                "crops": ["crops/ball/000000.jpg", "crops/robots/000000.jpg"],
            },
            {
                "video": "clip-a.mp4",
                "frame_index": 1,
                "split": "train",
                "image_path": "frames/clip-a/000001.jpg",
                "width": 100,
                "height": 80,
                "detections": [
                    {
                        "class_name": "ball",
                        "score": 0.55,
                        "box": [11, 12, 19, 20],
                        "source": "tracker",
                    },
                    {
                        "class_name": "robots",
                        "score": 0.90,
                        "box": [90, 20, 110, 40],
                    },
                ],
            },
            {
                "video": "clip-b.mp4",
                "frame_index": 2,
                "split": "val",
                "image_path": "frames/clip-b/000002.jpg",
                "width": 100,
                "height": 80,
                "detections": [],
            },
        ],
    }


if __name__ == "__main__":
    unittest.main()
