import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from samba_futbot.holdout import (
    build_human_holdout,
    select_ball_review_set,
    select_ball_review_set_file,
    select_human_holdout,
    select_human_holdout_file,
)
from samba_futbot.io_utils import read_json, write_json


class HumanHoldoutTest(unittest.TestCase):
    def test_selects_deterministically_balanced_from_preferred_split(self):
        manifest = _manifest()
        original = deepcopy(manifest)

        first, first_report = select_human_holdout(
            manifest,
            max_frames=4,
            preferred_split="val",
            seed=31,
        )
        second, second_report = select_human_holdout(
            manifest,
            max_frames=4,
            preferred_split="val",
            seed=31,
        )

        self.assertEqual(manifest, original)
        self.assertEqual(first, second)
        self.assertEqual(first_report, second_report)
        self.assertEqual(len(first["images"]), 4)
        self.assertTrue(all(image["split"] == "val" for image in first["images"]))
        self.assertEqual(first_report["selected_by_video"], {"a.mp4": 2, "b.mp4": 2})
        self.assertEqual(first["selection_fingerprint"]["algorithm"], "sha256")
        self.assertEqual(len(first["selection_fingerprint"]["digest"]), 64)

    def test_annotation_template_does_not_copy_pseudo_detections(self):
        holdout, report = build_human_holdout(_manifest(), max_frames=2, seed=5)

        for image in holdout["images"]:
            self.assertNotIn("detections", image)
            self.assertNotIn("crops", image)
            self.assertEqual(image["annotations"], [])
            self.assertEqual(image["annotation_status"], "pending")
            self.assertEqual(
                set(image),
                {
                    "video",
                    "frame_index",
                    "image_path",
                    "width",
                    "height",
                    "split",
                    "expected_classes",
                    "annotation_status",
                    "annotations",
                },
            )
        self.assertFalse(holdout["annotation_policy"]["pseudo_detections_copied"])
        self.assertGreaterEqual(
            report["selected_by_expected_class"].get("ball", 0),
            1,
        )

    def test_backfills_other_splits_when_preferred_split_is_small(self):
        manifest = _manifest()
        manifest["images"] = [
            image
            for image in manifest["images"]
            if image["split"] != "val"
            or (image["video"] == "a.mp4" and image["frame_index"] == 0)
        ]

        holdout, report = select_human_holdout(
            manifest,
            max_frames=3,
            preferred_split="val",
            seed=2,
        )

        self.assertEqual(report["summary"]["preferred_split_frames"], 1)
        self.assertEqual(report["summary"]["fallback_split_frames"], 2)
        self.assertEqual(len(holdout["images"]), 3)
        self.assertEqual(report["selected_by_split"]["val"], 1)

    def test_deduplicates_source_frame_and_unions_expected_classes(self):
        manifest = {
            "images": [
                _image("Game.MP4", 8, "variant-a.jpg", "val", ["ball"]),
                _image("game.mp4", 8, "variant-b.jpg", "val", ["robots"]),
                _image("other.mp4", 3, "other.jpg", "val", ["field"]),
            ]
        }

        holdout, report = select_human_holdout(manifest, max_frames=2, seed=9)

        duplicate = next(
            image
            for image in holdout["images"]
            if image["video"].casefold() == "game.mp4"
        )
        self.assertEqual(duplicate["expected_classes"], ["ball", "robots"])
        self.assertEqual(report["summary"]["input_frames"], 3)
        self.assertEqual(report["summary"]["unique_source_frames"], 2)
        self.assertEqual(report["summary"]["duplicate_source_frames_removed"], 1)

    def test_different_seed_changes_reproducible_selection(self):
        manifest = {
            "source_video": "single.mp4",
            "images": [
                _image("single.mp4", index, f"{index}.jpg", "val", ["ball"])
                for index in range(20)
            ],
        }

        first, _ = select_human_holdout(manifest, max_frames=5, seed=1)
        second, _ = select_human_holdout(manifest, max_frames=5, seed=2)
        repeated, _ = select_human_holdout(manifest, max_frames=5, seed=1)

        first_frames = [image["frame_index"] for image in first["images"]]
        self.assertNotEqual(
            first_frames,
            [image["frame_index"] for image in second["images"]],
        )
        self.assertEqual(
            first_frames,
            [image["frame_index"] for image in repeated["images"]],
        )

    def test_file_wrapper_writes_outputs_and_reports_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "manifest.json"
            annotations = root / "out" / "holdout.json"
            report_path = root / "out" / "report.json"
            write_json(source, _manifest())

            holdout, report = select_human_holdout_file(
                source,
                annotations,
                report_path,
                max_frames=3,
                seed=17,
            )

            self.assertEqual(read_json(annotations), holdout)
            self.assertEqual(read_json(report_path), report)
            self.assertEqual(report["inputs"]["manifest"], str(source))
            self.assertEqual(report["outputs"]["annotations"], str(annotations))
            self.assertEqual(report["outputs"]["report"], str(report_path))

    def test_validates_arguments_manifest_records_and_json(self):
        with self.assertRaisesRegex(ValueError, "JSON object"):
            select_human_holdout([], max_frames=1)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            select_human_holdout({"images": []}, max_frames=0)
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            select_human_holdout(
                {"images": []},
                max_frames=1,
                preferred_split=" ",
            )
        with self.assertRaisesRegex(ValueError, "seed must be an integer"):
            select_human_holdout({"images": []}, max_frames=1, seed=True)
        with self.assertRaisesRegex(ValueError, "images must be a list"):
            select_human_holdout({"images": {}}, max_frames=1)
        with self.assertRaisesRegex(ValueError, "invalid width"):
            select_human_holdout(
                {"images": [_image("a.mp4", 0, "a.jpg", "val", ["ball"], width=0)]},
                max_frames=1,
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            invalid = root / "invalid.json"
            invalid.write_text("{broken", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid JSON manifest"):
                select_human_holdout_file(
                    invalid,
                    root / "holdout.json",
                    root / "report.json",
                    max_frames=1,
                )

            array = root / "array.json"
            array.write_text(json.dumps([]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest object"):
                select_human_holdout_file(
                    array,
                    root / "holdout.json",
                    root / "report.json",
                    max_frames=1,
                )


class BallReviewSetTest(unittest.TestCase):
    def test_selects_positive_and_negative_review_frames_by_original_source(self):
        review, report = select_ball_review_set(
            _ball_manifest(),
            positive_frames=3,
            negative_frames=2,
            seed=11,
            min_frame_gap=3,
        )

        self.assertEqual(review["summary"]["positive_frames"], 3)
        self.assertEqual(review["summary"]["negative_frames"], 2)
        self.assertEqual(
            {image["review_task"] for image in review["images"]},
            {"verify_mask", "verify_absence"},
        )
        self.assertEqual(
            {
                image["source_group"]
                for image in review["images"]
                if image["video"].startswith("clips/IMG_9933")
            },
            {"clips/IMG_9933.mp4"},
        )
        for image in review["images"]:
            if image["review_task"] == "verify_mask":
                self.assertTrue(image["candidate_detections"])
                self.assertFalse(image["ball_absent_verified"])
            else:
                self.assertEqual(image["candidate_detections"], [])
                self.assertIsNone(image["ball_absent_verified"])
        self.assertEqual(report["summary"]["selected_positive_frames"], 3)
        self.assertEqual(report["selection"]["source_group_mode"], "original-video")

    def test_respects_minimum_frame_gap_within_source_group(self):
        review, _ = select_ball_review_set(
            _ball_manifest(),
            positive_frames=6,
            negative_frames=0,
            seed=5,
            min_frame_gap=100,
        )

        selected_by_group: dict[str, list[int]] = {}
        for image in review["images"]:
            selected_by_group.setdefault(image["source_group"], []).append(
                image["frame_index"]
            )

        self.assertTrue(
            all(len(frames) <= 1 for frames in selected_by_group.values())
        )

    def test_ball_review_file_wrapper_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "manifest.json"
            annotations = root / "ball-review.json"
            report_path = root / "ball-review-report.json"
            write_json(source, _ball_manifest())

            review, report = select_ball_review_set_file(
                source,
                annotations,
                report_path,
                positive_frames=2,
                negative_frames=1,
                seed=3,
            )

            self.assertEqual(read_json(annotations), review)
            self.assertEqual(read_json(report_path), report)
            self.assertEqual(report["inputs"]["manifest"], str(source))

    def test_ball_review_validates_arguments(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            select_ball_review_set(
                {"images": []},
                positive_frames=0,
                negative_frames=0,
            )
        with self.assertRaisesRegex(ValueError, "source_group_mode"):
            select_ball_review_set(
                {"images": []},
                positive_frames=1,
                negative_frames=0,
                source_group_mode="bad",
            )
        with self.assertRaisesRegex(ValueError, "min_frame_gap"):
            select_ball_review_set(
                {"images": []},
                positive_frames=1,
                negative_frames=0,
                min_frame_gap=-1,
            )


def _manifest() -> dict:
    images = []
    for video in ("a.mp4", "b.mp4"):
        for frame_index in range(3):
            images.append(
                _image(
                    video,
                    frame_index,
                    f"frames/{video}/{frame_index}.jpg",
                    "val",
                    ["ball", "robots"] if frame_index % 2 == 0 else ["field"],
                )
            )
        images.append(
            _image(
                video,
                100,
                f"frames/{video}/train.jpg",
                "train",
                ["goal_blue"],
            )
        )
    return {
        "schema": "samba_futbot.frame_dataset_merged.v1",
        "images": images,
    }


def _ball_manifest() -> dict:
    return {
        "schema": "samba_futbot.frame_dataset_merged.v1",
        "images": [
            _image("clips/IMG_9933_f000000_10s.mp4", 0, "a0.jpg", "train", ["robots"]),
            _image("clips/IMG_9933_f000000_10s.mp4", 4, "a4.jpg", "train", ["ball"]),
            _image("clips/IMG_9933_f000000_10s.mp4", 8, "a8.jpg", "train", ["ball"]),
            _image("clips/IMG_9933_f008995_10s.mp4", 12, "b12.jpg", "train", ["ball"]),
            _image("clips/IMG_9938_f001799_10s.mp4", 0, "c0.jpg", "val", ["robots"]),
            _image("clips/IMG_9938_f001799_10s.mp4", 5, "c5.jpg", "val", ["ball"]),
            _image("clips/IMG_9938_f001799_10s.mp4", 9, "c9.jpg", "val", ["robots"]),
        ],
    }


def _image(
    video: str,
    frame_index: int,
    image_path: str,
    split: str,
    classes: list[str],
    *,
    width: int = 1920,
    height: int = 1080,
) -> dict:
    return {
        "video": video,
        "frame_index": frame_index,
        "image_path": image_path,
        "width": width,
        "height": height,
        "split": split,
        "detections": [
            {
                "class_name": class_name,
                "score": 0.8,
                "box": [1, 2, 10, 20],
                "crop_path": f"crops/{class_name}.jpg",
            }
            for class_name in classes
        ],
        "crops": [f"crops/{class_name}.jpg" for class_name in classes],
    }
