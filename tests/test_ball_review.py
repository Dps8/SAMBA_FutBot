from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from samba_futbot.ball_review import (
    audit_ball_review,
    audit_ball_review_file,
    export_reviewed_ball_manifest,
    export_reviewed_ball_manifest_file,
    write_ball_review_audit_markdown,
)
from samba_futbot.io_utils import read_json, write_json


class BallReviewAuditTest(unittest.TestCase):
    def test_audit_marks_pending_review_not_ready(self):
        audit = audit_ball_review(_review(pending=True))

        self.assertFalse(audit["ready_for_training"])
        self.assertEqual(audit["summary"]["pending_frames"], 2)
        self.assertEqual(audit["summary"]["issues"], 2)
        self.assertEqual(
            [issue["code"] for issue in audit["issues"]],
            ["positive_without_annotation", "absence_not_verified"],
        )

    def test_audit_marks_completed_review_ready(self):
        audit = audit_ball_review(_review())

        self.assertTrue(audit["ready_for_training"])
        self.assertEqual(audit["summary"]["positive_frames"], 1)
        self.assertEqual(audit["summary"]["positive_annotations"], 1)
        self.assertEqual(audit["summary"]["verified_absence_frames"], 1)
        self.assertEqual(audit["by_task"], {"verify_absence": 1, "verify_mask": 1})

    def test_export_reviewed_ball_manifest_keeps_positive_and_verified_absence(self):
        manifest, report = export_reviewed_ball_manifest(_review())

        self.assertEqual(manifest["schema"], "samba_futbot.reviewed_ball_dataset.v1")
        self.assertEqual(manifest["summary"]["frames"], 2)
        self.assertEqual(manifest["summary"]["positive_frames"], 1)
        self.assertEqual(manifest["summary"]["verified_absence_frames"], 1)
        positive = next(image for image in manifest["images"] if image["detections"])
        negative = next(image for image in manifest["images"] if not image["detections"])
        self.assertEqual(positive["detections"][0]["class_name"], "ball")
        self.assertEqual(positive["detections"][0]["source"], "human_review")
        self.assertTrue(negative["ball_absent_verified"])
        self.assertTrue(report["audit"]["ready_for_training"])
        self.assertEqual(manifest["summary"]["mask_ready_annotations"], 1)
        self.assertEqual(manifest["summary"]["bbox_only_annotations"], 0)
        self.assertEqual(report["audit"]["summary"]["positive_mask_annotations"], 1)

    def test_export_can_reassign_splits_by_source_group(self):
        manifest, report = export_reviewed_ball_manifest(
            _multi_source_review(),
            split_strategy="by-source-balanced",
            train_ratio=0.6,
            val_ratio=0.2,
        )

        split_by_group = {}
        for image in manifest["images"]:
            group = image["source_group"]
            split_by_group.setdefault(group, image["split"])
            self.assertEqual(image["split"], split_by_group[group])

        self.assertEqual(set(split_by_group), {"match-a.mp4", "match-b.mp4", "match-c.mp4"})
        self.assertEqual(report["source_group_splits"], split_by_group)
        self.assertEqual(
            manifest["review_policy"]["split_strategy"],
            "by-source-balanced",
        )
        self.assertEqual(manifest["summary"]["mask_ready_annotations"], 0)
        self.assertEqual(manifest["summary"]["bbox_only_annotations"], 6)

    def test_export_rejects_unknown_split_strategy(self):
        with self.assertRaisesRegex(ValueError, "split_strategy"):
            export_reviewed_ball_manifest(_review(), split_strategy="random")

    def test_export_rejects_incomplete_review_by_default(self):
        with self.assertRaisesRegex(ValueError, "not ready"):
            export_reviewed_ball_manifest(_review(pending=True))

    def test_file_wrappers_write_audit_manifest_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review_path = root / "review.json"
            audit_path = root / "audit.json"
            audit_report_path = root / "audit.md"
            manifest_path = root / "manifest.json"
            report_path = root / "report.json"
            write_json(review_path, _review())

            audit = audit_ball_review_file(
                review_path,
                audit_path,
                report_path=audit_report_path,
            )
            manifest, report = export_reviewed_ball_manifest_file(
                review_path,
                manifest_path,
                report_path,
            )

            self.assertEqual(read_json(audit_path), audit)
            self.assertIn("Ready for training: yes", audit_report_path.read_text())
            self.assertEqual(read_json(manifest_path), manifest)
            self.assertEqual(read_json(report_path), report)
            self.assertEqual(report["inputs"]["review"], str(review_path))

    def test_markdown_report_lists_pending_issues(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "audit.md"
            audit = audit_ball_review(_review(pending=True))

            output = write_ball_review_audit_markdown(audit, report_path)

            text = output.read_text()
            self.assertIn("Ready for training: no", text)
            self.assertIn("positive_without_annotation", text)
            self.assertIn("absence_not_verified", text)


def _review(*, pending: bool = False) -> dict:
    positive_annotations = (
        []
        if pending
        else [{"box": [10, 20, 30, 40], "mask_path": "ball.npz", "mask_index": 0}]
    )
    absent_verified = None if pending else True
    return {
        "schema": "samba_futbot.ball_review_set.v1",
        "selection_fingerprint": {"algorithm": "sha256", "digest": "abc"},
        "images": [
            {
                "video": "clip.mp4",
                "source_group": "clip.mp4",
                "frame_index": 1,
                "split": "train",
                "image_path": "frames/clip_f000001.jpg",
                "width": 100,
                "height": 80,
                "review_task": "verify_mask",
                "annotation_status": "pending" if pending else "accepted",
                "candidate_detections": [{"class_name": "ball", "box": [9, 19, 31, 41]}],
                "annotations": positive_annotations,
                "ball_absent_verified": False,
            },
            {
                "video": "clip.mp4",
                "source_group": "clip.mp4",
                "frame_index": 5,
                "split": "train",
                "image_path": "frames/clip_f000005.jpg",
                "width": 100,
                "height": 80,
                "review_task": "verify_absence",
                "annotation_status": "pending" if pending else "accepted",
                "candidate_detections": [],
                "annotations": [],
                "ball_absent_verified": absent_verified,
            },
        ],
    }


def _multi_source_review() -> dict:
    images = []
    for source_index, source_group in enumerate(("match-a.mp4", "match-b.mp4", "match-c.mp4")):
        for frame_offset in range(2):
            frame_index = source_index * 100 + frame_offset
            images.append(
                {
                    "video": source_group.replace(".mp4", f"_f{frame_index:06d}_10s.mp4"),
                    "source_group": source_group,
                    "frame_index": frame_index,
                    "split": "train",
                    "image_path": f"frames/{source_group}_{frame_index:06d}.jpg",
                    "width": 100,
                    "height": 80,
                    "review_task": "verify_mask",
                    "annotation_status": "accepted",
                    "annotations": [{"box": [10, 20, 30, 40]}],
                }
            )
    return {
        "schema": "samba_futbot.ball_review_set.v1",
        "selection_fingerprint": {"algorithm": "sha256", "digest": "def"},
        "images": images,
    }


if __name__ == "__main__":
    unittest.main()
