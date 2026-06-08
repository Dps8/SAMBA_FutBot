import tempfile
import unittest
from pathlib import Path

import numpy as np

from samba_futbot.dataset import (
    clip_box,
    export_frame_dataset,
    merge_frame_dataset_manifests,
    selected_detections_by_frame,
    split_for_key,
)
from samba_futbot.io_utils import read_json, write_detections, write_json
from samba_futbot.types import Detection
from samba_futbot.video import require_cv2


class DatasetExportTest(unittest.TestCase):
    def test_selected_detections_filters_and_caps_per_class(self):
        detections = [
            Detection(0, "robots", 0.9, (0, 0, 10, 10)),
            Detection(0, "robots", 0.8, (10, 0, 20, 10)),
            Detection(0, "robots", 0.7, (20, 0, 30, 10)),
            Detection(1, "robots", 0.95, (0, 0, 10, 10)),
            Detection(2, "ball", 0.5, (0, 0, 5, 5)),
            Detection(2, "field", 0.99, (0, 0, 100, 100)),
        ]

        selected = selected_detections_by_frame(
            detections,
            classes=["robots", "ball"],
            min_score=0.6,
            frame_stride=2,
            max_frames=None,
            max_detections_per_class_per_frame=2,
        )

        self.assertEqual(list(selected), [0])
        self.assertEqual(len(selected[0]), 2)
        self.assertGreaterEqual(selected[0][0].score, selected[0][1].score)

    def test_split_for_key_is_deterministic_and_validates_ratios(self):
        first = split_for_key("IMG_9938", train_ratio=0.7, val_ratio=0.2)
        second = split_for_key("IMG_9938", train_ratio=0.7, val_ratio=0.2)

        self.assertEqual(first, second)
        self.assertIn(first, {"train", "val", "test"})
        with self.assertRaises(ValueError):
            split_for_key("bad", train_ratio=0.9, val_ratio=0.2)

    def test_clip_box_keeps_crop_inside_frame(self):
        self.assertEqual(
            clip_box((-5, -3, 12, 15), width=20, height=10, padding_px=4),
            (0.0, 0.0, 16.0, 10.0),
        )

    def test_export_frame_dataset_writes_manifest_frames_and_crops(self):
        cv2 = require_cv2()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            detections = root / "detections.jsonl"
            out_dir = root / "dataset"
            writer = cv2.VideoWriter(
                str(video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10,
                (64, 48),
            )
            for index in range(3):
                frame = np.full((48, 64, 3), 40 + index * 40, dtype=np.uint8)
                writer.write(frame)
            writer.release()
            write_detections(
                detections,
                [
                    Detection(0, "robots", 0.92, (5, 6, 25, 30), track_id=1, team="blue"),
                    Detection(1, "ball", 0.91, (30, 20, 36, 26)),
                    Detection(2, "robots", 0.20, (5, 6, 25, 30)),
                ],
            )

            manifest = export_frame_dataset(
                video_path=video,
                detections_path=detections,
                out_dir=out_dir,
                classes=["robots", "ball"],
                min_score=0.6,
                split_strategy="by-frame",
            )
            saved = read_json(out_dir / "manifest.json")
            self.assertTrue((out_dir / saved["images"][0]["image_path"]).exists())
            self.assertTrue((out_dir / saved["images"][0]["detections"][0]["crop_path"]).exists())

        self.assertEqual(manifest["summary"]["frames"], 2)
        self.assertEqual(saved["summary"]["detections_by_class"], {"ball": 1, "robots": 1})
        self.assertEqual(saved["summary"]["crops"], 2)

    def test_merge_frame_dataset_manifests_resolves_paths_and_summarizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first" / "manifest.json"
            second = root / "second" / "manifest.json"
            out = root / "merged" / "manifest.json"
            write_json(first, _dataset_manifest("frames/a.jpg", "crops/a.jpg", "train", "ball"))
            write_json(second, _dataset_manifest("frames/b.jpg", "crops/b.jpg", "val", "robots"))

            merged = merge_frame_dataset_manifests(
                [first, second],
                out,
                split_strategy="by-source-balanced",
                train_ratio=0.5,
                val_ratio=0.5,
            )
            saved = read_json(out)

        self.assertEqual(merged["summary"]["frames"], 2)
        self.assertEqual(saved["summary"]["detections_by_class"], {"ball": 1, "robots": 1})
        self.assertEqual(saved["summary"]["frames_by_split"], {"train": 1, "val": 1})
        self.assertEqual(saved["merge"]["split_strategy"], "by-source-balanced")
        self.assertTrue(Path(saved["images"][0]["image_path"]).is_absolute())
        self.assertTrue(Path(saved["images"][0]["detections"][0]["crop_path"]).is_absolute())


def _dataset_manifest(image_path: str, crop_path: str, split: str, class_name: str) -> dict:
    return {
        "schema": "samba_futbot.frame_dataset.v1",
        "summary": {},
        "images": [
            {
                "image_path": image_path,
                "width": 100,
                "height": 100,
                "split": split,
                "detections": [
                    {
                        "class_name": class_name,
                        "score": 0.9,
                        "box": [10, 10, 20, 20],
                        "crop_path": crop_path,
                    }
                ],
                "crops": [crop_path],
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
