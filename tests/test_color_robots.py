from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from samba_futbot.color_robots import (
    _box_inside_any_field,
    _keep_best_per_frame,
    _merge_nearby_candidates,
    detect_dark_robots,
)
from samba_futbot.io_utils import read_detections
from samba_futbot.types import Detection
from samba_futbot.video import require_cv2


class ColorRobotsTest(unittest.TestCase):
    def test_detect_dark_robots_finds_synthetic_top_camera_blob(self):
        cv2 = require_cv2()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "robots.mp4"
            writer = cv2.VideoWriter(
                str(video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10,
                (160, 120),
            )
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                frame = np.full((120, 160, 3), (45, 150, 45), dtype=np.uint8)
                cv2.circle(frame, (80, 70), 16, (20, 20, 20), -1)
                cv2.rectangle(frame, (75, 62), (90, 78), (210, 210, 210), -1)
                writer.write(frame)
            writer.release()

            out = root / "detections.jsonl"
            detections = detect_dark_robots(video, out, min_area=150, max_area=2000)

            self.assertGreaterEqual(len(detections), 3)
            self.assertTrue(out.exists())
            saved = read_detections(out)
            self.assertEqual(len(saved), len(detections))
            self.assertTrue(all(det.class_name == "robots" for det in saved))

    def test_field_gate_checks_box_centroid(self):
        field = Detection(0, "field", 1.0, (10, 10, 100, 100))

        self.assertTrue(_box_inside_any_field((20, 20, 40, 40), [field], margin_px=0))
        self.assertFalse(_box_inside_any_field((120, 120, 140, 140), [field], margin_px=0))

    def test_keep_best_per_frame_limits_candidates(self):
        detections = [
            Detection(0, "robots", 0.4, (0, 0, 10, 10), area=100),
            Detection(0, "robots", 0.9, (0, 0, 10, 10), area=50),
            Detection(0, "robots", 0.7, (0, 0, 10, 10), area=200),
        ]

        kept = _keep_best_per_frame(detections, max_per_frame=2)

        self.assertEqual([det.score for det in kept], [0.9, 0.7])

    def test_merge_nearby_candidates_unions_robot_parts(self):
        detections = [
            Detection(0, "robots", 0.7, (10, 10, 30, 30), area=100),
            Detection(0, "robots", 0.8, (24, 12, 45, 32), area=120),
            Detection(0, "robots", 0.6, (100, 100, 130, 130), area=150),
        ]

        merged = _merge_nearby_candidates(detections, distance_px=25)

        self.assertEqual(len(merged), 2)
        union = max(merged, key=lambda det: det.score)
        self.assertEqual(union.box, (10, 10, 45, 32))
        self.assertEqual(union.extra["merged_color_robot_parts"], 2)


if __name__ == "__main__":
    unittest.main()
