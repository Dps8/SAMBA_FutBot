import tempfile
import unittest
from pathlib import Path

import numpy as np

from samba_futbot.heatmap import render_activity_heatmap
from samba_futbot.field_analysis import FieldCalibration
from samba_futbot.types import Detection
from samba_futbot.video import require_cv2


class HeatmapTest(unittest.TestCase):
    def test_renders_dynamic_video_and_accumulated_image(self):
        cv2 = require_cv2()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "source.mp4"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (96, 64)
            )
            for _ in range(5):
                writer.write(np.full((64, 96, 3), (28, 100, 35), dtype=np.uint8))
            writer.release()
            detections = [
                Detection(frame, "robots", 0.9, (10 + frame * 5, 20, 30 + frame * 5, 45))
                for frame in range(5)
            ]
            out_video = root / "heatmap.mp4"
            out_image = root / "heatmap.png"

            report = render_activity_heatmap(
                video,
                detections,
                out_video,
                out_image,
                radius_px=6,
                write_every_n_frames=2,
                output_fps=10,
            )

            self.assertEqual(report["frames"], 5)
            self.assertEqual(report["output_frames"], 3)
            self.assertEqual(report["output_fps"], 10)
            self.assertEqual(report["samples"], 5)
            self.assertGreater(out_video.stat().st_size, 0)
            self.assertGreater(out_image.stat().st_size, 0)

    def test_rejects_invalid_decay(self):
        with self.assertRaisesRegex(ValueError, "decay"):
            render_activity_heatmap("missing.mp4", [], "out.mp4", "out.png", decay=0)

    def test_rejects_invalid_write_stride(self):
        with self.assertRaisesRegex(ValueError, "write_every_n_frames"):
            render_activity_heatmap(
                "missing.mp4",
                [],
                "out.mp4",
                "out.png",
                write_every_n_frames=0,
            )

    def test_filters_small_color_fallback_samples(self):
        detection = Detection(
            0,
            "robots",
            0.7,
            (10, 10, 30, 30),
            prompt="hsv_dark_robot_fallback",
            area=300,
            extra={"source": "color_robots", "extent": 0.8},
        )
        from samba_futbot.track_filter import is_tracking_artifact

        self.assertTrue(
            is_tracking_artifact(
                detection,
                robot_fallback_min_area=3500,
                robot_fallback_max_extent=0.72,
            )
        )

    def test_calibrated_field_filter_rejects_outside_detection(self):
        calibration = FieldCalibration.from_mapping(
            {"image_points": [[0, 0], [100, 0], [100, 50], [0, 50]]}
        )
        from samba_futbot.heatmap import _inside_calibrated_field

        inside = Detection(0, "robots", 0.9, (40, 20, 50, 30))
        outside = Detection(0, "robots", 0.9, (110, 20, 120, 30))

        self.assertTrue(_inside_calibrated_field(inside, calibration, 0.0))
        self.assertFalse(_inside_calibrated_field(outside, calibration, 0.0))


if __name__ == "__main__":
    unittest.main()
