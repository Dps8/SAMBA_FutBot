import unittest

from samba_futbot.metrics import summarize_tracks
from samba_futbot.types import Detection


class MetricsTest(unittest.TestCase):
    def test_ball_motion_and_coverage_are_reported(self):
        detections = [
            Detection(0, "ball", 0.9, (0, 0, 2, 2), track_id=1),
            Detection(2, "ball", 0.9, (6, 8, 8, 10), track_id=1),
            Detection(0, "robots", 0.8, (20, 20, 40, 40), track_id=2),
        ]

        summary = summarize_tracks(detections, fps=30)

        ball = summary["classes"]["ball"]
        self.assertEqual(ball["frames_with_detection"], 2)
        self.assertAlmostEqual(ball["frame_coverage_ratio"], 2 / 3)
        self.assertEqual(ball["track_fragmentation_gaps"], 1)
        self.assertEqual(summary["motion"]["ball"]["samples"], 1)
        self.assertAlmostEqual(summary["motion"]["ball"]["mean_speed_px_frame"], 5.0)
        self.assertAlmostEqual(summary["motion"]["ball"]["mean_speed_px_second"], 150.0)


if __name__ == "__main__":
    unittest.main()
