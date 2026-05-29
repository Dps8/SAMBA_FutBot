import unittest

from samba_futbot.metrics import summarize_tracks
from samba_futbot.types import Detection


class MetricsTest(unittest.TestCase):
    def test_ball_motion_and_coverage_are_reported_for_in_play_ball(self):
        detections = [
            Detection(0, "field", 0.8, (0, 0, 20, 20), track_id=20),
            Detection(0, "ball", 0.9, (0, 0, 2, 2), track_id=1),
            Detection(1, "ball", 0.9, (100, 100, 102, 102), track_id=99),
            Detection(2, "ball", 0.9, (6, 8, 8, 10), track_id=1),
            Detection(0, "robots", 0.8, (20, 20, 40, 40), track_id=2),
            Detection(2, "robots", 0.8, (20, 20, 40, 40), track_id=2),
        ]

        summary = summarize_tracks(detections, fps=30, possession_radius_px=35)

        ball = summary["classes"]["ball"]
        self.assertEqual(ball["frames_with_detection"], 3)
        self.assertAlmostEqual(ball["frame_coverage_ratio"], 1.0)
        self.assertEqual(ball["in_play_detections"], 2)
        self.assertEqual(ball["in_play_frames"], 2)
        self.assertAlmostEqual(ball["in_play_coverage_ratio"], 2 / 3)
        self.assertAlmostEqual(ball["in_play_duration_seconds"], 2 / 30)
        self.assertEqual(ball["track_fragmentation_gaps"], 1)
        self.assertEqual(summary["motion"]["ball"]["samples"], 1)
        self.assertEqual(summary["motion"]["ball"]["trajectory_scope"], "in_play")
        self.assertAlmostEqual(summary["motion"]["ball"]["mean_speed_px_frame"], 5.0)
        self.assertAlmostEqual(summary["motion"]["ball"]["mean_speed_px_second"], 150.0)
        self.assertEqual(summary["motion"]["ball"]["raw_candidates"]["samples"], 1)

    def test_possession_summary_groups_by_team(self):
        detections = [
            Detection(0, "ball", 0.9, (10, 10, 14, 14), track_id=1),
            Detection(0, "robots", 0.9, (0, 0, 20, 20), track_id=2, team="blue"),
            Detection(1, "ball", 0.9, (12, 10, 16, 14), track_id=1),
            Detection(1, "robots", 0.9, (0, 0, 20, 20), track_id=2, team="blue"),
            Detection(2, "ball", 0.9, (80, 10, 84, 14), track_id=1),
            Detection(2, "robots", 0.9, (70, 0, 90, 20), track_id=3, team="yellow"),
        ]

        summary = summarize_tracks(detections, fps=10, possession_radius_px=25)

        possession = summary["possession"]
        self.assertEqual(possession["frames_with_possession"], 3)
        self.assertAlmostEqual(possession["coverage_ratio"], 1.0)
        self.assertEqual(possession["by_team"]["blue"]["frames"], 2)
        self.assertAlmostEqual(possession["by_team"]["blue"]["seconds"], 0.2)
        self.assertEqual(possession["by_team"]["yellow"]["frames"], 1)
        self.assertEqual(possession["longest_streak"]["team"], "blue")
        self.assertEqual(possession["longest_streak"]["frames"], 2)


if __name__ == "__main__":
    unittest.main()
