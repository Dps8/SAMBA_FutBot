import unittest

from samba_futbot.types import Detection
from samba_futbot.visualize import (
    _event_label,
    _frame_header,
    _recent_event,
    robot_ball_distances,
    shot_probability,
)


class VisualizeTest(unittest.TestCase):
    def test_recent_event_is_held_for_demo_header(self):
        events_by_frame = {
            10: [{"frame_index": 10, "event_type": "shot", "metadata": {"shooting_team": "blue"}}]
        }

        event = _recent_event(events_by_frame, 20, hold_frames=15)

        self.assertEqual(event["event_type"], "shot")
        self.assertEqual(_event_label(event), "shot blue")

    def test_frame_header_combines_possession_and_event(self):
        owner = Detection(20, "robots", 0.9, (0, 0, 10, 10), track_id=7, team="yellow")
        event = {
            "frame_index": 20,
            "event_type": "goal_candidate",
            "metadata": {"scoring_team": "yellow"},
        }

        header = _frame_header(20, owner, event)

        self.assertIn("possession: yellow #7", header)
        self.assertIn("event: goal yellow", header)

    def test_analysis_header_uses_analysis_label(self):
        header = _frame_header(3, None, style="analysis")

        self.assertIn("SAMBA FutBot: analysis", header)

    def test_robot_ball_distances_rank_nearest_robots(self):
        frame = [
            Detection(0, "ball", 1.0, (10, 10, 14, 14), track_id=1),
            Detection(0, "robots", 1.0, (0, 0, 10, 10), track_id=2, team="blue"),
            Detection(0, "robots", 1.0, (90, 90, 110, 110), track_id=3, team="yellow"),
        ]

        distances = robot_ball_distances(frame)

        self.assertEqual([item["track_id"] for item in distances], [2, 3])
        self.assertEqual(distances[0]["team"], "blue")
        self.assertLess(distances[0]["distance_px"], distances[1]["distance_px"])

    def test_shot_probability_reports_direction_speed_and_probability(self):
        previous = Detection(0, "ball", 1.0, (40, 10, 44, 14), track_id=1)
        current = Detection(1, "ball", 1.0, (80, 10, 84, 14), track_id=1)

        pressure = shot_probability(current, previous, frame_width=100)

        self.assertEqual(pressure["target_side"], "right")
        self.assertGreater(pressure["speed_px_frame"], 0)
        self.assertGreater(pressure["probability"], 0.5)


if __name__ == "__main__":
    unittest.main()
