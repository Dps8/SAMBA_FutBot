import unittest

from samba_futbot.types import Detection
from samba_futbot.visualize import _event_label, _frame_header, _recent_event


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


if __name__ == "__main__":
    unittest.main()
