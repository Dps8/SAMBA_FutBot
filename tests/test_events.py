import unittest

from samba_futbot.events import detect_events, estimate_possession
from samba_futbot.types import Detection


class EventsTest(unittest.TestCase):
    def test_possession_and_pass(self):
        detections = [
            Detection(0, "ball", 1.0, (10, 10, 14, 14), track_id=10),
            Detection(0, "robots", 1.0, (0, 0, 20, 20), track_id=1, team="allied"),
            Detection(1, "ball", 1.0, (100, 10, 104, 14), track_id=10),
            Detection(1, "robots", 1.0, (90, 0, 110, 20), track_id=2, team="allied"),
        ]
        possession = estimate_possession(detections, possession_radius_px=30)
        self.assertEqual(possession[0].track_id, 1)
        self.assertEqual(possession[1].track_id, 2)
        events = detect_events(detections, possession_radius_px=30)
        self.assertEqual(events[0].event_type, "pass")

    def test_collision(self):
        detections = [
            Detection(0, "robots", 1.0, (0, 0, 20, 20), track_id=1),
            Detection(0, "robots", 1.0, (10, 0, 30, 20), track_id=2),
        ]
        events = detect_events(detections, collision_radius_px=25)
        self.assertTrue(any(event.event_type == "collision" for event in events))

    def test_off_field_ball_does_not_trigger_shot(self):
        detections = [
            Detection(0, "ball", 1.0, (5, 200, 9, 204), track_id=10),
            Detection(1, "ball", 1.0, (40, 200, 44, 204), track_id=10),
        ]

        events = detect_events(detections, frame_width=50, goal_x_margin_ratio=1.0)

        self.assertFalse(any(event.event_type == "shot" for event in events))


if __name__ == "__main__":
    unittest.main()
