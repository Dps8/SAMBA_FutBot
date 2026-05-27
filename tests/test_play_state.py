import unittest

from samba_futbot.play_state import ball_in_play, in_play_balls
from samba_futbot.types import Detection


class PlayStateTest(unittest.TestCase):
    def test_ball_is_in_play_on_field(self):
        ball = Detection(0, "ball", 0.9, (20, 20, 24, 24), track_id=1)
        field = Detection(0, "field", 0.8, (0, 0, 100, 60), track_id=2)

        self.assertTrue(ball_in_play(ball, [ball, field]))

    def test_ball_is_in_play_near_robot(self):
        ball = Detection(0, "ball", 0.9, (20, 20, 24, 24), track_id=1)
        robot = Detection(0, "robots", 0.8, (35, 20, 55, 40), track_id=2)

        self.assertTrue(ball_in_play(ball, [ball, robot], possession_radius_px=30))

    def test_off_field_ball_without_robot_is_filtered(self):
        detections = [
            Detection(0, "field", 0.8, (0, 0, 100, 60), track_id=1),
            Detection(0, "ball", 0.9, (150, 150, 154, 154), track_id=2),
        ]

        self.assertEqual(in_play_balls(detections), [])


if __name__ == "__main__":
    unittest.main()
