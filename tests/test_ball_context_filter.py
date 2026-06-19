import unittest

from samba_futbot.ball_context_filter import filter_contextual_ball_candidates
from samba_futbot.types import Detection


class BallContextFilterTest(unittest.TestCase):
    def test_rejects_orange_fragment_inside_robot(self):
        detections = [
            Detection(0, "field", 0.9, (0, 0, 200, 120)),
            Detection(0, "robots", 0.9, (50, 40, 110, 100)),
            Detection(0, "ball", 0.8, (70, 60, 80, 70)),
            Detection(0, "ball", 0.7, (140, 60, 150, 70)),
        ]

        filtered, report = filter_contextual_ball_candidates(detections)

        balls = [det for det in filtered if det.class_name == "ball"]
        self.assertEqual([ball.box for ball in balls], [(140, 60, 150, 70)])
        self.assertEqual(report["removed_by_reason"]["robot_overlap"], 1)

    def test_accepts_ball_near_referee_hand_without_field(self):
        detections = [
            Detection(0, "hand", 0.8, (90, 40, 110, 70)),
            Detection(0, "ball", 0.8, (105, 60, 115, 70)),
        ]

        filtered, _ = filter_contextual_ball_candidates(detections)

        self.assertEqual(len([det for det in filtered if det.class_name == "ball"]), 1)

    def test_rejects_ball_without_field_or_human_context(self):
        filtered, report = filter_contextual_ball_candidates(
            [Detection(0, "ball", 0.8, (10, 10, 20, 20))]
        )

        self.assertFalse([det for det in filtered if det.class_name == "ball"])
        self.assertEqual(report["removed_by_reason"]["outside_context"], 1)


if __name__ == "__main__":
    unittest.main()
