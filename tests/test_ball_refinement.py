import unittest

from samba_futbot.ball_refinement import refine_ball_trajectory
from samba_futbot.types import Detection


class BallRefinementTest(unittest.TestCase):
    def test_refinement_prefers_temporally_consistent_ball(self):
        detections = [
            Detection(0, "ball", 0.8, (10, 10, 20, 20), area=650),
            Detection(1, "ball", 0.9, (300, 300, 310, 310), area=650),
            Detection(1, "ball", 0.7, (12, 10, 22, 20), area=650),
            Detection(2, "ball", 0.8, (14, 10, 24, 20), area=650),
            Detection(1, "robots", 0.9, (0, 0, 50, 50)),
        ]

        refined = refine_ball_trajectory(detections, max_jump_px=30)

        balls = [det for det in refined if det.class_name == "ball"]
        self.assertEqual(len(balls), 3)
        self.assertEqual([round(det.centroid[0]) for det in balls], [15, 17, 19])
        self.assertTrue(any(det.class_name == "robots" for det in refined))


if __name__ == "__main__":
    unittest.main()
