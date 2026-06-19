import unittest

from samba_futbot.robot_filter import filter_robot_detections
from samba_futbot.types import Detection


class RobotFilterTest(unittest.TestCase):
    def test_filter_robot_detections_removes_nested_and_limits_per_frame(self):
        detections = [
            Detection(0, "robots", 0.90, (10, 10, 60, 60), track_id=1),
            Detection(0, "robots", 0.70, (14, 14, 56, 56), track_id=2),
            Detection(0, "robots", 0.88, (120, 10, 170, 60), track_id=3),
            Detection(0, "robots", 0.40, (220, 10, 270, 60), track_id=4),
            Detection(0, "ball", 0.95, (80, 80, 90, 90), track_id=9),
        ]

        filtered = filter_robot_detections(
            detections,
            max_per_frame=2,
            min_area=100,
            containment_threshold=0.80,
            iou_threshold=0.55,
        )

        robots = [det for det in filtered if det.class_name == "robots"]
        self.assertEqual([det.track_id for det in robots], [1, 3])
        self.assertEqual(len([det for det in filtered if det.class_name == "ball"]), 1)
        self.assertTrue(all(det.extra.get("robot_filter") == "kept" for det in robots))

    def test_filter_robot_detections_uses_center_distance_as_extra_gate(self):
        detections = [
            Detection(4, "robots", 0.90, (10, 10, 40, 40), track_id=1),
            Detection(4, "robots", 0.89, (42, 12, 72, 42), track_id=2),
            Detection(4, "robots", 0.88, (140, 10, 170, 40), track_id=3),
        ]

        filtered = filter_robot_detections(
            detections,
            max_per_frame=None,
            min_center_distance_px=45,
            iou_threshold=0.95,
            containment_threshold=0.95,
        )

        self.assertEqual([det.track_id for det in filtered], [1, 3])

    def test_filter_robot_detections_protects_robot_near_ball(self):
        detections = [
            Detection(0, "robots", 0.95, (10, 10, 40, 40), track_id=1),
            Detection(0, "robots", 0.93, (90, 10, 120, 40), track_id=2),
            Detection(0, "robots", 0.55, (205, 190, 245, 230), track_id=3),
            Detection(0, "ball", 0.90, (235, 210, 245, 220), track_id=9),
        ]

        filtered = filter_robot_detections(
            detections,
            max_per_frame=2,
            protect_near_ball_px=70,
            containment_threshold=0.95,
            iou_threshold=0.95,
        )

        robots = [det for det in filtered if det.class_name == "robots"]
        self.assertEqual([det.track_id for det in robots], [1, 3])

    def test_filter_robot_detections_keeps_non_robot_detections(self):
        detections = [
            Detection(0, "field", 0.9, (0, 0, 100, 100)),
            Detection(0, "goal_yellow", 0.9, (0, 0, 20, 20)),
        ]

        self.assertEqual(filter_robot_detections(detections), detections)

    def test_filter_uses_box_scaled_center_distance_for_fragments(self):
        detections = [
            Detection(0, "robots", 0.90, (100, 100, 260, 300), track_id=1),
            Detection(0, "robots", 0.80, (215, 245, 285, 305), track_id=2),
            Detection(0, "robots", 0.85, (500, 100, 650, 300), track_id=3),
        ]

        filtered = filter_robot_detections(
            detections,
            iou_threshold=0.95,
            containment_threshold=0.95,
            max_center_distance_ratio=0.55,
        )

        self.assertEqual([det.track_id for det in filtered], [1, 3])

    def test_filter_applies_vertical_play_area_gate(self):
        detections = [
            Detection(0, "robots", 0.90, (10, 50, 60, 120), track_id=1),
            Detection(0, "robots", 0.80, (10, 600, 60, 700), track_id=2),
        ]

        filtered = filter_robot_detections(
            detections,
            frame_height=1000,
            min_center_y_ratio=0.5,
        )

        self.assertEqual([det.track_id for det in filtered], [2])

    def test_filter_prefers_full_semantic_mask_over_color_fragment(self):
        detections = [
            Detection(0, "robots", 0.50, (100, 100, 260, 300), track_id=1),
            Detection(
                0,
                "robots",
                0.85,
                (210, 245, 280, 305),
                prompt="hsv_dark_robot_fallback",
                track_id=2,
                extra={"source": "color_robots"},
            ),
        ]

        filtered = filter_robot_detections(
            detections,
            max_per_frame=1,
        )

        self.assertEqual([det.track_id for det in filtered], [1])


if __name__ == "__main__":
    unittest.main()
