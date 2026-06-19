import unittest

from samba_futbot.track_filter import filter_tracking_artifacts
from samba_futbot.types import Detection


class TrackFilterTest(unittest.TestCase):
    def test_removes_small_rectangular_color_fallback(self):
        phone = Detection(
            0,
            "robots",
            0.7,
            (10, 10, 90, 60),
            prompt="hsv_dark_robot_fallback",
            area=3200,
            extra={"source": "color_robots", "extent": 0.78},
        )
        robot = Detection(
            0,
            "robots",
            0.8,
            (100, 100, 200, 200),
            prompt="hsv_dark_robot_fallback",
            area=5200,
            extra={"source": "color_robots", "extent": 0.61},
        )

        filtered, report = filter_tracking_artifacts(
            [phone, robot],
            robot_fallback_min_area=3500,
            robot_fallback_max_extent=0.72,
            robot_fallback_max_aspect_ratio=1.75,
        )

        self.assertEqual(filtered, [robot])
        self.assertEqual(report["removed"], 1)

    def test_keeps_sam_robot_regardless_of_fallback_thresholds(self):
        detection = Detection(0, "robots", 0.9, (0, 0, 10, 10), area=20)

        filtered, _ = filter_tracking_artifacts(
            [detection], robot_fallback_min_area=3500
        )

        self.assertEqual(filtered, [detection])

    def test_removes_elongated_color_fallback(self):
        detection = Detection(
            0,
            "robots",
            0.7,
            (10, 10, 100, 55),
            prompt="hsv_dark_robot_fallback",
            area=5000,
            extra={"source": "color_robots", "extent": 0.6},
        )

        filtered, report = filter_tracking_artifacts(
            [detection], robot_fallback_max_aspect_ratio=1.75
        )

        self.assertEqual(filtered, [])
        self.assertEqual(report["removed_by_reason"]["fallback_aspect"], 1)

    def test_removes_oversized_color_fallback(self):
        detection = Detection(
            0,
            "robots",
            0.7,
            (10, 10, 150, 150),
            prompt="hsv_dark_robot_fallback",
            area=18000,
            extra={"source": "color_robots", "extent": 0.6},
        )

        filtered, report = filter_tracking_artifacts(
            [detection], robot_fallback_max_area=15000
        )

        self.assertEqual(filtered, [])
        self.assertEqual(report["removed_by_reason"]["fallback_max_area"], 1)


if __name__ == "__main__":
    unittest.main()
