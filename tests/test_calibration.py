import tempfile
import unittest
from pathlib import Path

from samba_futbot.calibration import calibration_quality_report, render_calibration_frame
from samba_futbot.field_analysis import FieldCalibration
from samba_futbot.video import require_cv2


class CalibrationFrameTest(unittest.TestCase):
    def test_render_calibration_frame_from_synthetic_video(self):
        cv2 = require_cv2()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video = tmp_path / "clip.mp4"
            writer = cv2.VideoWriter(
                str(video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10,
                (160, 120),
            )
            frame = cv2.UMat(120, 160, cv2.CV_8UC3).get()
            frame[:] = (0, 130, 0)
            writer.write(frame)
            writer.release()

            calibration = FieldCalibration.from_mapping(
                {
                    "image_points": [[10, 10], [150, 10], [150, 110], [10, 110]],
                }
            )
            out = render_calibration_frame(
                video,
                tmp_path / "calibration.jpg",
                calibration=calibration,
            )

            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 1000)

    def test_calibration_quality_report_flags_points_outside_frame(self):
        calibration = FieldCalibration.from_mapping(
            {
                "field": {"length_m": 2.0, "width_m": 1.0},
                "image_points": [[0, 0], [100, 0], [100, 50], [-5, 50]],
            }
        )

        report = calibration_quality_report(calibration, frame_width=100, frame_height=50)

        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["points"], 4)
        self.assertEqual(len(report["outside_image_points"]), 1)

    def test_calibration_quality_report_accepts_rectangle(self):
        calibration = FieldCalibration.from_mapping(
            {
                "field": {"length_m": 2.0, "width_m": 1.0},
                "image_points": [[0, 0], [100, 0], [100, 50], [0, 50]],
            }
        )

        report = calibration_quality_report(calibration, frame_width=100, frame_height=50)

        self.assertEqual(report["status"], "good")
        self.assertAlmostEqual(report["reprojection_error_m"]["max"], 0.0, places=6)
        self.assertGreater(report["image_polygon"]["area_px"], 0)
        self.assertTrue(report["image_polygon"]["is_strictly_convex"])
        self.assertTrue(report["image_polygon"]["orientation_matches_field"])
        self.assertEqual(report["image_polygon"]["max_to_min_edge_ratio"], 2.0)
        self.assertEqual(report["image_polygon"]["frame_coverage_ratio"], 1.0)
        self.assertIn("max_edge_ratio_error", report["geometry_thresholds"])

    def test_calibration_quality_report_accepts_normal_perspective_trapezoid(self):
        calibration = FieldCalibration.from_mapping(
            {
                "field": {"length_m": 2.0, "width_m": 1.0},
                "image_points": [[30, 20], [170, 20], [195, 110], [5, 110]],
            }
        )

        report = calibration_quality_report(calibration, frame_width=200, frame_height=120)

        self.assertEqual(report["status"], "good")
        self.assertTrue(report["image_polygon"]["is_strictly_convex"])
        self.assertGreater(report["image_polygon"]["frame_coverage_ratio"], 0.5)

    def test_calibration_quality_report_rejects_crossed_corner_order(self):
        calibration = FieldCalibration.from_mapping(
            {
                "field": {"length_m": 2.0, "width_m": 1.0},
                "image_points": [[10, 10], [190, 110], [190, 10], [10, 110]],
            }
        )

        report = calibration_quality_report(calibration, frame_width=200, frame_height=120)
        codes = {issue["code"] for issue in report["issues"]}

        self.assertEqual(report["status"], "fail")
        self.assertIn("invalid_corner_order", codes)
        self.assertTrue(report["image_polygon"]["has_self_intersection"])

    def test_calibration_quality_report_rejects_reversed_orientation(self):
        calibration = FieldCalibration.from_mapping(
            {
                "field": {"length_m": 2.0, "width_m": 1.0},
                "image_points": [[10, 10], [10, 110], [190, 110], [190, 10]],
            }
        )

        report = calibration_quality_report(calibration, frame_width=200, frame_height=120)
        codes = {issue["code"] for issue in report["issues"]}

        self.assertEqual(report["status"], "fail")
        self.assertTrue(report["image_polygon"]["is_strictly_convex"])
        self.assertFalse(report["image_polygon"]["orientation_matches_field"])
        self.assertIn("orientation_mismatch", codes)

    def test_calibration_quality_report_rejects_tiny_frame_coverage(self):
        calibration = FieldCalibration.from_mapping(
            {
                "field": {"length_m": 2.0, "width_m": 1.0},
                "image_points": [[10, 10], [30, 10], [30, 20], [10, 20]],
            }
        )

        report = calibration_quality_report(
            calibration,
            frame_width=1920,
            frame_height=1080,
        )
        codes = {issue["code"] for issue in report["issues"]}

        self.assertEqual(report["status"], "fail")
        self.assertIn("insufficient_frame_coverage", codes)
        self.assertLess(report["image_polygon"]["frame_coverage_ratio"], 0.01)

    def test_calibration_quality_report_handles_repeated_points(self):
        calibration = FieldCalibration.from_mapping(
            {
                "field": {"length_m": 2.0, "width_m": 1.0},
                "image_points": [[10, 10], [10, 10], [100, 50], [0, 50]],
            }
        )

        report = calibration_quality_report(calibration, frame_width=100, frame_height=50)
        codes = {issue["code"] for issue in report["issues"]}

        self.assertEqual(report["status"], "fail")
        self.assertIn("invalid_homography", codes)
        self.assertIn("zero_length_edge", codes)
        self.assertFalse(report["reprojection_error_m"]["valid"])
        self.assertIsNotNone(report["reprojection_error_m"]["failure"])

    def test_calibration_quality_report_rejects_extreme_skew_at_zero_reprojection_error(self):
        calibration = FieldCalibration.from_mapping(
            {
                "field": {"length_m": 2.0, "width_m": 1.0},
                "image_points": [[10, 10], [310, 10], [311, 12], [10, 210]],
            }
        )

        report = calibration_quality_report(calibration, frame_width=320, frame_height=220)
        codes = {issue["code"] for issue in report["issues"]}

        self.assertAlmostEqual(report["reprojection_error_m"]["max"], 0.0, places=6)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(
            {"extreme_edge_ratio", "near_collinear_corner", "extreme_polygon_skew"} & codes
        )


if __name__ == "__main__":
    unittest.main()
