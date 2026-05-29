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


if __name__ == "__main__":
    unittest.main()
