import tempfile
import unittest
from pathlib import Path

from samba_futbot.field_analysis import (
    FieldCalibration,
    analyze_field_tracks,
    load_field_calibration,
    write_field_trajectory_csv,
)
from samba_futbot.types import Detection


class FieldAnalysisTest(unittest.TestCase):
    def test_calibration_projects_image_rectangle_to_field(self):
        calibration = FieldCalibration.from_mapping(
            {
                "field": {"length_m": 2.0, "width_m": 1.0},
                "image_points": [[0, 0], [100, 0], [100, 50], [0, 50]],
            }
        )

        x, y = calibration.transform_point((50, 25))

        self.assertAlmostEqual(x, 1.0, places=6)
        self.assertAlmostEqual(y, 0.5, places=6)

    def test_analyze_field_tracks_reports_metric_speed_and_zones(self):
        calibration = FieldCalibration.from_mapping(
            {
                "field": {"length_m": 2.0, "width_m": 1.0},
                "image_points": [[0, 0], [100, 0], [100, 50], [0, 50]],
            }
        )
        detections = [
            Detection(0, "field", 1.0, (0, 0, 100, 50)),
            Detection(10, "field", 1.0, (0, 0, 100, 50)),
            Detection(0, "ball", 1.0, (20, 20, 30, 30), track_id=1),
            Detection(10, "ball", 1.0, (70, 20, 80, 30), track_id=1),
        ]

        analysis = analyze_field_tracks(detections, calibration, fps=10, grid_cols=4, grid_rows=2)

        self.assertEqual(analysis["summary"]["path_samples"], 2)
        self.assertAlmostEqual(analysis["summary"]["distance_m"], 1.0, places=6)
        self.assertAlmostEqual(analysis["summary"]["mean_speed_m_s"], 1.0, places=6)
        self.assertEqual(analysis["grid"]["sample_counts"][1][1], 1)
        self.assertEqual(analysis["grid"]["sample_counts"][1][3], 1)

    def test_load_calibration_and_write_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            calibration_path = tmp_path / "calibration.yml"
            calibration_path.write_text(
                "\n".join(
                    [
                        "field:",
                        "  length_m: 2.0",
                        "  width_m: 1.0",
                        "image_points:",
                        "  - [0, 0]",
                        "  - [100, 0]",
                        "  - [100, 50]",
                        "  - [0, 50]",
                    ]
                ),
                encoding="utf-8",
            )

            calibration = load_field_calibration(calibration_path)
            analysis = analyze_field_tracks(
                [
                    Detection(0, "field", 1.0, (0, 0, 100, 50)),
                    Detection(0, "ball", 1.0, (20, 20, 30, 30), track_id=1),
                ],
                calibration,
                fps=10,
            )
            csv_path = tmp_path / "trajectory.csv"
            write_field_trajectory_csv(csv_path, analysis)
            csv_text = csv_path.read_text(encoding="utf-8")

        self.assertIn("field_x_m", csv_text)


if __name__ == "__main__":
    unittest.main()
