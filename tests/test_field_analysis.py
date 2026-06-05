import tempfile
import unittest
from pathlib import Path

from PIL import Image

from samba_futbot.field_analysis import (
    DEFAULT_FIELD_LENGTH_M,
    DEFAULT_FIELD_WIDTH_M,
    FieldCalibration,
    analyze_field_tracks,
    load_field_calibration,
    write_field_trajectory_csv,
    write_field_robot_csv,
    write_field_zone_control_csv,
)
from samba_futbot.field_viz import render_field_map
from samba_futbot.types import Detection


class FieldAnalysisTest(unittest.TestCase):
    def test_default_field_dimensions_follow_futbotmx_rules(self):
        calibration = FieldCalibration.from_mapping(
            {
                "image_points": [[0, 0], [243, 0], [243, 182], [0, 182]],
            }
        )

        self.assertEqual(DEFAULT_FIELD_LENGTH_M, 2.43)
        self.assertEqual(DEFAULT_FIELD_WIDTH_M, 1.82)
        self.assertEqual(calibration.field_points[2], (2.43, 1.82))
        self.assertEqual(calibration.center_circle_diameter_m, 0.60)
        self.assertEqual(calibration.penalty_area_depth_m, 0.25)
        self.assertEqual(calibration.penalty_area_width_m, 0.80)
        self.assertEqual(calibration.goal_width_m, 0.60)

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

    def test_analyze_field_tracks_reports_robot_penalty_samples(self):
        calibration = FieldCalibration.from_mapping(
            {
                "field": {
                    "length_m": 2.43,
                    "width_m": 1.82,
                    "penalty_area_depth_m": 0.25,
                    "penalty_area_width_m": 0.80,
                },
                "image_points": [[0, 0], [243, 0], [243, 182], [0, 182]],
            }
        )
        detections = [
            Detection(0, "field", 1.0, (0, 0, 243, 182)),
            Detection(0, "robots", 1.0, (5, 80, 15, 90), track_id=7, team="blue"),
            Detection(1, "robots", 1.0, (230, 80, 240, 90), track_id=8, team="yellow"),
            Detection(0, "ball", 1.0, (120, 90, 124, 94), track_id=1),
        ]

        analysis = analyze_field_tracks(detections, calibration, fps=10)

        self.assertEqual(analysis["robot_summary"]["path_samples"], 2)
        self.assertEqual(analysis["robot_summary"]["penalty_area_samples"], 2)
        self.assertEqual(analysis["robot_path"][0]["penalty_side"], "left")
        self.assertEqual(analysis["robot_summary"]["samples_by_team"]["blue"], 1)
        self.assertEqual(analysis["robot_summary"]["samples_by_team"]["yellow"], 1)
        self.assertEqual(
            analysis["robot_summary"]["penalty_area_samples_by_team"]["blue"]["left"],
            1,
        )
        self.assertEqual(
            analysis["robot_summary"]["penalty_area_samples_by_team"]["yellow"]["right"],
            1,
        )
        self.assertEqual(
            analysis["robot_summary"]["phase_samples_by_team"]["blue"]["attacking"],
            1,
        )
        self.assertEqual(
            analysis["robot_summary"]["phase_samples_by_team"]["yellow"]["attacking"],
            1,
        )
        self.assertEqual(
            analysis["robot_summary"]["phase_ratios_by_team"]["blue"]["attacking"],
            1.0,
        )
        self.assertEqual(analysis["robot_summary"]["attacking_pressure_by_team"]["yellow"], 1.0)
        self.assertTrue(analysis["robot_summary"]["zone_samples_by_team"]["blue"])
        self.assertEqual(analysis["robot_zone_control"][0]["leader"], "blue")
        self.assertEqual(analysis["robot_zone_control"][0]["leader_ratio"], 1.0)

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
            robot_csv_path = tmp_path / "robots.csv"
            write_field_robot_csv(robot_csv_path, analysis)
            zone_control_csv_path = tmp_path / "zone-control.csv"
            write_field_zone_control_csv(zone_control_csv_path, analysis)
            csv_text = csv_path.read_text(encoding="utf-8")
            robot_csv_text = robot_csv_path.read_text(encoding="utf-8")
            zone_control_text = zone_control_csv_path.read_text(encoding="utf-8")

        self.assertIn("field_x_m", csv_text)
        self.assertIn("penalty_side", robot_csv_text)
        self.assertIn("leader_ratio", zone_control_text)

    def test_render_field_map_writes_nonblank_png(self):
        calibration = FieldCalibration.from_mapping(
            {
                "field": {
                    "length_m": 2.0,
                    "width_m": 1.0,
                    "center_circle_diameter_m": 0.50,
                    "penalty_area_depth_m": 0.20,
                    "penalty_area_width_m": 0.60,
                    "goal_width_m": 0.50,
                    "goal_depth_m": 0.10,
                },
                "image_points": [[0, 0], [100, 0], [100, 50], [0, 50]],
            }
        )
        analysis = analyze_field_tracks(
            [
                Detection(0, "field", 1.0, (0, 0, 100, 50)),
                Detection(10, "field", 1.0, (0, 0, 100, 50)),
                Detection(0, "robots", 1.0, (15, 10, 25, 20), track_id=2, team="blue"),
                Detection(10, "robots", 1.0, (75, 30, 85, 40), track_id=3, team="yellow"),
                Detection(0, "ball", 1.0, (20, 20, 30, 30), track_id=1),
                Detection(10, "ball", 1.0, (70, 20, 80, 30), track_id=1),
            ],
            calibration,
            fps=10,
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = render_field_map(analysis, Path(tmp) / "field-map.png", width=500)

            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 1000)
            with Image.open(out) as image:
                pixel_bytes = image.convert("RGB").tobytes()

        blue_pixels = [
            pixel
            for pixel in zip(pixel_bytes[0::3], pixel_bytes[1::3], pixel_bytes[2::3])
            if pixel[2] > 170 and pixel[0] < 120 and pixel[1] < 170
        ]
        yellow_pixels = [
            pixel
            for pixel in zip(pixel_bytes[0::3], pixel_bytes[1::3], pixel_bytes[2::3])
            if pixel[0] > 180 and pixel[1] > 150 and pixel[2] < 120
        ]
        self.assertGreater(len(blue_pixels), 10)
        self.assertGreater(len(yellow_pixels), 10)


if __name__ == "__main__":
    unittest.main()
