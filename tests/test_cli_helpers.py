import unittest

from samba_futbot.cli import (
    _context_classes,
    _frame_anchors,
    _resolve_ball_color_profile,
    build_parser,
)


class CliHelpersTest(unittest.TestCase):
    def test_frame_anchors_uses_regular_step(self):
        self.assertEqual(_frame_anchors(926, start=150, step=150), [150, 300, 450, 600, 750, 900])

    def test_frame_anchors_falls_back_to_zero_for_short_video(self):
        self.assertEqual(_frame_anchors(102, start=150, step=150), [0])

    def test_process_top_camera_defaults_to_current_best_variant(self):
        args = build_parser().parse_args(["process-top-camera", "--video", "clip.mp4"])

        self.assertEqual(args.suffix, "top-hybrid-ball-v1")
        self.assertIsNone(args.sam3_ball)
        self.assertIsNone(args.color_ball)
        self.assertEqual(args.ball_window_size, 220)
        self.assertEqual(args.ball_step, 150)
        self.assertEqual(args.ball_threshold, 0.05)
        self.assertEqual(args.orange_min_area, 300.0)
        self.assertEqual(args.orange_max_per_frame, 6)
        self.assertEqual(args.refine_max_jump_px, 35.0)
        self.assertTrue(args.goals)
        self.assertTrue(args.qa)

    def test_process_top_camera_can_disable_ball_sources(self):
        args = build_parser().parse_args(
            [
                "process-top-camera",
                "--video",
                "clip.mp4",
                "--no-sam3-ball",
                "--ball-color-profile",
                "white",
                "--ball-hsv-lower",
                "0,0,170",
            ]
        )

        self.assertFalse(args.sam3_ball)
        self.assertIsNone(args.color_ball)
        self.assertEqual(args.ball_color_profile, "white")
        self.assertEqual(args.ball_hsv_lower, "0,0,170")

    def test_process_top_camera_can_disable_qa(self):
        args = build_parser().parse_args(["process-top-camera", "--video", "clip.mp4", "--no-qa"])

        self.assertFalse(args.qa)

    def test_context_classes_can_disable_goals(self):
        parser = build_parser()
        with_goals = parser.parse_args(["process-top-camera", "--video", "clip.mp4"])
        no_goals = parser.parse_args(["process-top-camera", "--video", "clip.mp4", "--no-goals"])

        self.assertEqual(_context_classes(with_goals), "field,robots,goal_blue,goal_yellow")
        self.assertEqual(_context_classes(no_goals), "field,robots")
        self.assertIsNone(with_goals.color_goals)

    def test_detect_orange_ball_accepts_custom_profile(self):
        args = build_parser().parse_args(
            [
                "detect-orange-ball",
                "--video",
                "clip.mp4",
                "--out",
                "ball.jsonl",
                "--color-profile",
                "white",
                "--hsv-lower",
                "0,0,160",
                "--hsv-upper",
                "180,80,255",
            ]
        )

        self.assertEqual(args.color_profile, "white")
        self.assertEqual(args.hsv_lower, "0,0,160")

    def test_resolve_ball_color_profile_uses_configured_hsv(self):
        profile, lower, upper = _resolve_ball_color_profile(
            {
                "default_profile": "white",
                "profiles": {
                    "white": {
                        "hsv_lower": [0, 0, 160],
                        "hsv_upper": [180, 80, 255],
                    }
                },
            },
            profile=None,
            hsv_lower=None,
            hsv_upper="180,70,250",
        )

        self.assertEqual(profile, "white")
        self.assertEqual(lower, (0, 0, 160))
        self.assertEqual(upper, (180, 70, 250))

    def test_resolve_ball_color_profile_has_builtin_profiles(self):
        profile, lower, upper = _resolve_ball_color_profile(
            {"default_profile": "yellow"},
            profile=None,
            hsv_lower=None,
            hsv_upper=None,
        )

        self.assertEqual(profile, "yellow")
        self.assertEqual(lower, (20, 80, 90))
        self.assertEqual(upper, (38, 255, 255))

    def test_field_analysis_command_parses_grid_options(self):
        args = build_parser().parse_args(
            [
                "field-analysis",
                "--tracks",
                "tracks.jsonl",
                "--calibration",
                "calibration.yml",
                "--out",
                "analysis.json",
                "--robot-csv-out",
                "robots.csv",
                "--map-out",
                "field-map.png",
                "--robot-anchor",
                "centroid",
                "--grid-cols",
                "8",
                "--grid-rows",
                "5",
            ]
        )

        self.assertEqual(args.grid_cols, 8)
        self.assertEqual(args.grid_rows, 5)
        self.assertEqual(args.map_out, "field-map.png")
        self.assertEqual(args.robot_csv_out, "robots.csv")
        self.assertEqual(args.robot_anchor, "centroid")

    def test_render_field_map_command_parses_width(self):
        args = build_parser().parse_args(
            [
                "render-field-map",
                "--analysis",
                "analysis.json",
                "--out",
                "field-map.png",
                "--width",
                "900",
            ]
        )

        self.assertEqual(args.width, 900)

    def test_render_calibration_frame_command_parses_frame_index(self):
        args = build_parser().parse_args(
            [
                "render-calibration-frame",
                "--video",
                "clip.mp4",
                "--out",
                "calibration.jpg",
                "--frame-index",
                "120",
            ]
        )

        self.assertEqual(args.frame_index, 120)

    def test_calibration_check_command_parses_optional_video(self):
        args = build_parser().parse_args(
            [
                "calibration-check",
                "--calibration",
                "calibration.yml",
                "--video",
                "clip.mp4",
                "--out",
                "calibration-quality.json",
            ]
        )

        self.assertEqual(args.calibration, "calibration.yml")
        self.assertEqual(args.video, "clip.mp4")
        self.assertEqual(args.out, "calibration-quality.json")

    def test_summarize_run_command_parses_optional_artifacts(self):
        args = build_parser().parse_args(
            [
                "summarize-run",
                "--out",
                "report.md",
                "--metrics",
                "metrics.json",
                "--events",
                "events.json",
                "--field-analysis",
                "field.json",
            ]
        )

        self.assertEqual(args.metrics, "metrics.json")
        self.assertEqual(args.field_analysis, "field.json")

    def test_qa_run_command_parses_thresholds(self):
        args = build_parser().parse_args(
            [
                "qa-run",
                "--out",
                "qa.json",
                "--report-out",
                "qa.md",
                "--metrics",
                "metrics.json",
                "--field-analysis",
                "field.json",
                "--min-ball-coverage",
                "0.7",
                "--max-ball-jump-px-frame",
                "40",
            ]
        )

        self.assertEqual(args.out, "qa.json")
        self.assertEqual(args.report_out, "qa.md")
        self.assertEqual(args.min_ball_coverage, 0.7)
        self.assertEqual(args.max_ball_jump_px_frame, 40.0)

    def test_event_summary_command_parses_paths(self):
        args = build_parser().parse_args(
            [
                "event-summary",
                "--events",
                "events.json",
                "--out",
                "summary.json",
            ]
        )

        self.assertEqual(args.events, "events.json")
        self.assertEqual(args.out, "summary.json")


if __name__ == "__main__":
    unittest.main()
