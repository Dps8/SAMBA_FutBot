import json
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

from samba_futbot.cli import (
    _context_classes,
    _frame_anchors,
    _git_snapshot,
    _game_state_summary,
    _jsonable,
    _resolve_ball_color_profile,
    _runtime_snapshot,
    _source_fingerprint,
    _write_pipeline_manifest,
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
        self.assertTrue(args.render_narrative)
        self.assertTrue(args.render_analysis)
        self.assertFalse(args.analysis_freeze)
        self.assertEqual(args.freeze_seconds, 1.5)
        self.assertTrue(args.generate_game_state)
        self.assertTrue(args.filter_by_game_state)
        self.assertEqual(args.game_state_missing_ball_frames, 12)

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
        args = build_parser().parse_args(
            [
                "process-top-camera",
                "--video",
                "clip.mp4",
                "--no-qa",
                "--run-report-out",
                "report.md",
                "--run-manifest-out",
                "manifest.json",
            ]
        )

        self.assertFalse(args.qa)
        self.assertEqual(args.run_report_out, "report.md")
        self.assertEqual(args.run_manifest_out, "manifest.json")

    def test_process_video_accepts_run_report_out(self):
        args = build_parser().parse_args(
            [
                "process-video",
                "--video",
                "clip.mp4",
                "--run-report-out",
                "report.md",
                "--run-manifest-out",
                "manifest.json",
            ]
        )

        self.assertEqual(args.run_report_out, "report.md")
        self.assertEqual(args.run_manifest_out, "manifest.json")

    def test_process_pipelines_accept_game_state_outputs(self):
        for command in ("process-video", "process-top-camera"):
            args = build_parser().parse_args(
                [
                    command,
                    "--video",
                    "clip.mp4",
                    "--no-generate-game-state",
                    "--no-filter-by-game-state",
                    "--game-state-out",
                    "game-state.json",
                    "--external-events-out",
                    "external-events.json",
                    "--game-segments-out",
                    "segments.json",
                    "--game-state-missing-ball-frames",
                    "7",
                    "--robot-disabled-after-frames",
                    "30",
                ]
            )

            self.assertFalse(args.generate_game_state)
            self.assertFalse(args.filter_by_game_state)
            self.assertEqual(args.game_state_out, "game-state.json")
            self.assertEqual(args.external_events_out, "external-events.json")
            self.assertEqual(args.game_segments_out, "segments.json")
            self.assertEqual(args.game_state_missing_ball_frames, 7)
            self.assertEqual(args.robot_disabled_after_frames, 30)

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

    def test_export_pseudolabels_command_parses_filters(self):
        args = build_parser().parse_args(
            [
                "export-pseudolabels",
                "--detections",
                "detections.jsonl",
                "--out",
                "pseudolabels.json",
                "--classes",
                "robots,ball",
                "--min-score",
                "0.75",
                "--no-require-mask",
            ]
        )

        self.assertEqual(args.detections, "detections.jsonl")
        self.assertEqual(args.out, "pseudolabels.json")
        self.assertEqual(args.classes, "robots,ball")
        self.assertEqual(args.min_score, 0.75)
        self.assertFalse(args.require_mask)

    def test_jsonable_serializes_private_cli_values(self):
        value = _jsonable({"path": __file__, "func": self.test_jsonable_serializes_private_cli_values})

        self.assertIn("path", value)
        self.assertIsInstance(value["func"], str)

    def test_git_snapshot_reports_current_repo(self):
        snapshot = _git_snapshot(Path(__file__).resolve().parents[1])

        if not snapshot["available"]:
            self.skipTest("Current test tree is not inside a Git checkout.")
        self.assertTrue(snapshot["available"])
        self.assertIn("branch", snapshot)
        self.assertIn("commit", snapshot)
        self.assertIn("dirty", snapshot)
        self.assertIn("changed_files", snapshot)

    def test_git_snapshot_handles_non_repo_paths(self):
        with TemporaryDirectory() as temp_dir:
            snapshot = _git_snapshot(Path(temp_dir))

        self.assertFalse(snapshot["available"])
        self.assertIn("error", snapshot)

    def test_runtime_snapshot_reports_python_environment(self):
        snapshot = _runtime_snapshot()

        self.assertIn("python", snapshot)
        self.assertIn("python_executable", snapshot)
        self.assertIn("platform", snapshot)

    def test_source_fingerprint_hashes_pipeline_files(self):
        fingerprint = _source_fingerprint(Path(__file__).resolve().parents[1])

        self.assertEqual(fingerprint["algorithm"], "sha256")
        self.assertEqual(len(fingerprint["digest"]), 64)
        self.assertGreater(fingerprint["files_hashed"], 0)
        self.assertIn("src/samba_futbot/cli.py", fingerprint["paths"])

    def test_write_pipeline_manifest_includes_reproducibility_context(self):
        with TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir)
            args = Namespace(
                run_manifest_out=str(results_dir / "manifest.json"),
                command="process-top-camera",
                video="clip.mp4",
                config="config/default.yml",
                suffix="demo",
            )
            manifest_out = _write_pipeline_manifest(
                args,
                results_dir=results_dir,
                stem="clip",
                artifacts={"tracks": results_dir / "tracks.jsonl"},
                metrics_summary={"frames_observed": 1, "detections": {}, "tracks": {}},
                event_summary={"events": 0},
                field_analysis_summary=None,
                qa_status="review",
            )
            manifest = json.loads(manifest_out.read_text(encoding="utf-8"))

        self.assertEqual(manifest["command_name"], "process-top-camera")
        self.assertEqual(manifest["qa_status"], "review")
        self.assertIn("generated_at_utc", manifest)
        self.assertIn("command_argv", manifest)
        self.assertIn("git", manifest)
        self.assertIn("runtime", manifest)
        self.assertIn("source_fingerprint", manifest)
        self.assertEqual(manifest["source_fingerprint"]["algorithm"], "sha256")
        self.assertEqual(manifest["artifacts"]["tracks"], str(results_dir / "tracks.jsonl"))

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
                "--video",
                "clip.mp4",
                "--config",
                "config/default.yml",
                "--out",
                "analysis.json",
                "--game-state",
                "game-state.json",
                "--robot-csv-out",
                "robots.csv",
                "--zone-control-csv-out",
                "zone-control.csv",
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
        self.assertEqual(args.video, "clip.mp4")
        self.assertEqual(args.config, "config/default.yml")
        self.assertEqual(args.map_out, "field-map.png")
        self.assertEqual(args.game_state, "game-state.json")
        self.assertEqual(args.robot_csv_out, "robots.csv")
        self.assertEqual(args.zone_control_csv_out, "zone-control.csv")
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
                "--qa",
                "qa.json",
            ]
        )

        self.assertEqual(args.metrics, "metrics.json")
        self.assertEqual(args.field_analysis, "field.json")
        self.assertEqual(args.qa, "qa.json")

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
                "--max-unknown-team-ratio",
                "0.4",
            ]
        )

        self.assertEqual(args.out, "qa.json")
        self.assertEqual(args.report_out, "qa.md")
        self.assertEqual(args.min_ball_coverage, 0.7)
        self.assertEqual(args.max_ball_jump_px_frame, 40.0)
        self.assertEqual(args.max_unknown_team_ratio, 0.4)

    def test_qa_index_command_parses_paths(self):
        args = build_parser().parse_args(
            [
                "qa-index",
                "--root",
                "outputs/review",
                "--out",
                "qa-index.json",
                "--report-out",
                "qa-index.md",
            ]
        )

        self.assertEqual(args.root, "outputs/review")
        self.assertEqual(args.out, "qa-index.json")
        self.assertEqual(args.report_out, "qa-index.md")

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

    def test_events_and_metrics_accept_game_state_filter(self):
        event_args = build_parser().parse_args(
            [
                "events",
                "--tracks",
                "tracks.jsonl",
                "--out",
                "events.json",
                "--game-state",
                "game-state.json",
            ]
        )
        metric_args = build_parser().parse_args(
            [
                "metrics",
                "--tracks",
                "tracks.jsonl",
                "--out",
                "metrics.json",
                "--game-state",
                "game-state.json",
            ]
        )

        self.assertEqual(event_args.game_state, "game-state.json")
        self.assertEqual(metric_args.game_state, "game-state.json")

    def test_game_state_command_parses_outputs_and_thresholds(self):
        args = build_parser().parse_args(
            [
                "game-state",
                "--tracks",
                "tracks.jsonl",
                "--out",
                "game-state.json",
                "--events-out",
                "external-events.json",
                "--segments-out",
                "segments.json",
                "--missing-ball-frames",
                "8",
            ]
        )

        self.assertEqual(args.tracks, "tracks.jsonl")
        self.assertEqual(args.out, "game-state.json")
        self.assertEqual(args.events_out, "external-events.json")
        self.assertEqual(args.segments_out, "segments.json")
        self.assertEqual(args.missing_ball_frames, 8)

    def test_game_state_summary_counts_playable_and_external_events(self):
        class State:
            def __init__(self, state):
                self.state = state

        class Event:
            def __init__(self, event_type):
                self.event_type = event_type

        summary = _game_state_summary(
            [State("in_play"), State("dead_ball"), State("in_play")],
            [object(), object()],
            [Event("human_intervention"), Event("human_intervention")],
            {0, 2},
        )

        self.assertEqual(summary["frames"], 3)
        self.assertEqual(summary["playable_frames"], 2)
        self.assertAlmostEqual(summary["playable_ratio"], 2 / 3)
        self.assertEqual(summary["states"]["in_play"], 2)
        self.assertEqual(summary["external_events"]["human_intervention"], 2)

    def test_render_demo_command_accepts_events_overlay(self):
        args = build_parser().parse_args(
            [
                "render-demo",
                "--video",
                "clip.mp4",
                "--tracks",
                "tracks.jsonl",
                "--events",
                "events.json",
                "--out",
                "demo.mp4",
                "--style",
                "analysis",
                "--analysis-freeze",
                "--freeze-seconds",
                "1.25",
                "--freeze-event-types",
                "shot,goal_candidate",
            ]
        )

        self.assertEqual(args.events, "events.json")
        self.assertEqual(args.style, "analysis")
        self.assertTrue(args.analysis_freeze)
        self.assertEqual(args.freeze_seconds, 1.25)
        self.assertEqual(args.freeze_event_types, "shot,goal_candidate")


if __name__ == "__main__":
    unittest.main()
