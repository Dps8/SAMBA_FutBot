import json
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

from samba_futbot.cli import (
    _context_classes,
    _filtered_prompts,
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
        self.assertFalse(args.robot_color_recovery)
        self.assertEqual(args.robot_recovery_min_area, 800.0)
        self.assertEqual(args.robot_recovery_min_circularity, 0.30)
        self.assertIsNone(args.robot_filter)
        self.assertIsNone(args.robot_filter_max_per_frame)
        self.assertEqual(args.refine_max_jump_px, 35.0)
        self.assertTrue(args.goals)
        self.assertIsNone(args.human_context)
        self.assertTrue(args.qa)
        self.assertTrue(args.render_narrative)
        self.assertTrue(args.render_analysis)
        self.assertFalse(args.analysis_freeze)
        self.assertEqual(args.freeze_seconds, 3.0)
        self.assertTrue(args.mask_overlay)
        self.assertEqual(args.mask_alpha, 0.35)
        self.assertEqual(args.label_scale, 1.05)
        self.assertEqual(args.box_thickness, 3)
        self.assertEqual(args.visual_hold_frames, 12)
        self.assertFalse(args.show_team_labels)
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

    def test_process_top_camera_parses_robot_color_recovery_settings(self):
        args = build_parser().parse_args(
            [
                "process-top-camera",
                "--video",
                "clip.mp4",
                "--robot-color-recovery",
                "--robot-recovery-min-area",
                "900",
                "--robot-recovery-min-circularity",
                "0.35",
                "--robot-recovery-hsv-upper",
                "179,255,120",
                "--robot-recovery-min-center-y-ratio",
                "0.38",
                "--robot-recovery-merge-distance-px",
                "42",
                "--robot-recovery-max-per-frame",
                "4",
            ]
        )

        self.assertTrue(args.robot_color_recovery)
        self.assertEqual(args.robot_recovery_min_area, 900)
        self.assertEqual(args.robot_recovery_min_circularity, 0.35)
        self.assertEqual(args.robot_recovery_hsv_upper, "179,255,120")
        self.assertEqual(args.robot_recovery_min_center_y_ratio, 0.38)
        self.assertEqual(args.robot_recovery_merge_distance_px, 42)
        self.assertEqual(args.robot_recovery_max_per_frame, 4)

    def test_process_top_camera_parses_robot_filter_settings(self):
        args = build_parser().parse_args(
            [
                "process-top-camera",
                "--video",
                "clip.mp4",
                "--no-robot-filter",
                "--robot-filter-max-per-frame",
                "2",
                "--robot-filter-min-area",
                "500",
                "--robot-filter-max-area-ratio",
                "0.07",
                "--robot-filter-containment-threshold",
                "0.8",
                "--robot-filter-iou-threshold",
                "0.5",
                "--robot-filter-min-center-distance-px",
                "30",
            ]
        )

        self.assertFalse(args.robot_filter)
        self.assertEqual(args.robot_filter_max_per_frame, 2)
        self.assertEqual(args.robot_filter_min_area, 500)
        self.assertEqual(args.robot_filter_max_area_ratio, 0.07)
        self.assertEqual(args.robot_filter_containment_threshold, 0.8)
        self.assertEqual(args.robot_filter_iou_threshold, 0.5)
        self.assertEqual(args.robot_filter_min_center_distance_px, 30)

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

    def test_context_classes_include_configured_human_intervention_classes(self):
        args = build_parser().parse_args(["process-top-camera", "--video", "clip.mp4"])
        config = {
            "human_detection": {
                "enabled": True,
                "classes": ["person", "referee", "hand", "person"],
            }
        }

        self.assertEqual(
            _context_classes(args, config),
            "field,robots,goal_blue,goal_yellow,person,referee,hand",
        )

    def test_context_classes_can_enable_human_context_per_run(self):
        args = build_parser().parse_args(
            ["process-top-camera", "--video", "clip.mp4", "--human-context"]
        )
        config = {
            "human_detection": {
                "enabled": False,
                "classes": ["person", "referee", "hand"],
            }
        }

        self.assertEqual(
            _context_classes(args, config),
            "field,robots,goal_blue,goal_yellow,person,referee,hand",
        )

    def test_filtered_prompts_limits_each_class_without_losing_classes(self):
        prompts = {
            "goal_blue": ["a", "b", "c"],
            "goal_yellow": ["d", "e", "f"],
            "ball": ["g"],
        }

        filtered = _filtered_prompts(
            prompts,
            "goal_blue,goal_yellow",
            max_per_class=2,
        )

        self.assertEqual(filtered, {"goal_blue": ["a", "b"], "goal_yellow": ["d", "e"]})

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

    def test_detect_dark_robots_command_parses_color_shape_settings(self):
        args = build_parser().parse_args(
            [
                "detect-dark-robots",
                "--video",
                "clip.mp4",
                "--out",
                "robots.jsonl",
                "--min-area",
                "500",
                "--max-area",
                "12000",
                "--hsv-upper",
                "179,255,115",
                "--field-detections",
                "field.jsonl",
                "--min-center-y-ratio",
                "0.25",
                "--merge-distance-px",
                "40",
                "--max-per-frame",
                "4",
            ]
        )

        self.assertEqual(args.command, "detect-dark-robots")
        self.assertEqual(args.min_area, 500)
        self.assertEqual(args.max_area, 12000)
        self.assertEqual(args.hsv_upper, "179,255,115")
        self.assertEqual(args.field_detections, "field.jsonl")
        self.assertEqual(args.min_center_y_ratio, 0.25)
        self.assertEqual(args.merge_distance_px, 40)
        self.assertEqual(args.max_per_frame, 4)

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

    def test_export_frame_dataset_command_parses_filters(self):
        args = build_parser().parse_args(
            [
                "export-frame-dataset",
                "--video",
                "clip.mp4",
                "--detections",
                "tracks.jsonl",
                "--out-dir",
                "dataset",
                "--classes",
                "robots,ball",
                "--min-score",
                "0.7",
                "--frame-stride",
                "3",
                "--max-frames",
                "20",
                "--no-crop",
                "--split-strategy",
                "by-frame",
            ]
        )

        self.assertEqual(args.video, "clip.mp4")
        self.assertEqual(args.detections, "tracks.jsonl")
        self.assertEqual(args.out_dir, "dataset")
        self.assertEqual(args.classes, "robots,ball")
        self.assertEqual(args.min_score, 0.7)
        self.assertEqual(args.frame_stride, 3)
        self.assertEqual(args.max_frames, 20)
        self.assertFalse(args.crop)
        self.assertEqual(args.split_strategy, "by-frame")

    def test_training_export_commands_parse_manifest_and_output(self):
        merge = build_parser().parse_args(
            [
                "merge-frame-datasets",
                "--manifests",
                "a/manifest.json,b/manifest.json",
                "--out",
                "merged/manifest.json",
                "--split-strategy",
                "by-source-balanced",
            ]
        )
        coco = build_parser().parse_args(
            [
                "export-coco",
                "--manifest",
                "dataset/manifest.json",
                "--out-dir",
                "coco",
                "--image-root",
                "dataset",
            ]
        )

        self.assertEqual(merge.manifests, "a/manifest.json,b/manifest.json")
        self.assertEqual(merge.out, "merged/manifest.json")
        self.assertEqual(merge.split_strategy, "by-source-balanced")
        self.assertEqual(coco.manifest, "dataset/manifest.json")
        self.assertEqual(coco.out_dir, "coco")
        self.assertEqual(coco.image_root, "dataset")

    def test_balance_coco_command_parses_focus_settings(self):
        args = build_parser().parse_args(
            [
                "balance-coco",
                "--annotations",
                "coco/train.json",
                "--out",
                "coco/train-balanced.json",
                "--focus-classes",
                "ball",
                "--negative-ratio",
                "1.5",
                "--seed",
                "44",
                "--focus-only",
            ]
        )

        self.assertEqual(args.focus_classes, "ball")
        self.assertEqual(args.negative_ratio, 1.5)
        self.assertEqual(args.seed, 44)
        self.assertTrue(args.focus_only)

    def test_prepare_sam3_finetune_command_parses_smoke_settings(self):
        args = build_parser().parse_args(
            [
                "prepare-sam3-finetune",
                "--template",
                "/opt/sam3/template.yaml",
                "--out",
                "/opt/sam3/configs/samba.yaml",
                "--data-root",
                "/data/samba",
                "--train-json",
                "/data/samba/train.json",
                "--val-json",
                "/data/samba/val.json",
                "--experiment-dir",
                "/runs/smoke",
                "--bpe-path",
                "/opt/sam3/bpe.gz",
                "--epochs",
                "2",
                "--train-limit",
                "16",
                "--val-limit",
                "10",
                "--resolution",
                "672",
                "--mode",
                "val",
            ]
        )

        self.assertEqual(args.epochs, 2)
        self.assertEqual(args.train_limit, 16)
        self.assertEqual(args.val_limit, 10)
        self.assertEqual(args.resolution, 672)
        self.assertEqual(args.mode, "val")

    def test_compare_sam3_finetune_command_parses_paths(self):
        args = build_parser().parse_args(
            [
                "compare-sam3-finetune",
                "--ground-truth",
                "val.json",
                "--baseline",
                "baseline.json",
                "--candidate",
                "candidate.json",
                "--out",
                "comparison.json",
                "--report-out",
                "comparison.md",
            ]
        )

        self.assertEqual(args.command, "compare-sam3-finetune")
        self.assertEqual(args.iou_type, "segm")

    def test_sam3_training_export_command_parses_portable_dataset_options(self):
        args = build_parser().parse_args(
            [
                "export-sam3-training",
                "--manifest",
                "dataset/manifest.json",
                "--out-dir",
                "dataset/sam3",
                "--image-root",
                "dataset",
                "--class-prompts",
                "prompts.json",
                "--include-negatives",
                "--negative-classes",
                "ball,robots",
                "--max-negative-pairs-per-class",
                "30",
            ]
        )

        self.assertEqual(args.image_root, "dataset")
        self.assertEqual(args.class_prompts, "prompts.json")
        self.assertTrue(args.include_negatives)
        self.assertEqual(args.negative_classes, "ball,robots")
        self.assertEqual(args.max_negative_pairs_per_class, 30)

    def test_finetune_preflight_command_parses_environment_paths(self):
        args = build_parser().parse_args(
            [
                "finetune-preflight",
                "--sam3-root",
                "/opt/sam3",
                "--checkpoint",
                "/models/sam3.pt",
                "--train-json",
                "train.json",
                "--val-json",
                "val.json",
                "--train-images",
                "images",
                "--val-images",
                "images",
                "--out",
                "preflight.json",
                "--no-check-cuda",
            ]
        )

        self.assertEqual(args.sam3_root, "/opt/sam3")
        self.assertEqual(args.checkpoint, "/models/sam3.pt")
        self.assertFalse(args.check_cuda)

    def test_dataset_quality_command_parses_manifest_outputs_and_thresholds(self):
        args = build_parser().parse_args(
            [
                "dataset-quality",
                "--manifest",
                "dataset/manifest.json",
                "--out",
                "dataset/quality.json",
                "--report-out",
                "dataset/quality.md",
                "--low-score-threshold",
                "0.72",
                "--max-review-examples",
                "9",
            ]
        )

        self.assertEqual(args.manifest, "dataset/manifest.json")
        self.assertEqual(args.out, "dataset/quality.json")
        self.assertEqual(args.report_out, "dataset/quality.md")
        self.assertEqual(args.low_score_threshold, 0.72)
        self.assertEqual(args.max_review_examples, 9)

    def test_curate_dataset_command_parses_filters_and_deduplication(self):
        args = build_parser().parse_args(
            [
                "curate-dataset",
                "--manifest",
                "dataset/manifest.json",
                "--out",
                "dataset/curated.json",
                "--report-out",
                "dataset/curation.json",
                "--classes",
                "ball,robots",
                "--min-score",
                "0.72",
                "--review-exclusions",
                "dataset/review.json",
                "--no-drop-empty-frames",
                "--no-deduplicate-source-frames",
            ]
        )

        self.assertEqual(args.classes, "ball,robots")
        self.assertEqual(args.min_score, 0.72)
        self.assertEqual(args.review_exclusions, "dataset/review.json")
        self.assertFalse(args.drop_empty_frames)
        self.assertFalse(args.deduplicate_source_frames)

    def test_select_holdout_command_parses_reproducible_selection(self):
        args = build_parser().parse_args(
            [
                "select-holdout",
                "--manifest",
                "dataset/curated.json",
                "--out",
                "dataset/holdout.json",
                "--report-out",
                "dataset/holdout-report.json",
                "--max-frames",
                "30",
                "--preferred-split",
                "test",
                "--seed",
                "44",
            ]
        )

        self.assertEqual(args.max_frames, 30)
        self.assertEqual(args.preferred_split, "test")
        self.assertEqual(args.seed, 44)

    def test_select_ball_review_command_parses_balanced_review_settings(self):
        args = build_parser().parse_args(
            [
                "select-ball-review",
                "--manifest",
                "dataset/manifest.json",
                "--out",
                "dataset/ball-review.json",
                "--report-out",
                "dataset/ball-review-report.json",
                "--positive-frames",
                "12",
                "--negative-frames",
                "18",
                "--seed",
                "9",
                "--class-name",
                "ball",
                "--source-group-mode",
                "original-video",
                "--min-frame-gap",
                "6",
            ]
        )

        self.assertEqual(args.command, "select-ball-review")
        self.assertEqual(args.positive_frames, 12)
        self.assertEqual(args.negative_frames, 18)
        self.assertEqual(args.min_frame_gap, 6)

    def test_ball_review_audit_and_export_commands_parse_paths(self):
        audit = build_parser().parse_args(
            [
                "audit-ball-review",
                "--review",
                "dataset/ball-review.json",
                "--out",
                "dataset/ball-review-audit.json",
                "--report-out",
                "dataset/ball-review-audit.md",
            ]
        )
        export = build_parser().parse_args(
            [
                "export-reviewed-ball",
                "--review",
                "dataset/ball-review.json",
                "--out",
                "dataset/reviewed-ball-manifest.json",
                "--report-out",
                "dataset/reviewed-ball-report.json",
                "--no-include-verified-absence",
                "--split-strategy",
                "by-source-balanced",
                "--train-ratio",
                "0.7",
                "--val-ratio",
                "0.2",
            ]
        )

        self.assertEqual(audit.command, "audit-ball-review")
        self.assertEqual(audit.class_name, "ball")
        self.assertEqual(audit.report_out, "dataset/ball-review-audit.md")
        self.assertEqual(export.command, "export-reviewed-ball")
        self.assertFalse(export.include_verified_absence)
        self.assertEqual(export.split_strategy, "by-source-balanced")
        self.assertEqual(export.train_ratio, 0.7)
        self.assertEqual(export.val_ratio, 0.2)

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

    def test_submission_report_command_parses_batch_and_training_roots(self):
        args = build_parser().parse_args(
            [
                "submission-report",
                "--batch-root",
                "outputs/review/batch",
                "--training-root",
                "outputs/review/training",
                "--out",
                "outputs/review/final.md",
                "--title",
                "Final Evidence",
                "--top",
                "3",
            ]
        )

        self.assertEqual(args.batch_root, "outputs/review/batch")
        self.assertEqual(args.training_root, "outputs/review/training")
        self.assertEqual(args.out, "outputs/review/final.md")
        self.assertEqual(args.title, "Final Evidence")
        self.assertEqual(args.top, 3)

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

    def test_showcase_index_command_parses_claims_and_limit(self):
        args = build_parser().parse_args(
            [
                "showcase-index",
                "--root",
                "outputs/review",
                "--out",
                "showcase.json",
                "--report-out",
                "showcase.md",
                "--limit",
                "5",
                "--required-claims",
                "ball_tracking,team_possession,shot_pressure",
            ]
        )

        self.assertEqual(args.root, "outputs/review")
        self.assertEqual(args.out, "showcase.json")
        self.assertEqual(args.report_out, "showcase.md")
        self.assertEqual(args.limit, 5)
        self.assertEqual(args.required_claims, "ball_tracking,team_possession,shot_pressure")

    def test_compare_qa_command_parses_inputs_and_outputs(self):
        args = build_parser().parse_args(
            [
                "compare-qa",
                "--baseline",
                "baseline-qa.json",
                "--candidate",
                "candidate-qa.json",
                "--out",
                "comparison.json",
                "--report-out",
                "comparison.md",
            ]
        )

        self.assertEqual(args.baseline, "baseline-qa.json")
        self.assertEqual(args.candidate, "candidate-qa.json")
        self.assertEqual(args.out, "comparison.json")
        self.assertEqual(args.report_out, "comparison.md")

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

    def test_assign_teams_command_parses_existing_tracks(self):
        args = build_parser().parse_args(
            [
                "assign-teams",
                "--video",
                "clip.mp4",
                "--tracks",
                "tracks.jsonl",
                "--out",
                "tracks-with-teams.jsonl",
                "--config",
                "config/default.yml",
            ]
        )

        self.assertEqual(args.video, "clip.mp4")
        self.assertEqual(args.tracks, "tracks.jsonl")
        self.assertEqual(args.out, "tracks-with-teams.jsonl")

    def test_validate_goals_command_parses_existing_artifacts(self):
        args = build_parser().parse_args(
            [
                "validate-goals",
                "--tracks",
                "tracks.jsonl",
                "--events",
                "events.json",
                "--out",
                "validated-events.json",
                "--config",
                "config/custom.yml",
            ]
        )

        self.assertEqual(args.tracks, "tracks.jsonl")
        self.assertEqual(args.events, "events.json")
        self.assertEqual(args.out, "validated-events.json")
        self.assertEqual(args.config, "config/custom.yml")

    def test_team_quality_command_parses_thresholds(self):
        args = build_parser().parse_args(
            [
                "team-quality",
                "--tracks",
                "tracks.jsonl",
                "--out",
                "team-quality.json",
                "--report-out",
                "team-quality.md",
                "--unknown-ratio-threshold",
                "0.3",
                "--ambiguous-track-dominance",
                "0.8",
                "--max-dominant-team-ratio",
                "0.9",
            ]
        )

        self.assertEqual(args.tracks, "tracks.jsonl")
        self.assertEqual(args.out, "team-quality.json")
        self.assertEqual(args.report_out, "team-quality.md")
        self.assertEqual(args.unknown_ratio_threshold, 0.3)
        self.assertEqual(args.ambiguous_track_dominance, 0.8)
        self.assertEqual(args.max_dominant_team_ratio, 0.9)

    def test_situation_analysis_command_parses_thresholds(self):
        args = build_parser().parse_args(
            [
                "situation-analysis",
                "--tracks",
                "tracks.jsonl",
                "--out",
                "situations.json",
                "--possession-radius-px",
                "80",
                "--dispute-margin-px",
                "18",
                "--frame-width",
                "1080",
            ]
        )

        self.assertEqual(args.tracks, "tracks.jsonl")
        self.assertEqual(args.out, "situations.json")
        self.assertEqual(args.possession_radius_px, 80.0)
        self.assertEqual(args.dispute_margin_px, 18.0)
        self.assertEqual(args.frame_width, 1080.0)

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
                "--mask-alpha",
                "0.42",
                "--label-scale",
                "0.9",
                "--box-thickness",
                "5",
                "--visual-hold-frames",
                "18",
            ]
        )

        self.assertEqual(args.events, "events.json")
        self.assertEqual(args.style, "analysis")
        self.assertTrue(args.analysis_freeze)
        self.assertEqual(args.freeze_seconds, 1.25)
        self.assertEqual(args.freeze_event_types, "shot,goal_candidate")
        self.assertTrue(args.mask_overlay)
        self.assertEqual(args.mask_alpha, 0.42)
        self.assertEqual(args.label_scale, 0.9)
        self.assertEqual(args.box_thickness, 5)
        self.assertEqual(args.visual_hold_frames, 18)
        self.assertFalse(args.show_team_labels)


if __name__ == "__main__":
    unittest.main()
