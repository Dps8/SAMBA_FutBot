import tempfile
import unittest
from pathlib import Path

import numpy as np

from samba_futbot.color_goals import (
    adapt_goal_color_profiles_from_detections,
    detect_colored_goals,
    enforce_goal_frame_constraints,
    stabilize_goal_detections,
)
from samba_futbot.types import Detection
from samba_futbot.video import require_cv2


class ColorGoalsTest(unittest.TestCase):
    def test_detect_colored_goals_finds_blue_and_yellow_regions(self):
        cv2 = require_cv2()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video = tmp_path / "goals.mp4"
            out = tmp_path / "goals.jsonl"
            writer = cv2.VideoWriter(
                str(video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                5,
                (120, 80),
            )
            frame = np.zeros((80, 120, 3), dtype=np.uint8)
            frame[10:50, 8:24] = (255, 0, 0)  # blue in BGR
            frame[20:60, 90:112] = (0, 255, 255)  # yellow in BGR
            writer.write(frame)
            writer.release()

            detections = detect_colored_goals(
                video,
                out,
                min_area=80,
                max_area=10_000,
                min_extent=0.2,
            )

        classes = {det.class_name for det in detections}
        self.assertIn("goal_blue", classes)
        self.assertIn("goal_yellow", classes)

    def test_adaptive_goal_profile_learns_color_from_seed_detection(self):
        cv2 = require_cv2()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video = tmp_path / "adaptive_goal.mp4"
            out = tmp_path / "adaptive_goal.jsonl"
            writer = cv2.VideoWriter(
                str(video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                5,
                (120, 80),
            )
            frame = np.zeros((80, 120, 3), dtype=np.uint8)
            shifted_yellow = cv2.cvtColor(
                np.array([[[50, 210, 210]]], dtype=np.uint8),
                cv2.COLOR_HSV2BGR,
            )[0, 0]
            frame[20:60, 76:110] = shifted_yellow
            writer.write(frame)
            writer.release()
            seed = [
                Detection(
                    0,
                    "goal_yellow",
                    0.8,
                    (72.0, 16.0, 114.0, 64.0),
                    prompt="yellow box",
                )
            ]

            profiles = adapt_goal_color_profiles_from_detections(
                video,
                seed,
                min_pixels=50,
            )
            detections = detect_colored_goals(
                video,
                out,
                seed_detections=seed,
                adaptive_color=True,
                adaptive_min_pixels=50,
                min_area=80,
                max_area=10_000,
                min_extent=0.2,
            )

        self.assertGreaterEqual(profiles["goal_yellow"]["hsv_upper"][0], 50)
        self.assertIn("goal_yellow", {det.class_name for det in detections})
        self.assertTrue(any(det.extra.get("adaptive_color") for det in detections))

    def test_adaptive_goal_profile_can_gate_detections_near_seed_box(self):
        cv2 = require_cv2()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video = tmp_path / "gated_goal.mp4"
            out = tmp_path / "gated_goal.jsonl"
            writer = cv2.VideoWriter(
                str(video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                5,
                (160, 80),
            )
            frame = np.zeros((80, 160, 3), dtype=np.uint8)
            frame[20:60, 12:44] = (0, 255, 255)
            frame[20:60, 116:148] = (0, 255, 255)
            writer.write(frame)
            writer.release()
            seed = [
                Detection(
                    0,
                    "goal_yellow",
                    0.8,
                    (10.0, 18.0, 46.0, 62.0),
                    prompt="yellow box",
                )
            ]

            detections = detect_colored_goals(
                video,
                out,
                seed_detections=seed,
                adaptive_color=True,
                adaptive_min_pixels=50,
                seed_spatial_margin_px=8,
                min_area=80,
                max_area=10_000,
                min_extent=0.2,
                max_per_frame_per_class=4,
            )

        yellow_boxes = [det.box for det in detections if det.class_name == "goal_yellow"]
        self.assertEqual(len(yellow_boxes), 1)
        self.assertLess(yellow_boxes[0][0], 60)

    def test_color_goals_can_require_seed_for_class(self):
        cv2 = require_cv2()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video = tmp_path / "require_seed.mp4"
            out = tmp_path / "require_seed.jsonl"
            writer = cv2.VideoWriter(
                str(video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                5,
                (160, 80),
            )
            frame = np.zeros((80, 160, 3), dtype=np.uint8)
            frame[20:60, 12:44] = (0, 255, 255)
            frame[20:60, 116:148] = (255, 0, 0)
            writer.write(frame)
            writer.release()
            seed = [
                Detection(
                    0,
                    "goal_yellow",
                    0.8,
                    (10.0, 18.0, 46.0, 62.0),
                    prompt="yellow box",
                )
            ]

            detections = detect_colored_goals(
                video,
                out,
                seed_detections=seed,
                adaptive_color=True,
                adaptive_min_pixels=50,
                require_seed_for_color=True,
                min_area=80,
                max_area=10_000,
                min_extent=0.2,
                max_per_frame_per_class=4,
            )

        classes = {det.class_name for det in detections}
        self.assertIn("goal_yellow", classes)
        self.assertNotIn("goal_blue", classes)

    def test_goal_constraints_keep_one_goal_per_color_on_field(self):
        detections = [
            Detection(0, "field", 0.9, (0, 0, 100, 100)),
            Detection(0, "goal_yellow", 0.4, (10, 10, 30, 30), area=100),
            Detection(0, "goal_yellow", 0.8, (40, 10, 60, 30), area=100),
            Detection(0, "goal_blue", 0.9, (120, 10, 140, 30), area=100),
            Detection(0, "robots", 0.7, (20, 50, 30, 70)),
        ]

        constrained = enforce_goal_frame_constraints(
            detections,
            field_detections=detections,
            require_field_overlap=True,
            max_per_frame_per_class=1,
        )

        goals = [det for det in constrained if det.class_name.startswith("goal_")]
        self.assertEqual([(det.class_name, det.score) for det in goals], [("goal_yellow", 0.8)])
        self.assertIn("robots", {det.class_name for det in constrained})

    def test_goal_constraints_infer_missing_opposite_goal_from_field_geometry(self):
        detections = [
            Detection(0, "field", 0.9, (0, 0, 160, 80)),
            Detection(0, "goal_yellow", 0.8, (10, 20, 30, 60), area=800),
        ]

        constrained = enforce_goal_frame_constraints(
            detections,
            field_detections=detections,
            require_field_overlap=True,
            infer_missing_opposite=True,
            inferred_goal_score=0.25,
        )

        goals = {det.class_name: det for det in constrained if det.class_name.startswith("goal_")}
        self.assertIn("goal_yellow", goals)
        self.assertIn("goal_blue", goals)
        self.assertEqual(goals["goal_blue"].box, (130, 20, 150, 60))
        self.assertEqual(goals["goal_blue"].score, 0.25)
        self.assertEqual(goals["goal_blue"].extra["source"], "goal_geometry")
        self.assertEqual(goals["goal_blue"].extra["inference_axis"], "horizontal")

    def test_goal_constraints_infer_vertical_opposite_goal_for_portrait_field(self):
        detections = [
            Detection(0, "field", 0.9, (0, 0, 80, 160)),
            Detection(0, "goal_yellow", 0.8, (20, 10, 60, 30), area=800),
        ]

        constrained = enforce_goal_frame_constraints(
            detections,
            field_detections=detections,
            infer_missing_opposite=True,
        )

        goals = {det.class_name: det for det in constrained if det.class_name.startswith("goal_")}
        self.assertEqual(goals["goal_blue"].box, (20, 130, 60, 150))
        self.assertEqual(goals["goal_blue"].extra["inference_axis"], "vertical")
        self.assertEqual(goals["goal_blue"].extra["evidence"], "geometry_only")

    def test_goal_constraints_prefer_end_line_over_larger_sideline_region(self):
        detections = [
            Detection(0, "field", 0.9, (0, 100, 1000, 1800), area=1_700_000),
            Detection(0, "goal_blue", 0.95, (850, 300, 990, 1150), area=60_000),
            Detection(0, "goal_blue", 0.85, (280, 1640, 720, 1860), area=50_000),
            Detection(0, "goal_blue", 0.46, (480, 1780, 500, 1800), area=400),
        ]

        constrained = enforce_goal_frame_constraints(
            detections,
            field_detections=detections,
            require_field_overlap=True,
            min_boundary_support=0.15,
        )

        blue = [det for det in constrained if det.class_name == "goal_blue"]
        self.assertEqual(len(blue), 1)
        self.assertEqual(blue[0].box, (280, 1640, 720, 1860))

    def test_goal_box_stabilization_smooths_jitter_and_replaces_jump(self):
        detections = [
            Detection(0, "goal_blue", 0.9, (20, 130, 60, 150)),
            Detection(1, "goal_blue", 0.9, (24, 126, 64, 154)),
            Detection(2, "goal_blue", 0.9, (120, 30, 150, 120)),
        ]

        stabilized = stabilize_goal_detections(
            detections,
            ema_alpha=0.25,
            max_center_jump_px=30,
        )

        self.assertEqual(stabilized[1].box, (21.0, 129.0, 61.0, 151.0))
        self.assertEqual(stabilized[2].box, stabilized[1].box)
        self.assertTrue(stabilized[2].extra["temporal_outlier_replaced"])

    def test_geometry_only_seed_does_not_block_observed_blue_goal(self):
        cv2 = require_cv2()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video = tmp_path / "portrait_goals.mp4"
            out = tmp_path / "portrait_goals.jsonl"
            writer = cv2.VideoWriter(
                str(video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                5,
                (160, 160),
            )
            frame = np.zeros((160, 160, 3), dtype=np.uint8)
            frame[130:152, 48:112] = (255, 0, 0)
            writer.write(frame)
            writer.release()
            seeds = [
                Detection(0, "field", 0.9, (0, 0, 160, 160)),
                Detection(
                    0,
                    "goal_blue",
                    0.2,
                    (110, 8, 150, 28),
                    prompt="geometry_inferred_opposite_goal",
                    extra={"source": "goal_geometry", "evidence": "geometry_only"},
                ),
            ]

            detections = detect_colored_goals(
                video,
                out,
                seed_detections=seeds,
                adaptive_color=True,
                spatial_gate_from_seeds=True,
                seed_spatial_margin_px=5,
                require_field_overlap=True,
                min_area=100,
                max_area=10_000,
                min_extent=0.2,
            )

        blue = [det for det in detections if det.class_name == "goal_blue"]
        self.assertEqual(len(blue), 1)
        self.assertGreater(blue[0].box[1], 120)
        self.assertEqual(blue[0].extra["source"], "color_goals")

    def test_goal_constraints_do_not_infer_opposite_goal_without_field(self):
        detections = [Detection(0, "goal_yellow", 0.8, (10, 20, 30, 60), area=800)]

        constrained = enforce_goal_frame_constraints(
            detections,
            field_detections=[],
            infer_missing_opposite=True,
        )

        self.assertEqual([det.class_name for det in constrained], ["goal_yellow"])

    def test_detect_colored_goals_can_require_field_overlap(self):
        cv2 = require_cv2()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video = tmp_path / "field_overlap.mp4"
            out = tmp_path / "field_overlap.jsonl"
            writer = cv2.VideoWriter(
                str(video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                5,
                (160, 80),
            )
            frame = np.zeros((80, 160, 3), dtype=np.uint8)
            frame[20:60, 12:44] = (0, 255, 255)
            frame[20:60, 116:148] = (0, 255, 255)
            writer.write(frame)
            writer.release()
            seeds = [
                Detection(0, "field", 0.9, (0, 0, 80, 80)),
                Detection(0, "goal_yellow", 0.8, (10, 18, 46, 62), prompt="yellow board"),
            ]

            detections = detect_colored_goals(
                video,
                out,
                seed_detections=seeds,
                adaptive_color=True,
                adaptive_min_pixels=50,
                require_seed_for_color=True,
                require_field_overlap=True,
                field_margin_px=0,
                min_area=80,
                max_area=10_000,
                min_extent=0.2,
                max_per_frame_per_class=1,
            )

        yellow_boxes = [det.box for det in detections if det.class_name == "goal_yellow"]
        self.assertEqual(len(yellow_boxes), 1)
        self.assertLess(yellow_boxes[0][0], 80)


if __name__ == "__main__":
    unittest.main()
