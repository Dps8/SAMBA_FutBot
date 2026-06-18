import tempfile
import unittest
from pathlib import Path

import numpy as np

from samba_futbot.types import Detection
from samba_futbot.visualize import (
    _blend_box,
    _blend_mask,
    _boxes_overlap,
    _choose_label_origin,
    _distance_label,
    _event_label,
    _freeze_event_candidates,
    _freeze_event_for_frame,
    _freeze_frame_count,
    _freeze_overlay_summary,
    _frame_header,
    _held_detection_conflicts,
    _load_detection_mask,
    _recent_event,
    _should_draw_detection,
    _should_overlay_detection,
    _visual_track_key,
    class_color,
    robot_ball_distances,
    shot_probability,
)


class VisualizeTest(unittest.TestCase):
    def test_recent_event_is_held_for_demo_header(self):
        events_by_frame = {
            10: [{"frame_index": 10, "event_type": "shot", "metadata": {"shooting_team": "blue"}}]
        }

        event = _recent_event(events_by_frame, 20, hold_frames=15)

        self.assertEqual(event["event_type"], "shot")
        self.assertEqual(_event_label(event), "shot blue")

    def test_frame_header_combines_possession_and_event(self):
        owner = Detection(20, "robots", 0.9, (0, 0, 10, 10), track_id=7, team="yellow")
        event = {
            "frame_index": 20,
            "event_type": "goal_candidate",
            "metadata": {"scoring_team": "yellow"},
        }

        header = _frame_header(20, owner, event)

        self.assertIn("possession: robot #7", header)
        self.assertIn("event: goal candidate yellow", header)

        team_header = _frame_header(20, owner, event, show_team_labels=True)

        self.assertIn("possession: yellow #7", team_header)

    def test_confirmed_goal_has_distinct_narrative_label(self):
        event = {
            "frame_index": 20,
            "event_type": "goal_confirmed",
            "metadata": {"scoring_team": "blue"},
        }

        self.assertEqual(_event_label(event), "confirmed goal blue")

    def test_rejected_goal_reports_reason(self):
        event = {
            "frame_index": 20,
            "event_type": "goal_rejected",
            "metadata": {"rejection_reason": "geometry_only_goal"},
        }

        self.assertEqual(_event_label(event), "goal rejected geometry_only_goal")

    def test_analysis_header_uses_analysis_label(self):
        header = _frame_header(3, None, style="analysis")

        self.assertIn("SAMBA FutBot: analysis", header)

    def test_narrative_header_includes_nearest_ball_distance(self):
        nearest = {"track_id": 4, "team": "blue", "distance_px": 37.8}

        header = _frame_header(3, None, nearest_distance=nearest, style="narrative")

        self.assertIn("nearest ball: robot #4 38px", header)
        self.assertEqual(_distance_label(nearest), "blue #4 38px")
        self.assertEqual(_distance_label(nearest, show_team=False), "robot #4 38px")

    def test_robot_ball_distances_rank_nearest_robots(self):
        frame = [
            Detection(0, "ball", 1.0, (10, 10, 14, 14), track_id=1),
            Detection(0, "robots", 1.0, (0, 0, 10, 10), track_id=2, team="blue"),
            Detection(0, "robots", 1.0, (90, 90, 110, 110), track_id=3, team="yellow"),
        ]

        distances = robot_ball_distances(frame)

        self.assertEqual([item["track_id"] for item in distances], [2, 3])
        self.assertEqual(distances[0]["team"], "blue")
        self.assertLess(distances[0]["distance_px"], distances[1]["distance_px"])

    def test_class_color_prioritizes_object_over_team(self):
        self.assertEqual(class_color("ball", team="blue"), (255, 130, 20))
        self.assertEqual(class_color("goal_yellow", team="blue"), (255, 220, 50))
        self.assertEqual(class_color("goal_blue", team="yellow"), (40, 90, 255))
        self.assertEqual(class_color("robots", team="blue", track_id=1), (255, 80, 80))
        self.assertEqual(class_color("robots", team="blue", track_id=2), (245, 245, 245))

    def test_inferred_geometry_goal_is_hidden_in_render(self):
        inferred = Detection(
            0,
            "goal_blue",
            0.28,
            (0, 0, 20, 20),
            extra={"source": "goal_geometry"},
        )
        measured = Detection(
            0,
            "goal_blue",
            0.8,
            (0, 0, 20, 20),
            extra={"source": "color_goal"},
        )

        self.assertFalse(_should_draw_detection(inferred, style="analysis"))
        self.assertFalse(_should_overlay_detection(inferred))
        self.assertTrue(_should_draw_detection(measured, style="analysis"))

    def test_shot_probability_reports_direction_speed_and_probability(self):
        previous = Detection(0, "ball", 1.0, (40, 10, 44, 14), track_id=1)
        current = Detection(1, "ball", 1.0, (80, 10, 84, 14), track_id=1)

        pressure = shot_probability(current, previous, frame_width=100)

        self.assertEqual(pressure["target_side"], "right")
        self.assertGreater(pressure["speed_px_frame"], 0)
        self.assertGreater(pressure["probability"], 0.5)

    def test_freeze_frame_count_uses_fps_and_seconds(self):
        self.assertEqual(_freeze_frame_count(30, 1.5), 45)
        self.assertEqual(_freeze_frame_count(30, 0), 0)

    def test_freeze_candidates_filter_and_rank_events(self):
        events = [
            {"event_type": "collision", "confidence": 0.9, "metadata": {}},
            {"event_type": "shot", "confidence": 0.3, "metadata": {}},
            {"event_type": "goal_candidate", "confidence": 0.7, "metadata": {}},
            {"event_type": "goal_confirmed", "confidence": 0.8, "metadata": {}},
            {
                "event_type": "shot_pressure",
                "confidence": 0.6,
                "metadata": {"priority": 120},
            },
        ]

        candidates = _freeze_event_candidates(events, min_confidence=0.45)

        self.assertEqual(
            [event["event_type"] for event in candidates],
            ["goal_confirmed", "shot_pressure", "goal_candidate", "collision"],
        )

    def test_freeze_is_only_active_for_analysis_style(self):
        events_by_frame = {12: [{"event_type": "shot", "confidence": 0.9}]}

        narrative = _freeze_event_for_frame(
            events_by_frame,
            12,
            style="narrative",
            analysis_freeze=True,
            event_types={"shot"},
            min_confidence=0.45,
        )
        analysis = _freeze_event_for_frame(
            events_by_frame,
            12,
            style="analysis",
            analysis_freeze=True,
            event_types={"shot"},
            min_confidence=0.45,
        )

        self.assertIsNone(narrative)
        self.assertEqual(analysis["event_type"], "shot")

    def test_freeze_overlay_summary_includes_probability_and_nearest(self):
        event = {
            "event_type": "shot_pressure",
            "description": "Balon avanza hacia porteria.",
            "metadata": {
                "goal_probability": 0.64,
                "target_side": "right",
                "ball_speed_px_frame": 19.2,
            },
        }
        distances = [{"track_id": 7, "team": "blue", "distance_px": 31.5}]

        lines = _freeze_overlay_summary(event, distances)

        self.assertIn("probability 64%", lines)
        self.assertIn("target right", lines)
        self.assertIn("nearest robot #7 32px", lines)
        self.assertNotIn("Balon avanza hacia porteria.", lines)

        team_lines = _freeze_overlay_summary(event, distances, show_team_labels=True)

        self.assertIn("nearest blue #7 32px", team_lines)

    def test_load_detection_mask_reads_npz_by_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            masks = np.zeros((2, 4, 6), dtype=np.uint8)
            masks[1, 1:3, 2:5] = 1
            np.savez_compressed(root / "masks.npz", masks=masks)
            det = Detection(
                0,
                "robots",
                0.9,
                (2, 1, 5, 3),
                mask_path="masks.npz",
                extra={"mask_index": 1},
            )

            mask = _load_detection_mask(det, mask_base_dir=root, frame_shape=(4, 6, 3))

            self.assertIsNotNone(mask)
            self.assertEqual(int(mask.sum()), 6)

    def test_blend_mask_and_box_change_only_expected_region(self):
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        mask = np.zeros((4, 6), dtype=np.uint8)
        mask[1:3, 2:5] = 1

        _blend_mask(frame, mask, (0, 0, 100), alpha=0.5)

        self.assertEqual(frame[1, 2, 2], 50)
        self.assertEqual(frame[0, 0, 2], 0)

        _blend_box(frame, (0, 0, 2, 2), (100, 0, 0), alpha=0.5)

        self.assertEqual(frame[0, 0, 0], 50)

    def test_visual_track_key_only_tracks_overlay_classes(self):
        robot = Detection(0, "robots", 0.9, (0, 0, 10, 10), track_id=4)
        field = Detection(0, "field", 0.9, (0, 0, 10, 10), track_id=2)
        untracked_ball = Detection(0, "ball", 0.9, (0, 0, 10, 10))

        self.assertEqual(_visual_track_key(robot), ("robots", 4))
        self.assertIsNone(_visual_track_key(field))
        self.assertIsNone(_visual_track_key(untracked_ball))

    def test_held_detection_conflicts_with_nearby_current_robot(self):
        held = Detection(4, "robots", 0.7, (100, 100, 180, 180), track_id=4)
        current = Detection(5, "robots", 0.8, (135, 120, 220, 205), track_id=9)
        far = Detection(5, "robots", 0.8, (400, 400, 480, 480), track_id=10)
        ball = Detection(5, "ball", 0.9, (135, 120, 150, 135), track_id=2)
        current_near_ball = Detection(5, "robots", 0.8, (155, 130, 225, 205), track_id=11)

        self.assertTrue(_held_detection_conflicts(held, [current]))
        self.assertFalse(_held_detection_conflicts(held, [far]))
        self.assertFalse(_held_detection_conflicts(held, [ball]))
        self.assertTrue(_held_detection_conflicts(held, [ball, current_near_ball]))

    def test_label_origin_avoids_occupied_box_when_possible(self):
        occupied = [(7, 0, 65, 25)]

        x, y, rect = _choose_label_origin(
            (120, 160, 3),
            (10, 20),
            text_w=50,
            text_h=12,
            baseline=4,
            anchor_box=(10, 28, 45, 60),
            occupied_boxes=occupied,
        )

        self.assertFalse(_boxes_overlap(rect, occupied[0]))
        self.assertGreaterEqual(y, 60)
        self.assertGreaterEqual(x, 0)

    def test_boxes_overlap_respects_separated_boxes(self):
        self.assertTrue(_boxes_overlap((0, 0, 20, 20), (10, 10, 30, 30)))
        self.assertFalse(_boxes_overlap((0, 0, 20, 20), (30, 30, 50, 50)))


if __name__ == "__main__":
    unittest.main()
