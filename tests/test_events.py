import unittest

from samba_futbot.events import (
    confirm_goal_candidates,
    detect_events,
    estimate_possession,
    summarize_events,
)
from samba_futbot.types import Detection


class EventsTest(unittest.TestCase):
    def test_possession_and_pass(self):
        detections = [
            Detection(0, "ball", 1.0, (10, 10, 14, 14), track_id=10),
            Detection(0, "robots", 1.0, (0, 0, 20, 20), track_id=1, team="allied"),
            Detection(1, "ball", 1.0, (100, 10, 104, 14), track_id=10),
            Detection(1, "robots", 1.0, (90, 0, 110, 20), track_id=2, team="allied"),
        ]
        possession = estimate_possession(detections, possession_radius_px=30)
        self.assertEqual(possession[0].track_id, 1)
        self.assertEqual(possession[1].track_id, 2)
        events = detect_events(detections, possession_radius_px=30)
        self.assertEqual(events[0].event_type, "pass")

    def test_collision(self):
        detections = [
            Detection(0, "robots", 1.0, (0, 0, 20, 20), track_id=1),
            Detection(0, "robots", 1.0, (10, 0, 30, 20), track_id=2),
        ]
        events = detect_events(detections, collision_radius_px=25)
        self.assertTrue(any(event.event_type == "collision" for event in events))

    def test_off_field_ball_does_not_trigger_shot(self):
        detections = [
            Detection(0, "ball", 1.0, (5, 200, 9, 204), track_id=10),
            Detection(1, "ball", 1.0, (40, 200, 44, 204), track_id=10),
        ]

        events = detect_events(detections, frame_width=50, goal_x_margin_ratio=1.0)

        self.assertFalse(any(event.event_type == "shot" for event in events))

    def test_shot_requires_motion_toward_goal_side(self):
        toward_goal = [
            Detection(0, "field", 1.0, (0, 0, 100, 50)),
            Detection(1, "field", 1.0, (0, 0, 100, 50)),
            Detection(0, "ball", 1.0, (82, 10, 88, 16), track_id=10),
            Detection(1, "ball", 1.0, (93, 10, 99, 16), track_id=10),
        ]
        away_from_goal = [
            Detection(0, "field", 1.0, (0, 0, 100, 50)),
            Detection(1, "field", 1.0, (0, 0, 100, 50)),
            Detection(0, "ball", 1.0, (93, 10, 99, 16), track_id=10),
            Detection(1, "ball", 1.0, (82, 10, 88, 16), track_id=10),
        ]

        shots = [
            event
            for event in detect_events(toward_goal, frame_width=100, goal_x_margin_ratio=0.1)
            if event.event_type == "shot"
        ]
        away_shots = [
            event
            for event in detect_events(away_from_goal, frame_width=100, goal_x_margin_ratio=0.1)
            if event.event_type == "shot"
        ]

        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0].metadata["target_side"], "right")
        self.assertEqual(shots[0].metadata["shooting_team"], "yellow")
        self.assertEqual(away_shots, [])

    def test_goal_candidate_reports_scoring_team(self):
        detections = [
            Detection(0, "field", 1.0, (0, 0, 100, 100), track_id=20),
            Detection(0, "ball", 1.0, (12, 12, 18, 18), track_id=10),
            Detection(0, "goal_blue", 1.0, (0, 0, 30, 30), track_id=30),
        ]

        events = detect_events(detections, possession_radius_px=30)
        goals = [event for event in events if event.event_type == "goal_candidate"]

        self.assertEqual(len(goals), 1)
        self.assertEqual(goals[0].metadata["goal_side"], "blue")
        self.assertEqual(goals[0].metadata["scoring_team"], "yellow")

    def test_temporal_goal_entry_confirms_candidate(self):
        detections = [
            Detection(frame, "field", 1.0, (0, 0, 100, 100))
            for frame in range(3)
        ]
        detections.extend(
            [
                Detection(0, "goal_blue", 0.9, (0, 20, 25, 60), track_id=30),
                Detection(1, "goal_blue", 0.9, (0, 20, 25, 60), track_id=30),
                Detection(2, "goal_blue", 0.9, (0, 20, 25, 60), track_id=30),
                Detection(0, "ball", 0.9, (40, 35, 46, 41), track_id=10),
                Detection(1, "ball", 0.9, (14, 35, 20, 41), track_id=10),
                Detection(2, "ball", 0.9, (10, 35, 16, 41), track_id=10),
            ]
        )

        candidates = detect_events(detections)
        events = confirm_goal_candidates(detections, candidates)
        confirmed = [event for event in events if event.event_type == "goal_confirmed"]

        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0].metadata["scoring_team"], "yellow")
        self.assertEqual(confirmed[0].metadata["inside_frames"], [1, 2])

    def test_inferred_goal_never_confirms_candidate(self):
        detections = [
            Detection(frame, "field", 1.0, (0, 0, 100, 100))
            for frame in range(3)
        ]
        detections.extend(
            [
                Detection(
                    frame,
                    "goal_blue",
                    0.9,
                    (0, 20, 25, 60),
                    track_id=30,
                    extra={"source": "goal_geometry"},
                )
                for frame in range(3)
            ]
        )
        detections.extend(
            [
                Detection(0, "ball", 0.9, (40, 35, 46, 41), track_id=10),
                Detection(1, "ball", 0.9, (14, 35, 20, 41), track_id=10),
                Detection(2, "ball", 0.9, (10, 35, 16, 41), track_id=10),
            ]
        )

        events = confirm_goal_candidates(detections, detect_events(detections))

        self.assertFalse(any(event.event_type == "goal_confirmed" for event in events))

    def test_non_consecutive_goal_presence_does_not_confirm(self):
        detections = [
            Detection(frame, "field", 1.0, (0, 0, 100, 100))
            for frame in range(4)
        ]
        detections.extend(
            [
                Detection(frame, "goal_blue", 0.9, (0, 20, 25, 60), track_id=30)
                for frame in range(4)
            ]
        )
        detections.extend(
            [
                Detection(0, "ball", 0.9, (40, 35, 46, 41), track_id=10),
                Detection(1, "ball", 0.9, (14, 35, 20, 41), track_id=10),
                Detection(2, "ball", 0.9, (40, 35, 46, 41), track_id=10),
                Detection(3, "ball", 0.9, (10, 35, 16, 41), track_id=10),
            ]
        )

        events = confirm_goal_candidates(
            detections,
            detect_events(detections),
            min_inside_frames=2,
        )

        self.assertFalse(any(event.event_type == "goal_confirmed" for event in events))

    def test_goal_confirmation_is_idempotent(self):
        detections = [
            Detection(frame, "field", 1.0, (0, 0, 100, 100))
            for frame in range(3)
        ]
        detections.extend(
            [
                Detection(frame, "goal_blue", 0.9, (0, 20, 25, 60), track_id=30)
                for frame in range(3)
            ]
        )
        detections.extend(
            [
                Detection(0, "ball", 0.9, (40, 35, 46, 41), track_id=10),
                Detection(1, "ball", 0.9, (14, 35, 20, 41), track_id=10),
                Detection(2, "ball", 0.9, (10, 35, 16, 41), track_id=10),
            ]
        )

        once = confirm_goal_candidates(detections, detect_events(detections))
        twice = confirm_goal_candidates(detections, once)

        self.assertEqual(
            sum(event.event_type == "goal_confirmed" for event in once),
            1,
        )
        self.assertEqual(
            sum(event.event_type == "goal_confirmed" for event in twice),
            1,
        )

    def test_summarize_events_reports_candidate_scoreboard(self):
        events = [
            {
                "frame_index": 10,
                "event_type": "goal_candidate",
                "description": "Balon entra",
                "confidence": 0.6,
                "metadata": {"goal_side": "blue", "scoring_team": "yellow"},
            },
            {
                "frame_index": 15,
                "event_type": "pass",
                "description": "Pase",
                "confidence": 0.65,
                "metadata": {},
            },
            {
                "frame_index": 20,
                "event_type": "interception",
                "description": "Intercepcion",
                "confidence": 0.7,
                "metadata": {},
            },
            {
                "frame_index": 25,
                "event_type": "shot",
                "description": "Tiro",
                "confidence": 0.5,
                "metadata": {"shooting_team": "blue", "target_side": "left"},
            },
        ]

        summary = summarize_events(events)

        self.assertEqual(summary["scoreboard"]["yellow"], 1)
        self.assertEqual(summary["scoreboard"]["blue"], 0)
        self.assertEqual(summary["confirmed_scoreboard"], {"blue": 0, "yellow": 0})
        self.assertEqual(summary["goals"]["by_goal_side"]["blue"], 1)
        self.assertEqual(summary["shots"]["by_team"]["blue"], 1)
        self.assertEqual(summary["shots"]["by_target_side"]["left"], 1)
        self.assertEqual(summary["possession_changes"]["passes"], 1)
        self.assertEqual(summary["possession_changes"]["interceptions"], 1)


if __name__ == "__main__":
    unittest.main()
