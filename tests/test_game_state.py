import unittest

from samba_futbot.game_state import (
    classify_frame_states,
    detect_external_events,
    detect_game_segments,
    play_mask_from_segments,
)
from samba_futbot.types import Detection


class GameStateTest(unittest.TestCase):
    def test_dead_ball_when_ball_outside_field_without_robot(self):
        detections = [
            Detection(0, "field", 1.0, (0, 0, 100, 60)),
            Detection(0, "ball", 1.0, (150, 150, 154, 154), track_id=9),
        ]

        states = classify_frame_states(detections)

        self.assertEqual(states[0].state, "dead_ball")
        self.assertIn("ball_out_of_play", states[0].reasons)

    def test_human_intervention_when_person_overlaps_field(self):
        detections = [
            Detection(0, "field", 1.0, (0, 0, 100, 60)),
            Detection(0, "person", 1.0, (10, 10, 30, 50)),
            Detection(0, "ball", 1.0, (40, 20, 44, 24), track_id=9),
        ]

        states = classify_frame_states(detections)
        events = detect_external_events(states)

        self.assertEqual(states[0].state, "human_intervention")
        self.assertTrue(states[0].human_intervention)
        self.assertEqual(events[0].event_type, "human_intervention")

    def test_robot_removed_after_track_disappears(self):
        detections = [
            Detection(0, "robots", 1.0, (10, 10, 30, 30), track_id=7),
            Detection(0, "ball", 1.0, (15, 15, 19, 19), track_id=9),
            Detection(1, "ball", 1.0, (16, 15, 20, 19), track_id=9),
            Detection(2, "ball", 1.0, (17, 15, 21, 19), track_id=9),
        ]

        states = classify_frame_states(detections, robot_removed_after_frames=2)
        events = detect_external_events(states)

        self.assertIn(7, states[-1].robot_removed)
        self.assertIn("robot_removed", {event.event_type for event in events})

    def test_robot_disabled_when_stationary_for_window(self):
        detections = []
        for frame in range(4):
            detections.append(Detection(frame, "robots", 1.0, (10, 10, 30, 30), track_id=7))
            detections.append(Detection(frame, "ball", 1.0, (15 + frame, 15, 19 + frame, 19), track_id=9))

        states = classify_frame_states(
            detections,
            robot_disabled_after_frames=3,
            stationary_threshold_px=1.0,
        )
        events = detect_external_events(states)

        self.assertIn(7, states[-1].robot_disabled)
        self.assertIn("robot_disabled", {event.event_type for event in events})

    def test_segments_and_play_mask_keep_only_in_play_frames(self):
        detections = [
            Detection(0, "field", 1.0, (0, 0, 100, 60)),
            Detection(0, "ball", 1.0, (20, 20, 24, 24), track_id=9),
            Detection(1, "field", 1.0, (0, 0, 100, 60)),
            Detection(1, "ball", 1.0, (150, 150, 154, 154), track_id=9),
        ]

        states = classify_frame_states(detections)
        segments = detect_game_segments(states)
        play_mask = play_mask_from_segments(segments)

        self.assertEqual([segment.state for segment in segments], ["in_play", "dead_ball"])
        self.assertEqual(play_mask, {0})


if __name__ == "__main__":
    unittest.main()
