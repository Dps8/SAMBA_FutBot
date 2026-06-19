import unittest

from samba_futbot.motion_prediction import (
    predict_robot_trajectories,
    select_robot_prediction_snapshot,
)


class MotionPredictionTest(unittest.TestCase):
    def test_predicts_three_metric_branches_with_normalized_probabilities(self):
        path = [
            {
                "frame_index": frame,
                "track_id": 7,
                "team": "blue",
                "field_x_m": 0.4 + frame * 0.01,
                "field_y_m": 0.8,
            }
            for frame in range(20)
        ]

        forecasts = predict_robot_trajectories(
            path,
            fps=10,
            field_length_m=2.43,
            field_width_m=1.82,
            horizon_s=1.0,
            step_s=0.25,
        )

        self.assertEqual(len(forecasts), 1)
        forecast = forecasts[0]
        self.assertEqual(forecast["probability_model"], "heuristic_kinematic_v1")
        self.assertAlmostEqual(forecast["state"]["speed_m_s"], 0.1, places=6)
        self.assertEqual(
            [branch["mode"] for branch in forecast["trajectories"]],
            ["continue", "turn_left", "turn_right"],
        )
        self.assertAlmostEqual(
            sum(branch["probability"] for branch in forecast["trajectories"]),
            1.0,
            places=8,
        )
        self.assertGreater(
            forecast["trajectories"][0]["probability"],
            forecast["trajectories"][1]["probability"],
        )

    def test_excludes_tracks_that_are_stale_at_reference_frame(self):
        path = [
            {
                "frame_index": frame,
                "track_id": 3,
                "team": "yellow",
                "field_x_m": 0.5,
                "field_y_m": 0.5,
            }
            for frame in range(8)
        ]

        forecasts = predict_robot_trajectories(
            path,
            fps=10,
            field_length_m=2.43,
            field_width_m=1.82,
            reference_frame=30,
            max_age_frames=5,
        )

        self.assertEqual(forecasts, [])

    def test_showcase_selects_observed_high_motion_moment(self):
        path = []
        for frame in range(40):
            step = 0.002 if frame < 20 else 0.015
            path.append(
                {
                    "frame_index": frame,
                    "track_id": 2,
                    "team": "blue",
                    "field_x_m": 0.4 + max(0, frame - 20) * step,
                    "field_y_m": 0.6,
                }
            )

        snapshot = select_robot_prediction_snapshot(
            path,
            fps=10,
            field_length_m=2.43,
            field_width_m=1.82,
            sample_stride_frames=2,
        )

        self.assertGreaterEqual(snapshot["reference_frame"], 20)
        self.assertEqual(len(snapshot["forecasts"]), 1)
        self.assertGreater(snapshot["forecasts"][0]["state"]["speed_m_s"], 0.05)


if __name__ == "__main__":
    unittest.main()
