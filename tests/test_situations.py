import unittest

from samba_futbot.situations import analyze_situations
from samba_futbot.types import Detection


def det(frame, class_name, box, *, track_id=None, team=None, score=1.0):
    return Detection(frame, class_name, score, box, track_id=track_id, team=team)


class SituationsTest(unittest.TestCase):
    def test_reports_controlled_disputed_and_free_possession(self):
        result = analyze_situations(
            _sample_detections(),
            possession_radius_px=18,
            dispute_margin_px=8,
        )

        states = [frame["possession"]["state"] for frame in result["frames"]]

        self.assertEqual(states, ["controlled", "disputed", "free"])
        self.assertEqual(result["frames"][0]["possession"]["owner_track_id"], 1)
        self.assertEqual(result["frames"][0]["possession"]["owner_team"], "blue")
        contenders = result["frames"][1]["possession"]["contenders"]
        self.assertEqual(
            [contender["track_id"] for contender in contenders],
            [3, 4],
        )
        self.assertIsNone(result["frames"][2]["possession"]["owner_track_id"])

    def test_robot_ball_distances_are_ranked(self):
        result = analyze_situations(
            _sample_detections(),
            possession_radius_px=18,
            dispute_margin_px=8,
        )

        ranked_ids = [record["track_id"] for record in result["frames"][0]["robot_ball_distances"]]
        ranked_distances = [
            record["distance_px"] for record in result["frames"][0]["robot_ball_distances"]
        ]

        self.assertEqual(ranked_ids, [1, 2, 5])
        self.assertEqual(ranked_distances, sorted(ranked_distances))

    def test_action_probabilities_and_loss_risk_are_normalized(self):
        result = analyze_situations(
            _sample_detections(),
            possession_radius_px=18,
            dispute_margin_px=8,
        )

        for frame in result["frames"]:
            self.assertGreaterEqual(frame["loss_risk"], 0.0)
            self.assertLessEqual(frame["loss_risk"], 1.0)
            for probability in frame["action_probabilities"].values():
                self.assertGreaterEqual(probability, 0.0)
                self.assertLessEqual(probability, 1.0)

    def test_summary_counts_frames_by_possession_state(self):
        result = analyze_situations(
            _sample_detections(),
            possession_radius_px=18,
            dispute_margin_px=8,
        )

        summary = result["summary"]

        self.assertEqual(summary["total_frames"], 3)
        self.assertEqual(summary["frames_with_ball"], 3)
        self.assertEqual(summary["possession_states"]["controlled"]["frames"], 1)
        self.assertEqual(summary["possession_states"]["disputed"]["frames"], 1)
        self.assertEqual(summary["possession_states"]["free"]["frames"], 1)
        self.assertAlmostEqual(summary["possession_states"]["controlled"]["ratio"], 1 / 3)
        self.assertEqual(summary["controlled_by_team"], {"blue": 1})


def _sample_detections():
    return [
        det(0, "field", (0, 0, 300, 180), track_id=100),
        det(0, "ball", (48, 48, 52, 52), track_id=10),
        det(0, "robots", (50, 45, 60, 55), track_id=1, team="blue"),
        det(0, "robots", (84, 45, 94, 55), track_id=2, team="yellow"),
        det(0, "robots", (125, 45, 135, 55), track_id=5, team="blue"),
        det(1, "field", (0, 0, 300, 180), track_id=101),
        det(1, "ball", (148, 48, 152, 52), track_id=10),
        det(1, "robots", (135, 45, 145, 55), track_id=3, team="blue"),
        det(1, "robots", (155, 45, 165, 55), track_id=4, team="yellow"),
        det(2, "field", (0, 0, 300, 180), track_id=102),
        det(2, "ball", (248, 48, 252, 52), track_id=10),
        det(2, "robots", (190, 45, 200, 55), track_id=6, team="blue"),
        det(2, "robots", (210, 45, 220, 55), track_id=7, team="yellow"),
    ]


if __name__ == "__main__":
    unittest.main()
