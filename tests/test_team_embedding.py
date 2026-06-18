import unittest

import numpy as np

from samba_futbot.team_embedding import (
    align_clusters_to_teams,
    assign_embedding_teams,
    cluster_track_embeddings,
    embedding_team_report,
)
from samba_futbot.types import Detection


class TeamEmbeddingTest(unittest.TestCase):
    def test_clusters_two_structural_groups(self):
        clusters = cluster_track_embeddings(
            {
                1: np.asarray([1.0, 0.0, 0.0]),
                2: np.asarray([0.95, 0.05, 0.0]),
                3: np.asarray([0.0, 1.0, 0.0]),
                4: np.asarray([0.05, 0.95, 0.0]),
            }
        )

        self.assertEqual(clusters[1], clusters[2])
        self.assertEqual(clusters[3], clusters[4])
        self.assertNotEqual(clusters[1], clusters[3])

    def test_aligns_clusters_with_existing_color_evidence(self):
        mapping, metadata = align_clusters_to_teams(
            {1: 0, 2: 0, 3: 1, 4: 1},
            {1: "yellow", 2: "yellow", 3: "blue"},
        )

        self.assertEqual(mapping, {0: "yellow", 1: "blue"})
        self.assertFalse(metadata["mapping_ambiguous"])

    def test_assignment_only_changes_robot_tracks_with_embeddings(self):
        detections = [
            Detection(0, "robots", 0.9, (0, 0, 10, 10), track_id=1, team="unknown"),
            Detection(0, "ball", 0.9, (20, 20, 25, 25), track_id=2),
        ]

        assigned = assign_embedding_teams(detections, {1: 0}, {0: "blue", 1: "yellow"})

        self.assertEqual(assigned[0].team, "blue")
        self.assertIsNone(assigned[1].team)

    def test_report_records_auditable_mapping(self):
        report = embedding_team_report(
            {1: np.asarray([1.0, 0.0]), 2: np.asarray([0.0, 1.0])},
            {1: 3, 2: 4},
            {1: 0, 2: 1},
            {0: "blue", 1: "yellow"},
            {"mapping_scores": [2, 0], "mapping_ambiguous": False},
            model_id="facebook/dinov2-small",
        )

        self.assertEqual(report["tracks_embedded"], 2)
        self.assertEqual(report["embedding_dimensions"], 2)
        self.assertEqual(report["cluster_to_team"], {"0": "blue", "1": "yellow"})


if __name__ == "__main__":
    unittest.main()
