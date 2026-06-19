import tempfile
import unittest
from pathlib import Path

import numpy as np

from samba_futbot.team import (
    assign_robot_teams_from_video,
    marker_ratio,
    nearest_palette_team,
    palette_team_vote,
)
from samba_futbot.types import Detection
from samba_futbot.video import require_cv2


class TeamTest(unittest.TestCase):
    def test_marker_ratio_separates_green_crop(self):
        green = np.zeros((20, 20, 3), dtype=np.uint8)
        green[:] = (50, 160, 70)
        dark = np.zeros((20, 20, 3), dtype=np.uint8)

        self.assertGreater(marker_ratio(green), 0.95)
        self.assertEqual(marker_ratio(dark), 0.0)

    def test_nearest_palette_team(self):
        team, distance = nearest_palette_team(
            (40, 100, 230),
            {"blue": (55, 115, 220), "yellow": (230, 210, 60)},
        )

        self.assertEqual(team, "blue")
        self.assertLess(distance, 40)

    def test_assign_robot_teams_from_video_uses_track_vote(self):
        cv2 = require_cv2()
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "teams.mp4"
            writer = cv2.VideoWriter(
                str(video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                5,
                (80, 60),
            )
            for _ in range(2):
                frame = np.zeros((60, 80, 3), dtype=np.uint8)
                frame[10:40, 10:40] = (230, 110, 50)  # BGR -> RGB near blue
                writer.write(frame)
            writer.release()

            detections = [
                Detection(0, "robots", 1.0, (10, 10, 40, 40), track_id=7),
                Detection(1, "robots", 1.0, (10, 10, 40, 40), track_id=7),
            ]
            assigned = assign_robot_teams_from_video(
                video,
                detections,
                palette={"blue": (50, 110, 230), "yellow": (230, 210, 60)},
            )

        self.assertEqual({det.team for det in assigned}, {"blue"})

    def test_palette_team_vote_ignores_field_background(self):
        crop = np.zeros((40, 40, 3), dtype=np.uint8)
        crop[:, :] = (15, 150, 40)
        crop[12:28, 12:28] = (55, 115, 220)

        team, distance = palette_team_vote(
            crop,
            {"blue": (55, 115, 220), "yellow": (230, 210, 60)},
            max_color_distance=90,
            min_saturation=40,
            min_value=30,
            min_pixels=20,
        )

        self.assertEqual(team, "blue")
        self.assertLess(distance, 1)


if __name__ == "__main__":
    unittest.main()
