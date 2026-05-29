import tempfile
import unittest
from pathlib import Path

import numpy as np

from samba_futbot.team import assign_robot_teams_from_video, nearest_palette_team
from samba_futbot.types import Detection
from samba_futbot.video import require_cv2


class TeamTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
