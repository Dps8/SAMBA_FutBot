import tempfile
import unittest
from pathlib import Path

from samba_futbot.color_ball import detect_orange_ball
from samba_futbot.color_ball import filter_robot_color_blobs
from samba_futbot.types import Detection
from samba_futbot.video import require_cv2


class ColorBallTest(unittest.TestCase):
    def test_detect_orange_ball_from_synthetic_video(self):
        cv2 = require_cv2()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video = tmp_path / "orange.mp4"
            writer = cv2.VideoWriter(
                str(video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,
                (120, 80),
            )
            for _ in range(3):
                frame = cv2.UMat(80, 120, cv2.CV_8UC3).get()
                frame[:] = (0, 120, 0)
                cv2.circle(frame, (60, 40), 8, (0, 90, 255), -1)
                writer.write(frame)
            writer.release()

            detections = detect_orange_ball(video, tmp_path / "detections.jsonl")

        self.assertEqual(len(detections), 3)
        self.assertTrue(all(det.class_name == "ball" for det in detections))

    def test_filter_robot_color_blobs_removes_blob_inside_robot(self):
        balls = [
            Detection(0, "ball", 0.9, (10, 10, 20, 20), prompt="hsv"),
            Detection(0, "ball", 0.9, (50, 50, 60, 60), prompt="hsv"),
        ]
        robots = [Detection(0, "robots", 0.9, (5, 5, 25, 25))]

        filtered = filter_robot_color_blobs(balls, robots)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].box, (50, 50, 60, 60))


if __name__ == "__main__":
    unittest.main()
