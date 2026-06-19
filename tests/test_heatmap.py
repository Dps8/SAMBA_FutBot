import tempfile
import unittest
from pathlib import Path

import numpy as np

from samba_futbot.heatmap import render_activity_heatmap
from samba_futbot.types import Detection
from samba_futbot.video import require_cv2


class HeatmapTest(unittest.TestCase):
    def test_renders_dynamic_video_and_accumulated_image(self):
        cv2 = require_cv2()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "source.mp4"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (96, 64)
            )
            for _ in range(5):
                writer.write(np.full((64, 96, 3), (28, 100, 35), dtype=np.uint8))
            writer.release()
            detections = [
                Detection(frame, "robots", 0.9, (10 + frame * 5, 20, 30 + frame * 5, 45))
                for frame in range(5)
            ]
            out_video = root / "heatmap.mp4"
            out_image = root / "heatmap.png"

            report = render_activity_heatmap(
                video,
                detections,
                out_video,
                out_image,
                radius_px=6,
            )

            self.assertEqual(report["frames"], 5)
            self.assertEqual(report["samples"], 5)
            self.assertGreater(out_video.stat().st_size, 0)
            self.assertGreater(out_image.stat().st_size, 0)

    def test_rejects_invalid_decay(self):
        with self.assertRaisesRegex(ValueError, "decay"):
            render_activity_heatmap("missing.mp4", [], "out.mp4", "out.png", decay=0)


if __name__ == "__main__":
    unittest.main()
