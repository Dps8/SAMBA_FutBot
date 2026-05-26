import unittest

from samba_futbot.tracking import iou, track_detections
from samba_futbot.types import Detection


class TrackingTest(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 5, 15, 15)), 25 / 175)
        self.assertEqual(iou((0, 0, 1, 1), (2, 2, 3, 3)), 0.0)

    def test_iou_tracker_keeps_identity(self):
        detections = [
            Detection(frame_index=0, class_name="robots", score=0.9, box=(0, 0, 10, 10)),
            Detection(frame_index=1, class_name="robots", score=0.9, box=(1, 1, 11, 11)),
            Detection(frame_index=2, class_name="ball", score=0.9, box=(50, 50, 55, 55)),
        ]
        tracked = track_detections(detections, iou_threshold=0.2, max_age=3)
        self.assertEqual(tracked[0].track_id, tracked[1].track_id)
        self.assertNotEqual(tracked[0].track_id, tracked[2].track_id)


if __name__ == "__main__":
    unittest.main()
