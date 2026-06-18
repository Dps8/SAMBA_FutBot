import unittest
from importlib.util import find_spec

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

    @unittest.skipUnless(find_spec("supervision"), "supervision is not installed")
    def test_bytetrack_keeps_ids_unique_across_classes(self):
        detections = [
            Detection(frame_index=0, class_name="robots", score=0.9, box=(0, 0, 20, 20)),
            Detection(frame_index=0, class_name="ball", score=0.9, box=(40, 40, 48, 48)),
            Detection(frame_index=1, class_name="robots", score=0.9, box=(3, 0, 23, 20)),
            Detection(frame_index=1, class_name="ball", score=0.9, box=(42, 40, 50, 48)),
        ]

        tracked = track_detections(detections, backend="bytetrack", max_age=5)

        self.assertEqual(tracked[0].track_id, tracked[2].track_id)
        self.assertEqual(tracked[1].track_id, tracked[3].track_id)
        self.assertNotEqual(tracked[0].track_id, tracked[1].track_id)


if __name__ == "__main__":
    unittest.main()
