import tempfile
import unittest
from pathlib import Path

from samba_futbot.io_utils import write_detections
from samba_futbot.types import Detection
from samba_futbot.windowing import (
    deduplicate_detections,
    merge_detection_files,
    offset_detections,
    parse_int_list,
)


class WindowingTest(unittest.TestCase):
    def test_parse_int_list(self):
        self.assertEqual(parse_int_list("0, 150,300"), [0, 150, 300])

    def test_deduplicate_keeps_best_overlapping_detection(self):
        detections = [
            Detection(0, "robots", 0.5, (10, 10, 20, 20), object_id=1),
            Detection(0, "robots", 0.9, (10, 10, 20, 20), object_id=2),
            Detection(0, "ball", 0.4, (10, 10, 20, 20), object_id=3),
        ]
        deduped = deduplicate_detections(detections, iou_threshold=0.9)
        self.assertEqual(len(deduped), 2)
        robot = [det for det in deduped if det.class_name == "robots"][0]
        self.assertEqual(robot.score, 0.9)

    def test_merge_detection_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first = tmp_path / "first.jsonl"
            second = tmp_path / "second.jsonl"
            out = tmp_path / "merged.jsonl"
            write_detections(first, [Detection(0, "robots", 0.5, (10, 10, 20, 20))])
            write_detections(second, [Detection(0, "robots", 0.9, (10, 10, 20, 20))])

            merged = merge_detection_files([first, second], out, iou_threshold=0.9)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].score, 0.9)

    def test_offset_detections_preserves_clip_frame(self):
        detections = [Detection(3, "ball", 0.8, (1, 2, 3, 4), extra={"source": "clip"})]

        shifted = offset_detections(detections, 150)

        self.assertEqual(shifted[0].frame_index, 153)
        self.assertEqual(shifted[0].extra["clip_frame_index"], 3)
        self.assertEqual(shifted[0].extra["source"], "clip")


if __name__ == "__main__":
    unittest.main()
