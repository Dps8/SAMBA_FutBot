import tempfile
import unittest
from pathlib import Path

from samba_futbot.io_utils import read_json, write_detections
from samba_futbot.pseudolabels import export_pseudolabel_candidates
from samba_futbot.types import Detection


class PseudolabelsTest(unittest.TestCase):
    def test_export_pseudolabel_candidates_filters_for_high_confidence_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            detections = root / "detections.jsonl"
            out = root / "manifest.json"
            mask = root / "masks" / "robots_000001.npz"
            mask.parent.mkdir()
            mask.write_bytes(b"placeholder")
            write_detections(
                detections,
                [
                    Detection(
                        1,
                        "robots",
                        0.92,
                        (10, 20, 30, 50),
                        area=400,
                        mask_path=str(mask),
                        extra={"mask_index": 2},
                    ),
                    Detection(1, "ball", 0.30, (40, 40, 45, 45), area=20),
                    Detection(1, "field", 0.99, (0, 0, 100, 100), area=10_000),
                    Detection(2, "robots", 0.95, (10, 20, 30, 50), area=0),
                ],
            )

            manifest = export_pseudolabel_candidates(
                detections,
                out,
                classes=["robots", "ball"],
                min_score=0.60,
                min_area=10,
                require_mask=True,
                root=root,
            )

            saved = read_json(out)

        self.assertEqual(manifest["summary"]["candidates"], 1)
        self.assertEqual(saved["summary"]["candidates_by_class"], {"robots": 1})
        self.assertEqual(saved["summary"]["rejected"]["low_score"], 1)
        self.assertEqual(saved["summary"]["rejected"]["class_filter"], 1)
        self.assertEqual(saved["summary"]["rejected"]["small_area"], 1)
        self.assertEqual(saved["candidates"][0]["mask_path"], "masks/robots_000001.npz")
        self.assertEqual(saved["candidates"][0]["mask_index"], 2)


if __name__ == "__main__":
    unittest.main()
