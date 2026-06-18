import tempfile
import unittest
from pathlib import Path

import numpy as np

from samba_futbot.sam3_adapter import (
    _configure_predictor_object_limit,
    _detections_from_processed,
)


class Sam3AdapterTest(unittest.TestCase):
    def test_configures_object_limit_on_legacy_official_predictor(self):
        class Model:
            max_num_objects = 10000
            num_obj_for_compile = 16
            world_size = 1

        class Predictor:
            model = Model()

        predictor = Predictor()
        _configure_predictor_object_limit(predictor, 12)

        self.assertEqual(predictor.model.max_num_objects, 12)
        self.assertEqual(predictor.model.num_obj_for_compile, 12)

    def test_sam3_xywh_probs_are_scaled_to_pixel_xyxy(self):
        masks = np.zeros((1, 100, 200), dtype=bool)
        masks[20:60, 20:80] = True
        with tempfile.TemporaryDirectory() as tmp:
            detections = _detections_from_processed(
                processed={
                    "out_boxes_xywh": np.asarray([[0.1, 0.2, 0.3, 0.4]]),
                    "out_probs": np.asarray([0.9]),
                    "out_binary_masks": masks,
                    "out_obj_ids": np.asarray([7]),
                },
                frame_index=3,
                class_name="robots",
                prompt="robot",
                out_dir=Path(tmp),
                threshold=0.5,
            )

        self.assertEqual(len(detections), 1)
        self.assertTrue(
            np.allclose(detections[0].box, (20.0, 20.0, 80.0, 60.0))
        )
        self.assertEqual(detections[0].object_id, 7)


if __name__ == "__main__":
    unittest.main()
