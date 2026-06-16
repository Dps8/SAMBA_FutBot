from __future__ import annotations

import unittest

from samba_futbot.finetune_evaluation import (
    COCO_METRICS,
    compare_coco_evaluations,
    subset_coco_ground_truth,
)


class FinetuneEvaluationTests(unittest.TestCase):
    def test_subset_coco_ground_truth_keeps_only_selected_images(self) -> None:
        ground_truth = {
            "info": {"name": "test"},
            "images": [{"id": 1}, {"id": 2}, {"id": 3}],
            "annotations": [
                {"id": 10, "image_id": 1},
                {"id": 20, "image_id": 2},
                {"id": 30, "image_id": 3},
            ],
            "categories": [{"id": 1, "name": "ball"}],
        }

        subset = subset_coco_ground_truth(ground_truth, [3, 1])

        self.assertEqual([image["id"] for image in subset["images"]], [1, 3])
        self.assertEqual(
            [annotation["id"] for annotation in subset["annotations"]],
            [10, 30],
        )
        self.assertEqual(subset["info"], {"name": "test"})

    def test_subset_rejects_unknown_prediction_image(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing from ground truth"):
            subset_coco_ground_truth(
                {"images": [{"id": 1}], "annotations": [], "categories": []},
                [2],
            )

    def test_compare_coco_evaluations_reports_relative_ap_gain(self) -> None:
        baseline_metrics = {metric: 0.1 for metric in COCO_METRICS}
        candidate_metrics = {metric: 0.15 for metric in COCO_METRICS}
        baseline = {
            "image_ids": [1, 2],
            "overall": baseline_metrics,
            "categories": {"ball": {"metrics": baseline_metrics}},
        }
        candidate = {
            "image_ids": [1, 2],
            "overall": candidate_metrics,
            "categories": {"ball": {"metrics": candidate_metrics}},
        }

        comparison = compare_coco_evaluations(baseline, candidate)

        self.assertEqual(comparison["verdict"], "improved")
        self.assertAlmostEqual(comparison["overall"]["AP"]["delta"], 0.05)
        self.assertAlmostEqual(
            comparison["categories"]["ball"]["AP"]["relative_change"],
            0.5,
        )

    def test_compare_requires_matching_image_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "same image IDs"):
            compare_coco_evaluations(
                {"image_ids": [1], "overall": {}, "categories": {}},
                {"image_ids": [2], "overall": {}, "categories": {}},
            )


if __name__ == "__main__":
    unittest.main()
