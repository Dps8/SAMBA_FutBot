import unittest

from samba_futbot.sam3_finetune_model import freeze_for_adaptation


class FakeParameter:
    def __init__(self, size):
        self.size = size
        self.requires_grad = True

    def numel(self):
        return self.size

    def requires_grad_(self, enabled):
        self.requires_grad = enabled
        return self


class FakeModel:
    def __init__(self):
        self.parameters = {
            "backbone.layer.weight": FakeParameter(100),
            "segmentation_head.pixel_decoder.weight": FakeParameter(2),
            "segmentation_head.mask_predictor.mask.weight": FakeParameter(20),
            "segmentation_head.cross_attend_prompt.weight": FakeParameter(2),
            "segmentation_head.cross_attn_norm.weight": FakeParameter(2),
            "segmentation_head.semantic_seg_head.weight": FakeParameter(3),
            "segmentation_head.instance_seg_head.weight": FakeParameter(2),
            "dot_prod_scoring.prompt.weight": FakeParameter(5),
        }

    def named_parameters(self):
        return self.parameters.items()


class Sam3FinetuneModelTest(unittest.TestCase):
    def test_freezes_backbone_and_keeps_adaptation_heads_trainable(self):
        model = FakeModel()

        summary = freeze_for_adaptation(model)

        self.assertFalse(model.parameters["backbone.layer.weight"].requires_grad)
        self.assertTrue(
            model.parameters["segmentation_head.mask_predictor.mask.weight"].requires_grad
        )
        self.assertFalse(
            model.parameters["segmentation_head.semantic_seg_head.weight"].requires_grad
        )
        self.assertTrue(model.parameters["dot_prod_scoring.prompt.weight"].requires_grad)
        self.assertEqual(
            summary,
            {
                "total_parameters": 136,
                "trainable_parameters": 33,
                "frozen_parameters": 103,
            },
        )

    def test_rejects_prefix_that_matches_nothing(self):
        with self.assertRaisesRegex(ValueError, "matched no parameters"):
            freeze_for_adaptation(FakeModel(), ("missing.",))


if __name__ == "__main__":
    unittest.main()
