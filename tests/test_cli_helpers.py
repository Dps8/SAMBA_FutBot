import unittest

from samba_futbot.cli import _frame_anchors, build_parser


class CliHelpersTest(unittest.TestCase):
    def test_frame_anchors_uses_regular_step(self):
        self.assertEqual(_frame_anchors(926, start=150, step=150), [150, 300, 450, 600, 750, 900])

    def test_frame_anchors_falls_back_to_zero_for_short_video(self):
        self.assertEqual(_frame_anchors(102, start=150, step=150), [0])

    def test_process_top_camera_defaults_to_current_best_variant(self):
        args = build_parser().parse_args(["process-top-camera", "--video", "clip.mp4"])

        self.assertEqual(args.suffix, "top-fusion-hsv-v3-minarea")
        self.assertEqual(args.orange_min_area, 300.0)
        self.assertEqual(args.orange_max_per_frame, 6)
        self.assertEqual(args.refine_max_jump_px, 35.0)


if __name__ == "__main__":
    unittest.main()
