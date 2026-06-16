import tempfile
import unittest
from pathlib import Path

from samba_futbot.io_utils import write_json
from samba_futbot.submission import write_submission_report


class SubmissionReportTest(unittest.TestCase):
    def test_write_submission_report_combines_batch_training_and_video_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch = root / "batch"
            training = root / "training"
            write_json(
                batch / "showcase-index.json",
                {
                    "runs": [
                        {
                            "path": "qa/clip-a-qa.json",
                            "status": "good",
                            "quality_score": 100,
                            "ball_coverage": 0.95,
                            "ready_claim_names": ["ball_tracking"],
                        }
                    ]
                },
            )
            write_json(batch / "qa-index.json", {"runs": [{"status": "good"}]})
            write_json(
                batch / "VIDEO_RENDER_SUMMARY.json",
                [
                    {
                        "stem": "clip-a",
                        "narrative": "videos/clip-a-narrative.mp4",
                        "analysis": "videos/clip-a-analysis.mp4",
                    }
                ],
            )
            write_json(
                batch / "BATCH_SUMMARY.json",
                [
                    {
                        "stem": "clip-a",
                        "frames": 30,
                        "detections": 60,
                        "ballCoverage": 0.95,
                        "possessionCoverage": 0.2,
                    }
                ],
            )
            write_json(
                batch / "situations" / "clip-a-situations.json",
                {
                    "summary": {
                        "frames_with_ball": 28,
                        "possession_states": {
                            "controlled": {"ratio": 0.25},
                            "disputed": {"ratio": 0.1},
                            "free": {"ratio": 0.65},
                        },
                        "average_loss_risk": 0.2,
                        "average_action_probabilities": {
                            "pass": 0.4,
                            "shot": 0.3,
                            "hold": 0.3,
                        },
                    }
                },
            )
            write_json(training / "TRAINING_DATASET_SUMMARY.json", [{"dataset": "clip-a"}])
            write_json(
                training / "merged_top_camera_balanced_manifest.json",
                {
                    "summary": {
                        "frames": 80,
                        "detections": 120,
                        "detections_by_class": {"ball": 30, "robots": 90},
                        "frames_by_split": {"train": 60, "val": 20},
                    }
                },
            )
            write_json(
                training / "merged_top_camera_balanced_quality.json",
                {"summary": {"invalid_boxes": 0, "low_scores": 2, "review_candidates": 2}},
            )

            out = write_submission_report(
                root / "report.md",
                batch_root=batch,
                training_root=training,
                title="Submission",
            )
            text = out.read_text(encoding="utf-8")

        self.assertIn("# Submission", text)
        self.assertIn("clip-a-narrative.mp4", text)
        self.assertIn("ball_tracking", text)
        self.assertIn("Tactical Situation Layer", text)
        self.assertIn("25.0%", text)
        self.assertIn("0.40", text)
        self.assertIn("Merged frames", text)
        self.assertIn("Dataset QA low-score detections", text)
        self.assertIn("train: 60, val: 20", text)
        self.assertIn("SAM3/SAM 3.1", text)


if __name__ == "__main__":
    unittest.main()
