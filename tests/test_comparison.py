import tempfile
import unittest
from pathlib import Path

from samba_futbot.comparison import (
    compare_qa_files,
    compare_qa_reports,
    write_qa_comparison_markdown,
)
from samba_futbot.io_utils import write_json


class QaComparisonTest(unittest.TestCase):
    def test_comparison_reports_improvements_and_claims(self):
        baseline = _report(score=80, ball=0.7, jump=30, claims=["ball_tracking"])
        candidate = _report(
            score=90,
            ball=0.9,
            jump=20,
            claims=["ball_tracking", "shot_pressure"],
        )

        comparison = compare_qa_reports(baseline, candidate)

        self.assertEqual(comparison["verdict"], "improved")
        self.assertEqual(comparison["metrics"]["ball_coverage"]["status"], "improved")
        self.assertEqual(comparison["metrics"]["max_ball_jump_px_frame"]["status"], "improved")
        self.assertEqual(comparison["claims"]["gained"], ["shot_pressure"])

    def test_lost_claim_and_lower_score_is_regression(self):
        baseline = _report(
            score=90,
            ball=0.9,
            jump=20,
            claims=["ball_tracking", "team_possession"],
        )
        candidate = _report(score=80, ball=0.8, jump=25, claims=["ball_tracking"])

        comparison = compare_qa_reports(baseline, candidate)

        self.assertEqual(comparison["verdict"], "regressed")
        self.assertEqual(comparison["claims"]["lost"], ["team_possession"])

    def test_compare_files_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            write_json(baseline, _report(score=80, ball=0.7, jump=30, claims=[]))
            write_json(candidate, _report(score=80, ball=0.8, jump=30, claims=[]))

            comparison = compare_qa_files(baseline, candidate)
            out = write_qa_comparison_markdown(root / "comparison.md", comparison)
            text = out.read_text(encoding="utf-8")

        self.assertEqual(comparison["inputs"]["baseline"], str(baseline))
        self.assertIn("# QA Comparison", text)
        self.assertIn("ball_coverage", text)


def _report(*, score: int, ball: float, jump: float, claims: list[str]) -> dict:
    return {
        "status": "good" if score == 100 else "review",
        "quality_score": score,
        "summary": {
            "ball_in_play_coverage_ratio": ball,
            "field_coverage_ratio": 1.0,
            "robot_coverage_ratio": 0.8,
            "possession_coverage_ratio": 0.4,
            "unknown_team_ratio": 0.1,
            "max_ball_speed_px_frame": jump,
            "ball_out_of_bounds_ratio": 0.0,
            "field_path_samples": 20,
        },
        "claim_readiness": {
            claim: {"status": "ready", "reason": "ok"} for claim in claims
        },
    }


if __name__ == "__main__":
    unittest.main()
