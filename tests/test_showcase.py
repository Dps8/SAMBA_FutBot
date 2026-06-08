import tempfile
import unittest
from pathlib import Path

from samba_futbot.io_utils import read_json, write_json
from samba_futbot.showcase import (
    collect_showcase_candidates,
    ready_claims,
    showcase_record,
    write_showcase_json,
    write_showcase_markdown,
)


class ShowcaseTest(unittest.TestCase):
    def test_ready_claims_returns_sorted_ready_names(self):
        claims = ready_claims(
            {
                "team_possession": {"status": "ready"},
                "ball_tracking": {"status": "ready"},
                "goal_scoring": {"status": "review"},
            }
        )

        self.assertEqual(claims, ["ball_tracking", "team_possession"])

    def test_showcase_record_marks_missing_required_claims(self):
        record = showcase_record(
            {
                "path": "clip-qa.json",
                "status": "good",
                "quality_score": 90,
                "summary": {"ball_in_play_coverage_ratio": 0.8},
                "claim_readiness": {"ball_tracking": {"status": "ready"}},
            },
            required_claims=["ball_tracking", "team_possession"],
        )

        self.assertFalse(record["showcase_ready"])
        self.assertEqual(record["missing_required_claims"], ["team_possession"])
        self.assertEqual(record["ready_claims"], 1)

    def test_collect_showcase_candidates_ranks_by_claims_status_and_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "a-qa.json",
                _qa_report(
                    status="good",
                    score=70,
                    claims=["ball_tracking", "team_possession"],
                    ball=0.9,
                ),
            )
            write_json(
                root / "b-qa.json",
                _qa_report(
                    status="review",
                    score=95,
                    claims=["ball_tracking", "team_possession", "shot_pressure"],
                    ball=0.8,
                ),
            )
            write_json(
                root / "c-qa.json",
                _qa_report(status="good", score=100, claims=["ball_tracking"], ball=0.95),
            )

            candidates = collect_showcase_candidates(root, limit=2)
            out_json = root / "showcase.json"
            out_md = root / "showcase.md"
            write_showcase_json(out_json, candidates)
            write_showcase_markdown(out_md, candidates)
            saved = read_json(out_json)
            markdown = out_md.read_text(encoding="utf-8")

        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["path"], "b-qa.json")
        self.assertTrue(candidates[0]["showcase_ready"])
        self.assertEqual(saved["schema"], "samba_futbot.showcase.v1")
        self.assertIn("Showcase Candidates", markdown)
        self.assertIn("shot_pressure", markdown)


def _qa_report(*, status: str, score: int, claims: list[str], ball: float) -> dict:
    return {
        "status": status,
        "quality_score": score,
        "summary": {
            "ball_in_play_coverage_ratio": ball,
            "max_ball_speed_px_frame": 10.0,
            "unknown_team_ratio": 0.1,
            "field_path_samples": 20,
        },
        "claim_readiness": {
            claim: {"status": "ready", "reason": "ok"} for claim in claims
        },
        "issues": [] if status == "good" else [{"severity": "warning"}],
    }


if __name__ == "__main__":
    unittest.main()
