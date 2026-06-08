import tempfile
import unittest
from pathlib import Path

from samba_futbot.io_utils import write_json
from samba_futbot.reporting import write_run_report


class ReportingTest(unittest.TestCase):
    def test_write_run_report_combines_metrics_events_and_field_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            metrics = tmp_path / "metrics.json"
            events = tmp_path / "events.json"
            field = tmp_path / "field.json"
            qa = tmp_path / "qa.json"
            report = tmp_path / "report.md"
            write_json(
                metrics,
                {
                    "frames_observed": 10,
                    "detections": 12,
                    "tracks": 3,
                    "classes": {"ball": {"in_play_coverage_ratio": 0.5}},
                    "motion": {
                        "ball": {
                            "mean_speed_px_second": 12.3,
                            "max_speed_px_second": 45.6,
                        }
                    },
                    "possession": {
                        "coverage_ratio": 0.6,
                        "by_team": {"blue": {"ratio": 1.0}},
                        "dominance": {
                            "team": "blue",
                            "ratio": 1.0,
                            "margin_ratio": 1.0,
                        },
                        "longest_streak": {
                            "team": "blue",
                            "track_id": 4,
                            "frames": 12,
                            "seconds": 0.4,
                        },
                    },
                },
            )
            write_json(
                events,
                [
                    {
                        "event_type": "shot",
                        "metadata": {"shooting_team": "blue", "target_side": "left"},
                    },
                    {
                        "event_type": "shot",
                        "metadata": {"shooting_team": "yellow", "target_side": "right"},
                    },
                    {
                        "frame_index": 2,
                        "event_type": "goal_candidate",
                        "metadata": {"goal_side": "blue", "scoring_team": "yellow"},
                    },
                ],
            )
            write_json(
                field,
                {
                    "calibration": {"field": {"length_m": 2.43, "width_m": 1.82}},
                    "summary": {
                        "path_samples": 8,
                        "distance_m": 1.2,
                        "mean_speed_m_s": 0.3,
                        "max_speed_m_s": 0.8,
                        "goal_zone_entries": 1,
                    },
                    "robot_summary": {
                        "penalty_area_samples": 2,
                        "samples_by_team": {"blue": 3, "yellow": 2},
                        "phase_samples_by_team": {
                            "blue": {"attacking": 2, "middle": 1},
                            "yellow": {"defensive": 2},
                        },
                        "attacking_pressure_by_team": {"blue": 0.67, "yellow": 0.0},
                    },
                    "robot_zone_control": [
                        {"zone": "r1c1", "leader": "blue", "leader_ratio": 0.75},
                        {"zone": "r1c2", "leader": "yellow", "leader_ratio": 1.0},
                    ],
                },
            )
            write_json(
                qa,
                {
                    "status": "review",
                    "quality_score": 90,
                    "summary": {
                        "ball_in_play_coverage_ratio": 0.7,
                        "max_ball_speed_px_frame": 12.0,
                        "unknown_team_ratio": 0.1,
                        "possession_coverage_ratio": 0.5,
                    },
                    "claim_readiness": {
                        "ball_tracking": {
                            "status": "ready",
                            "reason": "ball tracking evidence is stable",
                        },
                        "team_possession": {
                            "status": "ready",
                            "reason": "team possession evidence is stable",
                        },
                    },
                    "issues": [
                        {
                            "severity": "warning",
                            "code": "low_ball_coverage",
                            "message": "Ball coverage is low.",
                        }
                    ],
                },
            )

            out = write_run_report(
                report,
                title="Clip QA",
                metrics_path=metrics,
                events_path=events,
                field_analysis_path=field,
                qa_path=qa,
            )
            text = out.read_text(encoding="utf-8")

        self.assertIn("# Clip QA", text)
        self.assertIn("Ball in-play coverage", text)
        self.assertIn("Longest possession", text)
        self.assertIn("Possession dominance", text)
        self.assertIn("blue #4", text)
        self.assertIn("Candidate score", text)
        self.assertIn("blue 0 - 1 yellow", text)
        self.assertIn("Shots by team", text)
        self.assertIn("Goal-zone entries", text)
        self.assertIn("Robot samples by team", text)
        self.assertIn("Robot phases by team", text)
        self.assertIn("Attacking pressure by team", text)
        self.assertIn("Territorial control by leader", text)
        self.assertIn("Run QA", text)
        self.assertIn("Quality score", text)
        self.assertIn("Claim Readiness", text)
        self.assertIn("team_possession", text)
        self.assertIn("low_ball_coverage", text)


if __name__ == "__main__":
    unittest.main()
