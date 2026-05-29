import tempfile
import unittest
from pathlib import Path

from samba_futbot.io_utils import write_json
from samba_futbot.qa import evaluate_run_quality, write_quality_markdown


class RunQaTest(unittest.TestCase):
    def test_good_run_has_no_issues(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            metrics, events, field = _write_inputs(
                tmp_path,
                ball_coverage=0.92,
                max_jump=12.0,
                out_of_bounds=0,
                path_samples=100,
            )

            report = evaluate_run_quality(
                metrics_path=metrics,
                events_path=events,
                field_analysis_path=field,
            )

        self.assertEqual(report["status"], "good")
        self.assertEqual(report["quality_score"], 100)
        self.assertEqual(report["issues"], [])

    def test_low_coverage_and_large_jump_flag_review_or_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            metrics, events, field = _write_inputs(
                tmp_path,
                ball_coverage=0.25,
                max_jump=140.0,
                out_of_bounds=30,
                path_samples=100,
            )

            report = evaluate_run_quality(
                metrics_path=metrics,
                events_path=events,
                field_analysis_path=field,
            )

        self.assertEqual(report["status"], "fail")
        codes = {issue["code"] for issue in report["issues"]}
        self.assertIn("low_ball_coverage", codes)
        self.assertIn("large_ball_jump", codes)
        self.assertIn("ball_out_of_bounds", codes)

    def test_markdown_report_lists_status_and_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "qa.md"
            report = {
                "status": "review",
                "quality_score": 80,
                "summary": {
                    "frames_observed": 10,
                    "ball_in_play_coverage_ratio": 0.5,
                    "max_ball_speed_px_frame": 5.0,
                    "field_coverage_ratio": 1.0,
                    "robot_coverage_ratio": 1.0,
                    "field_path_samples": 5,
                    "ball_out_of_bounds_ratio": 0.0,
                    "robot_penalty_area_samples": 0,
                    "event_counts": {"shot": 2},
                },
                "issues": [{"severity": "warning", "code": "low_ball_coverage", "message": "Low."}],
            }

            write_quality_markdown(out, report)
            text = out.read_text(encoding="utf-8")

        self.assertIn("Status: `review`", text)
        self.assertIn("`shot`: `2`", text)


def _write_inputs(
    tmp_path: Path,
    *,
    ball_coverage: float,
    max_jump: float,
    out_of_bounds: int,
    path_samples: int,
) -> tuple[Path, Path, Path]:
    metrics = tmp_path / "metrics.json"
    events = tmp_path / "events.json"
    field = tmp_path / "field.json"
    write_json(
        metrics,
        {
            "frames_observed": 100,
            "detections": 300,
            "tracks": 4,
            "classes": {
                "ball": {
                    "in_play_coverage_ratio": ball_coverage,
                    "frame_coverage_ratio": ball_coverage,
                    "track_fragmentation_gaps": 1,
                },
                "field": {"frame_coverage_ratio": 1.0},
                "robots": {"frame_coverage_ratio": 0.9},
            },
            "motion": {"ball": {"max_speed_px_frame": max_jump}},
        },
    )
    write_json(events, [{"event_type": "shot"}])
    write_json(
        field,
        {
            "summary": {
                "path_samples": path_samples,
                "distance_m": 1.0,
                "ball_out_of_bounds_samples": out_of_bounds,
            },
            "robot_summary": {"penalty_area_samples": 0},
        },
    )
    return metrics, events, field


if __name__ == "__main__":
    unittest.main()
