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
                },
            )
            write_json(events, [{"event_type": "shot"}, {"event_type": "shot"}])
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
                    "robot_summary": {"penalty_area_samples": 2},
                },
            )

            out = write_run_report(
                report,
                title="Clip QA",
                metrics_path=metrics,
                events_path=events,
                field_analysis_path=field,
            )
            text = out.read_text(encoding="utf-8")

        self.assertIn("# Clip QA", text)
        self.assertIn("Ball in-play coverage", text)
        self.assertIn("Goal-zone entries", text)


if __name__ == "__main__":
    unittest.main()
