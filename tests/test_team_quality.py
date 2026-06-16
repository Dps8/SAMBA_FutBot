import json
import tempfile
import unittest
from pathlib import Path

from samba_futbot.io_utils import write_detections
from samba_futbot.team_quality import (
    analyze_team_quality,
    analyze_team_quality_file,
    analyze_team_quality_jsonl,
    write_team_quality_markdown,
)
from samba_futbot.types import Detection


class TeamQualityTest(unittest.TestCase):
    def test_summarizes_robot_samples_unknown_ratio_and_tracks_by_team(self):
        report = analyze_team_quality(
            [
                _robot(0, 1, "Blue"),
                _robot(1, 1, "blue"),
                _robot(0, 2, "yellow"),
                _robot(1, 2, None),
                Detection(0, "ball", 0.9, (0, 0, 2, 2), team=None),
            ]
        )

        summary = report["summary"]
        self.assertEqual(summary["robot_samples"], 4)
        self.assertEqual(summary["assigned_samples"], 3)
        self.assertEqual(summary["unknown_samples"], 1)
        self.assertAlmostEqual(summary["unknown_ratio"], 0.25)
        self.assertEqual(summary["tracks"], 2)
        self.assertEqual(report["by_team"]["blue"], {"samples": 2, "tracks": 1})
        self.assertEqual(report["by_team"]["yellow"], {"samples": 1, "tracks": 1})
        self.assertEqual(report["by_team"]["unknown"], {"samples": 1, "tracks": 0})

    def test_reports_temporal_team_changes_and_ambiguous_tracks(self):
        detections = [
            _robot(0, 7, "blue"),
            _robot(1, 7, "unknown"),
            _robot(2, 7, "yellow"),
            _robot(3, 7, "blue"),
        ]

        report = analyze_team_quality(
            detections,
            ambiguous_track_dominance_threshold=0.8,
            min_ambiguous_track_samples=3,
        )

        track = report["tracks"][0]
        self.assertEqual(track["team_changes"], 2)
        self.assertEqual(
            track["change_events"],
            [
                {
                    "from_frame": 0,
                    "to_frame": 2,
                    "from_team": "blue",
                    "to_team": "yellow",
                },
                {
                    "from_frame": 2,
                    "to_frame": 3,
                    "from_team": "yellow",
                    "to_team": "blue",
                },
            ],
        )
        self.assertTrue(track["ambiguous"])
        self.assertEqual(track["resolved_team"], "unknown")
        self.assertEqual(report["ambiguous_track_ids"], [7])
        self.assertEqual(report["summary"]["team_changes"], 2)
        self.assertEqual(
            [candidate["reason"] for candidate in report["review_candidates"][:2]],
            ["team_change", "ambiguous_track"],
        )

    def test_reports_team_coverage_for_each_robot_frame(self):
        report = analyze_team_quality(
            [
                _robot(5, 1, "blue"),
                _robot(5, 2, None),
                _robot(6, 1, "blue"),
                _robot(6, 2, "yellow"),
            ],
            min_frame_team_coverage=0.75,
        )

        coverage = report["frame_coverage"]
        self.assertAlmostEqual(coverage["mean_ratio"], 0.75)
        self.assertAlmostEqual(coverage["overall_ratio"], 0.75)
        self.assertEqual(
            [frame["coverage_ratio"] for frame in coverage["frames"]],
            [0.5, 1.0],
        )
        self.assertEqual(report["summary"]["frames_below_team_coverage"], 1)
        reasons = [candidate["reason"] for candidate in report["review_candidates"]]
        self.assertIn("low_frame_team_coverage", reasons)
        self.assertIn("unknown_team", reasons)

    def test_accepts_generators_limits_candidates_and_is_json_serializable(self):
        detections = (_robot(frame, None, None) for frame in range(3))

        report = analyze_team_quality(detections, max_review_candidates=2)

        self.assertEqual(report["summary"]["robot_samples"], 3)
        self.assertEqual(len(report["review_candidates"]), 2)
        self.assertEqual(report["summary"]["review_candidates"], 2)
        json.dumps(report)

    def test_file_helpers_read_jsonl_through_io_utils(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tracks.jsonl"
            write_detections(path, [_robot(0, 4, "blue"), _robot(1, 4, "blue")])

            report = analyze_team_quality_file(path)
            alias_report = analyze_team_quality_jsonl(path)

        self.assertEqual(report["inputs"]["detections"], str(path))
        self.assertEqual(report["by_team"]["blue"]["tracks"], 1)
        self.assertEqual(alias_report, report)

    def test_validates_configurable_thresholds(self):
        with self.assertRaisesRegex(ValueError, "unknown_ratio_threshold"):
            analyze_team_quality([], unknown_ratio_threshold=1.1)
        with self.assertRaisesRegex(ValueError, "min_ambiguous_track_samples"):
            analyze_team_quality([], min_ambiguous_track_samples=0)
        with self.assertRaisesRegex(ValueError, "max_review_candidates"):
            analyze_team_quality([], max_review_candidates=-1)

    def test_markdown_report_lists_summary_and_tracks(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = analyze_team_quality(
                [_robot(0, 4, "blue"), _robot(1, 4, "yellow")]
            )
            out = write_team_quality_markdown(report, Path(tmp) / "team-quality.md")
            text = out.read_text(encoding="utf-8")

        self.assertIn("# Team Quality Report", text)
        self.assertIn("Unknown ratio", text)
        self.assertIn("| 4 |", text)

    def test_flags_collapsed_assignment_to_one_team(self):
        report = analyze_team_quality(
            [
                _robot(0, 1, "blue"),
                _robot(0, 2, "blue"),
                _robot(1, 1, "blue"),
                _robot(1, 2, "yellow"),
            ],
            max_dominant_team_ratio=0.70,
        )

        self.assertTrue(report["summary"]["team_imbalance_above_threshold"])
        self.assertAlmostEqual(report["summary"]["dominant_team_ratio"], 0.75)
        self.assertEqual(report["review_candidates"][0]["reason"], "team_imbalance")


def _robot(
    frame_index: int,
    track_id: int | None,
    team: str | None,
) -> Detection:
    return Detection(
        frame_index,
        "robots",
        0.9,
        (10, 10, 30, 30),
        track_id=track_id,
        team=team,
    )


if __name__ == "__main__":
    unittest.main()
