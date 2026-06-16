# Final Clip Selection

Last updated: 2026-06-16

This shortlist is based on the current June 8 top-camera batch. It separates
defensible claims from claims that still need evidence.

## Recommended Clips

| Priority | Variant | Main claim | Use |
|---:|---|---|---|
| 1 | `IMG_9933_f000000_10s-top-fusion-hsv-v3-minarea` | Ball tracking | Clean narrative example with stable ball coverage. |
| 2 | `IMG_9933_f017990_10s-top-fusion-hsv-v3-minarea` | Ball tracking | Second stable tracking example; useful as backup or contrast. |
| 3 | `IMG_9938_f001799_10s-top-fusion-hsv-v2-refined` | Shot pressure | Best current analysis/freeze candidate for trajectory and pressure heuristics. |
| 4 | `IMG_9938_f001799_10s-top-fusion-hsv-v1` | Shot pressure backup | Keep as backup if the refined variant has visual issues. |

## Current June 16 Review Outputs

Use these as the most recent rendered videos for visual review. They use
robot-color recovery, semitransparent masks/boxes, larger labels, longer
analysis freezes and label-collision avoidance.

```text
outputs/review/2026-06-16/final_top_camera_robot_recovery_batch_v4/videos/IMG_9933_f000000_10s-top-final-robot-recovery-v4-narrative-demo.mp4
outputs/review/2026-06-16/final_top_camera_robot_recovery_batch_v4/videos/IMG_9933_f000000_10s-top-final-robot-recovery-v4-analysis-demo.mp4
outputs/review/2026-06-16/final_top_camera_robot_recovery_batch_v4/videos/IMG_9933_f017990_10s-top-final-robot-recovery-v4-narrative-demo.mp4
outputs/review/2026-06-16/final_top_camera_robot_recovery_batch_v4/videos/IMG_9933_f017990_10s-top-final-robot-recovery-v4-analysis-demo.mp4
outputs/review/2026-06-16/final_top_camera_robot_recovery_v2/videos/IMG_9938_f001799_10s-top-final-robot-recovery-v2-analysis-label-v4-demo.mp4
outputs/review/2026-06-16/final_top_camera_robot_recovery_v2/videos/IMG_9938_f001799_10s-top-final-robot-recovery-v2-narrative-label-v4-demo.mp4
```

## Current Visual-Fix Checks

These short June 16 checks were rendered locally from existing tracks after the
object-color and duplicate-robot fixes. They are not final statistical runs, but
they are useful for fast visual QA before launching full-video processing.

```text
outputs/review/2026-06-16/visual_fix_check/videos/IMG_9938-visual-fix-filtered-narrative.mp4
outputs/review/2026-06-16/visual_fix_check/videos/IMG_9938-visual-fix-filtered-analysis.mp4
outputs/review/2026-06-16/visual_fix_check/IMG_9938-visual-fix-filtered-narrative-frame90.jpg
outputs/review/2026-06-16/visual_fix_check/IMG_9938-visual-fix-filtered-analysis-frame90.jpg
```

QA status:

| Clip | QA | Ball coverage | Best use |
|---|---:|---:|---|
| `IMG_9933_f000000_10s` | `good` | 100.0% | Primary ball-tracking showcase. |
| `IMG_9933_f017990_10s` | `good` | 100.0% | Secondary ball-tracking showcase. |
| `IMG_9938_f001799_10s` | `review` | 68.3% | Shot-pressure/freezing demo only; do not claim robust tracking. |

## Artifact Paths

```text
outputs/review/2026-06-08/top_camera_batch/videos/IMG_9933_f000000_10s-top-fusion-hsv-v3-minarea-narrative-demo.mp4
outputs/review/2026-06-08/top_camera_batch/videos/IMG_9933_f017990_10s-top-fusion-hsv-v3-minarea-narrative-demo.mp4
outputs/review/2026-06-08/top_camera_batch/videos/IMG_9938_f001799_10s-top-fusion-hsv-v2-refined-narrative-demo.mp4
outputs/review/2026-06-08/top_camera_batch/videos/IMG_9938_f001799_10s-top-fusion-hsv-v2-refined-analysis-freeze-demo.mp4
outputs/review/2026-06-15/visual_refresh/videos/IMG_9938_f001799_10s-v2-refined-readable-analysis-mask-hold-freeze-demo.mp4
outputs/review/2026-06-15/robot_recovery/IMG_9938_f001799_10s-v2-refined-dark-robot-recovery-clean-v2-analysis-demo.mp4
outputs/review/2026-06-08/SUBMISSION_EVIDENCE.md
```

The June 15 refreshed analysis video uses larger labels, semitransparent
mask/box overlays and longer freeze frames. It should be reviewed before using
the older June 8 analysis render in the final presentation.

The robot-recovery preview adds an HSV/shape fallback for dark top-camera
robots and is useful for the late frames where SAM3 dropped robot boxes. Treat
it as a review candidate until false positives are checked visually.

## Claims To Defend Now

- Ball tracking on the two `IMG_9933` v3-minarea clips.
- Shot-pressure and trajectory heuristic on `IMG_9938_f001799_10s-top-fusion-hsv-v2-refined`.
- Dataset preparation, QA gates and SAM3-compatible adaptation path.
- Robot-mask adaptation improvement from the official-SAM3 head adaptation experiment.

## Claims To Keep In Review

- Team possession: current team assignment can collapse toward one color on hard
  clips, so only promote it after `team-quality` passes on the chosen final
  video.
- Goal scoring: no current June 8 top-camera showcase clip has a validated
  `goal_confirmed` event. Treat `goal_candidate` as visual evidence only.
- Learned ball segmentation: current adaptation experiments do not yet improve
  small-ball quality enough. The next step is human-reviewed masks and verified
  absence labels from more than one original recording.

## Avoid For Final Showcase

- `IMG_9938_f001799_10s-top-fusion-hsv-v3-minarea` as the primary analytical
  clip: it has weaker ball, field and robot coverage than the refined variant.
- `top-context-*` variants as final examples unless a manual visual check finds
  a specific advantage.
