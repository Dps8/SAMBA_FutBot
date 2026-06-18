# SAMBA FutBot Project Status

Last updated: 2026-06-18

## Current State

The project has a working SAM3-centered professional pipeline for top-camera
robot soccer videos. The current implementation covers:

- SAM3/SAM 3.1 video inference by windows and prompt anchors.
- Rotating prompt ensembles for long videos: every window records the exact
  prompt variant it used, while bounded 120-frame windows and explicit CUDA
  cache release keep official SAM3 viable on a 16 GB GPU.
- Hybrid ball recovery using SAM3 prompts plus configurable HSV/shape cues.
- Blue/yellow goal prompts, adaptive color fallback and optional geometric
  opposite-goal inference. The default is now conservative and does not infer
  the missing opposite goal automatically.
- Tracking, team assignment, possession metrics and event candidates.
- Optional class-isolated ByteTrack integration through Supervision, with the
  dependency-free IoU tracker retained as a reproducible fallback.
- Conservative temporal goal validation that separates `goal_candidate` from
  `goal_confirmed` and `goal_rejected`, records rejection reasons and never
  confirms geometry-only inferred goals. Event cooldown deduplication prevents
  repeated nearby candidates from inflating match counts.
- Game-state filtering for `in_play`, `dead_ball`, human intervention, removed
  robots and disabled robots. The top-camera context prompts now include
  configurable human/referee/hand classes so SAM3 can feed that state layer.
- Metric trajectory and speed when homography calibration is available.
- Narrative and analysis render modes, including analysis freeze frames.
- Semitransparent mask/box overlays, larger labels and short visual track hold
  for presentation videos.
- Object-priority render colors: orange ball, red/white robots and blue/yellow
  goals. Geometry-only inferred goals are hidden in demos to avoid false visual
  claims.
- Top-camera robot duplicate filtering using score, size, containment, IoU,
  center distance and max-per-frame gates.
- HSV/shape robot recovery for top-camera dark robots as a review-only fallback
  when SAM3 drops robots in crowded or late frames.
- Tactical situation analysis with robot-ball distances, controlled/disputed/free
  possession and heuristic pass/shot/hold probabilities.
- QA reports, QA indexes and showcase candidate selection.
- Final submission evidence report generation from processed batch artifacts.
- SAM-compatible dataset preparation through pseudo-label manifests, frame/crop
  exports, merged multi-video manifests, COCO exports, quality auditing and
  source-frame deduplication before adaptation.
- Mask-preserving frame manifests and COCO RLE segmentation export with
  per-mask failure auditing.
- Deterministic human-holdout selection that never copies pseudo labels into
  the annotation template.
- Team-assignment quality gates for missing evidence, temporal instability and
  collapse toward one color.
- Structural calibration checks for corner order, convexity, field coverage and
  extreme perspective skew.
- A working official-SAM3 adaptation path with portable COCO-RLE data,
  parameter-efficient segmentation-head training, checkpoints and exact-subset
  baseline comparison.

The repository intentionally avoids non-permitted detector-specific training
paths. The training preparation is kept neutral/SAM-compatible through
manifests, COCO boxes and RLE masks.

Current local verification: 256 `unittest` tests pass on Windows with
`PYTHONPATH=src`; the ByteTrack-only test is skipped when the optional
Supervision package is absent. The same test passes in the remote GPU venv with
Supervision 0.27.

## Generated Evidence

The strongest local review artifacts are currently under:

- `outputs/review/2026-06-08/top_camera_batch`
- `outputs/review/2026-06-08/training_datasets`

The current final-showcase shortlist is documented in
`docs/FINAL_CLIP_SELECTION.md`.

June 16 remote processing on `LadyGaga` generated three updated top-camera
review runs with the current no-YOLO SAM3 + color/geometry pipeline, robot
color recovery, semitransparent overlays and label-collision avoidance:

- `outputs/review/2026-06-16/final_top_camera_robot_recovery_v2`
- `outputs/review/2026-06-16/final_top_camera_robot_recovery_batch_v4`

`IMG_9933_f017990_10s` and `IMG_9933_f000000_10s` reached QA `good` with
100% ball coverage. `IMG_9938_f001799_10s` remains QA `review` because ball
coverage is 68.3%, but it is still useful as a shot-pressure/freezing visual
candidate. These outputs were copied back to the Windows workspace for review.

June 16 local visual-fix verification generated short render checks under:

- `outputs/review/2026-06-16/visual_fix_check`

The preview confirms object-priority colors, hidden geometry-only false goals
and filtered robot overlays on the existing `IMG_9938_f001799_10s` tracks.
Follow-up local work added ball-aware robot filtering, expanded HSV robot
recovery boxes and stale overlay suppression in the render. A full top-camera
and normal-view reprocessing pass is in progress on June 18 before claiming
final statistics. The first 300-frame normal-view attempt exposed a reproducible
16 GB CUDA limit; the long-video profile now uses 120-frame windows and one
rotating prompt per class.

The June 8 batch processed 14 top-camera track variants and identified four
strong showcase candidates. It also generated narrative and analysis-freeze
videos for those candidates, QA frames and an initial merged training manifest
with 320 frames and 800 detections/crops. The June 14 semantic audit found 80
repeated `video + frame_index` entries hidden behind different variant paths.
The curated manifest now contains 240 unique frames and 614 detections/crops in
a 160/80 train/validation split. Its quality report has 0 invalid boxes, 0
scores below 0.60, 0 videos shared between splits and 0 duplicate source frames.

A second mask-preserving pool contains 605 frames and 1,486 successful COCO
segmentations across four videos. It is heavily imbalanced toward robots
(1,452 robot masks versus 34 ball masks). A separate review candidate contains
124 ball masks from one clip. The frozen human holdout currently has 24 pending
frames from one validation video and no copied pseudo labels.

Official-SAM3 head adaptation now has three measured experiments on the remote
RTX 5080. The strongest mixed-class checkpoint used a 248-image ball-centered
train subset and all 128 validation images. Overall AP improved from 0.2299 to
0.3586 (+56.0%) and robot AP from 0.4496 to 0.7068 (+57.2%). Ball AP improved
only 2.0%, while ball small-object AP regressed 20.9%. A ball-only specialist
regressed 46.8% in ball AP and 92.5% in ball small-object AP, so it is explicitly
rejected. These results validate a strong robot adaptation and establish that
ball-data quality, verified negatives and source diversity are the next
bottleneck.

A ball-specific human-review selector is now available through
`select-ball-review`. It generated two June 15 review packages from
`mask_training_v2`: a strict set with 23 positive and 40 negative-candidate
frames, and a dense set with 40 positive and 40 negative-candidate frames. The
negative frames are deliberately marked as `verify_absence`, not as ground-truth
absence, because missing pseudo-labels can hide missed balls.
The companion `audit-ball-review` and `export-reviewed-ball` commands now block
training export until positives have human annotations and absence candidates
are explicitly verified. The export can now rewrite train/validation/test using
`--split-strategy by-source-balanced`, keeping all frames from the same original
recording in a single split before the next ball adaptation run. The audit also
writes an optional Markdown checklist so pending masks and absence labels can be
reviewed without reading the raw JSON.

## Remaining Work

### Critical

1. Review the generated narrative and analysis videos visually.
2. Select two or three final clips for the submission story.
3. Annotate the frozen 24-frame holdout and review the 124 ball-mask candidate
   frames. Add another independent ball clip.
4. Build a human-reviewed ball split with verified negatives and at least one
   additional original recording; the automatic balanced and ball-only
   variants have already been tested and rejected for ball promotion.
5. Re-run the best clips with the adapted checkpoint and compare QA results
   against the current baseline.

### Important

1. Improve team identification on hard clips if the final selected videos need
   stronger team-possession claims. The current `IMG_9938` reassignment
   collapses 99.7% of robot samples to blue and is correctly marked for review.
2. Validate the new temporal `goal_confirmed` path on a clip that actually
   contains a goal. The current 14-run top-camera batch contains no
   `goal_candidate` events, so it correctly produces zero confirmations.
3. Produce final report screenshots: QA table, tactical map, situation analysis
   snippet and selected video frame examples. A Markdown evidence report can now
   be generated with `submission-report`.
4. Package reproducible commands for the exact final clips.

### Optional

1. A lightweight dashboard/GUI for review.
2. Extra tracker comparison if time remains and rules allow it.
3. More top-camera clips in the dataset to improve validation coverage.

## Time Estimate

Estimated remaining time depends on whether the deliverable stops at a strong
SAM3 pipeline or includes a real fine-tuning experiment:

- Finalize current pipeline and presentation evidence: 6 to 10 focused hours.
- Human-review and freeze the curated SAM-compatible dataset: 2 to 5 focused
  hours.
- Run and evaluate the next ball-balanced adaptation experiment: 5 to 10
  focused hours, mostly constrained by annotation review and dataset quality.
- Polish final documentation and reproducible commands: 3 to 5 focused hours.

Practical total: 2 to 3 working days for a defensible final submission. A more
ambitious fine-tuning result with multiple iterations is closer to 4 to 5
working days.
