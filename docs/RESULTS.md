# SAMBA FutBot Results

Generated artifacts for the current SAM3 windowed pipeline are stored under
`outputs/`.

## Final submission v1.2.2 (19 June 2026)

The final delivery integrates two H.264 formats: a 116-second horizontal demo
and an 89-second vertical reel. Both show the narrative and analysis modules,
a real goal sequence with a scoreboard update, two camera views, motion
prediction, distances in meters, speed in m/s and a heatmap accumulated over
the complete 12:56 `IMG_9933.MOV` match.

Contextual cleaning limits the overhead clip to one ball and two robots per
frame. It retained 203 of 205 ball candidates and removed two false positives
overlapping robots. Robot filtering reduced 2,164 candidates to 1,697
detections: 292 frames contain two robots, eight contain one, and none contain
more than two.

Mask segmentation is evaluated with the COCO protocol over 128 annotated
images. The adapted variant reaches 92.1% global AR@100 and 75.1% robot AP75;
relative to the baseline, AP improves 56.0% and AP50 improves 52.7%. The small
ball regression remains in the evidence JSON so the presentation does not hide
the known limitation.

Final local artifacts are under
`outputs/review/2026-06-19/submission_v1_2_2/`. Small versioned evidence is under
`docs/evidence/`, with verified stills under `docs/assets/`.

The goal sequence in `video-427` is machine-verifiable rather than only a
visual edit. The tracked ball crosses an explicitly calibrated blue-goal line
at frame 356, after six outside frames, remains on the interior side for six
frames and contacts the calibrated back wall at frame 365. This final condition
implements rules 4.4.5 and 7.4.4. The emitted `goal_confirmed` event has
confidence 0.92 and updates the confirmed yellow scoreboard to 1-0. The demo
overlays the ball box, track, trajectory, goal region, goal line, back wall and
temporal persistence counter.

## Goal recovery and robot forecasts (18 June 2026)

The `IMG_9938` audit exposed a portrait-camera geometry error: the missing blue
goal had been mirrored horizontally beside the yellow goal. The corrected
constraint selects the symmetry axis from the detected field orientation and
does not let `geometry_only` seeds restrict color search. Broad blue HSV plus
field membership observes the physical blue goal in all 300 frames of the
reviewed clip; the yellow goal remains the higher-confidence SAM 3 result. The
new tracks contain zero geometry-only goals, one goal per color per frame and
render no goal trails.

Robot forecasting now uses metric robot paths instead of reusing only the ball
trajectory. A least-squares velocity estimate over recent observations creates
constant-heading, left-turn and right-turn branches over a 1.5-second horizon.
Branch weights combine fit consistency with the fraction of the path that
stays on the 2.43 x 1.82 m field and are normalized to one. They are explicitly
labeled heuristic and uncalibrated. The presentation snapshot suppresses
spatially coincident track hypotheses and displays at most two distinct robots,
including speed, branch weights and fit RMSE.

## Full-match top-camera evidence (18 June 2026)

`IMG_9933.MOV` was processed end to end for the submission heatmap: 23,278
declared frames (23,274 readable), 29.9876 fps and 776.3 seconds (12:56). The hybrid CPU fallback
produced ball detections on 97.2% of frames and filtered robot detections on
83.5% of frames. The accumulated heatmap uses 23,784 robot observations after
removing small, oversized, overfilled and elongated dark-object artifacts and
requiring the centroid to fall inside the calibrated field with a 0.10 m
margin. This removes the phone-shaped false positive seen in the previous
short demo.

The dynamic heatmap processes every source frame and writes one of every 30
frames at 30 fps, yielding a verifiable 30x timelapse without subsampling the
underlying accumulation. Outputs and their machine-readable report are under
`outputs/review/2026-06-19/full_match_IMG_9933_cpu/`.

The full color-only tracker is intentionally reported with its limitation:
2,772 robot track fragments make global team possession unsuitable as a final
ground-truth claim. Team possession, goals and interventions remain candidates
behind QA gates. Metric distances and speeds use the 2.43 x 1.82 m homography
only for valid in-play associations; pixel metrics are kept separate.

## Processed Videos

| Video | Frames | Detections | Ball detections | Ball tracks | Demo |
|---|---:|---:|---:|---:|---|
| `video-429_singular_display.mov` | 542 | 2214 | 342 | 20 | `outputs/videos/video-429-full-windowed-orange-prompt-demo.mp4` |
| `video-427_singular_display.mov` | 468 | 2115 | 292 | 5 | `outputs/videos/video-427_singular_display-full-windowed-orange-v2-demo.mp4` |
| `video-537_singular_display.mov` | 926 | 4297 | 645 | 5 | `outputs/videos/video-537_singular_display-full-windowed-orange-v2-clipped-demo.mp4` |
| `video-680_singular_display.mov` | 735 | 2066 | 767 | 21 | `outputs/videos/video-680_singular_display-full-windowed-orange-v2-clipped-demo.mp4` |
| `video-597_singular_display.mov` | 641 | 2131 | 546 | 7 | `outputs/videos/video-597_singular_display-full-windowed-orange-v2-clipped-demo.mp4` |
| `video-667_singular_display.mov` | 489 | 1821 | 126 | 2 | `outputs/videos/video-667_singular_display-full-windowed-orange-v2-clipped-demo.mp4` |
| `video-848_singular_display.mov` | 489 | 1348 | 244 | 3 | `outputs/videos/video-848_singular_display-full-windowed-orange-v2-clipped-demo.mp4` |
| `IMG_9866.MOV` | 102 | 276 | 74 | 2 | `outputs/videos/IMG_9866-top-camera-v1-demo.mp4` |
| `IMG_9866.MOV` | 102 | 365 | 74 | 2 | `outputs/videos/IMG_9866-top-camera-orange-context-v2-demo.mp4` |
| `IMG_9933_f000000_10s.mp4` | 300 | 1316 | 300 | 1 | `outputs/review/2026-05-27/18abril_top_camera/good/videos/IMG_9933_f000000_10s-top-fusion-hsv-v3-minarea-demo.mp4` |
| `IMG_9933_f008995_10s.mp4` | 300 | 931 | 124 | 1 | `outputs/review/2026-05-27/18abril_top_camera/good/videos/IMG_9933_f008995_10s-top-context-v1-demo.mp4` |
| `IMG_9933_f017990_10s.mp4` | 300 | 990 | 300 | 1 | `outputs/review/2026-05-27/18abril_top_camera/good/videos/IMG_9933_f017990_10s-top-fusion-hsv-v3-minarea-demo.mp4` |
| `IMG_9938_f001799_10s.mp4` | 300 | 842 | 218 | 3 | `outputs/review/2026-05-27/18abril_top_camera/good/videos/IMG_9938_f001799_10s-top-fusion-hsv-v3-minarea-demo.mp4` |

## Enriched Ball And Event Analysis

| Video | Ball coverage | Mean ball speed px/s | Max ball speed px/s | Events |
|---|---:|---:|---:|---:|
| `video-427_singular_display.mov` | 61.3% | 201.1 | 986.6 | 17 |
| `video-429_singular_display.mov` | 61.3% | 165.3 | 668.4 | 8 |
| `video-537_singular_display.mov` | 69.7% | 200.9 | 1233.1 | 17 |
| `video-597_singular_display.mov` | 72.7% | 251.9 | 1305.6 | 2 |
| `video-667_singular_display.mov` | 25.8% | 311.1 | 966.9 | 9 |
| `video-680_singular_display.mov` | 71.3% | 562.4 | 19092.0 | 17 |
| `video-848_singular_display.mov` | 46.3% | 164.1 | 755.7 | 7 |
| `IMG_9866.MOV` | 72.5% | 61.7 | 271.5 | 0 |
| `IMG_9866.MOV` contextual top-camera prompts | 59.8% in-play | 72.1 | 271.5 | 0 |
| `IMG_9933_f000000_10s.mp4` top-fusion HSV v3 | 100.0% in-play | 48.2 | 203.9 | 0 |
| `IMG_9933_f008995_10s.mp4` top-camera 18abril | 41.3% in-play | 11.9 | 33.5 | 0 |
| `IMG_9933_f017990_10s.mp4` top-fusion HSV v3 | 100.0% in-play | 2.6 | 33.5 | 0 |
| `IMG_9938_f001799_10s.mp4` top-fusion HSV v3 | 72.7% in-play | 31.5 | 261.9 | 1 |

## June 8 Top-Camera Batch

The local Windows batch under
`outputs/review/2026-06-08/top_camera_batch` reprocessed 14 existing top-camera
track variants through game-state filtering, metrics, event detection and
`situation-analysis`. The strongest ball-tracking variants were:

| Variant | Ball coverage | QA | Ready claims |
|---|---:|---|---|
| `IMG_9933_f000000_10s-top-fusion-hsv-v3-minarea` | 100.0% | good / 100 | ball_tracking |
| `IMG_9933_f017990_10s-top-fusion-hsv-v3-minarea` | 100.0% | good / 100 | ball_tracking |
| `IMG_9938_f001799_10s-top-fusion-hsv-v2-refined` | 94.0% | review / 90 | ball_tracking, shot_pressure |
| `IMG_9938_f001799_10s-top-fusion-hsv-v3-minarea` | 72.7% | review / 80 | ball_tracking, shot_pressure |

The same batch currently contains 12 presentation videos: narrative and
analysis-freeze outputs for the strongest variants, plus QA frames for quick
review. It also has a generated submission evidence report at
`outputs/review/2026-06-08/SUBMISSION_EVIDENCE.md`, which links the selected
showcase clips, QA status and dataset preparation in one Markdown artifact. The
batch also produced a merged top-camera training manifest with 320 frames and
800 detections/crops (`259` ball and `541` robot samples). A deeper June 14
audit found `80` duplicate source-frame groups: two processing variants of
`IMG_9938` contained the same 80 video frames under different image paths.
The new `curate-dataset` pass kept the strongest variant per source frame and
produced 240 unique frames with 614 detections/crops (`231` ball and `383`
robot samples), split as `160` train / `80` validation. The post-curation
quality report found `0` invalid boxes, `0` detections below `0.60`, `0` videos
shared between splits, `0` duplicate image paths and `0` duplicate
source-frames.

A mask-preserving export was then generated directly from the existing SAM3
tracks. The dense multi-video manifest contains `605` unique frames and `1,486`
mask annotations (`1,452` robots and `34` ball), with `485` train and `120`
validation frames. COCO segmentation export completed with `1,486/1,486`
masks and `0` failures. Because this pool is strongly robot-heavy, a separate
ball-review candidate set was also exported from `IMG_9933_f008995`: `124`
frames, `124` ball masks and `124/124` successful COCO RLE exports at score
`>=0.50`. It remains a single-video pseudo-label pool and must be reviewed or
combined with independent ball annotations before a final adaptation claim.

The first independent human holdout template contains `24` validation frames
from one held-out source video. It copies no pseudo detections, leaves every
annotation `pending`, and is frozen by SHA-256
`9fc753bc4b6ecadd2f49855e90b40188360b215ea4b4dd22a64ae44eb8adfba9`.
This is enough to start manual labeling, but broader camera/video coverage is
still desirable for a final benchmark.

## SAM3 Head Adaptation

On June 15, 2026, the first official-SAM3 adaptation completed on the remote
RTX 5080 using the repository's portable COCO-RLE export. Full 840 M-parameter
AdamW training exceeded the 16 GB memory budget when optimizer states were
created, so the successful run trained approximately 3.5 M segmentation and
prompt-scoring head parameters while freezing approximately 837 M backbone
parameters.

The reproducible run used 3 epochs, 64 train datapoints, 64 validation
datapoints and 1008-pixel inputs. Re-evaluation against only the 64 image IDs
actually inferred produced:

| Metric | SAM3 baseline | Adapted | Relative change |
|---|---:|---:|---:|
| Overall mask AP | 0.2364 | 0.3043 | +28.8% |
| Overall mask AP50 | 0.2589 | 0.3400 | +31.3% |
| Robot mask AP | 0.4625 | 0.5979 | +29.3% |
| Ball mask AP | 0.0102 | 0.0108 | +6.2% |
| Small-object AP | 0.0128 | 0.0109 | -14.7% |

The result validates the training/configuration/checkpoint path and gives a
meaningful robot improvement. It does not yet justify claiming robust learned
ball segmentation: the ball subset remains small and imbalanced, and
small-object quality must improve before promotion to the main video pipeline.

A follow-up mixed-class run centered training on all 124 ball-positive images
plus 124 images without a ball annotation, while retaining the robot masks.
Against the full 128-image validation set, overall AP improved from 0.2299 to
0.3586 (+56.0%) and robot AP improved from 0.4496 to 0.7068 (+57.2%). Ball AP
only moved from 0.0101 to 0.0103 (+2.0%), and ball small-object AP fell from
0.0592 to 0.0468 (-20.9%). This is the current strongest robot checkpoint, not
a promoted ball checkpoint.

A separate ball-only specialist was also tested and rejected. Its ball AP fell
from 0.0099 to 0.0052 (-46.8%), and ball small-object AP fell from 0.0587 to
0.0044 (-92.5%). The experiment demonstrates that missing pseudo-labels cannot
be treated as trustworthy negative examples. The next ball adaptation requires
manual mask review, verified absence labels and another original recording with
different ball scale and occlusion conditions.

The follow-up data-preparation step generated human-review packages for the
ball bottleneck. The dense package contains 40 `verify_mask` frames and 40
`verify_absence` frames, with contact sheets under
`outputs/review/2026-06-15/ball_review`. These are review artifacts, not final
training labels.

The QA comparison tool was tested on `IMG_9938` using the `top-fusion-hsv-v1`
run as baseline and `top-fusion-hsv-v3-minarea` as candidate. It classified the
candidate as `regressed`: quality score dropped from `90` to `80`, ball
coverage from `94.0%` to `72.7%`, field coverage from `100.0%` to `82.0%`, and
robot coverage from `40.0%` to `22.0%`.

`video-680` has an unrealistic max ball speed, which flags a likely ball-track
jump. This is useful as an automatic QA signal for track fragmentation.

Current pipelines now write an automatic QA report in addition to metrics and
events. The QA score combines ball in-play coverage, maximum ball jump,
field/robot coverage, homography out-of-bounds samples and rule candidates.
The `qa-index` command can now scan a results folder and rank QA JSON reports by
status and score, which helps triage many prompt/model variants before manual
video review. Its Markdown table also exposes the unknown-team ratio so weak
team-color assignments are visible before opening the videos.
QA also checks the ratio of robot samples that remain `unknown` after team
assignment, so possession-by-team and tactical claims are flagged when color
classification is not reliable enough.
The standalone `team-quality` audit additionally detects temporal label changes,
ambiguous tracks and assignments collapsed toward one color. On the historical
`IMG_9938` refined tracks, the original file had 324 robot samples with no team
evidence. Reassigning colors classified 323 as blue and 1 as yellow; the new
dominant-team check correctly flags that `99.7%` concentration for review
instead of treating full assignment coverage as trustworthy.
Run Markdown reports can now include the QA JSON directly, so a single report
contains tracking metrics, events, field analysis and quality gates.
QA now also emits `claim_readiness`, a compact evidence matrix that marks
whether each run is ready for ball-tracking, metric trajectory/speed,
team-possession, goal-scoring and shot-pressure claims. This is meant to keep
the professional submission honest: strong clips can be promoted quickly, while
weak clips remain candidates instead of becoming unsupported claims.
The repo also includes an `export-pseudolabels` CLI command that converts
high-confidence SAM3 detections with saved masks into a compact pseudo-label
manifest. This does not train a model yet, but it reduces the setup time for the
next fine-tuning block by making candidate masks auditable and filterable by
class, score, area and mask availability.
The `export-frame-dataset` command extends that preparation step by exporting
full frames, class-specific crops and a dataset manifest from any video plus
detections/tracks file. Its default split is by video, which avoids leaking
near-duplicate frames from the same clip across train/validation/test when the
fine-tuning block starts.
Dataset manifests can now be converted to COCO detection JSON with
`export-coco`, keeping the fine-tuning preparation neutral and compatible with
SAM-style mask/box review without introducing an extra detector family.
The dataset tools also support `merge-frame-datasets --split-strategy
by-source-balanced`. The current top-camera batch exported a merged manifest
with `320` frames, `800` detections/crops, `259` ball samples and `541` robot
samples split as `240` train frames and `80` validation frames.
Full processing commands also write that integrated report by default under
`outputs/reports`, reducing the number of manual post-processing commands needed
for reproducible review.
They also write a JSON run manifest with the invoked command, normalized
arguments, UTC timestamp, runtime, local Git branch/commit/dirty status, a
SHA256 fingerprint of the local code/config files, artifact paths and key
summaries for auditability.
For the reviewed top-camera smoke run
`IMG_9938_f001799_10s-rules-smoke`, QA returns `review` with score `90`
because ball coverage is `68.7%`, below the professional default of `75%`,
even though the refined trajectory has no large jumps. This matches the visual
observation that some overhead frames still miss the ball and should not be
promoted blindly.

Team-aware analysis is now part of the main pipeline. Robot detections are
assigned to `blue` or `yellow` from color evidence inside each tracked robot
box, using the configurable palette in `config/default.yml`. The sampler votes
with pixels close to the blue/yellow palettes and ignores low-saturation or
dark pixels, reducing contamination from field/background inside wide boxes.
Metrics include possession coverage, possession by team and longest possession streaks. Event detection can also report
`goal_candidate` events when visual `goal_blue` or `goal_yellow` detections are
available; metric goal claims should still be validated with calibrated
homography before final presentation. QA therefore keeps `goal_scoring` in
review for candidate-only events and only marks it ready for an explicit
`goal_confirmed` event. Confirmation now requires a non-inferred goal detection,
a tracked ball entering from outside with measurable motion toward the goal,
and persistence inside the goal across multiple frames.
Applying the semantic-goal validator to the 14 existing June 8 top-camera runs
found zero goal candidates. The normal-view `video-427` sequence now supplies
the missing end-to-end validation through the explicit calibrated-geometry
route: one tracked ball, directed exterior-to-interior crossing, temporal
persistence and contact with the back wall produce a real `goal_confirmed`
event.
Possession metrics also expose a dominant team with frame and ratio margin,
which gives a compact game-control signal for reports.
Shot candidates now require the ball to move toward the left/right goal side,
instead of only being fast near a goal margin, and include estimated target side
and shooting team metadata. Event summaries also aggregate shots by team and
target side.
Rendered demo videos can now read `events.json` and keep the latest event in
the overlay header for a short window, turning the visual output into a more
narrative review artifact.
Rendering is now split into two complementary styles. The narrative demo keeps
the overlay lighter for presentation: teams, possession and recent match events.
The analysis demo keeps the technical detail: boxes, scores, team labels, ball
trajectory, per-robot distance to the ball, ball speed in px/frame and a
heuristic shot-pressure probability toward the left or right goal. Full
pipelines can generate both videos in one run through `--render-narrative` and
`--render-analysis`, while `render-demo --style narrative|analysis` can
re-render either view from existing tracks.
The `situation-analysis` command adds a separate tactical JSON layer: per-frame
robot-ball distance ranking, `controlled/disputed/free` possession state, loss
risk and heuristic pass/shot/hold probabilities. This is intended as the
structured source for richer analysis overlays and final reports.
The tactical field map also overlays robot-density heat by team, so heatmaps no
longer depend only on the ball trajectory; they show where blue, yellow or
unknown robots spent time across the calibrated field grid.
The analysis render can now add optional freeze frames through
`--analysis-freeze`. These pauses reuse the already-rendered frame, add a short
explanation panel, highlight the ball trajectory/shot lane, and are limited by
cooldown and maximum-event controls so the video stays reviewable.

SAM3 prompts alone did not reliably detect the blue/yellow goals in short smoke
clips, so the pipeline now includes a configurable HSV fallback for
`goal_blue` and `goal_yellow`. On the `video_429_side_smoke_2s` remote smoke,
the fallback produced `240` colored-goal detections and the integrated pipeline
tracked both goal classes through the rendered demo.

The goal-color fallback is now adaptive. If SAM3 finds a `goal_blue` or
`goal_yellow` box in a video, the pipeline samples HSV pixels inside that box
and expands the learned range before running the color pass. This keeps the
fixed YAML profiles as a fallback, while letting each video adjust to its own
lighting, camera compression and actual goal material color.

Visual QA also showed that pure color search can pick up similarly colored
objects outside the field. The current adaptive path therefore uses SAM3 goal
boxes as spatial gates: learned HSV detections must fall near the seed box when
one exists. In conservative mode, a goal class also requires its own SAM3 seed
before color detections of that class are accepted; this prevents a blue phone
or background object from being promoted to `goal_blue`.

The latest goal post-processing also applies two domain constraints from the
tournament setup: at most one `goal_blue` and one `goal_yellow` are kept per
frame, and accepted goals must overlap or touch the detected green field. If
only one colored goal is visible in a frame with a detected field, the pipeline
can add the opposite goal by mirroring the detected box across the field center;
the inferred goal is explicitly marked with `source: goal_geometry`.

## Top Camera Notes

`IMG_9866.MOV` is a short high-angle camera clip. The first run validated that
the pipeline can process the camera-superior view and recover robots plus ball.
The contextual prompt run (`top-camera-orange-context-v2`) adds green-field and
robot-possession context for the orange ball, detects the field in 89/102 frames,
and reports ball trajectory only when the ball is on the field or close enough to
a robot to count as possession. This view is the best candidate for the next
professional visual layer: heatmaps, zones, possession maps and homography.

For `18abril/Camara_superior`, long videos are first sampled into contact
sheets and promoted to short clips before running SAM3. Current reviewed outputs
are organized under `outputs/review/2026-05-27/18abril_top_camera/` with
`good`, `latest` and `needs_review` folders. The strongest current clip is
`IMG_9933_f000000_10s-top-fusion-hsv-v3-minarea`. The original top-fusion path
combined SAM3 field/robot detections with an HSV orange-ball fallback, removed
orange blobs inside robot boxes, rejected edge-touching and undersized ball
candidates, refined the ball path with temporal dynamic programming and then
computed in-play trajectories. The current default evolves this into
`top-hybrid-ball-v1`: SAM3 also attempts semantic ball detection, while the
HSV/color source stays as a configurable cue for the current orange ball. New
overhead clips can therefore run SAM3 ball prompts, color/shape detection,
merge, refinement, tracking and rendering through a single
`samba-futbot process-top-camera` command.

June 2026 smoke tests on the top-camera clips showed that full goal prompting can
slow down or destabilize short reprocessing passes. A lighter top-camera variant
with `--no-goals` over two-second windows recovered the orange ball in every
frame for two previously weak clips: `IMG_9933_f000000_2s` reached QA `good`
with 100% in-play ball coverage, and `IMG_9933_f017990_2s` reached QA `review`
with 100% in-play ball coverage but low robot coverage. This supports a
two-stage strategy: first recover ball/robots/field in short windows, then add
goal evidence through color, seeds or field geometry.

The June 16 top-camera refresh reprocessed the final review clips on the remote
GPU with the current no-YOLO pipeline: SAM3 contextual prompts, orange-ball
color fallback, blue/yellow goal color geometry, dark-robot recovery,
semi-transparent overlays and label-collision avoidance. The two `IMG_9933`
clips reached QA `good` with 100% ball coverage and are the strongest current
ball-tracking examples. `IMG_9938_f001799_10s` remains QA `review` because ball
coverage is 68.3%, but its analysis render is still useful for shot-pressure
and pause-overlay demonstration. The latest videos are under
`outputs/review/2026-06-16/final_top_camera_robot_recovery_batch_v4/videos/`
and `outputs/review/2026-06-16/final_top_camera_robot_recovery_v2/videos/`.

The next tactical layer is implemented as an optional homography analysis:
`samba-futbot field-analysis` or `process-top-camera --field-calibration`.
Given four calibrated field corners, it converts in-play ball centroids from
pixels to field meters, exports a trajectory CSV, reports speed in `m/s`,
distance traveled, a zone-occupancy grid and a tactical PNG field map. This
keeps the professional analysis grounded in field geometry instead of only
screen-space pixels. The default field template now follows the official
FutBotMX field dimensions, `2.43 m x 1.82 m`, with official center-circle,
penalty-area and goal markings. Final metric claims still require replacing the
template image points with calibrated corners from each real top-camera setup.
Calibration can now be checked separately with `samba-futbot calibration-check`.
In addition to reprojection error and points outside the frame, the check
validates corner order, convexity, frame coverage, edge ratios, compressed
angles and polygon skew. This prevents a four-point homography from passing
only because it is evaluated on the same points used to fit it.
Robot field projection also reports samples by team, zone samples by team,
penalty-area samples by team and defensive/middle/attacking thirds relative to
each team's defended side. It also derives an attacking-pressure ratio per team,
the share of robot samples in that team's offensive third. The same analysis now
exports territorial control by grid zone: leader team, margin and leader ratio.
This control layer is also exported as CSV for spreadsheet review.
Standalone `field-analysis` can also receive the original video and color/team
configuration, allowing old track files without `team` metadata to be
reclassified before metric projection.
The tactical field-map PNG now plots robot samples and territorial-control
cells over the calibrated pitch with team colors, so the same artifact shows
ball trajectory, ball-zone occupancy and blue/yellow robot positioning.

## Artifact Types

- `outputs/detections/`: SAM3 detections as JSONL.
- `outputs/tracks/`: IoU tracker output as JSONL.
- `outputs/metrics/`: per-video summary metrics as JSON.
- `outputs/events/`: event candidates such as possession changes, collisions
  and shots. Pipelines also write `*-event-summary.json` with candidate score,
  goals by side, passes, interceptions, shots and collisions.
- `outputs/field_analysis/`: homography metrics, trajectory CSV files and
  tactical field-map PNGs. It can also include robot projection CSVs,
  calibration-frame JPGs and Markdown run reports.
- `outputs/qa/`: automatic run-quality JSON/Markdown reports.
- `outputs/videos/`: rendered demo videos with tracking overlays.
- `outputs/videos/qa_frames/`: QA screenshots sampled from rendered demos.

Large MP4 files are tracked with Git LFS.
