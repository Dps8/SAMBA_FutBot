# SAMBA FutBot Results

Generated artifacts for the current SAM3 windowed pipeline are stored under
`outputs/`.

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
Run Markdown reports can now include the QA JSON directly, so a single report
contains tracking metrics, events, field analysis and quality gates.
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
homography before final presentation.
Possession metrics also expose a dominant team with frame and ratio margin,
which gives a compact game-control signal for reports.
Shot candidates now require the ball to move toward the left/right goal side,
instead of only being fast near a goal margin, and include estimated target side
and shooting team metadata. Event summaries also aggregate shots by team and
target side.
Rendered demo videos can now read `events.json` and keep the latest event in
the overlay header for a short window, turning the visual output into a more
narrative review artifact.

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
frame, and accepted goals must overlap or touch the detected green field.

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
Calibration can now be checked separately with `samba-futbot calibration-check`;
the check reports reprojection error, image polygon area and calibration points
outside the frame so metric claims can be marked as calibrated or template-only.
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
