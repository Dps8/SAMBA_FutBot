# 18abril Top-Camera Review - 2026-05-27

Source folder:

`data/raw/Meta_Glasses/18abril/Camara_superior`

## Videos Found

| Video | Frames | Duration | Resolution |
|---|---:|---:|---|
| `IMG_9933.MOV` | 23278 | 776.3 s | 1080x1920 |
| `IMG_9938.MOV` | 19407 | 647.2 s | 1080x1920 |

## Review Structure

| Folder | Meaning |
|---|---|
| `samples/` | Sparse frames sampled from the long source videos. |
| `contact_sheets/` | Visual sheets used to choose relevant segments. |
| `clips/` | Ten-second clips cut from selected source frames. |
| `runs/` | Full SAM3 detections, tracks, metrics, events and demos. |
| `latest/` | Latest usable result promoted for quick review. |
| `good/` | Strong demos suitable for presentation. |
| `needs_review/` | Runs with useful field/robot detection but weak or false ball detection. |

## Current Ranking

| Rank | Clip | Status | Notes |
|---|---|---|---|
| 1 | `IMG_9933_f000000_10s-top-fusion-hsv-v1` | good | SAM3 field/robots plus HSV ball fallback; 300/300 in-play ball frames. |
| 2 | `IMG_9933_f017990_10s-top-fusion-hsv-v1` | good | Recovers the orange ball that SAM3 text prompts missed; 300/300 in-play ball frames. |
| 3 | `IMG_9938_f001799_10s-top-fusion-hsv-v1` | good | Recovers ball in 282/300 frames after robot-context filtering. |
| 4 | `IMG_9933_f008995_10s-top-context-v1` | good | Pure SAM3 contextual prompt baseline; 124 in-play ball frames. |
| 5 | `IMG_9938_f001799_10s-top-context-v2-edgefiltered` | needs_review | Useful QA case: edge filter removes 13 false ball detections and 5 false events. |
| 6 | `IMG_9933_f017990_10s-top-context-v2` | needs_review | Field and robots are stable; SAM3 text prompts still miss the visible ball. |

## Best Artifact

Demo:

`outputs/review/2026-05-27/18abril_top_camera/good/videos/IMG_9933_f000000_10s-top-fusion-hsv-v1-demo.mp4`

QA frame:

`outputs/review/2026-05-27/18abril_top_camera/good/qa_frames/IMG_9933_f000000_10s-top-fusion-hsv-v1-demo-frame-000149.jpg`

Metrics:

`outputs/review/2026-05-27/18abril_top_camera/good/metrics/IMG_9933_f000000_10s-top-fusion-hsv-v1-metrics.json`

## Competitive Notes

- The pipeline now uses top-camera sampling before GPU processing, which avoids spending SAM3 time on low-value intervals.
- Ball trajectory and time-in-play are filtered by field geometry or robot possession, so orange objects outside play do not automatically become game time.
- The prompt set now includes small bright orange ball, orange soccer ball and sideline/white-line context. This helped recover some hard cases, but QA still flags false positives near borders and hands.
- ROI-aware ball filtering rejects ball boxes that touch the frame border, preventing false events from hand/edge artifacts.
- The new `top-fusion-hsv-v1` path combines SAM3 for field/robots with an HSV color/shape fallback for the orange ball, then removes blobs inside robot boxes. This is the current differentiator versus a prompt-only SAM3 solution.
- Next improvement should add homography/zone coordinates from the top camera so the trajectories can be reported in field space instead of pixels.
