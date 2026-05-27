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
| 1 | `IMG_9933_f008995_10s` | good | Field, robots and real orange ball are visible; 124 in-play ball frames. |
| 2 | `IMG_9938_f001799_10s` | needs_review | Field is stable; ball retry found 13 candidates but QA shows edge/hand false positives. |
| 3 | `IMG_9933_f017990_10s` | needs_review | Field and robots are stable; ball remains undetected even after v3 prompt retry. |

## Best Artifact

Demo:

`outputs/review/2026-05-27/18abril_top_camera/good/videos/IMG_9933_f008995_10s-top-context-v1-demo.mp4`

QA frame:

`outputs/review/2026-05-27/18abril_top_camera/good/qa_frames/IMG_9933_f008995_10s-top-context-v1-demo-frame-000149.jpg`

Metrics:

`outputs/review/2026-05-27/18abril_top_camera/good/metrics/IMG_9933_f008995_10s-top-context-v1-metrics.json`

## Competitive Notes

- The pipeline now uses top-camera sampling before GPU processing, which avoids spending SAM3 time on low-value intervals.
- Ball trajectory and time-in-play are filtered by field geometry or robot possession, so orange objects outside play do not automatically become game time.
- The prompt set now includes small bright orange ball, orange soccer ball and sideline/white-line context. This helped recover some hard cases, but QA still flags false positives near borders and hands.
- Next improvement should be ROI-aware ball filtering: reject ball boxes that are too close to the frame border or outside the detected field mask/box, and add a color/shape sanity check after SAM3.
