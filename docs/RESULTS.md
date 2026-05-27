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

`video-680` has an unrealistic max ball speed, which flags a likely ball-track
jump. This is useful as an automatic QA signal for track fragmentation.

## Artifact Types

- `outputs/detections/`: SAM3 detections as JSONL.
- `outputs/tracks/`: IoU tracker output as JSONL.
- `outputs/metrics/`: per-video summary metrics as JSON.
- `outputs/events/`: event candidates such as possession changes, collisions
  and shots.
- `outputs/videos/`: rendered demo videos with tracking overlays.
- `outputs/videos/qa_frames/`: QA screenshots sampled from rendered demos.

Large MP4 files are tracked with Git LFS.
