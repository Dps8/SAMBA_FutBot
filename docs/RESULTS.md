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
`IMG_9933_f000000_10s-top-fusion-hsv-v3-minarea`. The top-fusion path combines SAM3
field/robot detections with an HSV orange-ball fallback, removes orange blobs
inside robot boxes, rejects edge-touching and undersized ball candidates, refines
the ball path with temporal dynamic programming and then computes in-play
trajectories. This avoids relying only on text prompts when SAM3 misses the
small orange ball from the overhead camera.

## Artifact Types

- `outputs/detections/`: SAM3 detections as JSONL.
- `outputs/tracks/`: IoU tracker output as JSONL.
- `outputs/metrics/`: per-video summary metrics as JSON.
- `outputs/events/`: event candidates such as possession changes, collisions
  and shots.
- `outputs/videos/`: rendered demo videos with tracking overlays.
- `outputs/videos/qa_frames/`: QA screenshots sampled from rendered demos.

Large MP4 files are tracked with Git LFS.
