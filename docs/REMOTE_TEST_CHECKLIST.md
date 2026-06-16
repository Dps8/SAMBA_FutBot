# Remote Validation Checklist

Use this checklist after the Windows changes have been copied or pulled into
the remote repository.

## 1. Enter The Environment

```bash
cd /home/robocup/.samba_futbot_work/SAMBA_FutBot
source /home/robocup/ComputerVision/venv/bin/activate
export PYTHONPATH=src
```

Confirm that the new commands and prompts are available:

```bash
python -m samba_futbot.cli --help | grep -E \
  'process-top-camera|dataset-quality|curate-dataset|select-holdout|select-ball-review|audit-ball-review|export-reviewed-ball|team-quality|submission-report'

python - <<'PY'
from argparse import Namespace
from samba_futbot.cli import _context_classes
from samba_futbot.config import load_config

config = load_config("config/default.yml")
print(_context_classes(Namespace(goals=True, human_context=True), config))
PY
```

Expected context classes:

```text
field,robots,goal_blue,goal_yellow,person,human,referee,hand
```

## 2. Locate A Top-Camera Clip

```bash
find outputs data -type f \
  \( -iname '*IMG_9938*f001799*10s*.mp4' -o -iname '*IMG_9933*f000000*10s*.mp4' \) \
  | head -20
```

The commands below use this known reviewed clip:

```text
outputs/review/2026-05-27/18abril_top_camera/clips/IMG_9938_f001799_10s.mp4
```

## 3. Three-Second Smoke Test

```bash
python -m samba_futbot.cli process-top-camera \
  --config config/default.yml \
  --video "outputs/review/2026-05-27/18abril_top_camera/clips/IMG_9938_f001799_10s.mp4" \
  --results-dir "outputs/review/2026-06-14/remote_smoke_3s" \
  --suffix "top-human-context-smoke-v1" \
  --human-context \
  --max-seconds 3 \
  --render \
  --render-narrative \
  --render-analysis \
  --analysis-freeze
```

Check that the run produced tracks, game state, QA and videos:

```bash
find "outputs/review/2026-06-14/remote_smoke_3s" -type f | sort
```

## 4. Ten-Second Validation

Run this only after the smoke test completes:

```bash
python -m samba_futbot.cli process-top-camera \
  --config config/default.yml \
  --video "outputs/review/2026-05-27/18abril_top_camera/clips/IMG_9938_f001799_10s.mp4" \
  --results-dir "outputs/review/2026-06-14/remote_validation_10s" \
  --suffix "top-human-context-v1" \
  --human-context \
  --max-seconds 10 \
  --render \
  --render-narrative \
  --render-analysis \
  --analysis-freeze
```

Review these outputs first:

- Narrative video under `videos/`.
- Analysis video under `videos/`.
- QA JSON and Markdown under `qa/`.
- Game-state JSON and external events.
- Tracks JSONL, especially `person`, `human`, `referee` or `hand` detections.

Search human-intervention evidence:

```bash
grep -RniE \
  '"class_name": "(person|human|referee|hand)"|human_intervention' \
  "outputs/review/2026-06-14/remote_validation_10s" \
  | head -80
```

Compare candidate and confirmed goals:

```bash
grep -RniE 'goal_candidate|goal_confirmed|Candidate score|Confirmed score' \
  "outputs/review/2026-06-14/remote_validation_10s" \
  | head -80
```

`goal_confirmed` should appear only when the ball enters a directly detected
goal from outside and remains inside across multiple frames. A goal created
only by field geometry must remain a candidate.

If SAM3 drops robots late in the top-camera clip, run the same validation with
the optional robot color recovery enabled:

```bash
python -m samba_futbot.cli process-top-camera \
  --config config/default.yml \
  --video "outputs/review/2026-05-27/18abril_top_camera/clips/IMG_9938_f001799_10s.mp4" \
  --results-dir "outputs/review/2026-06-14/remote_robot_recovery_10s" \
  --suffix "top-robot-recovery-v1" \
  --human-context \
  --robot-color-recovery \
  --robot-recovery-min-area 800 \
  --robot-recovery-min-circularity 0.30 \
  --robot-recovery-hsv-upper "179,255,125" \
  --robot-recovery-min-center-y-ratio 0.38 \
  --robot-recovery-merge-distance-px 42 \
  --robot-recovery-max-per-frame 4 \
  --render \
  --render-narrative \
  --render-analysis \
  --analysis-freeze \
  --mask-overlay \
  --label-scale 0.82 \
  --box-thickness 4
```

Treat this as a review candidate until the late-frame boxes are visually
checked. It is a SAM-compatible post-processing fallback, not YOLO or external
detector training.

Existing tracks and events can be validated without rerunning SAM3:

```bash
python -m samba_futbot.cli validate-goals \
  --tracks "PATH/TO/clip-tracks.jsonl" \
  --events "PATH/TO/clip-events.json" \
  --out "outputs/review/2026-06-14/clip-validated-events.json" \
  --config config/default.yml
```

## 5. Dataset Gate

```bash
python -m samba_futbot.cli dataset-quality \
  --manifest "outputs/review/2026-06-08/training_datasets/merged_top_camera_balanced_manifest.json" \
  --out "outputs/review/2026-06-08/training_datasets/merged_top_camera_balanced_quality.json" \
  --report-out "outputs/review/2026-06-08/training_datasets/merged_top_camera_balanced_quality.md" \
  --low-score-threshold 0.60
```

Expected critical values:

```text
invalid_boxes: 0
low_scores: 0
videos_in_multiple_splits: 0
duplicate_image_paths: 0
duplicate_source_frame_groups: 80
duplicate_source_frame_extras: 80
```

Create and re-audit the unique-frame manifest:

```bash
python -m samba_futbot.cli curate-dataset \
  --manifest "outputs/review/2026-06-08/training_datasets/merged_top_camera_balanced_manifest.json" \
  --out "outputs/review/2026-06-14/training_datasets/merged_top_camera-curated-v1.json" \
  --report-out "outputs/review/2026-06-14/training_datasets/merged_top_camera-curation-v1.json" \
  --classes "ball,robots" \
  --min-score 0.60 \
  --deduplicate-source-frames

python -m samba_futbot.cli dataset-quality \
  --manifest "outputs/review/2026-06-14/training_datasets/merged_top_camera-curated-v1.json" \
  --out "outputs/review/2026-06-14/training_datasets/merged_top_camera-curated-v1-quality.json" \
  --report-out "outputs/review/2026-06-14/training_datasets/merged_top_camera-curated-v1-quality.md"
```

Expected curated values:

```text
frames: 240
detections: 614
videos_in_multiple_splits: 0
duplicate_source_frame_groups: 0
```

Freeze the same human-review selection:

```bash
python -m samba_futbot.cli select-holdout \
  --manifest "outputs/review/2026-06-14/training_datasets/merged_top_camera-curated-v1.json" \
  --out "outputs/review/2026-06-14/training_datasets/human-holdout-v1.json" \
  --report-out "outputs/review/2026-06-14/training_datasets/human-holdout-v1-report.json" \
  --max-frames 24 \
  --preferred-split val \
  --seed 2026
```

Expected fingerprint:

```text
9fc753bc4b6ecadd2f49855e90b40188360b215ea4b4dd22a64ae44eb8adfba9
```

## 6. Ball-Review Gate

Do not train a promoted ball checkpoint from pseudo-negatives. First audit the
human-review package and generate the Markdown checklist:

```bash
python -m samba_futbot.cli audit-ball-review \
  --review "outputs/review/2026-06-15/ball_review/ball-review-v2-dense.json" \
  --out "outputs/review/2026-06-15/ball_review/ball-review-v2-dense-audit.json" \
  --report-out "outputs/review/2026-06-15/ball_review/ball-review-v2-dense-audit.md"
```

Expected until manual review is complete:

```text
ready_for_training: false
pending_frames: 80
verify_mask: 40
verify_absence: 40
```

After manual review, export with source-group isolation:

```bash
python -m samba_futbot.cli export-reviewed-ball \
  --review "outputs/review/2026-06-15/ball_review/ball-review-v2-dense.json" \
  --out "outputs/review/2026-06-15/ball_review/reviewed-ball-manifest.json" \
  --report-out "outputs/review/2026-06-15/ball_review/reviewed-ball-report.json" \
  --split-strategy by-source-balanced \
  --train-ratio 0.8 \
  --val-ratio 0.1
```

Only proceed to SAM3 segmentation adaptation when `mask_ready_annotations` is
non-zero and enough verified absence labels come from more than one original
recording.

## 7. Team Gate

Existing tracks can be reclassified and audited without repeating SAM3:

```bash
python -m samba_futbot.cli assign-teams \
  --video "outputs/review/2026-05-27/18abril_top_camera/clips/IMG_9938_f001799_10s.mp4" \
  --tracks "PATH/TO/clip-tracks.jsonl" \
  --out "outputs/review/2026-06-14/clip-tracks-with-teams.jsonl" \
  --config config/default.yml

python -m samba_futbot.cli team-quality \
  --tracks "outputs/review/2026-06-14/clip-tracks-with-teams.jsonl" \
  --out "outputs/review/2026-06-14/clip-team-quality.json" \
  --report-out "outputs/review/2026-06-14/clip-team-quality.md"
```

Do not promote team-possession claims when `unknown_ratio_above_threshold` or
`team_imbalance_above_threshold` is true.

## 8. Local Tests

```bash
python -m unittest discover -s tests
python -m compileall -q src tests
```

Do not treat `goal_candidate` as a confirmed score. The QA claim
`goal_scoring` must remain in review until the event evidence is explicitly
confirmed.

Do not introduce YOLO or detector-family-specific training. The tournament path
for this repo stays SAM3/SAM-compatible through prompts, masks, COCO/RLE exports
and official SAM3 adaptation.
