# SAM3 Fine-Tuning Runbook

Last updated: 2026-06-15

## Scope

This runbook prepares an image-segmentation adaptation using only the official
Meta SAM3 training repository and SAM-compatible masks. It does not introduce
an external detector.

Official references:

- https://github.com/facebookresearch/sam3
- https://github.com/facebookresearch/sam3/blob/main/README_TRAIN.md
- https://github.com/facebookresearch/sam3/blob/main/scripts/eval/gold/README.md

The official training path uses Hydra configuration and is launched locally
with a command shaped like:

```bash
python sam3/train/train.py -c PATH/TO/CONFIG.yaml --use-cluster 0 --num-gpus 1
```

The installed SAM3 commit uses a fused ViT MLP operation that rejects autograd.
Use the repository launcher for training so the normal PyTorch MLP path is used
when gradients are enabled, while inference keeps the fused operation:

```bash
python scripts/run_sam3_finetune.py \
  -c configs/samba_futbot/robots_ball_smoke_v1.yaml \
  --use-cluster 0 \
  --num-gpus 1
```

The official SA-Co annotation format is not ordinary multi-class COCO. It is
appropriate for noun-phrase evaluation, but the installed Roboflow training
loader expects ordinary multi-class COCO. Each SA-Co image entry represents an
image plus one noun phrase and includes
`text_input`, `is_instance_exhaustive` and `is_pixel_exhaustive`. Boxes and
areas are normalized, and masks use COCO RLE.

For the current robot/ball fine-tuning run, use `export-coco --image-root` so
the training loader receives distinct categories and portable image paths.

The 16 GB remote GPU can complete a 1008-pixel forward/backward pass, but full
840 M-parameter AdamW adaptation exhausts memory when optimizer states are
created. The generated config therefore freezes the SAM3 backbones and trains
the segmentation and prompt-scoring heads first. This is a parameter-efficient
SAM3 adaptation, not an external detector.

Keep the current backbone at resolution 1008. A 504-pixel smoke run is not
compatible with the fixed RoPE frequency shape in the installed SAM3 commit.

## Current Data Gates

Box/crop manifest:

```text
outputs/review/2026-06-14/training_datasets/merged_top_camera-curated-v1.json
```

- 240 unique frames.
- 614 detections.
- No duplicate source frames.
- No source video shared between train and validation.

Mask-preserving multi-video manifest:

```text
outputs/review/2026-06-14/mask_training_datasets_dense/merged_mask_manifest.json
```

- 605 frames from four clips.
- 1,486 masks exported successfully.
- 1,452 robot masks and 34 ball masks.
- 485 train frames and 120 validation frames.

Ball review candidate:

```text
outputs/review/2026-06-14/ball_mask_candidates/IMG_9933_f008995/manifest.json
```

- 124 ball frames and masks.
- One source clip only.
- Score threshold 0.50.
- Must be reviewed before it supports a final adaptation claim.

Frozen human holdout:

```text
outputs/review/2026-06-14/training_datasets/human-holdout-v1.json
```

Selection fingerprint:

```text
9fc753bc4b6ecadd2f49855e90b40188360b215ea4b4dd22a64ae44eb8adfba9
```

The holdout contains no copied pseudo detections. Its 24 frames remain pending
human annotation.

## Experiment Order

### Experiment A: Robot Segmentation

Use the four-video mask pool and train only after exporting it to the
SAM3-specific image/noun-phrase format.

Success conditions:

- Validation mask quality improves on the held-out video.
- Robot coverage does not decrease in `compare-qa`.
- Team assignment is evaluated separately; segmentation quality alone does not
  prove blue/yellow classification.

### Experiment B: Ball Segmentation

Use the 124-frame ball candidate only after visual review. Keep this experiment
separate from the robot-heavy pool or use balanced sampling.

Success conditions:

- Ball coverage improves over the current baseline.
- Maximum frame-to-frame jump does not regress.
- The improvement transfers to a different clip, not only the source video.

## Official Config Checklist

Generate a config from the official image fine-tuning YAML that matches the
installed SAM3 commit. `prepare-sam3-finetune` patches the installed template
instead of maintaining a copied 500-line config in this repository.

For segmentation:

1. Install the official training extras with `pip install -e ".[train]"`.
2. Set the dataset image roots and train/validation annotation JSON paths.
3. Set the BPE asset path.
4. Set the pretrained SAM3 checkpoint path.
5. Enable segmentation in the model, dataset and collator.
6. Enable the official mask and Dice loss block.
7. Set `skip_saving_ckpts: false`.
8. Start with batch size 1 and one GPU.
9. Keep a new experiment directory for every run.
10. Save the source commit, resolved Hydra config and checkpoint together.

## Evaluation Contract

Every adapted checkpoint must first be compared with the same SAM3 baseline
using the exact inferred image IDs:

```bash
PYTHONPATH=src python -m samba_futbot.cli compare-sam3-finetune \
  --ground-truth /data/coco/annotations/val.json \
  --baseline /runs/baseline/dumps/samba_futbot/coco_predictions_segm.json \
  --candidate /runs/adapted/dumps/samba_futbot/coco_predictions_segm.json \
  --out /runs/comparison/comparison.json \
  --report-out /runs/comparison/comparison.md
```

The report includes global and per-class COCO AP/AR. It restricts ground truth
to the image IDs present in both prediction files, which makes absolute metrics
valid when the official dataset loader uses `limit_ids`.

The adapted checkpoint must then be compared end to end:

```powershell
python -m samba_futbot.cli compare-qa `
  --baseline "outputs\qa\baseline-qa.json" `
  --candidate "outputs\qa\adapted-qa.json" `
  --out "outputs\qa\baseline-vs-adapted.json" `
  --report-out "outputs\qa\baseline-vs-adapted.md"
```

Do not accept a checkpoint only because training loss decreases. Promote it
only if holdout segmentation and end-to-end video QA both improve.

## First Adaptation Result

The first reproducible head-adaptation run used:

- 3 epochs.
- 64 training datapoints and 64 validation datapoints.
- 1008-pixel resolution.
- 3.5 M trainable parameters and approximately 837 M frozen parameters.
- The official SAM3 model, losses, COCO evaluator and checkpoint format.

Exact-subset segmentation comparison:

| Metric | SAM3 baseline | Adapted | Change |
|---|---:|---:|---:|
| Overall AP | 0.2364 | 0.3043 | +28.8% |
| Overall AP50 | 0.2589 | 0.3400 | +31.3% |
| Robot AP | 0.4625 | 0.5979 | +29.3% |
| Ball AP | 0.0102 | 0.0108 | +6.2% |
| Small-object AP | 0.0128 | 0.0109 | -14.7% |

This checkpoint is a successful robot-domain adaptation, but it is not yet a
strong ball checkpoint. The next experiment must increase reviewed ball masks,
balance sampling by class/source and keep an independent ball-heavy holdout.

### Balanced and Specialist Follow-up

A second mixed-class run used all 124 ball-positive train images plus 124
images without a ball annotation, while retaining robot masks. Validation used
all 128 images:

| Metric | SAM3 baseline | Balanced adapted | Change |
|---|---:|---:|---:|
| Overall AP | 0.2299 | 0.3586 | +56.0% |
| Robot AP | 0.4496 | 0.7068 | +57.2% |
| Ball AP | 0.0101 | 0.0103 | +2.0% |
| Ball AP small | 0.0592 | 0.0468 | -20.9% |

This is the strongest robot checkpoint so far, but the balance did not solve
small-ball segmentation.

A ball-only specialist trained on the same 248 images regressed:

| Metric | SAM3 baseline | Ball specialist | Change |
|---|---:|---:|---:|
| Ball AP | 0.0099 | 0.0052 | -46.8% |
| Ball AP small | 0.0587 | 0.0044 | -92.5% |
| Ball AR100 | 0.8692 | 0.8308 | -4.4% |

Do not promote the specialist checkpoint. Images without a pseudo-label are
not proven negatives, and 124 of 137 available ball masks come from one
continuous source clip. The next ball experiment is blocked on manual review,
verified negatives and additional source/scale diversity.

## Ball Review Package

Use `select-ball-review` before the next ball experiment. It creates two human
tasks:

- `verify_mask`: review or correct the existing ball candidate mask.
- `verify_absence`: confirm that a frame without a pseudo-label truly has no
  visible in-play ball.

The first generated packages are:

```text
outputs/review/2026-06-15/ball_review/ball-review-v1.json
outputs/review/2026-06-15/ball_review/ball-review-v2-dense.json
```

`v1` keeps a stricter 5-frame separation and selected 23 positive plus 40
negative-candidate frames. `v2-dense` uses a 2-frame separation and selected 40
positive plus 40 negative-candidate frames. Both group top-camera clips by
original recording, so `IMG_9933_f000000_10s`, `IMG_9933_f008995_10s` and
`IMG_9933_f017990_10s` are treated as one source.

Promote a frame into ball training only after:

1. The mask is corrected or accepted by a human reviewer.
2. Negative frames have `ball_absent_verified: true`.
3. Train, validation and test splits do not share the same original recording.

After editing the review package, run:

```bash
PYTHONPATH=src python -m samba_futbot.cli audit-ball-review \
  --review outputs/review/2026-06-15/ball_review/ball-review-v2-dense.json \
  --out outputs/review/2026-06-15/ball_review/ball-review-v2-dense-audit.json \
  --report-out outputs/review/2026-06-15/ball_review/ball-review-v2-dense-audit.md
```

Only when `ready_for_training` is `true`, export:

```bash
PYTHONPATH=src python -m samba_futbot.cli export-reviewed-ball \
  --review outputs/review/2026-06-15/ball_review/ball-review-v2-dense.json \
  --out outputs/review/2026-06-15/ball_review/reviewed-ball-manifest.json \
  --report-out outputs/review/2026-06-15/ball_review/reviewed-ball-report.json \
  --split-strategy by-source-balanced \
  --train-ratio 0.8 \
  --val-ratio 0.1
```

The exported manifest can then go through `export-coco --image-root` and the
normal `prepare-sam3-finetune` flow. `by-source-balanced` rewrites the reviewed
manifest split by original source recording, so near-duplicate clips from the
same match segment do not leak across train, validation and test. Check
`mask_ready_annotations` before starting a segmentation run: `bbox_only`
annotations are useful for detection QA, but they are not enough for SAM3 mask
adaptation.
