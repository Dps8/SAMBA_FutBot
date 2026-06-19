from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .ball_refinement import refine_ball_trajectory
from .ball_review import audit_ball_review_file, export_reviewed_ball_manifest_file
from .calibration import calibration_quality_report, render_calibration_frame, write_calibration_quality
from .config import deep_get, load_config
from .color_ball import detect_orange_ball
from .color_goals import detect_colored_goals, enforce_goal_frame_constraints
from .color_robots import detect_dark_robots
from .comparison import (
    compare_qa_files,
    write_qa_comparison_json,
    write_qa_comparison_markdown,
)
from .dataset import export_frame_dataset
from .dataset import merge_frame_dataset_manifests
from .dataset_curation import curate_dataset_manifest_file
from .dataset_quality import analyze_dataset_quality_file, write_dataset_quality_markdown
from .drive import (
    download_manifest_files,
    download_drive_file,
    find_manifest_item,
    index_public_folder,
    load_manifest,
    write_manifest,
)
from .events import confirm_goal_candidates, detect_events, summarize_events
from .field_analysis import (
    analyze_field_tracks,
    load_field_calibration,
    write_field_robot_csv,
    write_field_trajectory_csv,
    write_field_zone_control_csv,
)
from .field_viz import render_field_map
from .finetune_config import prepare_sam3_finetune_config
from .finetune_evaluation import evaluate_and_compare_coco_files
from .finetune_preflight import write_finetune_preflight
from .game_state import (
    classify_frame_states,
    detect_external_events,
    detect_game_segments,
    filter_detections_to_playable_frames,
    play_mask_from_segments,
    playable_frames_from_game_state,
)
from .holdout import select_ball_review_set_file, select_human_holdout_file
from .heatmap import render_activity_heatmap
from .io_utils import read_detections, read_json, write_detections, write_events, write_json
from .metrics import summarize_tracks
from .play_state import ROBOT_CLASSES
from .pseudolabels import export_pseudolabel_candidates
from .qa import (
    collect_quality_reports,
    evaluate_run_quality,
    write_quality_index_json,
    write_quality_index_markdown,
    write_quality_json,
    write_quality_markdown,
)
from .reporting import write_run_report
from .robot_filter import filter_robot_detections
from .sam3_adapter import run_sam3_video
from .sam3_training_export import export_sam3_training
from .showcase import (
    collect_showcase_candidates,
    write_showcase_json,
    write_showcase_markdown,
)
from .situations import analyze_situations
from .submission import write_submission_report
from .team import assign_marker_teams_from_video, assign_robot_teams_from_video
from .team_embedding import (
    align_clusters_to_teams,
    assign_embedding_teams,
    cluster_track_embeddings,
    embedding_team_report,
    extract_dinov2_track_embeddings,
)
from .team_quality import analyze_team_quality_file, write_team_quality_markdown
from .track_filter import filter_tracking_artifacts
from .tracking import track_detections
from .training_export import export_balanced_coco_subset, export_coco_detection
from .types import Event
from .video import extract_video_clip, sample_frames, video_info
from .visualize import render_demo_video
from .windowing import (
    filter_edge_ball_detections,
    merge_detection_files,
    offset_detections,
    parse_int_list,
    write_window_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="samba-futbot")
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index-drive", help="Indexar carpeta publica de Google Drive.")
    index.add_argument("--config", default=None)
    index.add_argument("--folder-id", default=None)
    index.add_argument("--root-name", default="Meta_Glasses")
    index.add_argument("--out", default="data/manifests/drive_index.json")
    index.set_defaults(func=cmd_index_drive)

    download = sub.add_parser("download", help="Descargar un archivo del manifest.")
    download.add_argument("--manifest", default="data/manifests/drive_index.json")
    download.add_argument("--id", default=None)
    download.add_argument("--name", default=None)
    download.add_argument("--out-dir", default="data/raw")
    download.add_argument("--max-bytes", type=int, default=None)
    download.add_argument("--force", action="store_true")
    download.set_defaults(func=cmd_download)

    download_all = sub.add_parser("download-all", help="Descargar todos los videos del manifest.")
    download_all.add_argument("--manifest", default="data/manifests/drive_index.json")
    download_all.add_argument("--out-dir", default="data/raw")
    download_all.add_argument("--extensions", default=".mov,.mp4,.avi,.mkv")
    download_all.add_argument("--strip-root", action="store_true")
    download_all.add_argument("--limit", type=int, default=None)
    download_all.add_argument("--force", action="store_true")
    download_all.set_defaults(func=cmd_download_all)

    sample = sub.add_parser("sample-frames", help="Extraer frames para calibracion.")
    sample.add_argument("--video", required=True)
    sample.add_argument("--out-dir", required=True)
    sample.add_argument("--every", type=float, default=None)
    sample.add_argument("--stride", type=int, default=None)
    sample.add_argument("--max-frames", type=int, default=None)
    sample.set_defaults(func=cmd_sample_frames)

    run = sub.add_parser("run-sam3", help="Ejecutar SAM 3/SAM 3.1 sobre un video.")
    run.add_argument("--config", default="config/default.yml")
    run.add_argument("--video", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--backend", choices=["official", "transformers"], default=None)
    run.add_argument("--model-id", default=None)
    run.add_argument("--max-frames", type=int, default=None)
    run.add_argument("--stride", type=int, default=None)
    run.add_argument("--prompt-frame-index", type=int, default=None)
    run.add_argument("--use-fa3", action=argparse.BooleanOptionalAction, default=None)
    run.add_argument("--max-num-objects", type=int, default=None)
    run.add_argument("--offload-video-to-cpu", action=argparse.BooleanOptionalAction, default=None)
    run.add_argument("--offload-state-to-cpu", action=argparse.BooleanOptionalAction, default=None)
    run.set_defaults(func=cmd_run_sam3)

    sweep = sub.add_parser("run-sam3-sweep", help="Ejecutar SAM 3 por ventanas.")
    sweep.add_argument("--config", default="config/default.yml")
    sweep.add_argument("--video", required=True)
    sweep.add_argument("--out", required=True)
    sweep.add_argument("--prompt-frames", required=True, help="Lista CSV de frames ancla.")
    sweep.add_argument("--window-size", type=int, default=None)
    sweep.add_argument("--end-frame", type=int, default=None)
    sweep.add_argument("--classes", default=None, help="Clases CSV a conservar del config.")
    sweep.add_argument("--threshold", type=float, default=None)
    sweep.add_argument("--dedupe-iou", type=float, default=0.9)
    sweep.add_argument("--backend", choices=["official", "transformers"], default=None)
    sweep.add_argument("--model-id", default=None)
    sweep.add_argument("--use-fa3", action=argparse.BooleanOptionalAction, default=None)
    sweep.add_argument("--max-num-objects", type=int, default=None)
    sweep.add_argument("--offload-video-to-cpu", action=argparse.BooleanOptionalAction, default=None)
    sweep.add_argument("--offload-state-to-cpu", action=argparse.BooleanOptionalAction, default=None)
    sweep.add_argument("--clip-windows", action=argparse.BooleanOptionalAction, default=True)
    sweep.set_defaults(func=cmd_run_sam3_sweep)

    merge = sub.add_parser("merge-detections", help="Fusionar JSONL de detecciones.")
    merge.add_argument("--inputs", required=True, help="Archivos JSONL separados por coma.")
    merge.add_argument("--out", required=True)
    merge.add_argument("--dedupe-iou", type=float, default=0.9)
    merge.set_defaults(func=cmd_merge_detections)

    filter_dets = sub.add_parser("filter-detections", help="Filtrar falsos positivos geometricos.")
    filter_dets.add_argument("--detections", required=True)
    filter_dets.add_argument("--out", required=True)
    filter_dets.add_argument("--frame-width", type=int, required=True)
    filter_dets.add_argument("--frame-height", type=int, required=True)
    filter_dets.add_argument("--ball-border-margin-px", type=float, default=4.0)
    filter_dets.set_defaults(func=cmd_filter_detections)

    filter_tracks = sub.add_parser(
        "filter-track-artifacts",
        help="Filtrar artefactos geometricos del fallback cromatico de robots.",
    )
    filter_tracks.add_argument("--tracks", required=True)
    filter_tracks.add_argument("--out", required=True)
    filter_tracks.add_argument("--robot-fallback-min-area", type=float, default=0.0)
    filter_tracks.add_argument(
        "--robot-fallback-max-area", type=float, default=float("inf")
    )
    filter_tracks.add_argument("--robot-fallback-max-extent", type=float, default=1.0)
    filter_tracks.add_argument(
        "--robot-fallback-max-aspect-ratio", type=float, default=float("inf")
    )
    filter_tracks.set_defaults(func=cmd_filter_track_artifacts)

    pseudolabels = sub.add_parser(
        "export-pseudolabels",
        help="Exportar candidatos de pseudo-etiquetas desde detecciones SAM3.",
    )
    pseudolabels.add_argument("--detections", required=True)
    pseudolabels.add_argument("--out", required=True)
    pseudolabels.add_argument("--classes", default="robots,ball,goal_blue,goal_yellow")
    pseudolabels.add_argument("--min-score", type=float, default=0.60)
    pseudolabels.add_argument("--min-area", type=float, default=1.0)
    pseudolabels.add_argument("--require-mask", action=argparse.BooleanOptionalAction, default=True)
    pseudolabels.add_argument(
        "--root",
        default=None,
        help="Base para escribir mask_path relativo; por defecto usa la carpeta del JSONL.",
    )
    pseudolabels.set_defaults(func=cmd_export_pseudolabels)

    frame_dataset = sub.add_parser(
        "export-frame-dataset",
        help="Exportar frames/crops auditables desde video y detecciones.",
    )
    frame_dataset.add_argument("--video", required=True)
    frame_dataset.add_argument("--detections", required=True)
    frame_dataset.add_argument("--out-dir", required=True)
    frame_dataset.add_argument("--classes", default="robots,ball,goal_blue,goal_yellow")
    frame_dataset.add_argument("--min-score", type=float, default=0.60)
    frame_dataset.add_argument("--frame-stride", type=int, default=1)
    frame_dataset.add_argument("--max-frames", type=int, default=None)
    frame_dataset.add_argument("--crop", action=argparse.BooleanOptionalAction, default=True)
    frame_dataset.add_argument("--crop-padding-px", type=int, default=8)
    frame_dataset.add_argument("--max-detections-per-class-per-frame", type=int, default=8)
    frame_dataset.add_argument(
        "--split-strategy",
        choices=["by-video", "by-frame"],
        default="by-video",
    )
    frame_dataset.add_argument("--train-ratio", type=float, default=0.80)
    frame_dataset.add_argument("--val-ratio", type=float, default=0.10)
    frame_dataset.set_defaults(func=cmd_export_frame_dataset)

    merge_dataset = sub.add_parser(
        "merge-frame-datasets",
        help="Unir manifests de export-frame-dataset en un dataset multi-video.",
    )
    merge_dataset.add_argument("--manifests", required=True, help="Rutas separadas por coma.")
    merge_dataset.add_argument("--out", required=True)
    merge_dataset.add_argument(
        "--split-strategy",
        choices=["preserve", "by-source-balanced"],
        default="preserve",
    )
    merge_dataset.add_argument("--train-ratio", type=float, default=0.80)
    merge_dataset.add_argument("--val-ratio", type=float, default=0.10)
    merge_dataset.set_defaults(func=cmd_merge_frame_datasets)

    coco = sub.add_parser(
        "export-coco",
        help="Convertir manifest de export-frame-dataset a COCO detection.",
    )
    coco.add_argument("--manifest", required=True)
    coco.add_argument("--out-dir", required=True)
    coco.add_argument("--image-root", default=None)
    coco.set_defaults(func=cmd_export_coco)

    balanced_coco = sub.add_parser(
        "balance-coco",
        help="Crear subconjunto COCO de entrenamiento balanceado por clase objetivo.",
    )
    balanced_coco.add_argument("--annotations", required=True)
    balanced_coco.add_argument("--out", required=True)
    balanced_coco.add_argument("--focus-classes", default="ball")
    balanced_coco.add_argument("--negative-ratio", type=float, default=1.0)
    balanced_coco.add_argument("--max-positive-images", type=int, default=None)
    balanced_coco.add_argument("--seed", type=int, default=123)
    balanced_coco.add_argument(
        "--focus-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    balanced_coco.set_defaults(func=cmd_balance_coco)

    finetune_config = sub.add_parser(
        "prepare-sam3-finetune",
        help="Derivar un YAML de segmentacion desde la configuracion oficial de SAM3.",
    )
    finetune_config.add_argument("--template", required=True)
    finetune_config.add_argument("--out", required=True)
    finetune_config.add_argument("--data-root", required=True)
    finetune_config.add_argument("--train-json", required=True)
    finetune_config.add_argument("--val-json", required=True)
    finetune_config.add_argument("--experiment-dir", required=True)
    finetune_config.add_argument("--bpe-path", required=True)
    finetune_config.add_argument("--epochs", type=int, default=1)
    finetune_config.add_argument("--train-limit", type=int, default=8)
    finetune_config.add_argument("--val-limit", type=int, default=8)
    finetune_config.add_argument("--resolution", type=int, default=1008)
    finetune_config.add_argument("--num-workers", type=int, default=0)
    finetune_config.add_argument("--class-prompts", default=None)
    finetune_config.add_argument("--mode", choices=["train", "val"], default="train")
    finetune_config.set_defaults(func=cmd_prepare_sam3_finetune)

    finetune_compare = sub.add_parser(
        "compare-sam3-finetune",
        help="Comparar predicciones COCO de SAM3 base y adaptado sobre el subconjunto exacto.",
    )
    finetune_compare.add_argument("--ground-truth", required=True)
    finetune_compare.add_argument("--baseline", required=True)
    finetune_compare.add_argument("--candidate", required=True)
    finetune_compare.add_argument("--out", required=True)
    finetune_compare.add_argument("--report-out", required=True)
    finetune_compare.add_argument("--iou-type", choices=["segm", "bbox"], default="segm")
    finetune_compare.set_defaults(func=cmd_compare_sam3_finetune)

    sam3_training = sub.add_parser(
        "export-sam3-training",
        help="Exportar manifest al formato oficial imagen-frase de SAM3.",
    )
    sam3_training.add_argument("--manifest", required=True)
    sam3_training.add_argument("--out-dir", required=True)
    sam3_training.add_argument("--image-root", default=None)
    sam3_training.add_argument("--class-prompts", default=None)
    sam3_training.add_argument(
        "--include-negatives",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    sam3_training.add_argument(
        "--negative-classes",
        default="ball,robots,goal_blue,goal_yellow",
    )
    sam3_training.add_argument("--max-negative-classes-per-image", type=int, default=1)
    sam3_training.add_argument("--max-negative-pairs-per-class", type=int, default=100)
    sam3_training.set_defaults(func=cmd_export_sam3_training)

    preflight = sub.add_parser(
        "finetune-preflight",
        help="Validar entorno y datos antes de entrenar SAM3 oficial.",
    )
    preflight.add_argument("--sam3-root", required=True)
    preflight.add_argument("--checkpoint", required=True)
    preflight.add_argument("--train-json", required=True)
    preflight.add_argument("--val-json", required=True)
    preflight.add_argument("--train-images", required=True)
    preflight.add_argument("--val-images", required=True)
    preflight.add_argument("--out", required=True)
    preflight.add_argument(
        "--check-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    preflight.add_argument("--python-executable", default=sys.executable)
    preflight.set_defaults(func=cmd_finetune_preflight)

    dataset_quality = sub.add_parser(
        "dataset-quality",
        help="Auditar calidad de manifest de dataset antes de adaptacion/fine-tuning.",
    )
    dataset_quality.add_argument("--manifest", required=True)
    dataset_quality.add_argument("--out", required=True)
    dataset_quality.add_argument("--report-out", default=None)
    dataset_quality.add_argument("--low-score-threshold", type=float, default=0.60)
    dataset_quality.add_argument("--max-review-examples", type=int, default=25)
    dataset_quality.set_defaults(func=cmd_dataset_quality)

    curate_dataset = sub.add_parser(
        "curate-dataset",
        help="Filtrar y deduplicar un manifest antes de adaptacion/fine-tuning.",
    )
    curate_dataset.add_argument("--manifest", required=True)
    curate_dataset.add_argument("--out", required=True)
    curate_dataset.add_argument("--report-out", required=True)
    curate_dataset.add_argument("--classes", default="robots,ball,goal_blue,goal_yellow")
    curate_dataset.add_argument("--min-score", type=float, default=0.60)
    curate_dataset.add_argument("--review-exclusions", default=None)
    curate_dataset.add_argument(
        "--drop-empty-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    curate_dataset.add_argument(
        "--deduplicate-source-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    curate_dataset.set_defaults(func=cmd_curate_dataset)

    holdout = sub.add_parser(
        "select-holdout",
        help="Seleccionar frames independientes para anotacion humana.",
    )
    holdout.add_argument("--manifest", required=True)
    holdout.add_argument("--out", required=True)
    holdout.add_argument("--report-out", required=True)
    holdout.add_argument("--max-frames", type=int, default=24)
    holdout.add_argument("--preferred-split", default="val")
    holdout.add_argument("--seed", type=int, default=2026)
    holdout.set_defaults(func=cmd_select_holdout)

    ball_review = sub.add_parser(
        "select-ball-review",
        help="Seleccionar positivos y negativos candidatos para revisar pelota.",
    )
    ball_review.add_argument("--manifest", required=True)
    ball_review.add_argument("--out", required=True)
    ball_review.add_argument("--report-out", required=True)
    ball_review.add_argument("--positive-frames", type=int, default=40)
    ball_review.add_argument("--negative-frames", type=int, default=40)
    ball_review.add_argument("--seed", type=int, default=2027)
    ball_review.add_argument("--class-name", default="ball")
    ball_review.add_argument(
        "--source-group-mode",
        choices=["video", "original-video"],
        default="original-video",
    )
    ball_review.add_argument("--min-frame-gap", type=int, default=5)
    ball_review.set_defaults(func=cmd_select_ball_review)

    audit_ball_review = sub.add_parser(
        "audit-ball-review",
        help="Auditar un paquete de revision humana de pelota.",
    )
    audit_ball_review.add_argument("--review", required=True)
    audit_ball_review.add_argument("--out", required=True)
    audit_ball_review.add_argument("--report-out", default=None)
    audit_ball_review.add_argument("--class-name", default="ball")
    audit_ball_review.set_defaults(func=cmd_audit_ball_review)

    export_ball_review = sub.add_parser(
        "export-reviewed-ball",
        help="Convertir revision humana de pelota en manifest entrenable.",
    )
    export_ball_review.add_argument("--review", required=True)
    export_ball_review.add_argument("--out", required=True)
    export_ball_review.add_argument("--report-out", required=True)
    export_ball_review.add_argument("--class-name", default="ball")
    export_ball_review.add_argument(
        "--require-complete",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    export_ball_review.add_argument(
        "--include-verified-absence",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    export_ball_review.add_argument(
        "--split-strategy",
        choices=["preserve", "by-source-balanced"],
        default="preserve",
    )
    export_ball_review.add_argument("--train-ratio", type=float, default=0.8)
    export_ball_review.add_argument("--val-ratio", type=float, default=0.1)
    export_ball_review.set_defaults(func=cmd_export_reviewed_ball)

    color_ball = sub.add_parser(
        "detect-orange-ball",
        help="Detectar pelota por color/forma configurable.",
    )
    color_ball.add_argument("--video", required=True)
    color_ball.add_argument("--out", required=True)
    color_ball.add_argument("--max-frames", type=int, default=None)
    color_ball.add_argument("--min-area", type=float, default=80.0)
    color_ball.add_argument("--max-area", type=float, default=2200.0)
    color_ball.add_argument("--min-circularity", type=float, default=0.45)
    color_ball.add_argument("--color-profile", default="orange")
    color_ball.add_argument("--hsv-lower", default=None, help="HSV lower bound as H,S,V.")
    color_ball.add_argument("--hsv-upper", default=None, help="HSV upper bound as H,S,V.")
    color_ball.add_argument("--context-detections", default=None)
    color_ball.add_argument("--robot-margin-px", type=float, default=8.0)
    color_ball.add_argument("--border-margin-px", type=float, default=4.0)
    color_ball.add_argument("--max-per-frame", type=int, default=1)
    color_ball.set_defaults(func=cmd_detect_orange_ball)

    dark_robots = sub.add_parser(
        "detect-dark-robots",
        help="Recuperar robots oscuros en vista superior con color/forma.",
    )
    dark_robots.add_argument("--video", required=True)
    dark_robots.add_argument("--out", required=True)
    dark_robots.add_argument("--max-frames", type=int, default=None)
    dark_robots.add_argument("--min-area", type=float, default=350.0)
    dark_robots.add_argument("--max-area", type=float, default=18000.0)
    dark_robots.add_argument("--min-extent", type=float, default=0.18)
    dark_robots.add_argument("--max-extent", type=float, default=0.92)
    dark_robots.add_argument("--min-circularity", type=float, default=0.12)
    dark_robots.add_argument("--hsv-lower", default=None, help="HSV lower bound as H,S,V.")
    dark_robots.add_argument("--hsv-upper", default=None, help="HSV upper bound as H,S,V.")
    dark_robots.add_argument("--field-detections", default=None)
    dark_robots.add_argument("--field-margin-px", type=float, default=8.0)
    dark_robots.add_argument("--border-margin-px", type=float, default=4.0)
    dark_robots.add_argument("--min-center-y-ratio", type=float, default=0.0)
    dark_robots.add_argument("--max-center-y-ratio", type=float, default=1.0)
    dark_robots.add_argument("--merge-distance-px", type=float, default=32.0)
    dark_robots.add_argument("--max-per-frame", type=int, default=6)
    dark_robots.add_argument("--box-expand-x-px", type=float, default=0.0)
    dark_robots.add_argument("--box-expand-top-px", type=float, default=0.0)
    dark_robots.add_argument("--box-expand-bottom-px", type=float, default=0.0)
    dark_robots.set_defaults(func=cmd_detect_dark_robots)

    refine_ball = sub.add_parser("refine-ball", help="Refinar trayectoria temporal de pelota.")
    refine_ball.add_argument("--detections", required=True)
    refine_ball.add_argument("--out", required=True)
    refine_ball.add_argument("--max-jump-px", type=float, default=45.0)
    refine_ball.add_argument("--preferred-area", type=float, default=650.0)
    refine_ball.add_argument("--score-weight", type=float, default=2.0)
    refine_ball.add_argument("--area-weight", type=float, default=1.0)
    refine_ball.add_argument("--max-candidates-per-frame", type=int, default=6)
    refine_ball.set_defaults(func=cmd_refine_ball)

    process = sub.add_parser("process-video", help="Pipeline completo: SAM3, merge, tracking, metricas y demo.")
    process.add_argument("--config", default="config/default.yml")
    process.add_argument("--video", required=True)
    process.add_argument("--results-dir", default="outputs")
    process.add_argument("--suffix", default="full-windowed-orange-v2-clipped")
    process.add_argument("--field-window-size", type=int, default=120)
    process.add_argument("--ball-window-size", type=int, default=120)
    process.add_argument("--field-step", type=int, default=120)
    process.add_argument("--ball-step", type=int, default=120)
    process.add_argument("--field-start", type=int, default=0)
    process.add_argument("--ball-start", type=int, default=0)
    process.add_argument("--field-threshold", type=float, default=0.45)
    process.add_argument("--ball-threshold", type=float, default=0.05)
    process.add_argument("--field-dedupe-iou", type=float, default=0.90)
    process.add_argument("--ball-dedupe-iou", type=float, default=0.70)
    process.add_argument("--goals", action=argparse.BooleanOptionalAction, default=True)
    process.add_argument("--human-context", action=argparse.BooleanOptionalAction, default=None)
    process.add_argument("--color-goals", action=argparse.BooleanOptionalAction, default=None)
    process.add_argument("--merge-dedupe-iou", type=float, default=0.85)
    process.add_argument("--track-iou-threshold", type=float, default=0.05)
    process.add_argument("--track-max-age", type=int, default=20)
    process.add_argument("--tracker-backend", choices=["iou", "bytetrack"], default=None)
    process.add_argument("--track-activation-threshold", type=float, default=None)
    process.add_argument("--track-minimum-matching-threshold", type=float, default=None)
    process.add_argument("--possession-radius-px", type=float, default=None)
    process.add_argument("--collision-radius-px", type=float, default=None)
    process.add_argument("--goal-x-margin-ratio", type=float, default=None)
    process.add_argument("--in-play-field-margin-px", type=float, default=None)
    process.add_argument("--ball-border-margin-px", type=float, default=None)
    process.add_argument("--field-calibration", default=None)
    process.add_argument("--field-analysis-out", default=None)
    process.add_argument("--field-trajectory-csv", default=None)
    process.add_argument("--field-robot-csv", default=None)
    process.add_argument("--field-zone-control-csv", default=None)
    process.add_argument("--field-map-out", default=None)
    process.add_argument("--qa", action=argparse.BooleanOptionalAction, default=True)
    process.add_argument("--qa-out", default=None)
    process.add_argument("--qa-report-out", default=None)
    process.add_argument("--run-report-out", default=None)
    process.add_argument("--run-manifest-out", default=None)
    _add_pipeline_game_state_args(process)
    process.add_argument("--field-grid-cols", type=int, default=6)
    process.add_argument("--field-grid-rows", type=int, default=4)
    process.add_argument(
        "--field-robot-anchor",
        choices=["centroid", "bottom_center"],
        default="bottom_center",
    )
    process.add_argument("--trail-length", type=int, default=45)
    process.add_argument("--max-seconds", type=float, default=None)
    process.add_argument("--clip-windows", action=argparse.BooleanOptionalAction, default=True)
    process.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    process.add_argument("--render-narrative", action=argparse.BooleanOptionalAction, default=True)
    process.add_argument("--render-analysis", action=argparse.BooleanOptionalAction, default=True)
    process.add_argument("--analysis-freeze", action=argparse.BooleanOptionalAction, default=False)
    process.add_argument("--freeze-seconds", type=float, default=3.0)
    process.add_argument("--freeze-min-confidence", type=float, default=0.45)
    process.add_argument("--freeze-cooldown-frames", type=int, default=60)
    process.add_argument("--freeze-max-events", type=int, default=20)
    process.add_argument(
        "--freeze-event-types",
        default=None,
        help="CSV de eventos que activan freeze frames en render analysis.",
    )
    process.add_argument("--mask-overlay", action=argparse.BooleanOptionalAction, default=True)
    process.add_argument("--mask-alpha", type=float, default=0.35)
    process.add_argument("--label-scale", type=float, default=1.05)
    process.add_argument("--box-thickness", type=int, default=3)
    process.add_argument("--visual-hold-frames", type=int, default=12)
    process.add_argument("--show-team-labels", action=argparse.BooleanOptionalAction, default=False)
    process.set_defaults(func=cmd_process_video)

    top_camera = sub.add_parser(
        "process-top-camera",
        help="Pipeline para camara superior: SAM3 pelota + color configurable + refinamiento.",
    )
    top_camera.add_argument("--config", default="config/default.yml")
    top_camera.add_argument("--video", required=True)
    top_camera.add_argument("--results-dir", default="outputs")
    top_camera.add_argument("--suffix", default="top-hybrid-ball-v1")
    top_camera.add_argument("--field-window-size", type=int, default=120)
    top_camera.add_argument("--field-step", type=int, default=120)
    top_camera.add_argument("--field-start", type=int, default=0)
    top_camera.add_argument("--field-threshold", type=float, default=0.45)
    top_camera.add_argument("--field-dedupe-iou", type=float, default=0.90)
    top_camera.add_argument("--goals", action=argparse.BooleanOptionalAction, default=True)
    top_camera.add_argument(
        "--human-context",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    top_camera.add_argument("--color-goals", action=argparse.BooleanOptionalAction, default=None)
    top_camera.add_argument("--sam3-ball", action=argparse.BooleanOptionalAction, default=None)
    top_camera.add_argument("--color-ball", action=argparse.BooleanOptionalAction, default=None)
    top_camera.add_argument("--ball-window-size", type=int, default=120)
    top_camera.add_argument("--ball-step", type=int, default=120)
    top_camera.add_argument("--ball-start", type=int, default=0)
    top_camera.add_argument("--ball-threshold", type=float, default=0.05)
    top_camera.add_argument("--ball-dedupe-iou", type=float, default=0.70)
    top_camera.add_argument("--merge-dedupe-iou", type=float, default=0.85)
    top_camera.add_argument("--track-iou-threshold", type=float, default=0.05)
    top_camera.add_argument("--track-max-age", type=int, default=20)
    top_camera.add_argument("--tracker-backend", choices=["iou", "bytetrack"], default=None)
    top_camera.add_argument("--track-activation-threshold", type=float, default=None)
    top_camera.add_argument("--track-minimum-matching-threshold", type=float, default=None)
    top_camera.add_argument("--possession-radius-px", type=float, default=None)
    top_camera.add_argument("--collision-radius-px", type=float, default=None)
    top_camera.add_argument("--goal-x-margin-ratio", type=float, default=None)
    top_camera.add_argument("--in-play-field-margin-px", type=float, default=None)
    top_camera.add_argument("--ball-border-margin-px", type=float, default=None)
    top_camera.add_argument("--orange-min-area", type=float, default=300.0)
    top_camera.add_argument("--orange-max-area", type=float, default=2200.0)
    top_camera.add_argument("--orange-min-circularity", type=float, default=0.45)
    top_camera.add_argument("--orange-robot-margin-px", type=float, default=8.0)
    top_camera.add_argument("--orange-max-per-frame", type=int, default=6)
    top_camera.add_argument("--ball-color-profile", default=None)
    top_camera.add_argument("--ball-hsv-lower", default=None, help="HSV lower bound as H,S,V.")
    top_camera.add_argument("--ball-hsv-upper", default=None, help="HSV upper bound as H,S,V.")
    top_camera.add_argument(
        "--robot-color-recovery",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Agregar recuperacion HSV/forma de robots oscuros como fuente opcional.",
    )
    top_camera.add_argument("--robot-recovery-min-area", type=float, default=800.0)
    top_camera.add_argument("--robot-recovery-max-area", type=float, default=18000.0)
    top_camera.add_argument("--robot-recovery-min-circularity", type=float, default=0.30)
    top_camera.add_argument("--robot-recovery-hsv-lower", default=None)
    top_camera.add_argument("--robot-recovery-hsv-upper", default="179,255,125")
    top_camera.add_argument("--robot-recovery-min-center-y-ratio", type=float, default=0.0)
    top_camera.add_argument("--robot-recovery-max-center-y-ratio", type=float, default=1.0)
    top_camera.add_argument("--robot-recovery-merge-distance-px", type=float, default=32.0)
    top_camera.add_argument("--robot-recovery-max-per-frame", type=int, default=6)
    top_camera.add_argument("--robot-recovery-box-expand-x-px", type=float, default=36.0)
    top_camera.add_argument("--robot-recovery-box-expand-top-px", type=float, default=90.0)
    top_camera.add_argument("--robot-recovery-box-expand-bottom-px", type=float, default=20.0)
    top_camera.add_argument("--robot-filter", action=argparse.BooleanOptionalAction, default=None)
    top_camera.add_argument("--robot-filter-max-per-frame", type=int, default=None)
    top_camera.add_argument("--robot-filter-min-area", type=float, default=None)
    top_camera.add_argument("--robot-filter-max-area-ratio", type=float, default=None)
    top_camera.add_argument("--robot-filter-containment-threshold", type=float, default=None)
    top_camera.add_argument("--robot-filter-iou-threshold", type=float, default=None)
    top_camera.add_argument("--robot-filter-min-center-distance-px", type=float, default=None)
    top_camera.add_argument("--robot-filter-protect-near-ball-px", type=float, default=None)
    top_camera.add_argument("--refine-max-jump-px", type=float, default=35.0)
    top_camera.add_argument("--refine-preferred-area", type=float, default=680.0)
    top_camera.add_argument("--refine-score-weight", type=float, default=2.0)
    top_camera.add_argument("--refine-area-weight", type=float, default=1.0)
    top_camera.add_argument("--refine-max-candidates-per-frame", type=int, default=6)
    top_camera.add_argument("--field-calibration", default=None)
    top_camera.add_argument("--field-analysis-out", default=None)
    top_camera.add_argument("--field-trajectory-csv", default=None)
    top_camera.add_argument("--field-robot-csv", default=None)
    top_camera.add_argument("--field-zone-control-csv", default=None)
    top_camera.add_argument("--field-map-out", default=None)
    top_camera.add_argument("--qa", action=argparse.BooleanOptionalAction, default=True)
    top_camera.add_argument("--qa-out", default=None)
    top_camera.add_argument("--qa-report-out", default=None)
    top_camera.add_argument("--run-report-out", default=None)
    top_camera.add_argument("--run-manifest-out", default=None)
    _add_pipeline_game_state_args(top_camera)
    top_camera.add_argument("--field-grid-cols", type=int, default=6)
    top_camera.add_argument("--field-grid-rows", type=int, default=4)
    top_camera.add_argument(
        "--field-robot-anchor",
        choices=["centroid", "bottom_center"],
        default="bottom_center",
    )
    top_camera.add_argument("--trail-length", type=int, default=45)
    top_camera.add_argument("--max-seconds", type=float, default=None)
    top_camera.add_argument("--clip-windows", action=argparse.BooleanOptionalAction, default=True)
    top_camera.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    top_camera.add_argument("--render-narrative", action=argparse.BooleanOptionalAction, default=True)
    top_camera.add_argument("--render-analysis", action=argparse.BooleanOptionalAction, default=True)
    top_camera.add_argument("--analysis-freeze", action=argparse.BooleanOptionalAction, default=False)
    top_camera.add_argument("--freeze-seconds", type=float, default=3.0)
    top_camera.add_argument("--freeze-min-confidence", type=float, default=0.45)
    top_camera.add_argument("--freeze-cooldown-frames", type=int, default=60)
    top_camera.add_argument("--freeze-max-events", type=int, default=20)
    top_camera.add_argument(
        "--freeze-event-types",
        default=None,
        help="CSV de eventos que activan freeze frames en render analysis.",
    )
    top_camera.add_argument("--mask-overlay", action=argparse.BooleanOptionalAction, default=True)
    top_camera.add_argument("--mask-alpha", type=float, default=0.35)
    top_camera.add_argument("--label-scale", type=float, default=1.05)
    top_camera.add_argument("--box-thickness", type=int, default=3)
    top_camera.add_argument("--visual-hold-frames", type=int, default=12)
    top_camera.add_argument("--show-team-labels", action=argparse.BooleanOptionalAction, default=False)
    top_camera.set_defaults(func=cmd_process_top_camera)

    field_analysis = sub.add_parser(
        "field-analysis",
        help="Convertir trayectoria de pelota a coordenadas de cancha con homografia.",
    )
    field_analysis.add_argument("--tracks", required=True)
    field_analysis.add_argument("--calibration", required=True)
    field_analysis.add_argument("--video", default=None)
    field_analysis.add_argument("--config", default="config/default.yml")
    field_analysis.add_argument("--out", required=True)
    field_analysis.add_argument("--game-state", default=None)
    field_analysis.add_argument("--csv-out", default=None)
    field_analysis.add_argument("--robot-csv-out", default=None)
    field_analysis.add_argument("--zone-control-csv-out", default=None)
    field_analysis.add_argument("--map-out", default=None)
    field_analysis.add_argument("--fps", type=float, default=None)
    field_analysis.add_argument("--possession-radius-px", type=float, default=90.0)
    field_analysis.add_argument("--in-play-field-margin-px", type=float, default=8.0)
    field_analysis.add_argument("--grid-cols", type=int, default=6)
    field_analysis.add_argument("--grid-rows", type=int, default=4)
    field_analysis.add_argument(
        "--robot-anchor",
        choices=["centroid", "bottom_center"],
        default="bottom_center",
    )
    field_analysis.set_defaults(func=cmd_field_analysis)

    field_map = sub.add_parser(
        "render-field-map",
        help="Renderizar PNG tactico desde un JSON de field-analysis.",
    )
    field_map.add_argument("--analysis", required=True)
    field_map.add_argument("--out", required=True)
    field_map.add_argument("--width", type=int, default=1200)
    field_map.set_defaults(func=cmd_render_field_map)

    calibration_frame = sub.add_parser(
        "render-calibration-frame",
        help="Extraer un frame con guias para calibrar esquinas del campo.",
    )
    calibration_frame.add_argument("--video", required=True)
    calibration_frame.add_argument("--out", required=True)
    calibration_frame.add_argument("--frame-index", type=int, default=0)
    calibration_frame.add_argument("--calibration", default=None)
    calibration_frame.set_defaults(func=cmd_render_calibration_frame)

    calibration_check = sub.add_parser(
        "calibration-check",
        help="Validar una homografia/calibracion de cancha.",
    )
    calibration_check.add_argument("--calibration", required=True)
    calibration_check.add_argument("--out", default=None)
    calibration_check.add_argument("--video", default=None)
    calibration_check.add_argument("--frame-width", type=int, default=None)
    calibration_check.add_argument("--frame-height", type=int, default=None)
    calibration_check.set_defaults(func=cmd_calibration_check)

    report = sub.add_parser("summarize-run", help="Crear resumen Markdown de una corrida.")
    report.add_argument("--out", required=True)
    report.add_argument("--title", default="SAMBA FutBot Run")
    report.add_argument("--metrics", default=None)
    report.add_argument("--events", default=None)
    report.add_argument("--field-analysis", default=None)
    report.add_argument("--qa", default=None)
    report.add_argument("--demo", default=None)
    report.add_argument("--field-map", default=None)
    report.set_defaults(func=cmd_summarize_run)

    submission = sub.add_parser(
        "submission-report",
        help="Crear reporte Markdown de evidencia final desde un batch procesado.",
    )
    submission.add_argument("--batch-root", required=True)
    submission.add_argument("--training-root", default=None)
    submission.add_argument("--out", required=True)
    submission.add_argument("--title", default="SAMBA FutBot Submission Evidence")
    submission.add_argument("--top", type=int, default=4)
    submission.set_defaults(func=cmd_submission_report)

    qa = sub.add_parser("qa-run", help="Evaluar calidad automatica de una corrida.")
    qa.add_argument("--out", required=True)
    qa.add_argument("--report-out", default=None)
    qa.add_argument("--metrics", default=None)
    qa.add_argument("--events", default=None)
    qa.add_argument("--field-analysis", default=None)
    qa.add_argument("--min-ball-coverage", type=float, default=None)
    qa.add_argument("--fail-ball-coverage", type=float, default=None)
    qa.add_argument("--max-ball-jump-px-frame", type=float, default=None)
    qa.add_argument("--fail-ball-jump-px-frame", type=float, default=None)
    qa.add_argument("--max-out-of-bounds-ratio", type=float, default=None)
    qa.add_argument("--fail-out-of-bounds-ratio", type=float, default=None)
    qa.add_argument("--max-unknown-team-ratio", type=float, default=None)
    qa.add_argument("--fail-unknown-team-ratio", type=float, default=None)
    qa.set_defaults(func=cmd_qa_run)

    qa_index = sub.add_parser(
        "qa-index",
        help="Indexar y ordenar reportes QA de una carpeta de resultados.",
    )
    qa_index.add_argument("--root", default="outputs")
    qa_index.add_argument("--pattern", default="*.json")
    qa_index.add_argument("--out", required=True)
    qa_index.add_argument("--report-out", default=None)
    qa_index.set_defaults(func=cmd_qa_index)

    compare_qa = sub.add_parser(
        "compare-qa",
        help="Comparar QA baseline contra candidato para medir mejoras o regresiones.",
    )
    compare_qa.add_argument("--baseline", required=True)
    compare_qa.add_argument("--candidate", required=True)
    compare_qa.add_argument("--out", required=True)
    compare_qa.add_argument("--report-out", default=None)
    compare_qa.set_defaults(func=cmd_compare_qa)

    showcase = sub.add_parser(
        "showcase-index",
        help="Seleccionar corridas candidatas para demo final desde reportes QA.",
    )
    showcase.add_argument("--root", default="outputs/review")
    showcase.add_argument("--out", required=True)
    showcase.add_argument("--report-out", default=None)
    showcase.add_argument("--limit", type=int, default=12)
    showcase.add_argument("--required-claims", default="ball_tracking,team_possession")
    showcase.set_defaults(func=cmd_showcase_index)

    track = sub.add_parser("track", help="Reparar/asignar IDs con IoU o ByteTrack.")
    track.add_argument("--detections", required=True)
    track.add_argument("--out", required=True)
    track.add_argument("--iou-threshold", type=float, default=0.25)
    track.add_argument("--max-age", type=int, default=12)
    track.add_argument("--backend", choices=["iou", "bytetrack"], default="iou")
    track.add_argument("--frame-rate", type=int, default=30)
    track.add_argument("--activation-threshold", type=float, default=0.05)
    track.add_argument("--minimum-matching-threshold", type=float, default=0.8)
    track.set_defaults(func=cmd_track)

    assign_teams = sub.add_parser(
        "assign-teams",
        help="Asignar equipos azul/amarillo a tracks existentes usando el video.",
    )
    assign_teams.add_argument("--video", required=True)
    assign_teams.add_argument("--tracks", required=True)
    assign_teams.add_argument("--out", required=True)
    assign_teams.add_argument("--config", default="config/default.yml")
    assign_teams.set_defaults(func=cmd_assign_teams)

    marker_teams = sub.add_parser(
        "assign-teams-marker",
        help="Asignar equipos por proporcion medida de un marcador HSV.",
    )
    marker_teams.add_argument("--video", required=True)
    marker_teams.add_argument("--tracks", required=True)
    marker_teams.add_argument("--out", required=True)
    marker_teams.add_argument("--report-out", required=True)
    marker_teams.add_argument("--marker-team", default="green_marker")
    marker_teams.add_argument("--other-team", default="unmarked")
    marker_teams.add_argument("--marker-ratio-threshold", type=float, default=0.20)
    marker_teams.add_argument("--hsv-lower", default="35,65,45")
    marker_teams.add_argument("--hsv-upper", default="90,255,255")
    marker_teams.add_argument("--samples-per-track", type=int, default=20)
    marker_teams.add_argument("--min-frame-gap", type=int, default=10)
    marker_teams.set_defaults(func=cmd_assign_teams_marker)

    embedding_teams = sub.add_parser(
        "assign-teams-embedding",
        help="Evaluar equipos por apariencia DINOv2 y alinearlos con votos HSV.",
    )
    embedding_teams.add_argument("--video", required=True)
    embedding_teams.add_argument("--tracks", required=True)
    embedding_teams.add_argument("--out", required=True)
    embedding_teams.add_argument("--report-out", required=True)
    embedding_teams.add_argument("--model-id", default="facebook/dinov2-small")
    embedding_teams.add_argument("--samples-per-track", type=int, default=8)
    embedding_teams.add_argument("--min-frame-gap", type=int, default=10)
    embedding_teams.add_argument("--batch-size", type=int, default=16)
    embedding_teams.add_argument("--device", default=None)
    embedding_teams.set_defaults(func=cmd_assign_teams_embedding)

    events = sub.add_parser("events", help="Detectar eventos de juego.")
    events.add_argument("--tracks", required=True)
    events.add_argument("--out", required=True)
    events.add_argument("--game-state", default=None)
    events.add_argument("--summary-out", default=None)
    events.add_argument("--possession-radius-px", type=float, default=90)
    events.add_argument("--collision-radius-px", type=float, default=55)
    events.add_argument("--frame-width", type=int, default=None)
    events.add_argument("--in-play-field-margin-px", type=float, default=8.0)
    events.set_defaults(func=cmd_events)

    event_summary = sub.add_parser(
        "event-summary",
        help="Resumir eventos en marcador candidato y conteos deportivos.",
    )
    event_summary.add_argument("--events", required=True)
    event_summary.add_argument("--out", required=True)
    event_summary.set_defaults(func=cmd_event_summary)

    validate_goals = sub.add_parser(
        "validate-goals",
        help="Validar goal_candidate temporalmente usando tracks existentes.",
    )
    validate_goals.add_argument("--tracks", required=True)
    validate_goals.add_argument("--events", required=True)
    validate_goals.add_argument("--out", required=True)
    validate_goals.add_argument("--config", default="config/default.yml")
    validate_goals.set_defaults(func=cmd_validate_goals)

    team_quality = sub.add_parser(
        "team-quality",
        help="Auditar cobertura y consistencia temporal de equipos en tracks.",
    )
    team_quality.add_argument("--tracks", required=True)
    team_quality.add_argument("--out", required=True)
    team_quality.add_argument("--report-out", default=None)
    team_quality.add_argument("--unknown-ratio-threshold", type=float, default=0.20)
    team_quality.add_argument("--ambiguous-track-dominance", type=float, default=0.75)
    team_quality.add_argument("--min-ambiguous-track-samples", type=int, default=2)
    team_quality.add_argument("--min-frame-team-coverage", type=float, default=0.80)
    team_quality.add_argument("--max-dominant-team-ratio", type=float, default=0.85)
    team_quality.add_argument("--max-review-candidates", type=int, default=100)
    team_quality.set_defaults(func=cmd_team_quality)

    situations = sub.add_parser(
        "situation-analysis",
        help="Analizar posesion fina, distancias y probabilidades tacticas.",
    )
    situations.add_argument("--tracks", required=True)
    situations.add_argument("--out", required=True)
    situations.add_argument("--possession-radius-px", type=float, default=90.0)
    situations.add_argument("--dispute-margin-px", type=float, default=22.0)
    situations.add_argument("--frame-width", type=float, default=None)
    situations.set_defaults(func=cmd_situation_analysis)

    game_state = sub.add_parser(
        "game-state",
        help="Detectar estado de juego y eventos externos desde tracks.",
    )
    game_state.add_argument("--tracks", required=True)
    game_state.add_argument("--out", required=True)
    game_state.add_argument("--events-out", default=None)
    game_state.add_argument("--segments-out", default=None)
    game_state.add_argument("--possession-radius-px", type=float, default=90.0)
    game_state.add_argument("--in-play-field-margin-px", type=float, default=8.0)
    game_state.add_argument("--missing-ball-frames", type=int, default=12)
    game_state.add_argument("--robot-removed-after-frames", type=int, default=18)
    game_state.add_argument("--robot-disabled-after-frames", type=int, default=45)
    game_state.add_argument("--stationary-threshold-px", type=float, default=2.0)
    game_state.add_argument("--field-calibration", default=None)
    game_state.set_defaults(func=cmd_game_state)

    metrics = sub.add_parser("metrics", help="Calcular metricas operativas.")
    metrics.add_argument("--tracks", required=True)
    metrics.add_argument("--out", required=True)
    metrics.add_argument("--game-state", default=None)
    metrics.add_argument("--fps", type=float, default=None)
    metrics.add_argument("--possession-radius-px", type=float, default=90.0)
    metrics.add_argument("--in-play-field-margin-px", type=float, default=8.0)
    metrics.set_defaults(func=cmd_metrics)

    render = sub.add_parser("render-demo", help="Renderizar video lado a lado.")
    render.add_argument("--video", required=True)
    render.add_argument("--tracks", required=True)
    render.add_argument("--out", required=True)
    render.add_argument("--events", default=None)
    render.add_argument("--max-seconds", type=float, default=120)
    render.add_argument("--trail-length", type=int, default=45)
    render.add_argument("--style", choices=["narrative", "analysis"], default="narrative")
    render.add_argument("--analysis-freeze", action=argparse.BooleanOptionalAction, default=False)
    render.add_argument("--freeze-seconds", type=float, default=3.0)
    render.add_argument("--freeze-min-confidence", type=float, default=0.45)
    render.add_argument("--freeze-cooldown-frames", type=int, default=60)
    render.add_argument("--freeze-max-events", type=int, default=20)
    render.add_argument("--freeze-event-types", default=None)
    render.add_argument("--mask-overlay", action=argparse.BooleanOptionalAction, default=True)
    render.add_argument("--mask-alpha", type=float, default=0.35)
    render.add_argument("--label-scale", type=float, default=1.05)
    render.add_argument("--box-thickness", type=int, default=3)
    render.add_argument("--visual-hold-frames", type=int, default=12)
    render.add_argument("--show-team-labels", action=argparse.BooleanOptionalAction, default=False)
    render.set_defaults(func=cmd_render_demo)

    heatmap = sub.add_parser(
        "render-heatmap",
        help="Renderizar mapa de calor dinamico y acumulado desde tracks.",
    )
    heatmap.add_argument("--video", required=True)
    heatmap.add_argument("--tracks", required=True)
    heatmap.add_argument("--out-video", required=True)
    heatmap.add_argument("--out-image", required=True)
    heatmap.add_argument("--report-out", default=None)
    heatmap.add_argument("--class-name", default="robots")
    heatmap.add_argument("--team", default=None)
    heatmap.add_argument("--radius-px", type=int, default=28)
    heatmap.add_argument("--decay", type=float, default=0.997)
    heatmap.add_argument("--alpha", type=float, default=0.48)
    heatmap.add_argument("--max-seconds", type=float, default=None)
    heatmap.add_argument("--robot-fallback-min-area", type=float, default=0.0)
    heatmap.add_argument("--robot-fallback-max-area", type=float, default=float("inf"))
    heatmap.add_argument("--robot-fallback-max-extent", type=float, default=1.0)
    heatmap.add_argument(
        "--robot-fallback-max-aspect-ratio", type=float, default=float("inf")
    )
    heatmap.add_argument(
        "--write-every-n-frames",
        type=int,
        default=1,
        help="Escribir un cuadro por cada N cuadros procesados, acumulando todos.",
    )
    heatmap.add_argument("--output-fps", type=float, default=None)
    heatmap.add_argument("--field-calibration", default=None)
    heatmap.add_argument("--field-margin-m", type=float, default=0.0)
    heatmap.set_defaults(func=cmd_render_heatmap)

    info = sub.add_parser("video-info", help="Mostrar metadata de video.")
    info.add_argument("--video", required=True)
    info.set_defaults(func=cmd_video_info)
    return parser


def _add_pipeline_game_state_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--generate-game-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generar estado de juego y eventos externos desde tracks.",
    )
    parser.add_argument(
        "--filter-by-game-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Calcular metricas/eventos/field-analysis solo con frames in_play.",
    )
    parser.add_argument("--game-state-out", default=None)
    parser.add_argument("--external-events-out", default=None)
    parser.add_argument("--game-segments-out", default=None)
    parser.add_argument("--game-state-missing-ball-frames", type=int, default=12)
    parser.add_argument("--robot-removed-after-frames", type=int, default=18)
    parser.add_argument("--robot-disabled-after-frames", type=int, default=45)
    parser.add_argument("--stationary-threshold-px", type=float, default=2.0)


def cmd_index_drive(args: argparse.Namespace) -> None:
    config = load_config(args.config) if args.config else {}
    folder_id = args.folder_id or deep_get(config, "project.drive_root_id")
    if not folder_id:
        raise SystemExit("Provide --folder-id or project.drive_root_id in config.")
    items = index_public_folder(folder_id, root_name=args.root_name)
    write_manifest(items, args.out)
    files = [item for item in items if not item.is_folder]
    print(json.dumps({"out": args.out, "items": len(items), "files": len(files)}, indent=2))


def cmd_download(args: argparse.Namespace) -> None:
    if not args.id and not args.name:
        raise SystemExit("Provide --id or --name.")
    item = find_manifest_item(load_manifest(args.manifest), file_id=args.id, name=args.name)
    out_path = Path(args.out_dir) / item.name
    if out_path.exists() and not args.force:
        print(json.dumps({"path": str(out_path), "status": "exists"}, indent=2))
        return
    download_drive_file(item.id, out_path, max_bytes=args.max_bytes)
    print(json.dumps({"path": str(out_path), "source": item.path}, ensure_ascii=False, indent=2))


def cmd_download_all(args: argparse.Namespace) -> None:
    extensions = {ext.strip() for ext in args.extensions.split(",") if ext.strip()}

    def progress(result: dict[str, str | int]) -> None:
        print(
            f"[{result['index']}/{result['total']}] {result['status']}: "
            f"{result['path']} ({result['bytes']} bytes)",
            flush=True,
        )

    results = download_manifest_files(
        load_manifest(args.manifest),
        args.out_dir,
        extensions=extensions,
        strip_root=args.strip_root,
        limit=args.limit,
        force=args.force,
        progress=progress,
    )
    downloaded = sum(1 for result in results if result["status"] == "downloaded")
    existing = sum(1 for result in results if result["status"] == "exists")
    total_bytes = sum(int(result.get("bytes", 0)) for result in results)
    print(
        json.dumps(
            {
                "files": len(results),
                "downloaded": downloaded,
                "existing": existing,
                "bytes": total_bytes,
                "out_dir": args.out_dir,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_sample_frames(args: argparse.Namespace) -> None:
    frames = sample_frames(
        args.video,
        args.out_dir,
        every_seconds=args.every,
        stride=args.stride,
        max_frames=args.max_frames,
    )
    print(json.dumps({"frames": [str(path) for path in frames]}, indent=2))


def cmd_run_sam3(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    sam_config = config.get("sam3", {})
    detections = run_sam3_video(
        args.video,
        args.out,
        prompts=_filtered_prompts(
            sam_config.get("prompts", {}),
            None,
            max_per_class=sam_config.get("max_prompts_per_class"),
        ),
        backend=args.backend or sam_config.get("backend", "official"),
        model_id=args.model_id or sam_config.get("model_id", "facebook/sam3.1"),
        max_frames=args.max_frames or sam_config.get("max_frames"),
        stride=args.stride or sam_config.get("stride", 1),
        threshold=float(sam_config.get("threshold", 0.45)),
        mask_threshold=float(sam_config.get("mask_threshold", 0.5)),
        prompt_frame_index=(
            args.prompt_frame_index
            if args.prompt_frame_index is not None
            else int(sam_config.get("prompt_frame_index", 0))
        ),
        use_fa3=(
            args.use_fa3
            if args.use_fa3 is not None
            else bool(sam_config.get("use_fa3", False))
        ),
        max_num_objects=(
            args.max_num_objects
            if args.max_num_objects is not None
            else int(sam_config.get("max_num_objects", 16))
        ),
        offload_video_to_cpu=(
            args.offload_video_to_cpu
            if args.offload_video_to_cpu is not None
            else bool(sam_config.get("offload_video_to_cpu", True))
        ),
        offload_state_to_cpu=(
            args.offload_state_to_cpu
            if args.offload_state_to_cpu is not None
            else bool(sam_config.get("offload_state_to_cpu", True))
        ),
    )
    print(json.dumps({"detections": len(detections), "out": str(Path(args.out) / "detections.jsonl")}, indent=2))


def cmd_run_sam3_sweep(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    sam_config = config.get("sam3", {})
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_frames = parse_int_list(args.prompt_frames)
    end_frame = args.end_frame or int(video_info(args.video)["frames"])
    window_size = int(args.window_size or sam_config.get("max_frames", 300))
    prompts = _filtered_prompts(
        sam_config.get("prompts", {}),
        args.classes,
        max_per_class=sam_config.get("max_prompts_per_class"),
    )
    threshold = args.threshold if args.threshold is not None else float(sam_config.get("threshold", 0.45))

    detection_files: list[Path] = []
    windows: list[dict] = []
    prompt_strategy = str(sam_config.get("prompt_window_strategy", "all"))
    prompts_per_window = sam_config.get("prompts_per_class_per_window")
    long_video_threshold = int(sam_config.get("long_video_threshold_frames", 0))
    if long_video_threshold > 0 and end_frame >= long_video_threshold:
        prompts_per_window = sam_config.get(
            "long_video_prompts_per_class_per_window", prompts_per_window
        )
    for window_index, prompt_frame in enumerate(prompt_frames):
        if prompt_frame >= end_frame:
            continue
        window_end = min(prompt_frame + window_size, end_frame)
        window_dir = output_dir / f"window_{prompt_frame:06d}_{window_end:06d}"
        source_video: str | Path = args.video
        source_max_frames = window_end
        source_prompt_frame = prompt_frame
        frame_offset = 0
        clip_path: Path | None = None
        if args.clip_windows:
            clip_path = output_dir / "clips" / f"window_{prompt_frame:06d}_{window_end:06d}.mp4"
            extract_video_clip(
                args.video,
                clip_path,
                start_frame=prompt_frame,
                end_frame=window_end,
            )
            source_video = clip_path
            source_max_frames = window_end - prompt_frame
            source_prompt_frame = 0
            frame_offset = prompt_frame
        window_prompts = _prompts_for_window(
            prompts,
            window_index=window_index,
            strategy=prompt_strategy,
            per_class=prompts_per_window,
        )
        detections = run_sam3_video(
            source_video,
            window_dir,
            prompts=window_prompts,
            backend=args.backend or sam_config.get("backend", "official"),
            model_id=args.model_id or sam_config.get("model_id", "facebook/sam3"),
            max_frames=source_max_frames,
            stride=sam_config.get("stride", 1),
            threshold=threshold,
            mask_threshold=float(sam_config.get("mask_threshold", 0.5)),
            prompt_frame_index=source_prompt_frame,
            use_fa3=(
                args.use_fa3
                if args.use_fa3 is not None
                else bool(sam_config.get("use_fa3", False))
            ),
            max_num_objects=(
                args.max_num_objects
                if args.max_num_objects is not None
                else int(sam_config.get("max_num_objects", 16))
            ),
            offload_video_to_cpu=(
                args.offload_video_to_cpu
                if args.offload_video_to_cpu is not None
                else bool(sam_config.get("offload_video_to_cpu", True))
            ),
            offload_state_to_cpu=(
                args.offload_state_to_cpu
                if args.offload_state_to_cpu is not None
                else bool(sam_config.get("offload_state_to_cpu", True))
            ),
        )
        detections_path = window_dir / "detections.jsonl"
        detections = offset_detections(detections, frame_offset)
        write_detections(detections_path, detections)
        detection_files.append(detections_path)
        windows.append(
            {
                "prompt_frame": prompt_frame,
                "end_frame": window_end,
                "clip_path": str(clip_path) if clip_path else None,
                "detections": len(detections),
                "detections_path": str(detections_path),
                "prompts": window_prompts,
            }
        )

    merged = merge_detection_files(
        detection_files,
        output_dir / "detections.jsonl",
        iou_threshold=args.dedupe_iou,
    )
    write_window_manifest(
        output_dir / "manifest.json",
        video=args.video,
        windows=windows,
        detections=len(merged),
    )
    print(
        json.dumps(
            {
                "out": str(output_dir / "detections.jsonl"),
                "detections": len(merged),
                "windows": len(windows),
            },
            indent=2,
        )
    )


def cmd_merge_detections(args: argparse.Namespace) -> None:
    inputs = [Path(part.strip()) for part in args.inputs.split(",") if part.strip()]
    merged = merge_detection_files(inputs, args.out, iou_threshold=args.dedupe_iou)
    print(json.dumps({"out": args.out, "inputs": len(inputs), "detections": len(merged)}, indent=2))


def cmd_filter_detections(args: argparse.Namespace) -> None:
    detections = read_detections(args.detections)
    filtered = filter_edge_ball_detections(
        detections,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        border_margin_px=args.ball_border_margin_px,
    )
    write_detections(args.out, filtered)
    print(
        json.dumps(
            {
                "out": args.out,
                "input_detections": len(detections),
                "detections": len(filtered),
                "removed": len(detections) - len(filtered),
            },
            indent=2,
        )
    )


def cmd_filter_track_artifacts(args: argparse.Namespace) -> None:
    filtered, report = filter_tracking_artifacts(
        read_detections(args.tracks),
        robot_fallback_min_area=args.robot_fallback_min_area,
        robot_fallback_max_area=args.robot_fallback_max_area,
        robot_fallback_max_extent=args.robot_fallback_max_extent,
        robot_fallback_max_aspect_ratio=args.robot_fallback_max_aspect_ratio,
    )
    write_detections(args.out, filtered)
    report["out"] = args.out
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_export_pseudolabels(args: argparse.Namespace) -> None:
    classes = [part.strip() for part in args.classes.split(",") if part.strip()]
    manifest = export_pseudolabel_candidates(
        args.detections,
        args.out,
        classes=classes,
        min_score=args.min_score,
        min_area=args.min_area,
        require_mask=args.require_mask,
        root=args.root,
    )
    print(
        json.dumps(
            {
                "out": args.out,
                "input_detections": manifest["summary"]["input_detections"],
                "candidates": manifest["summary"]["candidates"],
                "candidates_by_class": manifest["summary"]["candidates_by_class"],
                "rejected": manifest["summary"]["rejected"],
            },
            indent=2,
        )
    )


def cmd_export_frame_dataset(args: argparse.Namespace) -> None:
    classes = [part.strip() for part in args.classes.split(",") if part.strip()]
    manifest = export_frame_dataset(
        video_path=args.video,
        detections_path=args.detections,
        out_dir=args.out_dir,
        classes=classes,
        min_score=args.min_score,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        crop=args.crop,
        crop_padding_px=args.crop_padding_px,
        max_detections_per_class_per_frame=args.max_detections_per_class_per_frame,
        split_strategy=args.split_strategy,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print(
        json.dumps(
            {
                "out": str(Path(args.out_dir) / "manifest.json"),
                "frames": manifest["summary"]["frames"],
                "detections": manifest["summary"]["detections"],
                "crops": manifest["summary"]["crops"],
                "detections_by_class": manifest["summary"]["detections_by_class"],
                "frames_by_split": manifest["summary"]["frames_by_split"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_merge_frame_datasets(args: argparse.Namespace) -> None:
    manifests = [part.strip() for part in args.manifests.split(",") if part.strip()]
    manifest = merge_frame_dataset_manifests(
        manifests,
        args.out,
        split_strategy=args.split_strategy,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print(
        json.dumps(
            {
                "out": args.out,
                "sources": len(manifests),
                "frames": manifest["summary"]["frames"],
                "detections": manifest["summary"]["detections"],
                "crops": manifest["summary"]["crops"],
                "detections_by_class": manifest["summary"]["detections_by_class"],
                "frames_by_split": manifest["summary"]["frames_by_split"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_dataset_quality(args: argparse.Namespace) -> None:
    report = analyze_dataset_quality_file(
        args.manifest,
        low_score_threshold=args.low_score_threshold,
        max_review_examples=args.max_review_examples,
    )
    write_json(args.out, report)
    markdown = None
    if args.report_out:
        markdown = write_dataset_quality_markdown(report, args.report_out)
    print(
        json.dumps(
            {
                "out": args.out,
                "report": str(markdown) if markdown else None,
                "summary": report["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_curate_dataset(args: argparse.Namespace) -> None:
    exclusions = None
    if args.review_exclusions:
        raw_exclusions = read_json(args.review_exclusions)
        if isinstance(raw_exclusions, list):
            exclusions = raw_exclusions
        elif isinstance(raw_exclusions, dict):
            exclusions = (
                raw_exclusions.get("review_exclusions")
                or raw_exclusions.get("review_candidates")
                or []
            )
        else:
            raise ValueError("review exclusions must be a JSON list or object")
    classes = [part.strip() for part in args.classes.split(",") if part.strip()]
    curated, report = curate_dataset_manifest_file(
        args.manifest,
        args.out,
        args.report_out,
        classes=classes,
        min_score=args.min_score,
        review_exclusions=exclusions,
        drop_empty_frames=args.drop_empty_frames,
        deduplicate_source_frames=args.deduplicate_source_frames,
    )
    print(
        json.dumps(
            {
                "out": args.out,
                "report": args.report_out,
                "summary": curated["summary"],
                "curation": report["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_select_holdout(args: argparse.Namespace) -> None:
    holdout, report = select_human_holdout_file(
        args.manifest,
        args.out,
        args.report_out,
        max_frames=args.max_frames,
        preferred_split=args.preferred_split,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "out": args.out,
                "report": args.report_out,
                "summary": holdout["summary"],
                "selection_fingerprint": report["selection_fingerprint"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_select_ball_review(args: argparse.Namespace) -> None:
    review, report = select_ball_review_set_file(
        args.manifest,
        args.out,
        args.report_out,
        positive_frames=args.positive_frames,
        negative_frames=args.negative_frames,
        seed=args.seed,
        class_name=args.class_name,
        source_group_mode=args.source_group_mode,
        min_frame_gap=args.min_frame_gap,
    )
    print(
        json.dumps(
            {
                "out": args.out,
                "report": args.report_out,
                "summary": review["summary"],
                "selection_fingerprint": report["selection_fingerprint"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_audit_ball_review(args: argparse.Namespace) -> None:
    audit = audit_ball_review_file(
        args.review,
        args.out,
        class_name=args.class_name,
        report_path=args.report_out,
    )
    print(
        json.dumps(
            {
                "out": args.out,
                "report": args.report_out,
                "ready_for_training": audit["ready_for_training"],
                "summary": audit["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_export_reviewed_ball(args: argparse.Namespace) -> None:
    manifest, report = export_reviewed_ball_manifest_file(
        args.review,
        args.out,
        args.report_out,
        class_name=args.class_name,
        require_complete=args.require_complete,
        include_verified_absence=args.include_verified_absence,
        split_strategy=args.split_strategy,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print(
        json.dumps(
            {
                "out": args.out,
                "report": args.report_out,
                "summary": manifest["summary"],
                "ready_for_training": report["audit"]["ready_for_training"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_export_coco(args: argparse.Namespace) -> None:
    paths = export_coco_detection(
        args.manifest,
        args.out_dir,
        image_root=args.image_root,
    )
    print(
        json.dumps(
            {"out_dir": args.out_dir, "annotations": {key: str(value) for key, value in paths.items()}},
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_balance_coco(args: argparse.Namespace) -> None:
    focus_classes = [
        class_name.strip()
        for class_name in args.focus_classes.split(",")
        if class_name.strip()
    ]
    summary = export_balanced_coco_subset(
        args.annotations,
        args.out,
        focus_classes=focus_classes,
        negative_ratio=args.negative_ratio,
        max_positive_images=args.max_positive_images,
        seed=args.seed,
        focus_only=args.focus_only,
    )
    print(
        json.dumps(
            {"out": args.out, "summary": summary},
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_prepare_sam3_finetune(args: argparse.Namespace) -> None:
    prompt_overrides = None
    if args.class_prompts:
        raw_prompts = read_json(args.class_prompts)
        if not isinstance(raw_prompts, dict):
            raise ValueError("class prompts JSON must be an object")
        prompt_overrides = {
            str(class_name): str(prompt)
            for class_name, prompt in raw_prompts.items()
        }
    report = prepare_sam3_finetune_config(
        args.template,
        args.out,
        data_root=args.data_root,
        train_json=args.train_json,
        val_json=args.val_json,
        experiment_dir=args.experiment_dir,
        bpe_path=args.bpe_path,
        epochs=args.epochs,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        resolution=args.resolution,
        num_workers=args.num_workers,
        prompts=prompt_overrides,
        mode=args.mode,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_compare_sam3_finetune(args: argparse.Namespace) -> None:
    comparison = evaluate_and_compare_coco_files(
        args.ground_truth,
        args.baseline,
        args.candidate,
        out_path=args.out,
        report_path=args.report_out,
        iou_type=args.iou_type,
    )
    overall = comparison["overall"]
    print(
        json.dumps(
            {
                "out": args.out,
                "report": args.report_out,
                "verdict": comparison["verdict"],
                "images": comparison["images"],
                "baseline_ap": overall["AP"]["baseline"],
                "candidate_ap": overall["AP"]["candidate"],
                "ap_delta": overall["AP"]["delta"],
                "ap_relative_change": overall["AP"]["relative_change"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_export_sam3_training(args: argparse.Namespace) -> None:
    prompt_overrides = None
    if args.class_prompts:
        raw_prompts = read_json(args.class_prompts)
        if not isinstance(raw_prompts, dict):
            raise ValueError("class prompts JSON must be an object")
        prompt_overrides = {
            str(class_name): str(prompt)
            for class_name, prompt in raw_prompts.items()
        }
    negative_classes = [
        part.strip()
        for part in args.negative_classes.split(",")
        if part.strip()
    ]
    result = export_sam3_training(
        args.manifest,
        args.out_dir,
        class_prompts=prompt_overrides,
        include_negatives=args.include_negatives,
        negative_classes=negative_classes,
        max_negative_classes_per_image=args.max_negative_classes_per_image,
        max_negative_pairs_per_class=args.max_negative_pairs_per_class,
        image_root=args.image_root,
    )
    print(
        json.dumps(
            {
                "out_dir": args.out_dir,
                "annotations": {
                    key: str(path)
                    for key, path in result["annotations"].items()
                },
                "report": str(result["report"]),
                "summary": result["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_finetune_preflight(args: argparse.Namespace) -> None:
    report = write_finetune_preflight(
        args.out,
        sam3_root=args.sam3_root,
        checkpoint=args.checkpoint,
        train_json=args.train_json,
        val_json=args.val_json,
        train_images=args.train_images,
        val_images=args.val_images,
        check_cuda=args.check_cuda,
        python_executable=args.python_executable,
    )
    print(
        json.dumps(
            {
                "out": args.out,
                "status": report["status"],
                "summary": report["summary"],
                "cuda": report["cuda"],
                "suggested_command": report["suggested_command"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_detect_orange_ball(args: argparse.Namespace) -> None:
    _, hsv_lower, hsv_upper = _resolve_ball_color_profile(
        {"default_profile": args.color_profile},
        profile=args.color_profile,
        hsv_lower=args.hsv_lower,
        hsv_upper=args.hsv_upper,
    )
    detections = detect_orange_ball(
        args.video,
        args.out,
        max_frames=args.max_frames,
        min_area=args.min_area,
        max_area=args.max_area,
        min_circularity=args.min_circularity,
        hsv_lower=hsv_lower,
        hsv_upper=hsv_upper,
        color_profile=args.color_profile,
        context_detections_path=args.context_detections,
        robot_margin_px=args.robot_margin_px,
        border_margin_px=args.border_margin_px,
        max_per_frame=args.max_per_frame,
    )
    print(json.dumps({"out": args.out, "detections": len(detections)}, indent=2))


def cmd_detect_dark_robots(args: argparse.Namespace) -> None:
    hsv_lower = _parse_hsv_bound(args.hsv_lower, fallback=(0, 0, 0))
    hsv_upper = _parse_hsv_bound(args.hsv_upper, fallback=(179, 255, 105))
    detections = detect_dark_robots(
        args.video,
        args.out,
        max_frames=args.max_frames,
        min_area=args.min_area,
        max_area=args.max_area,
        min_extent=args.min_extent,
        max_extent=args.max_extent,
        min_circularity=args.min_circularity,
        hsv_lower=hsv_lower,
        hsv_upper=hsv_upper,
        field_detections_path=args.field_detections,
        field_margin_px=args.field_margin_px,
        border_margin_px=args.border_margin_px,
        min_center_y_ratio=args.min_center_y_ratio,
        max_center_y_ratio=args.max_center_y_ratio,
        merge_distance_px=args.merge_distance_px,
        max_per_frame=args.max_per_frame,
        box_expand_x_px=args.box_expand_x_px,
        box_expand_top_px=args.box_expand_top_px,
        box_expand_bottom_px=args.box_expand_bottom_px,
    )
    by_late_frame = sum(1 for det in detections if det.frame_index >= 120)
    print(
        json.dumps(
            {
                "out": args.out,
                "detections": len(detections),
                "detections_from_frame_120": by_late_frame,
            },
            indent=2,
        )
    )


def cmd_refine_ball(args: argparse.Namespace) -> None:
    detections = read_detections(args.detections)
    refined = refine_ball_trajectory(
        detections,
        max_jump_px=args.max_jump_px,
        preferred_area=args.preferred_area,
        score_weight=args.score_weight,
        area_weight=args.area_weight,
        max_candidates_per_frame=args.max_candidates_per_frame,
    )
    write_detections(args.out, refined)
    ball_in = sum(1 for det in detections if det.class_name in {"ball", "balon", "soccer_ball"})
    ball_out = sum(1 for det in refined if det.class_name in {"ball", "balon", "soccer_ball"})
    print(
        json.dumps(
            {
                "out": args.out,
                "input_detections": len(detections),
                "detections": len(refined),
                "input_ball_detections": ball_in,
                "ball_detections": ball_out,
            },
            indent=2,
        )
    )


def cmd_process_video(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    analysis_config = config.get("analysis", {})
    tracking_config = config.get("tracking", {})
    info = video_info(args.video)
    end_frame = int(info["frames"])
    fps = float(info.get("fps") or analysis_config.get("fps", 30))
    duration_seconds = info.get("duration_seconds")
    stem = Path(args.video).stem
    results_dir = Path(args.results_dir)

    field_prompt_frames = _frame_anchors(
        end_frame,
        start=args.field_start,
        step=args.field_step,
    )
    ball_prompt_frames = _frame_anchors(
        end_frame,
        start=args.ball_start,
        step=args.ball_step,
    )

    field_out = results_dir / "detections" / f"{stem}-field-robots-sweep-clipped"
    ball_out = results_dir / "detections" / f"{stem}-ball-sweep-orange-v2-clipped"
    color_goals_out = results_dir / "detections" / f"{stem}-{args.suffix}-color-goals.jsonl"
    merged_out = results_dir / "detections" / f"{stem}-{args.suffix}" / "detections.jsonl"
    tracks_out = results_dir / "tracks" / f"{stem}-{args.suffix}-tracks.jsonl"
    metrics_out = results_dir / "metrics" / f"{stem}-{args.suffix}-metrics.json"
    events_out = results_dir / "events" / f"{stem}-{args.suffix}-events.json"
    event_summary_out = results_dir / "events" / f"{stem}-{args.suffix}-event-summary.json"

    field_detections = _run_sweep_for_process(
        args,
        classes=_context_classes(args, config),
        prompt_frames=field_prompt_frames,
        window_size=args.field_window_size,
        end_frame=end_frame,
        out=field_out,
        threshold=args.field_threshold,
        dedupe_iou=args.field_dedupe_iou,
    )
    _run_sweep_for_process(
        args,
        classes="ball",
        prompt_frames=ball_prompt_frames,
        window_size=args.ball_window_size,
        end_frame=end_frame,
        out=ball_out,
        threshold=args.ball_threshold,
        dedupe_iou=args.ball_dedupe_iou,
    )
    color_goal_detections = _detect_color_goals_for_process(
        args,
        config,
        end_frame=end_frame,
        out=color_goals_out,
        seed_detections=field_detections,
    )

    merge_inputs = [field_out / "detections.jsonl", ball_out / "detections.jsonl"]
    if color_goal_detections is not None:
        merge_inputs.append(color_goals_out)
    merged = merge_detection_files(merge_inputs, merged_out, iou_threshold=args.merge_dedupe_iou)
    ball_border_margin_px = (
        args.ball_border_margin_px
        if args.ball_border_margin_px is not None
        else float(analysis_config.get("ball_border_margin_px", 4))
    )
    field_margin_px = (
        args.in_play_field_margin_px
        if args.in_play_field_margin_px is not None
        else float(analysis_config.get("in_play_field_margin_px", 8))
    )
    filtered = filter_edge_ball_detections(
        merged,
        frame_width=int(info.get("width") or 0) or None,
        frame_height=int(info.get("height") or 0) or None,
        border_margin_px=ball_border_margin_px,
    )
    edge_ball_filter_removed = len(merged) - len(filtered)
    if len(filtered) != len(merged):
        write_detections(merged_out, filtered)
    merged = filtered
    constrained = _enforce_goal_constraints_for_process(merged, config)
    goal_constraints_removed = len(merged) - len(constrained)
    if goal_constraints_removed:
        write_detections(merged_out, constrained)
    merged = constrained
    tracked = track_detections(
        merged,
        iou_threshold=args.track_iou_threshold,
        max_age=args.track_max_age,
        backend=args.tracker_backend or str(tracking_config.get("backend", "iou")),
        frame_rate=max(1, round(fps)),
        track_activation_threshold=(
            args.track_activation_threshold
            if args.track_activation_threshold is not None
            else float(tracking_config.get("track_activation_threshold", 0.05))
        ),
        minimum_matching_threshold=(
            args.track_minimum_matching_threshold
            if args.track_minimum_matching_threshold is not None
            else float(tracking_config.get("minimum_matching_threshold", 0.8))
        ),
    )
    tracked = _assign_teams_for_process(args.video, tracked, config)
    write_detections(tracks_out, tracked)

    possession_radius_px = (
        args.possession_radius_px
        if args.possession_radius_px is not None
        else float(analysis_config.get("possession_radius_px", 90))
    )
    game_state_result = _write_pipeline_game_state(
        args,
        tracked=tracked,
        results_dir=results_dir,
        stem=stem,
        possession_radius_px=possession_radius_px,
        field_margin_px=field_margin_px,
    )
    analysis_tracks, analysis_tracks_out = _analysis_tracks_for_pipeline(
        args,
        tracked=tracked,
        tracks_out=tracks_out,
        game_state_result=game_state_result,
        results_dir=results_dir,
        stem=stem,
    )
    summary = summarize_tracks(
        analysis_tracks,
        fps=fps,
        possession_radius_px=possession_radius_px,
        field_margin_px=field_margin_px,
    )
    summary["game_state"] = game_state_result["summary"]
    write_json(metrics_out, summary)
    events = detect_events(
        analysis_tracks,
        possession_radius_px=possession_radius_px,
        collision_radius_px=(
            args.collision_radius_px
            if args.collision_radius_px is not None
            else float(analysis_config.get("collision_radius_px", 55))
        ),
        frame_width=int(info.get("width") or 0) or None,
        goal_x_margin_ratio=(
            args.goal_x_margin_ratio
            if args.goal_x_margin_ratio is not None
            else float(analysis_config.get("goal_x_margin_ratio", 0.08))
        ),
        field_margin_px=field_margin_px,
    )
    events = _confirm_goals_for_process(analysis_tracks, events, analysis_config)
    write_events(events_out, events)
    event_summary = summarize_events(events)
    event_summary["external_events"] = len(game_state_result["external_events"])
    event_summary["game_state"] = game_state_result["summary"]
    write_json(event_summary_out, event_summary)
    all_events_out = _write_combined_events_for_pipeline(
        results_dir=results_dir,
        stem=stem,
        suffix=args.suffix,
        events=events,
        external_events=game_state_result["external_events"],
    )
    field_analysis_result = None
    field_analysis_out = None
    field_trajectory_csv = None
    field_robot_csv = None
    field_zone_control_csv = None
    field_map_out = None
    if args.field_calibration:
        field_analysis_out = Path(
            args.field_analysis_out
            or results_dir / "field_analysis" / f"{stem}-{args.suffix}-field-analysis.json"
        )
        field_trajectory_csv = Path(
            args.field_trajectory_csv
            or results_dir / "field_analysis" / f"{stem}-{args.suffix}-trajectory.csv"
        )
        field_robot_csv = Path(
            args.field_robot_csv
            or results_dir / "field_analysis" / f"{stem}-{args.suffix}-robots.csv"
        )
        field_zone_control_csv = Path(
            args.field_zone_control_csv
            or results_dir / "field_analysis" / f"{stem}-{args.suffix}-zone-control.csv"
        )
        field_map_out = Path(
            args.field_map_out
            or results_dir / "field_analysis" / f"{stem}-{args.suffix}-field-map.png"
        )
        field_analysis_result = analyze_field_tracks(
            analysis_tracks,
            load_field_calibration(args.field_calibration),
            fps=fps,
            possession_radius_px=possession_radius_px,
            field_margin_px=field_margin_px,
            grid_cols=args.field_grid_cols,
            grid_rows=args.field_grid_rows,
            robot_anchor=args.field_robot_anchor,
        )
        write_json(field_analysis_out, field_analysis_result)
        write_field_trajectory_csv(field_trajectory_csv, field_analysis_result)
        write_field_robot_csv(field_robot_csv, field_analysis_result)
        write_field_zone_control_csv(field_zone_control_csv, field_analysis_result)
        render_field_map(field_analysis_result, field_map_out)

    rendered_videos = _render_pipeline_videos(
        args,
        video_path=args.video,
        tracks_out=analysis_tracks_out,
        events_out=all_events_out or events_out,
        results_dir=results_dir,
        stem=stem,
        duration_seconds=duration_seconds,
    )
    rendered = rendered_videos.get("narrative") or rendered_videos.get("analysis")

    qa_out, qa_report_out, qa_report = _write_pipeline_qa(
        args,
        results_dir=results_dir,
        stem=stem,
        metrics_out=metrics_out,
        events_out=events_out,
        field_analysis_out=field_analysis_out,
    )
    run_report_out = _write_pipeline_run_report(
        args,
        results_dir=results_dir,
        stem=stem,
        metrics_out=metrics_out,
        events_out=events_out,
        field_analysis_out=field_analysis_out,
        qa_out=qa_out,
        rendered=rendered,
        field_map_out=field_map_out,
    )
    run_manifest_out = _write_pipeline_manifest(
        args,
        results_dir=results_dir,
        stem=stem,
        artifacts={
            "detections": merged_out,
            "color_goals": color_goals_out if color_goal_detections is not None else None,
            "tracks": tracks_out,
            "in_play_tracks": analysis_tracks_out if analysis_tracks_out != tracks_out else None,
            "metrics": metrics_out,
            "events": events_out,
            "all_events": all_events_out,
            "external_events": game_state_result["external_events_out"],
            "game_state": game_state_result["game_state_out"],
            "game_segments": game_state_result["segments_out"],
            "event_summary": event_summary_out,
            "field_analysis": field_analysis_out,
            "field_trajectory_csv": field_trajectory_csv,
            "field_robot_csv": field_robot_csv,
            "field_zone_control_csv": field_zone_control_csv,
            "field_map": field_map_out,
            "qa": qa_out,
            "qa_report": qa_report_out,
            "run_report": run_report_out,
            "demo": rendered,
            "narrative_demo": rendered_videos.get("narrative"),
            "analysis_demo": rendered_videos.get("analysis"),
        },
        metrics_summary=summary,
        event_summary=event_summary,
        field_analysis_summary=(
            field_analysis_result.get("summary") if field_analysis_result else None
        ),
        qa_status=qa_report.get("status") if qa_report else None,
    )

    print(
        json.dumps(
            {
                "video": args.video,
                "frames": end_frame,
                "field_prompt_frames": field_prompt_frames,
                "ball_prompt_frames": ball_prompt_frames,
                "detections": len(merged),
                "analysis_detections": len(analysis_tracks),
                "edge_ball_filter_removed": edge_ball_filter_removed,
                "goal_constraints_removed": goal_constraints_removed,
                "tracks": len({det.track_id for det in tracked if det.track_id is not None}),
                "paths": {
                    "detections": str(merged_out),
                    "color_goals": str(color_goals_out) if color_goal_detections is not None else None,
                    "tracks": str(tracks_out),
                    "in_play_tracks": (
                        str(analysis_tracks_out) if analysis_tracks_out != tracks_out else None
                    ),
                    "metrics": str(metrics_out),
                    "events": str(events_out),
                    "all_events": str(all_events_out) if all_events_out else None,
                    "external_events": (
                        str(game_state_result["external_events_out"])
                        if game_state_result["external_events_out"]
                        else None
                    ),
                    "game_state": (
                        str(game_state_result["game_state_out"])
                        if game_state_result["game_state_out"]
                        else None
                    ),
                    "game_segments": (
                        str(game_state_result["segments_out"])
                        if game_state_result["segments_out"]
                        else None
                    ),
                    "event_summary": str(event_summary_out),
                    "field_analysis": str(field_analysis_out) if field_analysis_out else None,
                    "field_trajectory_csv": (
                        str(field_trajectory_csv) if field_trajectory_csv else None
                    ),
                    "field_robot_csv": str(field_robot_csv) if field_robot_csv else None,
                    "field_zone_control_csv": (
                        str(field_zone_control_csv) if field_zone_control_csv else None
                    ),
                    "field_map": str(field_map_out) if field_map_out else None,
                    "qa": str(qa_out) if qa_out else None,
                    "qa_report": str(qa_report_out) if qa_report_out else None,
                    "run_report": str(run_report_out) if run_report_out else None,
                    "run_manifest": str(run_manifest_out) if run_manifest_out else None,
                    "demo": str(rendered) if rendered else None,
                    "narrative_demo": (
                        str(rendered_videos["narrative"])
                        if rendered_videos.get("narrative")
                        else None
                    ),
                    "analysis_demo": (
                        str(rendered_videos["analysis"])
                        if rendered_videos.get("analysis")
                        else None
                    ),
                },
                "metrics": summary,
                "game_state": game_state_result["summary"],
                "field_analysis_summary": (
                    field_analysis_result.get("summary") if field_analysis_result else None
                ),
                "qa_status": qa_report.get("status") if qa_report else None,
                "events": len(events),
                "event_summary": event_summary,
                "color_goal_detections": (
                    len(color_goal_detections) if color_goal_detections is not None else None
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_process_top_camera(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    analysis_config = config.get("analysis", {})
    tracking_config = config.get("tracking", {})
    ball_detection_config = config.get("ball_detection", {})
    sam3_ball_enabled = (
        bool(ball_detection_config.get("sam3_enabled", True))
        if args.sam3_ball is None
        else bool(args.sam3_ball)
    )
    color_ball_enabled = (
        bool(ball_detection_config.get("color_enabled", True))
        if args.color_ball is None
        else bool(args.color_ball)
    )
    if not sam3_ball_enabled and not color_ball_enabled:
        raise SystemExit("Enable at least one ball source: --sam3-ball or --color-ball.")

    info = video_info(args.video)
    end_frame = int(info["frames"])
    fps = float(info.get("fps") or analysis_config.get("fps", 30))
    duration_seconds = info.get("duration_seconds")
    stem = Path(args.video).stem
    results_dir = Path(args.results_dir)

    field_prompt_frames = _frame_anchors(
        end_frame,
        start=args.field_start,
        step=args.field_step,
    )
    ball_prompt_frames = _frame_anchors(
        end_frame,
        start=args.ball_start,
        step=args.ball_step,
    )

    field_out = results_dir / "detections" / f"{stem}-field-robots-sweep-clipped"
    sam3_ball_out = results_dir / "detections" / f"{stem}-{args.suffix}-sam3-ball-sweep-clipped"
    color_ball_out = results_dir / "detections" / f"{stem}-{args.suffix}-color-ball.jsonl"
    robot_recovery_out = results_dir / "detections" / f"{stem}-{args.suffix}-dark-robots.jsonl"
    color_goals_out = results_dir / "detections" / f"{stem}-{args.suffix}-color-goals.jsonl"
    merged_dir = results_dir / "detections" / f"{stem}-{args.suffix}"
    merged_out = merged_dir / "detections.jsonl"
    refined_out = merged_dir / "detections-refined.jsonl"
    tracks_out = results_dir / "tracks" / f"{stem}-{args.suffix}-tracks.jsonl"
    metrics_out = results_dir / "metrics" / f"{stem}-{args.suffix}-metrics.json"
    events_out = results_dir / "events" / f"{stem}-{args.suffix}-events.json"
    event_summary_out = results_dir / "events" / f"{stem}-{args.suffix}-event-summary.json"

    field_detections = _run_sweep_for_process(
        args,
        classes=_context_classes(args, config),
        prompt_frames=field_prompt_frames,
        window_size=args.field_window_size,
        end_frame=end_frame,
        out=field_out,
        threshold=args.field_threshold,
        dedupe_iou=args.field_dedupe_iou,
    )

    sam3_ball_detections = []
    if sam3_ball_enabled:
        sam3_ball_detections = _run_sweep_for_process(
            args,
            classes="ball",
            prompt_frames=ball_prompt_frames,
            window_size=args.ball_window_size,
            end_frame=end_frame,
            out=sam3_ball_out,
            threshold=args.ball_threshold,
            dedupe_iou=args.ball_dedupe_iou,
        )

    ball_border_margin_px = (
        args.ball_border_margin_px
        if args.ball_border_margin_px is not None
        else float(analysis_config.get("ball_border_margin_px", 4))
    )
    field_margin_px = (
        args.in_play_field_margin_px
        if args.in_play_field_margin_px is not None
        else float(analysis_config.get("in_play_field_margin_px", 8))
    )
    color_profile, hsv_lower, hsv_upper = _resolve_ball_color_profile(
        ball_detection_config,
        profile=args.ball_color_profile,
        hsv_lower=args.ball_hsv_lower,
        hsv_upper=args.ball_hsv_upper,
    )
    color_ball_detections = []
    if color_ball_enabled:
        color_ball_detections = detect_orange_ball(
            args.video,
            color_ball_out,
            max_frames=end_frame,
            min_area=args.orange_min_area,
            max_area=args.orange_max_area,
            min_circularity=args.orange_min_circularity,
            hsv_lower=hsv_lower,
            hsv_upper=hsv_upper,
            color_profile=color_profile,
            context_detections_path=field_out / "detections.jsonl",
            robot_margin_px=args.orange_robot_margin_px,
            border_margin_px=ball_border_margin_px,
            max_per_frame=args.orange_max_per_frame,
        )
    robot_recovery_detections = []
    if args.robot_color_recovery:
        robot_recovery_detections = detect_dark_robots(
            args.video,
            robot_recovery_out,
            max_frames=end_frame,
            min_area=args.robot_recovery_min_area,
            max_area=args.robot_recovery_max_area,
            min_circularity=args.robot_recovery_min_circularity,
            hsv_lower=_parse_hsv_bound(args.robot_recovery_hsv_lower, fallback=(0, 0, 0)),
            hsv_upper=_parse_hsv_bound(args.robot_recovery_hsv_upper, fallback=(179, 255, 125)),
            field_detections_path=field_out / "detections.jsonl",
            field_margin_px=field_margin_px,
            border_margin_px=4.0,
            min_center_y_ratio=args.robot_recovery_min_center_y_ratio,
            max_center_y_ratio=args.robot_recovery_max_center_y_ratio,
            merge_distance_px=args.robot_recovery_merge_distance_px,
            max_per_frame=args.robot_recovery_max_per_frame,
            box_expand_x_px=args.robot_recovery_box_expand_x_px,
            box_expand_top_px=args.robot_recovery_box_expand_top_px,
            box_expand_bottom_px=args.robot_recovery_box_expand_bottom_px,
        )
    color_goal_detections = _detect_color_goals_for_process(
        args,
        config,
        end_frame=end_frame,
        out=color_goals_out,
        seed_detections=field_detections,
    )

    merge_inputs = [field_out / "detections.jsonl"]
    if sam3_ball_enabled:
        merge_inputs.append(sam3_ball_out / "detections.jsonl")
    if color_ball_enabled:
        merge_inputs.append(color_ball_out)
    if args.robot_color_recovery:
        merge_inputs.append(robot_recovery_out)
    if color_goal_detections is not None:
        merge_inputs.append(color_goals_out)
    merged = merge_detection_files(
        merge_inputs,
        merged_out,
        iou_threshold=args.merge_dedupe_iou,
    )
    robot_filter_config = config.get("robot_filter", {})
    robot_filter_enabled = (
        args.robot_filter
        if args.robot_filter is not None
        else bool(robot_filter_config.get("enabled", True))
    )
    robot_filter_removed = 0
    if robot_filter_enabled:
        before_robot_filter = len(merged)
        merged = filter_robot_detections(
            merged,
            frame_width=int(info.get("width") or 0) or None,
            frame_height=int(info.get("height") or 0) or None,
            max_per_frame=_resolve_optional_int(
                args.robot_filter_max_per_frame,
                robot_filter_config.get("max_per_frame"),
            ),
            min_area=_resolve_optional_float(
                args.robot_filter_min_area,
                robot_filter_config.get("min_area"),
                default=0.0,
            ),
            max_area_ratio=_resolve_optional_float(
                args.robot_filter_max_area_ratio,
                robot_filter_config.get("max_area_ratio"),
                default=None,
            ),
            containment_threshold=_resolve_optional_float(
                args.robot_filter_containment_threshold,
                robot_filter_config.get("containment_threshold"),
                default=0.82,
            ),
            iou_threshold=_resolve_optional_float(
                args.robot_filter_iou_threshold,
                robot_filter_config.get("iou_threshold"),
                default=0.55,
            ),
            min_center_distance_px=_resolve_optional_float(
                args.robot_filter_min_center_distance_px,
                robot_filter_config.get("min_center_distance_px"),
                default=0.0,
            ),
            protect_near_ball_px=_resolve_optional_float(
                args.robot_filter_protect_near_ball_px,
                robot_filter_config.get("protect_near_ball_px"),
                default=0.0,
            ),
        )
        robot_filter_removed = before_robot_filter - len(merged)
        write_detections(merged_out, merged)
    refined = refine_ball_trajectory(
        merged,
        max_jump_px=args.refine_max_jump_px,
        preferred_area=args.refine_preferred_area,
        score_weight=args.refine_score_weight,
        area_weight=args.refine_area_weight,
        max_candidates_per_frame=args.refine_max_candidates_per_frame,
    )
    constrained_refined = _enforce_goal_constraints_for_process(refined, config)
    goal_constraints_removed = len(refined) - len(constrained_refined)
    refined = constrained_refined
    write_detections(refined_out, refined)

    tracked = track_detections(
        refined,
        iou_threshold=args.track_iou_threshold,
        max_age=args.track_max_age,
        backend=args.tracker_backend or str(tracking_config.get("backend", "iou")),
        frame_rate=max(1, round(fps)),
        track_activation_threshold=(
            args.track_activation_threshold
            if args.track_activation_threshold is not None
            else float(tracking_config.get("track_activation_threshold", 0.05))
        ),
        minimum_matching_threshold=(
            args.track_minimum_matching_threshold
            if args.track_minimum_matching_threshold is not None
            else float(tracking_config.get("minimum_matching_threshold", 0.8))
        ),
    )
    tracked = _assign_teams_for_process(args.video, tracked, config)
    write_detections(tracks_out, tracked)

    possession_radius_px = (
        args.possession_radius_px
        if args.possession_radius_px is not None
        else float(analysis_config.get("possession_radius_px", 90))
    )
    game_state_result = _write_pipeline_game_state(
        args,
        tracked=tracked,
        results_dir=results_dir,
        stem=stem,
        possession_radius_px=possession_radius_px,
        field_margin_px=field_margin_px,
    )
    analysis_tracks, analysis_tracks_out = _analysis_tracks_for_pipeline(
        args,
        tracked=tracked,
        tracks_out=tracks_out,
        game_state_result=game_state_result,
        results_dir=results_dir,
        stem=stem,
    )
    summary = summarize_tracks(
        analysis_tracks,
        fps=fps,
        possession_radius_px=possession_radius_px,
        field_margin_px=field_margin_px,
    )
    summary["game_state"] = game_state_result["summary"]
    write_json(metrics_out, summary)
    events = detect_events(
        analysis_tracks,
        possession_radius_px=possession_radius_px,
        collision_radius_px=(
            args.collision_radius_px
            if args.collision_radius_px is not None
            else float(analysis_config.get("collision_radius_px", 55))
        ),
        frame_width=int(info.get("width") or 0) or None,
        goal_x_margin_ratio=(
            args.goal_x_margin_ratio
            if args.goal_x_margin_ratio is not None
            else float(analysis_config.get("goal_x_margin_ratio", 0.08))
        ),
        field_margin_px=field_margin_px,
    )
    events = _confirm_goals_for_process(analysis_tracks, events, analysis_config)
    write_events(events_out, events)
    event_summary = summarize_events(events)
    event_summary["external_events"] = len(game_state_result["external_events"])
    event_summary["game_state"] = game_state_result["summary"]
    write_json(event_summary_out, event_summary)
    all_events_out = _write_combined_events_for_pipeline(
        results_dir=results_dir,
        stem=stem,
        suffix=args.suffix,
        events=events,
        external_events=game_state_result["external_events"],
    )

    field_analysis_result = None
    field_analysis_out = None
    field_trajectory_csv = None
    field_robot_csv = None
    field_zone_control_csv = None
    field_map_out = None
    if args.field_calibration:
        field_analysis_out = Path(
            args.field_analysis_out
            or results_dir / "field_analysis" / f"{stem}-{args.suffix}-field-analysis.json"
        )
        field_trajectory_csv = Path(
            args.field_trajectory_csv
            or results_dir / "field_analysis" / f"{stem}-{args.suffix}-trajectory.csv"
        )
        field_robot_csv = Path(
            args.field_robot_csv
            or results_dir / "field_analysis" / f"{stem}-{args.suffix}-robots.csv"
        )
        field_zone_control_csv = Path(
            args.field_zone_control_csv
            or results_dir / "field_analysis" / f"{stem}-{args.suffix}-zone-control.csv"
        )
        field_map_out = Path(
            args.field_map_out
            or results_dir / "field_analysis" / f"{stem}-{args.suffix}-field-map.png"
        )
        field_analysis_result = analyze_field_tracks(
            analysis_tracks,
            load_field_calibration(args.field_calibration),
            fps=fps,
            possession_radius_px=possession_radius_px,
            field_margin_px=field_margin_px,
            grid_cols=args.field_grid_cols,
            grid_rows=args.field_grid_rows,
            robot_anchor=args.field_robot_anchor,
        )
        write_json(field_analysis_out, field_analysis_result)
        write_field_trajectory_csv(field_trajectory_csv, field_analysis_result)
        write_field_robot_csv(field_robot_csv, field_analysis_result)
        write_field_zone_control_csv(field_zone_control_csv, field_analysis_result)
        render_field_map(field_analysis_result, field_map_out)

    rendered_videos = _render_pipeline_videos(
        args,
        video_path=args.video,
        tracks_out=analysis_tracks_out,
        events_out=all_events_out or events_out,
        results_dir=results_dir,
        stem=stem,
        duration_seconds=duration_seconds,
    )
    rendered = rendered_videos.get("narrative") or rendered_videos.get("analysis")

    qa_out, qa_report_out, qa_report = _write_pipeline_qa(
        args,
        results_dir=results_dir,
        stem=stem,
        metrics_out=metrics_out,
        events_out=events_out,
        field_analysis_out=field_analysis_out,
    )
    run_report_out = _write_pipeline_run_report(
        args,
        results_dir=results_dir,
        stem=stem,
        metrics_out=metrics_out,
        events_out=events_out,
        field_analysis_out=field_analysis_out,
        qa_out=qa_out,
        rendered=rendered,
        field_map_out=field_map_out,
    )
    run_manifest_out = _write_pipeline_manifest(
        args,
        results_dir=results_dir,
        stem=stem,
        artifacts={
            "field_detections": field_out / "detections.jsonl",
            "sam3_ball_detections": (
                sam3_ball_out / "detections.jsonl" if sam3_ball_enabled else None
            ),
            "color_ball_detections": color_ball_out if color_ball_enabled else None,
            "robot_recovery_detections": (
                robot_recovery_out if args.robot_color_recovery else None
            ),
            "color_goal_detections": (
                color_goals_out if color_goal_detections is not None else None
            ),
            "detections": merged_out,
            "refined_detections": refined_out,
            "tracks": tracks_out,
            "in_play_tracks": analysis_tracks_out if analysis_tracks_out != tracks_out else None,
            "metrics": metrics_out,
            "events": events_out,
            "all_events": all_events_out,
            "external_events": game_state_result["external_events_out"],
            "game_state": game_state_result["game_state_out"],
            "game_segments": game_state_result["segments_out"],
            "event_summary": event_summary_out,
            "field_analysis": field_analysis_out,
            "field_trajectory_csv": field_trajectory_csv,
            "field_robot_csv": field_robot_csv,
            "field_zone_control_csv": field_zone_control_csv,
            "field_map": field_map_out,
            "qa": qa_out,
            "qa_report": qa_report_out,
            "run_report": run_report_out,
            "demo": rendered,
            "narrative_demo": rendered_videos.get("narrative"),
            "analysis_demo": rendered_videos.get("analysis"),
        },
        metrics_summary=summary,
        event_summary=event_summary,
        field_analysis_summary=(
            field_analysis_result.get("summary") if field_analysis_result else None
        ),
        qa_status=qa_report.get("status") if qa_report else None,
    )

    ball_in = sum(1 for det in merged if det.class_name in {"ball", "balon", "soccer_ball"})
    ball_out = sum(1 for det in refined if det.class_name in {"ball", "balon", "soccer_ball"})
    print(
        json.dumps(
            {
                "video": args.video,
                "frames": end_frame,
                "field_prompt_frames": field_prompt_frames,
                "ball_sources": {
                    "sam3": sam3_ball_enabled,
                    "color": color_ball_enabled,
                },
                "ball_prompt_frames": ball_prompt_frames if sam3_ball_enabled else [],
                "sam3_ball_candidates": len(sam3_ball_detections),
                "color_ball_candidates": len(color_ball_detections),
                "robot_recovery_candidates": len(robot_recovery_detections),
                "color_goal_candidates": (
                    len(color_goal_detections) if color_goal_detections is not None else None
                ),
                "ball_color_profile": color_profile if color_ball_enabled else None,
                "ball_candidates_before_refine": ball_in,
                "ball_detections_after_refine": ball_out,
                "robot_filter_removed": robot_filter_removed,
                "goal_constraints_removed": goal_constraints_removed,
                "detections": len(refined),
                "analysis_detections": len(analysis_tracks),
                "tracks": len({det.track_id for det in tracked if det.track_id is not None}),
                "paths": {
                    "field_detections": str(field_out / "detections.jsonl"),
                    "sam3_ball_detections": (
                        str(sam3_ball_out / "detections.jsonl") if sam3_ball_enabled else None
                    ),
                    "color_ball_detections": str(color_ball_out) if color_ball_enabled else None,
                    "robot_recovery_detections": (
                        str(robot_recovery_out) if args.robot_color_recovery else None
                    ),
                    "color_goal_detections": (
                        str(color_goals_out) if color_goal_detections is not None else None
                    ),
                    "detections": str(merged_out),
                    "refined_detections": str(refined_out),
                    "tracks": str(tracks_out),
                    "in_play_tracks": (
                        str(analysis_tracks_out) if analysis_tracks_out != tracks_out else None
                    ),
                    "metrics": str(metrics_out),
                    "events": str(events_out),
                    "all_events": str(all_events_out) if all_events_out else None,
                    "external_events": (
                        str(game_state_result["external_events_out"])
                        if game_state_result["external_events_out"]
                        else None
                    ),
                    "game_state": (
                        str(game_state_result["game_state_out"])
                        if game_state_result["game_state_out"]
                        else None
                    ),
                    "game_segments": (
                        str(game_state_result["segments_out"])
                        if game_state_result["segments_out"]
                        else None
                    ),
                    "event_summary": str(event_summary_out),
                    "field_analysis": str(field_analysis_out) if field_analysis_out else None,
                    "field_trajectory_csv": (
                        str(field_trajectory_csv) if field_trajectory_csv else None
                    ),
                    "field_robot_csv": str(field_robot_csv) if field_robot_csv else None,
                    "field_zone_control_csv": (
                        str(field_zone_control_csv) if field_zone_control_csv else None
                    ),
                    "field_map": str(field_map_out) if field_map_out else None,
                    "qa": str(qa_out) if qa_out else None,
                    "qa_report": str(qa_report_out) if qa_report_out else None,
                    "run_report": str(run_report_out) if run_report_out else None,
                    "run_manifest": str(run_manifest_out) if run_manifest_out else None,
                    "demo": str(rendered) if rendered else None,
                    "narrative_demo": (
                        str(rendered_videos["narrative"])
                        if rendered_videos.get("narrative")
                        else None
                    ),
                    "analysis_demo": (
                        str(rendered_videos["analysis"])
                        if rendered_videos.get("analysis")
                        else None
                    ),
                },
                "metrics": summary,
                "game_state": game_state_result["summary"],
                "field_analysis_summary": (
                    field_analysis_result.get("summary") if field_analysis_result else None
                ),
                "qa_status": qa_report.get("status") if qa_report else None,
                "events": len(events),
                "event_summary": event_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _write_pipeline_qa(
    args: argparse.Namespace,
    *,
    results_dir: Path,
    stem: str,
    metrics_out: Path,
    events_out: Path,
    field_analysis_out: Path | None,
) -> tuple[Path | None, Path | None, dict | None]:
    if not args.qa:
        return None, None, None
    qa_out = Path(args.qa_out or results_dir / "qa" / f"{stem}-{args.suffix}-qa.json")
    qa_report_out = Path(
        args.qa_report_out or results_dir / "qa" / f"{stem}-{args.suffix}-qa.md"
    )
    report = evaluate_run_quality(
        metrics_path=metrics_out,
        events_path=events_out,
        field_analysis_path=field_analysis_out,
    )
    write_quality_json(qa_out, report)
    write_quality_markdown(qa_report_out, report)
    return qa_out, qa_report_out, report


def _write_pipeline_game_state(
    args: argparse.Namespace,
    *,
    tracked: list,
    results_dir: Path,
    stem: str,
    possession_radius_px: float,
    field_margin_px: float,
) -> dict:
    if not getattr(args, "generate_game_state", True):
        return {
            "game_state_out": None,
            "segments_out": None,
            "external_events_out": None,
            "playable_frames": None,
            "external_events": [],
            "summary": {"enabled": False},
        }

    calibration_path = getattr(args, "field_calibration", None)
    calibration = load_field_calibration(calibration_path) if calibration_path else None
    states = classify_frame_states(
        tracked,
        possession_radius_px=possession_radius_px,
        field_margin_px=field_margin_px,
        missing_ball_frames=args.game_state_missing_ball_frames,
        robot_removed_after_frames=args.robot_removed_after_frames,
        robot_disabled_after_frames=args.robot_disabled_after_frames,
        stationary_threshold_px=args.stationary_threshold_px,
        field_polygon=calibration.image_points if calibration else None,
    )
    segments = detect_game_segments(states)
    external_events = detect_external_events(states)
    playable_frames = play_mask_from_segments(segments)
    summary = _game_state_summary(states, segments, external_events, playable_frames)

    game_state_out = Path(
        args.game_state_out or results_dir / "events" / f"{stem}-{args.suffix}-game-state.json"
    )
    segments_out = Path(
        args.game_segments_out
        or results_dir / "events" / f"{stem}-{args.suffix}-game-segments.json"
    )
    external_events_out = Path(
        args.external_events_out
        or results_dir / "events" / f"{stem}-{args.suffix}-external-events.json"
    )

    write_json(
        game_state_out,
        {
            "schema": "samba_futbot.game_state.v1",
            "summary": summary,
            "states": [state.to_record() for state in states],
            "segments": [segment.to_record() for segment in segments],
            "events": [event.to_record() for event in external_events],
        },
    )
    write_json(segments_out, [segment.to_record() for segment in segments])
    write_events(external_events_out, external_events)

    return {
        "game_state_out": game_state_out,
        "segments_out": segments_out,
        "external_events_out": external_events_out,
        "playable_frames": playable_frames,
        "external_events": external_events,
        "summary": summary,
    }


def _analysis_tracks_for_pipeline(
    args: argparse.Namespace,
    *,
    tracked: list,
    tracks_out: Path,
    game_state_result: dict,
    results_dir: Path,
    stem: str,
) -> tuple[list, Path]:
    playable_frames = game_state_result.get("playable_frames")
    if not getattr(args, "filter_by_game_state", True) or playable_frames is None:
        return tracked, tracks_out

    filtered = filter_detections_to_playable_frames(tracked, playable_frames)
    filtered_out = results_dir / "tracks" / f"{stem}-{args.suffix}-in-play-tracks.jsonl"
    write_detections(filtered_out, filtered)
    return filtered, filtered_out


def _write_combined_events_for_pipeline(
    *,
    results_dir: Path,
    stem: str,
    suffix: str,
    events: list,
    external_events: list,
) -> Path | None:
    if not external_events:
        return None
    out = results_dir / "events" / f"{stem}-{suffix}-all-events.json"
    write_events(out, [*events, *external_events])
    return out


def _game_state_summary(
    states: list,
    segments: list,
    external_events: list,
    playable_frames: set[int],
) -> dict:
    state_counts: dict[str, int] = {}
    for state in states:
        state_counts[state.state] = state_counts.get(state.state, 0) + 1
    event_counts: dict[str, int] = {}
    for event in external_events:
        event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
    total_frames = len(states)
    playable_count = len(playable_frames)
    return {
        "enabled": True,
        "frames": total_frames,
        "playable_frames": playable_count,
        "playable_ratio": playable_count / total_frames if total_frames else 0.0,
        "states": state_counts,
        "segments": len(segments),
        "external_events": event_counts,
    }


def _write_pipeline_run_report(
    args: argparse.Namespace,
    *,
    results_dir: Path,
    stem: str,
    metrics_out: Path,
    events_out: Path,
    field_analysis_out: Path | None,
    qa_out: Path | None,
    rendered: Path | None,
    field_map_out: Path | None,
) -> Path:
    report_out = Path(
        args.run_report_out or results_dir / "reports" / f"{stem}-{args.suffix}-report.md"
    )
    return write_run_report(
        report_out,
        title=f"{stem} {args.suffix}",
        metrics_path=metrics_out,
        events_path=events_out,
        field_analysis_path=field_analysis_out,
        qa_path=qa_out,
        demo_path=rendered,
        field_map_path=field_map_out,
    )


def _render_pipeline_videos(
    args: argparse.Namespace,
    *,
    video_path: str,
    tracks_out: Path,
    events_out: Path,
    results_dir: Path,
    stem: str,
    duration_seconds: float | None,
) -> dict[str, Path]:
    if not args.render:
        return {}
    render_seconds = args.max_seconds
    if render_seconds is None:
        render_seconds = duration_seconds

    rendered: dict[str, Path] = {}
    styles = []
    if getattr(args, "render_narrative", True):
        styles.append("narrative")
    if getattr(args, "render_analysis", True):
        styles.append("analysis")
    for style in styles:
        out_path = results_dir / "videos" / f"{stem}-{args.suffix}-{style}-demo.mp4"
        rendered[style] = render_demo_video(
            video_path,
            tracks_out,
            out_path,
            events_path=events_out,
            max_seconds=render_seconds,
            trail_length=args.trail_length,
            style=style,
            analysis_freeze=args.analysis_freeze,
            freeze_seconds=args.freeze_seconds,
            freeze_min_confidence=args.freeze_min_confidence,
            freeze_cooldown_frames=args.freeze_cooldown_frames,
            freeze_max_events=args.freeze_max_events,
            freeze_event_types=args.freeze_event_types,
            mask_overlay=args.mask_overlay,
            mask_alpha=args.mask_alpha,
            label_scale=args.label_scale,
            box_thickness=args.box_thickness,
            visual_hold_frames=args.visual_hold_frames,
            show_team_labels=args.show_team_labels,
        )
    return rendered


def _write_pipeline_manifest(
    args: argparse.Namespace,
    *,
    results_dir: Path,
    stem: str,
    artifacts: dict,
    metrics_summary: dict,
    event_summary: dict,
    field_analysis_summary: dict | None,
    qa_status: str | None,
) -> Path:
    manifest_out = Path(
        args.run_manifest_out
        or results_dir / "reports" / f"{stem}-{args.suffix}-manifest.json"
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "command_argv": list(sys.argv),
        "command_name": getattr(args, "command", None),
        "video": getattr(args, "video", None),
        "config": getattr(args, "config", None),
        "results_dir": str(results_dir),
        "suffix": getattr(args, "suffix", None),
        "git": _git_snapshot(Path(__file__).resolve().parents[2]),
        "source_fingerprint": _source_fingerprint(Path(__file__).resolve().parents[2]),
        "runtime": _runtime_snapshot(),
        "args": _jsonable(vars(args)),
        "artifacts": {
            key: str(value) if value is not None else None for key, value in sorted(artifacts.items())
        },
        "metrics": {
            "frames_observed": metrics_summary.get("frames_observed"),
            "detections": metrics_summary.get("detections"),
            "tracks": metrics_summary.get("tracks"),
            "possession": metrics_summary.get("possession", {}),
        },
        "event_summary": event_summary,
        "field_analysis_summary": field_analysis_summary,
        "qa_status": qa_status,
    }
    write_json(manifest_out, manifest)
    return manifest_out


def _runtime_snapshot() -> dict:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }


def _source_fingerprint(repo_root: Path) -> dict:
    patterns = [
        "src/samba_futbot/*.py",
        "config/*.yml",
        "config/*.yaml",
        "pyproject.toml",
        "requirements*.txt",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path for path in repo_root.glob(pattern) if path.is_file())

    digest = hashlib.sha256()
    hashed_files = []
    for path in sorted(set(files), key=lambda item: item.as_posix()):
        relative = path.relative_to(repo_root).as_posix()
        data = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
        hashed_files.append(relative)

    return {
        "algorithm": "sha256",
        "digest": digest.hexdigest(),
        "files_hashed": len(hashed_files),
        "paths": hashed_files,
    }


def _git_snapshot(repo_root: Path) -> dict:
    def run_git(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return completed.stdout.strip()

    try:
        branch = run_git("rev-parse", "--abbrev-ref", "HEAD")
        commit = run_git("rev-parse", "HEAD")
        status = run_git("status", "--short")
    except Exception as exc:
        return {
            "available": False,
            "repo_root": str(repo_root),
            "error": f"{type(exc).__name__}: {exc}",
        }

    changed_files = [line for line in status.splitlines() if line.strip()]
    return {
        "available": True,
        "repo_root": str(repo_root),
        "branch": branch,
        "commit": commit,
        "dirty": bool(changed_files),
        "changed_files": len(changed_files),
    }


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if callable(value):
        return getattr(value, "__name__", str(value))
    return value


def _assign_teams_for_process(video_path: str, detections: list, config: dict) -> list:
    team_config = config.get("team_detection", {})
    if not bool(team_config.get("enabled", True)):
        return detections
    palette = _team_palette(team_config.get("palette", {}))
    max_distance = float(team_config.get("max_color_distance", 170.0))
    return assign_robot_teams_from_video(
        video_path,
        detections,
        palette=palette,
        max_color_distance=max_distance,
        min_saturation=int(team_config.get("min_saturation", 45)),
        min_value=int(team_config.get("min_value", 40)),
        min_pixels=int(team_config.get("min_pixels", 8)),
    )


def _detect_color_goals_for_process(
    args: argparse.Namespace,
    config: dict,
    *,
    end_frame: int,
    out: Path,
    seed_detections: list | None = None,
) -> list | None:
    if not getattr(args, "goals", False):
        return None
    goal_config = config.get("goal_detection", {})
    enabled = (
        bool(goal_config.get("color_enabled", True))
        if args.color_goals is None
        else bool(args.color_goals)
    )
    if not enabled:
        return None
    return detect_colored_goals(
        args.video,
        out,
        max_frames=end_frame,
        profiles=_goal_color_profiles(goal_config.get("profiles", {})) or None,
        seed_detections=seed_detections,
        adaptive_color=bool(goal_config.get("adaptive_color", True)),
        broad_profiles=_goal_color_profiles(goal_config.get("broad_profiles", {})) or None,
        adaptive_hsv_margin=_parse_hsv_bound(
            None,
            fallback=goal_config.get("adaptive_hsv_margin", [12, 45, 45]),
        ),
        adaptive_min_pixels=int(goal_config.get("adaptive_min_pixels", 120)),
        spatial_gate_from_seeds=bool(goal_config.get("spatial_gate_from_seeds", True)),
        seed_spatial_margin_px=float(goal_config.get("seed_spatial_margin_px", 90.0)),
        require_seed_for_color=bool(goal_config.get("require_seed_for_color", False)),
        require_field_overlap=bool(goal_config.get("require_field_overlap", True)),
        field_margin_px=float(goal_config.get("field_margin_px", 18.0)),
        min_area=float(goal_config.get("min_area", 180.0)),
        max_area=float(goal_config.get("max_area", 80_000.0)),
        min_extent=float(goal_config.get("min_extent", 0.18)),
        max_per_frame_per_class=int(goal_config.get("max_per_frame_per_class", 1)),
    )


def _enforce_goal_constraints_for_process(detections: list, config: dict) -> list:
    goal_config = config.get("goal_detection", {})
    return enforce_goal_frame_constraints(
        detections,
        field_detections=detections,
        max_per_frame_per_class=int(goal_config.get("max_per_frame_per_class", 1)),
        require_field_overlap=bool(goal_config.get("require_field_overlap", True)),
        infer_missing_opposite=bool(goal_config.get("infer_missing_opposite", False)),
        inferred_goal_score=float(goal_config.get("inferred_goal_score", 0.28)),
        field_margin_px=float(goal_config.get("field_margin_px", 18.0)),
    )


def _goal_color_profiles(raw_profiles: dict) -> dict[str, dict[str, tuple[int, int, int]]]:
    profiles: dict[str, dict[str, tuple[int, int, int]]] = {}
    for class_name, profile in raw_profiles.items():
        if not isinstance(profile, dict):
            continue
        profiles[str(class_name)] = {
            "hsv_lower": _parse_hsv_bound(None, fallback=profile.get("hsv_lower", [])),
            "hsv_upper": _parse_hsv_bound(None, fallback=profile.get("hsv_upper", [])),
        }
    return profiles


def _team_palette(raw_palette: dict) -> dict[str, tuple[int, int, int]]:
    palette: dict[str, tuple[int, int, int]] = {}
    for team, value in raw_palette.items():
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            continue
        palette[str(team)] = (int(value[0]), int(value[1]), int(value[2]))
    return palette


def _context_classes(args: argparse.Namespace, config: dict | None = None) -> str:
    classes = ["field", "robots"]
    if getattr(args, "goals", False):
        classes.extend(["goal_blue", "goal_yellow"])
    human_config = (config or {}).get("human_detection", {})
    human_enabled = (
        bool(human_config.get("enabled", False))
        if getattr(args, "human_context", None) is None
        else bool(args.human_context)
    )
    if human_enabled:
        classes.extend(str(item) for item in human_config.get("classes", []) if str(item))
    classes = list(dict.fromkeys(classes))
    return ",".join(classes)


def _confirm_goals_for_process(detections: list, events: list, analysis_config: dict) -> list:
    goal_config = analysis_config.get("goal_confirmation", {})
    if not bool(goal_config.get("enabled", True)):
        return events
    return confirm_goal_candidates(
        detections,
        events,
        lookback_frames=int(goal_config.get("lookback_frames", 8)),
        confirmation_frames=int(goal_config.get("confirmation_frames", 4)),
        min_inside_frames=int(goal_config.get("min_inside_frames", 2)),
        min_entry_motion_px=float(goal_config.get("min_entry_motion_px", 3.0)),
        min_goal_score=float(goal_config.get("min_goal_score", 0.45)),
    )


def _run_sweep_for_process(
    args: argparse.Namespace,
    *,
    classes: str,
    prompt_frames: list[int],
    window_size: int,
    end_frame: int,
    out: Path,
    threshold: float,
    dedupe_iou: float,
) -> list:
    sweep_args = argparse.Namespace(
        config=args.config,
        video=args.video,
        out=str(out),
        prompt_frames=",".join(str(frame) for frame in prompt_frames),
        window_size=window_size,
        end_frame=end_frame,
        classes=classes,
        threshold=threshold,
        dedupe_iou=dedupe_iou,
        backend=None,
        model_id=None,
        use_fa3=None,
        max_num_objects=None,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        clip_windows=args.clip_windows,
    )
    cmd_run_sam3_sweep(sweep_args)
    return read_detections(out / "detections.jsonl")


def _frame_anchors(end_frame: int, *, start: int, step: int) -> list[int]:
    if end_frame <= 0:
        return []
    if step <= 0:
        raise ValueError("Anchor step must be positive.")
    anchors = list(range(start, end_frame, step))
    if not anchors:
        anchors = [0]
    return anchors


def _filtered_prompts(
    prompts: dict,
    classes_csv: str | None,
    *,
    max_per_class: int | None = None,
) -> dict:
    filtered = prompts
    if classes_csv:
        classes = {part.strip() for part in classes_csv.split(",") if part.strip()}
        filtered = {
            class_name: value for class_name, value in prompts.items() if class_name in classes
        }
    if max_per_class is None:
        return filtered
    limit = int(max_per_class)
    if limit <= 0:
        return filtered
    return {
        class_name: value[:limit] if isinstance(value, list) else value
        for class_name, value in filtered.items()
    }


def _prompts_for_window(
    prompts: dict,
    *,
    window_index: int,
    strategy: str = "all",
    per_class: int | None = None,
) -> dict:
    """Select a bounded, rotating prompt ensemble for one video window."""
    if strategy == "all" or per_class is None:
        return prompts
    if strategy != "rotate":
        raise ValueError(f"Unknown prompt window strategy: {strategy}")
    limit = int(per_class)
    if limit <= 0:
        return prompts

    selected: dict = {}
    for class_name, value in prompts.items():
        if not isinstance(value, list) or len(value) <= limit:
            selected[class_name] = value
            continue
        if limit == 1:
            selected[class_name] = [value[window_index % len(value)]]
            continue
        variants = value[1:]
        start = window_index % len(variants)
        rotating = [variants[(start + offset) % len(variants)] for offset in range(limit - 1)]
        selected[class_name] = [value[0], *rotating]
    return selected


def _resolve_ball_color_profile(
    color_config: dict,
    *,
    profile: str | None,
    hsv_lower: str | None,
    hsv_upper: str | None,
) -> tuple[str, tuple[int, int, int], tuple[int, int, int]]:
    resolved_profile = profile or str(color_config.get("default_profile", "orange"))
    builtin_profiles = {
        "orange": {"hsv_lower": [0, 90, 90], "hsv_upper": [25, 255, 255]},
        "white": {"hsv_lower": [0, 0, 160], "hsv_upper": [180, 80, 255]},
        "yellow": {"hsv_lower": [20, 80, 90], "hsv_upper": [38, 255, 255]},
    }
    profiles = color_config.get("profiles", {})
    profile_config = profiles.get(resolved_profile, {}) if isinstance(profiles, dict) else {}
    if not profile_config:
        profile_config = builtin_profiles.get(resolved_profile, builtin_profiles["orange"])
    lower = _parse_hsv_bound(
        hsv_lower,
        fallback=profile_config.get("hsv_lower", [0, 90, 90]),
    )
    upper = _parse_hsv_bound(
        hsv_upper,
        fallback=profile_config.get("hsv_upper", [25, 255, 255]),
    )
    return resolved_profile, lower, upper


def _resolve_optional_float(
    cli_value: float | None,
    config_value: object,
    *,
    default: float | None,
) -> float | None:
    if cli_value is not None:
        return float(cli_value)
    if config_value is None:
        return default
    return float(config_value)


def _resolve_optional_int(cli_value: int | None, config_value: object) -> int | None:
    if cli_value is not None:
        return int(cli_value)
    if config_value is None:
        return None
    return int(config_value)


def _parse_hsv_bound(value: str | None, *, fallback: object) -> tuple[int, int, int]:
    if value:
        items = parse_int_list(value)
    elif isinstance(fallback, (list, tuple)):
        items = [int(item) for item in fallback]
    else:
        raise ValueError("HSV bounds must be a 3-item list.")
    if len(items) != 3:
        raise ValueError("HSV bounds must have exactly 3 values: H,S,V.")
    return (items[0], items[1], items[2])


def _filter_by_game_state(detections: list, game_state_path: str | None) -> list:
    if not game_state_path:
        return detections
    playable_frames = playable_frames_from_game_state(game_state_path)
    return filter_detections_to_playable_frames(detections, playable_frames)


def cmd_field_analysis(args: argparse.Namespace) -> None:
    detections = read_detections(args.tracks)
    detections = _filter_by_game_state(detections, args.game_state)
    if args.video:
        detections = _assign_teams_for_process(args.video, detections, load_config(args.config))
    analysis = analyze_field_tracks(
        detections,
        load_field_calibration(args.calibration),
        fps=args.fps,
        possession_radius_px=args.possession_radius_px,
        field_margin_px=args.in_play_field_margin_px,
        grid_cols=args.grid_cols,
        grid_rows=args.grid_rows,
        robot_anchor=args.robot_anchor,
    )
    write_json(args.out, analysis)
    if args.csv_out:
        write_field_trajectory_csv(args.csv_out, analysis)
    if args.robot_csv_out:
        write_field_robot_csv(args.robot_csv_out, analysis)
    if args.zone_control_csv_out:
        write_field_zone_control_csv(args.zone_control_csv_out, analysis)
    if args.map_out:
        render_field_map(analysis, args.map_out)
    print(
        json.dumps(
            {
                "out": args.out,
                "csv_out": args.csv_out,
                "robot_csv_out": args.robot_csv_out,
                "zone_control_csv_out": args.zone_control_csv_out,
                "map_out": args.map_out,
                "game_state": args.game_state,
                "summary": analysis["summary"],
                "robot_summary": analysis["robot_summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_render_field_map(args: argparse.Namespace) -> None:
    out = render_field_map(read_json(args.analysis), args.out, width=args.width)
    print(json.dumps({"map": str(out)}, indent=2))


def cmd_render_calibration_frame(args: argparse.Namespace) -> None:
    calibration = load_field_calibration(args.calibration) if args.calibration else None
    out = render_calibration_frame(
        args.video,
        args.out,
        frame_index=args.frame_index,
        calibration=calibration,
    )
    print(json.dumps({"calibration_frame": str(out)}, indent=2))


def cmd_calibration_check(args: argparse.Namespace) -> None:
    frame_width = args.frame_width
    frame_height = args.frame_height
    if args.video and (frame_width is None or frame_height is None):
        info = video_info(args.video)
        frame_width = int(info.get("width") or 0) or frame_width
        frame_height = int(info.get("height") or 0) or frame_height
    report = calibration_quality_report(
        load_field_calibration(args.calibration),
        frame_width=frame_width,
        frame_height=frame_height,
    )
    if args.out:
        write_calibration_quality(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_summarize_run(args: argparse.Namespace) -> None:
    out = write_run_report(
        args.out,
        title=args.title,
        metrics_path=args.metrics,
        events_path=args.events,
        field_analysis_path=args.field_analysis,
        qa_path=args.qa,
        demo_path=args.demo,
        field_map_path=args.field_map,
    )
    print(json.dumps({"report": str(out)}, indent=2))


def cmd_submission_report(args: argparse.Namespace) -> None:
    out = write_submission_report(
        args.out,
        batch_root=args.batch_root,
        training_root=args.training_root,
        title=args.title,
        top=args.top,
    )
    print(json.dumps({"report": str(out)}, ensure_ascii=False, indent=2))


def cmd_qa_run(args: argparse.Namespace) -> None:
    thresholds = _qa_thresholds(args)
    report = evaluate_run_quality(
        metrics_path=args.metrics,
        events_path=args.events,
        field_analysis_path=args.field_analysis,
        thresholds=thresholds,
    )
    out = write_quality_json(args.out, report)
    report_out = None
    if args.report_out:
        report_out = write_quality_markdown(args.report_out, report)
    print(
        json.dumps(
            {
                "qa": str(out),
                "report": str(report_out) if report_out else None,
                "status": report["status"],
                "quality_score": report["quality_score"],
                "issues": len(report["issues"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_qa_index(args: argparse.Namespace) -> None:
    reports = collect_quality_reports(args.root, pattern=args.pattern)
    out = write_quality_index_json(args.out, reports)
    report_out = None
    if args.report_out:
        report_out = write_quality_index_markdown(args.report_out, reports)
    print(
        json.dumps(
            {
                "qa_index": str(out),
                "report": str(report_out) if report_out else None,
                "runs": len(reports),
                "best": reports[0] if reports else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_compare_qa(args: argparse.Namespace) -> None:
    comparison = compare_qa_files(args.baseline, args.candidate)
    write_qa_comparison_json(args.out, comparison)
    markdown = None
    if args.report_out:
        markdown = write_qa_comparison_markdown(args.report_out, comparison)
    print(
        json.dumps(
            {
                "out": args.out,
                "report": str(markdown) if markdown else None,
                "verdict": comparison["verdict"],
                "summary": comparison["summary"],
                "claims": comparison["claims"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_showcase_index(args: argparse.Namespace) -> None:
    required_claims = [part.strip() for part in args.required_claims.split(",") if part.strip()]
    candidates = collect_showcase_candidates(
        args.root,
        limit=args.limit,
        required_claims=required_claims,
    )
    out = write_showcase_json(args.out, candidates)
    report_out = None
    if args.report_out:
        report_out = write_showcase_markdown(args.report_out, candidates)
    print(
        json.dumps(
            {
                "showcase_index": str(out),
                "report": str(report_out) if report_out else None,
                "runs": len(candidates),
                "best": candidates[0] if candidates else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _qa_thresholds(args: argparse.Namespace) -> dict[str, float]:
    mapping = {
        "min_ball_coverage": args.min_ball_coverage,
        "fail_ball_coverage": args.fail_ball_coverage,
        "max_ball_jump_px_frame": args.max_ball_jump_px_frame,
        "fail_ball_jump_px_frame": args.fail_ball_jump_px_frame,
        "max_out_of_bounds_ratio": args.max_out_of_bounds_ratio,
        "fail_out_of_bounds_ratio": args.fail_out_of_bounds_ratio,
        "max_unknown_team_ratio": args.max_unknown_team_ratio,
        "fail_unknown_team_ratio": args.fail_unknown_team_ratio,
    }
    return {key: value for key, value in mapping.items() if value is not None}


def cmd_track(args: argparse.Namespace) -> None:
    tracked = track_detections(
        read_detections(args.detections),
        iou_threshold=args.iou_threshold,
        max_age=args.max_age,
        backend=args.backend,
        frame_rate=args.frame_rate,
        track_activation_threshold=args.activation_threshold,
        minimum_matching_threshold=args.minimum_matching_threshold,
    )
    write_detections(args.out, tracked)
    print(json.dumps({"tracks_out": args.out, "detections": len(tracked)}, indent=2))


def cmd_assign_teams(args: argparse.Namespace) -> None:
    assigned = _assign_teams_for_process(
        args.video,
        read_detections(args.tracks),
        load_config(args.config),
    )
    write_detections(args.out, assigned)
    counts: dict[str, int] = {}
    for detection in assigned:
        if detection.class_name not in ROBOT_CLASSES:
            continue
        team = detection.team or "unknown"
        counts[team] = counts.get(team, 0) + 1
    print(
        json.dumps(
            {
                "out": args.out,
                "robot_samples_by_team": dict(sorted(counts.items())),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_assign_teams_marker(args: argparse.Namespace) -> None:
    hsv_lower = tuple(parse_int_list(args.hsv_lower))
    hsv_upper = tuple(parse_int_list(args.hsv_upper))
    if len(hsv_lower) != 3 or len(hsv_upper) != 3:
        raise ValueError("HSV bounds must contain exactly three integers")
    assigned, report = assign_marker_teams_from_video(
        args.video,
        read_detections(args.tracks),
        marker_team=args.marker_team,
        other_team=args.other_team,
        marker_ratio_threshold=args.marker_ratio_threshold,
        hsv_lower=hsv_lower,
        hsv_upper=hsv_upper,
        samples_per_track=args.samples_per_track,
        min_frame_gap=args.min_frame_gap,
    )
    write_detections(args.out, assigned)
    write_json(args.report_out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_assign_teams_embedding(args: argparse.Namespace) -> None:
    tracks = read_detections(args.tracks)
    existing_votes: dict[int, list[str]] = {}
    for detection in tracks:
        if (
            detection.class_name in ROBOT_CLASSES
            and detection.track_id is not None
            and detection.team in {"blue", "yellow"}
        ):
            existing_votes.setdefault(detection.track_id, []).append(detection.team)
    existing_team_by_track = {
        track_id: sorted(set(votes), key=lambda team: (-votes.count(team), team))[0]
        for track_id, votes in existing_votes.items()
    }
    embeddings, samples_by_track = extract_dinov2_track_embeddings(
        args.video,
        tracks,
        model_id=args.model_id,
        samples_per_track=args.samples_per_track,
        min_frame_gap=args.min_frame_gap,
        batch_size=args.batch_size,
        device=args.device,
    )
    cluster_by_track = cluster_track_embeddings(embeddings)
    cluster_to_team, mapping_metadata = align_clusters_to_teams(
        cluster_by_track,
        existing_team_by_track,
    )
    assigned = assign_embedding_teams(tracks, cluster_by_track, cluster_to_team)
    report = embedding_team_report(
        embeddings,
        samples_by_track,
        cluster_by_track,
        cluster_to_team,
        mapping_metadata,
        model_id=args.model_id,
    )
    write_detections(args.out, assigned)
    write_json(args.report_out, report)
    print(json.dumps({"tracks_out": args.out, "report_out": args.report_out, **report}, indent=2))


def cmd_events(args: argparse.Namespace) -> None:
    detections = _filter_by_game_state(read_detections(args.tracks), args.game_state)
    events = detect_events(
        detections,
        possession_radius_px=args.possession_radius_px,
        collision_radius_px=args.collision_radius_px,
        frame_width=args.frame_width,
        field_margin_px=args.in_play_field_margin_px,
    )
    write_events(args.out, events)
    summary = summarize_events(events)
    if args.summary_out:
        write_json(args.summary_out, summary)
    print(
        json.dumps(
            {
                "events_out": args.out,
                "summary_out": args.summary_out,
                "game_state": args.game_state,
                "events": len(events),
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_event_summary(args: argparse.Namespace) -> None:
    events = read_json(args.events)
    if not isinstance(events, list):
        raise ValueError(f"Expected JSON array: {args.events}")
    summary = summarize_events(events)
    write_json(args.out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_validate_goals(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    records = read_json(args.events)
    if not isinstance(records, list):
        raise ValueError(f"Expected events JSON array: {args.events}")
    events = [Event(**record) for record in records if isinstance(record, dict)]
    validated = _confirm_goals_for_process(
        read_detections(args.tracks),
        events,
        config.get("analysis", {}),
    )
    write_events(args.out, validated)
    summary = summarize_events(validated)
    print(
        json.dumps(
            {
                "out": args.out,
                "goal_candidates": summary["goals"]["total"],
                "goal_confirmed": summary["goals"]["confirmed"],
                "confirmed_scoreboard": summary["confirmed_scoreboard"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_team_quality(args: argparse.Namespace) -> None:
    report = analyze_team_quality_file(
        args.tracks,
        unknown_ratio_threshold=args.unknown_ratio_threshold,
        ambiguous_track_dominance_threshold=args.ambiguous_track_dominance,
        min_ambiguous_track_samples=args.min_ambiguous_track_samples,
        min_frame_team_coverage=args.min_frame_team_coverage,
        max_dominant_team_ratio=args.max_dominant_team_ratio,
        max_review_candidates=args.max_review_candidates,
    )
    write_json(args.out, report)
    markdown = None
    if args.report_out:
        markdown = write_team_quality_markdown(report, args.report_out)
    print(
        json.dumps(
            {
                "out": args.out,
                "report": str(markdown) if markdown else None,
                "summary": report["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_situation_analysis(args: argparse.Namespace) -> None:
    analysis = analyze_situations(
        read_detections(args.tracks),
        possession_radius_px=args.possession_radius_px,
        dispute_margin_px=args.dispute_margin_px,
        frame_width=args.frame_width,
    )
    write_json(args.out, analysis)
    print(
        json.dumps(
            {
                "out": args.out,
                "frames": analysis["summary"]["total_frames"],
                "frames_with_ball": analysis["summary"]["frames_with_ball"],
                "possession_states": analysis["summary"]["possession_states"],
                "average_action_probabilities": analysis["summary"][
                    "average_action_probabilities"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_game_state(args: argparse.Namespace) -> None:
    detections = read_detections(args.tracks)
    calibration = (
        load_field_calibration(args.field_calibration) if args.field_calibration else None
    )
    states = classify_frame_states(
        detections,
        possession_radius_px=args.possession_radius_px,
        field_margin_px=args.in_play_field_margin_px,
        missing_ball_frames=args.missing_ball_frames,
        robot_removed_after_frames=args.robot_removed_after_frames,
        robot_disabled_after_frames=args.robot_disabled_after_frames,
        stationary_threshold_px=args.stationary_threshold_px,
        field_polygon=calibration.image_points if calibration else None,
    )
    segments = detect_game_segments(states)
    external_events = detect_external_events(states)
    payload = {
        "schema": "samba_futbot.game_state.v1",
        "tracks": args.tracks,
        "summary": {
            "frames": len(states),
            "segments": len(segments),
            "external_events": len(external_events),
            "playable_frames": len(play_mask_from_segments(segments)),
        },
        "states": [state.to_record() for state in states],
        "segments": [segment.to_record() for segment in segments],
        "events": [event.to_record() for event in external_events],
    }
    write_json(args.out, payload)
    if args.segments_out:
        write_json(args.segments_out, [segment.to_record() for segment in segments])
    if args.events_out:
        write_events(args.events_out, external_events)
    print(
        json.dumps(
            {
                "out": args.out,
                "events_out": args.events_out,
                "segments_out": args.segments_out,
                **payload["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_metrics(args: argparse.Namespace) -> None:
    detections = _filter_by_game_state(read_detections(args.tracks), args.game_state)
    summary = summarize_tracks(
        detections,
        fps=args.fps,
        possession_radius_px=args.possession_radius_px,
        field_margin_px=args.in_play_field_margin_px,
    )
    if args.game_state:
        summary["game_state"] = {
            "path": args.game_state,
            "playable_detections": len(detections),
        }
    write_json(args.out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_render_demo(args: argparse.Namespace) -> None:
    out = render_demo_video(
        args.video,
        args.tracks,
        args.out,
        events_path=args.events,
        max_seconds=args.max_seconds,
        trail_length=args.trail_length,
        style=args.style,
        analysis_freeze=args.analysis_freeze,
        freeze_seconds=args.freeze_seconds,
        freeze_min_confidence=args.freeze_min_confidence,
        freeze_cooldown_frames=args.freeze_cooldown_frames,
        freeze_max_events=args.freeze_max_events,
        freeze_event_types=args.freeze_event_types,
        mask_overlay=args.mask_overlay,
        mask_alpha=args.mask_alpha,
        label_scale=args.label_scale,
        box_thickness=args.box_thickness,
        visual_hold_frames=args.visual_hold_frames,
        show_team_labels=args.show_team_labels,
    )
    print(json.dumps({"video": str(out), "style": args.style}, indent=2))


def cmd_render_heatmap(args: argparse.Namespace) -> None:
    calibration = (
        load_field_calibration(args.field_calibration) if args.field_calibration else None
    )
    result = render_activity_heatmap(
        args.video,
        read_detections(args.tracks),
        args.out_video,
        args.out_image,
        class_name=args.class_name,
        team=args.team,
        radius_px=args.radius_px,
        decay=args.decay,
        alpha=args.alpha,
        max_seconds=args.max_seconds,
        robot_fallback_min_area=args.robot_fallback_min_area,
        robot_fallback_max_area=args.robot_fallback_max_area,
        robot_fallback_max_extent=args.robot_fallback_max_extent,
        robot_fallback_max_aspect_ratio=args.robot_fallback_max_aspect_ratio,
        write_every_n_frames=args.write_every_n_frames,
        output_fps=args.output_fps,
        calibration=calibration,
        field_margin_m=args.field_margin_m,
    )
    if args.report_out:
        write_json(args.report_out, result)
        result["report"] = args.report_out
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_video_info(args: argparse.Namespace) -> None:
    print(json.dumps(video_info(args.video), indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
